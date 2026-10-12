from __future__ import annotations

"""单进程、有界内存 Trace。重启清空；查询返回快照，不暴露内部可变对象。"""

import contextvars
import json
import logging
import math
import threading
import time
from itertools import islice
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# 小型调试缓存，不作为审计数据库。限制记录数量、生命周期及序列化载荷。
_max_traces = 500
_max_steps = 128
_max_step_bytes = 16 * 1024
_max_trace_bytes = 256 * 1024
_max_store_bytes = 8 * 1024 * 1024
_ttl_seconds = 3600
_max_sources = 64
_store: dict[str, dict[str, Any]] = {}
_store_bytes = 0
_lock = threading.Lock()
_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)
_SECRET_FIELDS = {"authorization", "api_key", "apikey", "token", "password", "secret", "cookie"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def source_refs(sources: list[Any] | None) -> list[dict[str, Any]]:
    """只取来源定位信息；坏元数据不应打断回答，NaN/Infinity 不进入 JSON。"""
    refs = []
    try:
        for rank, item in enumerate(islice(sources or [], _max_sources), start=1):
            try:
                get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
                try:
                    chunk = int(get("chunk", 0) or 0)
                except (ValueError, TypeError, OverflowError):
                    chunk = 0
                ref = {"rank": rank, "source": str(get("source") or "未知来源")[:512], "chunk": chunk}
                for field in ("rerank_score", "distance"):
                    number = _number(get(field))
                    if number is not None:
                        ref[field] = round(number, 4)
                refs.append(ref)
            except Exception:
                refs.append({"rank": rank, "source": "未知来源", "chunk": 0, "metadata_invalid": True})
    except Exception:
        pass  # 可观测性辅助函数始终保持尽力记录，不影响业务。
    return refs


def _safe_value(value: Any, depth: int = 0, budget: list[int] | None = None) -> Any:
    """拷贝为有界的 JSON 基本类型；不调用未知对象的 repr，也不展开无限嵌套。"""
    if budget is None:
        budget = [512]
    budget[0] -= 1
    if budget[0] < 0 or depth > 5:
        return "[truncated]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value if len(value) <= 2048 else value[:2048] + "[truncated]"
    if isinstance(value, int):
        return value if value.bit_length() <= 64 else "[large integer]"
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        result = {}
        for key, item in islice(value.items(), 64):
            if not isinstance(key, str):
                continue
            result[key[:128]] = ("[redacted]" if key.lower() in _SECRET_FIELDS
                                else _safe_value(item, depth + 1, budget))
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth + 1, budget) for item in value[:64]]
    return "[unsupported]"


def _info(message: str, *args: Any) -> None:
    try:
        logger.info(message, *args)
    except Exception:
        pass  # 日志输出故障不向聊天链路传播。


def _drop(trace_id: str) -> None:
    global _store_bytes
    _store_bytes -= _store.pop(trace_id)["payload_bytes"]


def _trim() -> None:
    """调用方持锁；TTL 使用单调时钟，不受系统时间校准影响。"""
    now = time.perf_counter()
    for tid, record in list(_store.items()):
        if now - record["started_clock"] >= _ttl_seconds:
            _drop(tid)
    while len(_store) > _max_traces or _store_bytes > _max_store_bytes:
        _drop(next(iter(_store)))


def new_trace() -> str:
    trace_id = uuid4().hex[:16]
    with _lock:
        _store[trace_id] = {
            "trace_id": trace_id, "started_clock": time.perf_counter(),
            "steps": [], "payload_bytes": 0, "dropped_steps": 0,
        }
        _trim()
    set_trace(trace_id)
    _info("trace_start: trace_id=%s", trace_id)
    return trace_id


def set_trace(trace_id: str) -> None:
    _current_trace_id.set(trace_id)


def clear_trace() -> None:
    _current_trace_id.set(None)


def log_step(trace_id: str, step: str, ok: bool = True, duration_ms: float = 0, **extra: Any) -> None:
    """尽力记录：限流、序列化或日志异常都不应导致聊天失败。"""
    global _store_bytes
    try:
        entry = _safe_value(extra)
        entry.update(step=str(step)[:128].replace("\n", " ").replace("\r", " "), ok=bool(ok), duration_ms=round(max(0, _number(duration_ms) or 0), 1))
        payload = json.dumps(entry, ensure_ascii=False, allow_nan=False)
        size = len(payload.encode("utf-8"))
        if size > _max_step_bytes:
            entry = {key: entry[key] for key in ("step", "ok", "duration_ms")}
            entry["payload_truncated"] = True
            payload = json.dumps(entry, ensure_ascii=False)
            size = len(payload.encode("utf-8"))
        with _lock:
            _trim()
            record = _store.get(trace_id)
            if record is None or "total_ms" in record:
                return
            if len(record["steps"]) >= _max_steps or record["payload_bytes"] + size > _max_trace_bytes:
                record["dropped_steps"] += 1
                return
            record["steps"].append(payload)
            record["payload_bytes"] += size
            _store_bytes += size
            _trim()
        # 正文/问题/错误详情仅进入受保护的内存查询，不复制进常规 INFO 日志。
        _info("trace_step: trace_id=%s step=%s ok=%s duration_ms=%.1f", trace_id,
              entry["step"], entry["ok"], entry["duration_ms"])
    except Exception:
        pass


def log_step_ctx(step: str, ok: bool = True, duration_ms: float = 0, **extra: Any) -> None:
    trace_id = _current_trace_id.get()
    if trace_id is not None:
        log_step(trace_id, step, ok=ok, duration_ms=duration_ms, **extra)


def _total_ms(record: dict[str, Any]) -> float:
    if "total_ms" in record:
        return record["total_ms"]
    return round(max(0, time.perf_counter() - record["started_clock"]) * 1000, 1)


def finish_trace(trace_id: str) -> None:
    """幂等结束；清理自己的上下文，即使记录已被缓存淘汰。"""
    with _lock:
        record = _store.get(trace_id)
        if record is not None and "total_ms" not in record:
            record["total_ms"] = _total_ms(record)
    if _current_trace_id.get() == trace_id:
        clear_trace()
    _info("trace_end: trace_id=%s", trace_id)


def get_trace(trace_id: str) -> dict[str, Any] | None:
    with _lock:
        _trim()
        record = _store.get(trace_id)
        if record is None:
            return None
        # 在锁内获得一致快照，返回内容与缓存没有共享的 list/dict。
        return {"trace_id": trace_id, "total_ms": _total_ms(record),
                "steps": [json.loads(step) for step in record["steps"]],
                "dropped_steps": record["dropped_steps"]}


def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    limit = max(0, min(int(limit), 100))
    if limit == 0:
        return []
    with _lock:
        _trim()
        return [{"trace_id": tid, "total_ms": _total_ms(record)}
                for tid, record in list(_store.items())[-limit:][::-1]]
