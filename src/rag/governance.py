from __future__ import annotations

"""知识治理：严格校验配置，异常时停止检索，而不是重新启用失效知识。"""

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

METADATA_FILE_NAME = "knowledge_metadata.json"
# 业务日期固定使用北京时间，避免服务器部署在 UTC 时跨日判断不同。
BUSINESS_TIMEZONE = timezone(timedelta(hours=8))
DEFAULT_METADATA = {
    "knowledge_id": "",
    "owner": "",
    "authority_rank": 0,
    "version": "",
    "status": "active",
    "published_at": "",
    "effective_from": "",
    "effective_to": "",
    "region": "",
    "product": "",
    "source_url": "",
}
STATUSES = {"active", "inactive", "archived", "draft"}
DATE_FIELDS = ("published_at", "effective_from", "effective_to")


class MetadataError(ValueError):
    """配置错误；调用方应提示维护配置，不得回退到不带治理的检索。"""


def business_date() -> date:
    return datetime.now(BUSINESS_TIMEZONE).date()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetadataError(f"配置出现重复键：{key}")
        result[key] = value
    return result


def load_metadata(source_dir: Path) -> dict[str, Any]:
    """一次读取并校验整份配置；缺失/损坏均报错，保留原索引。"""
    path = source_dir / METADATA_FILE_NAME
    try:
        content = json.loads(
            path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise MetadataError(f"知识治理配置读取失败：{path.name}：{exc}") from exc
    if not isinstance(content, dict) or set(content) - {"default", "documents"}:
        raise MetadataError("配置根节点应为对象，且只接受 default 和 documents")
    default = _validate_fields(content.get("default", {}), "default")
    if "knowledge_id" in default:
        raise MetadataError("knowledge_id 应逐文档配置，不应放在 default 中")
    # 即使没有任何文档，也要校验默认日期区间。
    _merge_metadata(Path("default"), default, {})
    documents = content.get("documents", {})
    if not isinstance(documents, dict):
        raise MetadataError("documents 应为文件名到配置对象的映射")
    normalized: dict[str, dict[str, Any]] = {}
    ids: set[str] = set()
    for filename, overrides in documents.items():
        if (
            not filename or "/" in filename or "\\" in filename
            or Path(filename).suffix.lower() not in {".txt", ".md", ".docx"}
        ):
            raise MetadataError(f"documents 中需要完整文件名（含扩展名）：{filename}")
        fields = _validate_fields(overrides, f"documents.{filename}")
        metadata = _merge_metadata(Path(filename), default, fields)
        if metadata["knowledge_id"] in ids:
            raise MetadataError(f"knowledge_id 重复：{metadata['knowledge_id']}")
        ids.add(metadata["knowledge_id"])
        normalized[filename] = fields
    return {"default": default, "documents": normalized}


def _validate_fields(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetadataError(f"{location} 应为对象")
    fields = dict(value)
    # 兼容上一版 JSON 的业务 document_id；内部同步 document_id 由入库代码生成。
    if "document_id" in fields:
        legacy_id = fields.pop("document_id")
        if "knowledge_id" in fields and fields["knowledge_id"] != legacy_id:
            raise MetadataError(f"{location} 的 document_id 与 knowledge_id 冲突")
        fields["knowledge_id"] = legacy_id
    unknown = set(fields) - set(DEFAULT_METADATA)
    if unknown:
        raise MetadataError(f"{location} 有未知字段：{', '.join(sorted(unknown))}")
    for key, item in fields.items():
        if key == "authority_rank":
            if type(item) is not int or item < 0:
                raise MetadataError(f"{location}.authority_rank 应为非负整数")
        else:
            if not isinstance(item, str):
                raise MetadataError(f"{location}.{key} 应为字符串")
            fields[key] = item.strip()
    if "status" in fields:
        fields["status"] = fields["status"].lower()
        if fields["status"] not in STATUSES:
            raise MetadataError(f"{location}.status 应为 active/inactive/archived/draft")
    if "version" in fields:
        version_key(fields["version"])
    for key in DATE_FIELDS:
        if key in fields:
            _parse_date(fields[key])
    url = fields.get("source_url", "")
    if url:
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"https", "http"} and bool(parsed.hostname)
        except ValueError:
            valid = False
        if not valid:
            raise MetadataError(f"{location}.source_url 应为 http/https 原文链接或空字符串")
    return fields


def _merge_metadata(path: Path, default: dict, overrides: dict) -> dict[str, Any]:
    metadata = {**DEFAULT_METADATA, **default, **overrides}
    metadata["knowledge_id"] = metadata["knowledge_id"] or path.name
    start = _parse_date(metadata["effective_from"])
    end = _parse_date(metadata["effective_to"])
    if start and end and start > end:
        raise MetadataError(f"{path.name} 的 effective_from 晚于 effective_to")
    return metadata


def metadata_for_document(path: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    """逐文档覆盖默认配置，返回适合写入每个 Chunk 的标量字段。"""
    default = _validate_fields(catalog.get("default", {}), "default")
    documents = catalog.get("documents", {})
    if not isinstance(documents, dict):
        raise MetadataError("documents 应为对象")
    overrides = _validate_fields(documents.get(path.name, {}), path.name)
    return _merge_metadata(path, default, overrides)


def is_effective(metadata: dict[str, Any], today: date | None = None) -> bool:
    """缺状态、坏日期、坏排序字段都排除；结束日期当天仍有效。"""
    if not isinstance(metadata, dict) or metadata.get("status") != "active":
        return False
    try:
        start = _parse_date(metadata.get("effective_from", ""))
        end = _parse_date(metadata.get("effective_to", ""))
        governance_sort_key(metadata)
    except MetadataError:
        return False
    current = today or business_date()
    return (start is None or start <= current) and (end is None or current <= end)


def governance_sort_key(metadata: dict[str, Any]) -> tuple:
    """治理优先级；语义相关性阈值在调用方先行判断。"""
    rank = metadata.get("authority_rank", 0)
    if type(rank) is not int or rank < 0:
        raise MetadataError("authority_rank 应为非负整数")
    start = _parse_date(metadata.get("effective_from", ""))
    return rank, version_key(metadata.get("version", "")), start or date.min


def version_key(value: Any) -> tuple[int, ...]:
    """支持 1、1.2、v1.10；1.0 与 1.0.0 等价，不猜测自定义版本顺序。"""
    if not isinstance(value, str):
        raise MetadataError("version 应为字符串")
    if not value:
        return (0,)
    if not re.fullmatch(r"[vV]?[0-9]+(?:\.[0-9]+)*", value) or len(value) > 100:
        raise MetadataError("version 应为数字点分版本，例如 1.0 或 v1.10")
    parts = [int(part) for part in value.lstrip("vV").split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _parse_date(value: Any) -> date | None:
    if value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise MetadataError("日期应为 YYYY-MM-DD 或空字符串")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise MetadataError(f"无效日期：{value}") from exc
