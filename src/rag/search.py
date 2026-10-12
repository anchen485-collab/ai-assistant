from __future__ import annotations

"""检索模块：向量召回 + BM25 关键词召回 → RRF 融合 → Rerank 精排。"""

import logging
import math
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from requests.exceptions import RequestException

from src.core.config import settings
from src.rag.keyword_cache import keyword_cache
from src.rag.embeddings import _ensure_store as ensure_store
from src.rag.embeddings import get_index_revision, knowledge_read_context
from src.rag.governance import business_date, governance_sort_key, is_effective
from src.trace import log_step_ctx, source_refs

MMR_FETCH_MULTIPLIER = 4
MMR_LAMBDA = 0.6
RRF_K = 60
RERANK_TIMEOUT_SECONDS = 10
RERANK_MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)

def _valid_score(value: Any) -> bool:
    """分数必须是 0～1 的有限数字；拒绝缺失值、字符串和布尔值。"""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= value <= 1
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class SearchHit:
    """对外暴露的检索命中结构：文本 + 来源文件 + 片段序号 + 可选分数。"""
    text: str
    source: str
    chunk: int
    distance: float | None = None
    rerank_score: float | None = None
    governance: dict[str, Any] = field(default_factory=dict)
    # 只用于内部去重；避免不同文档的相同正文被错误合并。
    parent_id: str = ""


def serialize_hits(hits: list[SearchHit]) -> list[dict[str, Any]]:
    """将检索结果转为可序列化的字典列表。"""
    from dataclasses import asdict
    return [
        {key: value for key, value in asdict(hit).items() if key != "parent_id"}
        for hit in hits
    ]


def search(query: str, k: int | None = None) -> list[SearchHit]:
    """混合召回 → rerank 精排 → 阈值过滤 → 返回 top-K。"""
    if not isinstance(query, str):
        raise TypeError("query 必须为字符串")
    k = settings.retrieval_k if k is None else k
    if type(k) is not int:
        raise ValueError("k 必须为整数")
    threshold = settings.rerank_threshold
    if not _valid_score(threshold):
        raise ValueError("RERANK_THRESHOLD 必须是 0 到 1 之间的有限数字")
    query = query.strip()
    if not query or k <= 0:
        return []
    if type(settings.rerank_fetch_k) is not int or settings.rerank_fetch_k <= 0:
        raise ValueError("RERANK_FETCH_K 必须为正整数")
    limit = max(k, settings.rerank_fetch_k)
    recall_started = time.perf_counter()
    mmr_hits, bm25_hits, rrf_hits, candidates = _hybrid_recall(query, limit)
    log_step_ctx(
        "rag:recall",
        duration_ms=(time.perf_counter() - recall_started) * 1000,
        ok=bool(candidates),
        query=query,
        mmr=source_refs(mmr_hits),
        bm25=source_refs(bm25_hits),
        rrf=source_refs(rrf_hits),
        selected=source_refs(candidates),
    )
    if not candidates:
        log_step_ctx(
            "rag:quality_gate", query=query, ok=True, decision="blocked_no_candidates",
            threshold=threshold, rerank=[], selected=[],
        )
        return []

    started_at = time.perf_counter()
    # 全部候选先过相关性门槛，再按治理规则选 k 条，避免提前丢掉权威文档。
    ranked = _rerank(query, _sort_by_governance(candidates), top_n=len(candidates))
    rerank_ms = (time.perf_counter() - started_at) * 1000
    if ranked is None:
        selected = _sort_by_governance(candidates)[:k]
        log_step_ctx(
            "rag:quality_gate", query=query,
            ok=False,
            duration_ms=rerank_ms,
            decision="rerank_unavailable_use_governed_rrf",
            quality_gate_applied=False,
            threshold=threshold, rerank=[], selected=source_refs(selected),
        )
        return selected

    if not ranked:
        log_step_ctx(
            "rag:quality_gate", query=query, ok=False, duration_ms=rerank_ms,
            decision="blocked_empty_or_invalid_rerank", threshold=threshold, rerank=[], selected=[],
        )
        return []

    filtered = [hit for hit in ranked if hit.rerank_score >= threshold]
    if not filtered:
        highest_score = max(hit.rerank_score for hit in ranked)
        logger.info(
            "Rerank 最高分 %.2f 未达到阈值 %.2f，拦截低质量候选。",
            highest_score, threshold,
        )
        log_step_ctx(
            "rag:quality_gate", query=query, ok=True, duration_ms=rerank_ms,
            decision="blocked_low_rerank_score", threshold=threshold,
            highest_score=highest_score, rerank=source_refs(ranked), selected=[],
        )
        return []

    selected = _sort_by_governance(filtered)[:k]
    log_step_ctx(
        "rag:quality_gate", query=query,
        ok=True,
        duration_ms=rerank_ms,
        decision="accepted",
        threshold=threshold,
        rerank=source_refs(ranked),
        selected=source_refs(selected),
    )
    return selected


def _mmr_recall(query: str, *, store=None, revision=None, today=None,
                limit: int | None = None) -> list[SearchHit]:
    """先用有效文档 ID 筛选，再做 MMR，避免失效文档占满候选池。"""
    limit = settings.rerank_fetch_k if limit is None else limit
    if not query.strip() or limit <= 0:
        return []
    if store is None:
        with knowledge_read_context() as current:
            return _mmr_recall(query, store=current, revision=get_index_revision(current),
                               today=business_date(), limit=limit)
    snapshot = keyword_cache.get(store, revision=revision, today=today)
    if not snapshot.document_ids:
        return []
    results = store.max_marginal_relevance_search(
        query,
        k=limit * MMR_FETCH_MULTIPLIER,
        fetch_k=limit * MMR_FETCH_MULTIPLIER,
        lambda_mult=MMR_LAMBDA,
        filter={"document_id": {"$in": list(snapshot.document_ids)}},
    )
    hits = [_to_hit(doc.page_content, doc.metadata) for doc in results
            if doc.page_content.strip() and is_effective(doc.metadata, today)]
    return _rrf_merge(hits, limit=limit)


def _bm25_recall(query: str, *, store=None, revision=None, today=None,
                 limit: int | None = None) -> list[SearchHit]:
    """关键词分数只在有效文档中计算；同一父块只保留一次。"""
    limit = settings.rerank_fetch_k if limit is None else limit
    if not query.strip() or limit <= 0:
        return []
    if store is None:
        with knowledge_read_context() as current:
            return _bm25_recall(query, store=current, revision=get_index_revision(current),
                                today=business_date(), limit=limit)
    snapshot = keyword_cache.get(store, revision=revision, today=today)
    indices = snapshot.index.top_indices(query, len(snapshot.texts))
    hits = [_to_hit(snapshot.texts[index], snapshot.metadatas[index]) for index in indices]
    return _rrf_merge(hits, limit=limit)


def _rrf_merge(*groups: list[SearchHit], limit: int | None = None) -> list[SearchHit]:
    """同一文档父块每路只计分一次；不同文档的相同文本保留独立来源。"""
    limit = settings.rerank_fetch_k if limit is None else limit
    if limit <= 0:
        return []
    scores, hits = {}, {}
    for group in groups:
        seen = set()
        for hit in group:
            document_id = hit.governance.get("document_id")
            # 优先保留 Small-to-Big 的父块身份；正文相同不代表来自同一文档。
            if hit.parent_id:
                key = ("parent", str(document_id or ""), hit.parent_id)
            elif document_id:
                key = ("document", str(document_id), hit.chunk)
            else:
                key = ("legacy", hit.source, hit.chunk, hit.text)
            if not hit.text.strip() or key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + len(seen))
            hits.setdefault(key, hit)
    keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [hits[key] for key in keys[:limit]]


def _to_hit(text: str, metadata: dict) -> SearchHit:
    """把 Chroma 文档统一转换为项目对外使用的 SearchHit。"""

    metadata = metadata or {}
    source_path = metadata.get("source_path") or metadata.get("source", "")
    parent_text = metadata.get("parent_text")
    has_parent = isinstance(parent_text, str) and bool(parent_text.strip())
    chunk = _chunk_number(metadata.get("chunk"))
    if has_parent:
        chunk = _chunk_number(metadata.get("parent_chunk")) or chunk
    text = parent_text if has_parent else text
    parent_id = metadata.get("parent_id") if has_parent else None
    if not isinstance(parent_id, str) or not parent_id:
        # 兼容没有 parent_id 的旧索引；保留完整路径，区分同名文件。
        parent_id = json.dumps([str(source_path), chunk, text], ensure_ascii=False)
    return SearchHit(
        text=text,
        source=Path(str(source_path)).name if source_path else "未知来源",
        chunk=chunk,
        parent_id=parent_id,
        governance={
            key: metadata.get(key, "")
            for key in (
                "document_id", "knowledge_id", "owner", "authority_rank", "version", "status",
                "published_at", "effective_from", "effective_to", "region",
                "product", "source_url",
            )
        },
    )


def _chunk_number(value: Any) -> int:
    """旧索引中的缺失/脏编号回退为 0，避免一次查询整体失败。"""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        return 0


def _filter_effective(hits: list[SearchHit]) -> list[SearchHit]:
    """先排除已停用或未在生效期内的文档。"""

    return [hit for hit in hits if is_effective(hit.governance)]


def _sort_by_governance(hits: list[SearchHit]) -> list[SearchHit]:
    """在精排结果内，优先权威等级和版本，再保留语义分数顺序。"""

    return sorted(
        hits,
        key=lambda hit: (
            *governance_sort_key(hit.governance),
            hit.rerank_score or 0,
        ),
        reverse=True,
    )


def _rerank(query: str, candidates: list[SearchHit], top_n: int) -> list[SearchHit] | None:
    """返回精排结果；[] 表示没有可信结果，None 表示服务不可用、沿用降级。"""
    if not candidates:
        return []
    if not settings.dashscope_api_key:
        logger.warning("未配置 DASHSCOPE_API_KEY，跳过 rerank")
        return None

    from dashscope.rerank import TextReRank

    for attempt in range(RERANK_MAX_ATTEMPTS):
        try:
            response = TextReRank.call(
                model=settings.rerank_model,
                query=query,
                documents=[hit.text for hit in candidates],
                top_n=min(top_n, len(candidates)),
                return_documents=False,
                api_key=settings.dashscope_api_key,
                request_timeout=RERANK_TIMEOUT_SECONDS,
            )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            # SDK 可能在构造响应对象时就报错；这不是服务不可用，不能放行原候选。
            logger.warning("Rerank 参数或响应格式异常，拦截本次结果：%s", type(exc).__name__)
            return []
        except (RequestException, TimeoutError, ConnectionError) as exc:
            # 不记录异常正文：第三方异常可能包含请求文本或密钥。
            logger.warning("Rerank 网络异常：%s", type(exc).__name__)
        except Exception as exc:
            logger.warning("Rerank 未知异常，拦截本次结果：%s", type(exc).__name__)
            return []  # 未确认是网络/服务故障时，不把未知错误当成降级理由。
        else:
            status = getattr(response, "status_code", None)
            if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
                logger.warning("Rerank 响应状态无效，拦截本次结果")
                return []
            if status == 200:
                return _parse_rerank_results(getattr(response, "output", None), candidates)
            logger.warning("Rerank API 返回 HTTP %s", status)
            if status not in (408, 429) and not 500 <= status <= 599:
                return None  # 例如鉴权失败，保留降级，但不要重复请求。

        # 只重试网络异常、超时、限流和 5xx；最多 3 次，间隔 0.5、1 秒。
        if attempt + 1 < RERANK_MAX_ATTEMPTS:
            time.sleep(0.5 * (2 ** attempt))

    logger.warning("Rerank 重试耗尽，使用治理后的 RRF")
    return None


def _parse_rerank_results(output: Any, candidates: list[SearchHit]) -> list[SearchHit]:
    """校验服务返回值，避免错误索引或异常分数绕过质量门槛。"""
    results = output.get("results") if isinstance(output, Mapping) else None
    if not isinstance(results, list):
        logger.warning("Rerank 响应缺少有效 results，拦截本次结果")
        return []

    ranked: list[SearchHit] = []
    seen: set[int] = set()
    for item in results:
        if not isinstance(item, Mapping):
            continue
        index, score = item.get("index"), item.get("relevance_score")
        if (
            not isinstance(index, int) or isinstance(index, bool)
            or not 0 <= index < len(candidates)
            or not _valid_score(score) or index in seen
        ):
            continue
        seen.add(index)
        # 只更新分数：正文、来源、父片段和治理元数据始终沿用本地候选。
        ranked.append(replace(candidates[index], rerank_score=float(score)))

    if len(ranked) != len(results):
        logger.warning("Rerank 丢弃 %d 条异常或重复结果", len(results) - len(ranked))
    return sorted(ranked, key=lambda hit: hit.rerank_score, reverse=True)


def _hybrid_recall(query: str, limit: int):
    """锁内两路读同一知识库版本；模型精排在释放锁后执行。"""
    today = business_date()
    with knowledge_read_context() as store:
        revision = get_index_revision(store)
        groups, errors = [], []
        for name, recall in (("MMR", _mmr_recall), ("BM25", _bm25_recall)):
            try:
                groups.append(recall(query, store=store, revision=revision, today=today, limit=limit))
            except Exception as exc:
                logger.warning("%s 召回失败，保留另一路结果：%s", name, type(exc).__name__)
                groups.append([])
                errors.append(exc)
        if len(errors) == 2:
            raise RuntimeError("向量与关键词召回均失败") from errors[0]
        mmr_hits, bm25_hits = groups
        rrf_hits = _rrf_merge(mmr_hits, bm25_hits, limit=limit)
        candidates = [hit for hit in rrf_hits if is_effective(hit.governance, today)]
    return mmr_hits, bm25_hits, rrf_hits, candidates
