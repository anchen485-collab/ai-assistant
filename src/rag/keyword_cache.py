"""混合检索的只读快照：分页加载，按已提交版本和业务日期失效。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from threading import Lock
from typing import Any

from src.rag.bm25 import BM25Index
from src.rag.governance import is_effective

PAGE_SIZE = 500
CACHE_SECONDS = 60


@dataclass(frozen=True)
class KeywordSnapshot:
    texts: tuple[str, ...]
    metadatas: tuple[dict[str, Any], ...]
    document_ids: tuple[str, ...]
    index: BM25Index


class KeywordCache:
    """每进程只保留一个快照，避免历史库常驻内存。

    调用者须持有知识库读写共享锁；锁内发布完整快照，查询只读。
    revision 用于同步后的即时失效，TTL 用于兜底，日期用于到期下线。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._key: tuple[Any, ...] | None = None
        self._snapshot: KeywordSnapshot | None = None
        self._expires_at = 0.0

    def get(self, store: Any, *, revision: str, today: date) -> KeywordSnapshot:
        if not isinstance(revision, str) or not revision:
            raise ValueError("关键词缓存需要已提交的索引版本")
        if not isinstance(today, date):
            raise ValueError("关键词缓存需要明确的业务日期")
        collection = store._collection
        with self._lock:
            count = collection.count()
            key = (str(store._persist_directory), str(collection.id), revision, today, count)
            if key == self._key and time.monotonic() < self._expires_at:
                assert self._snapshot is not None
                return self._snapshot

            texts, metadatas, document_ids, seen = [], [], set(), set()
            for offset in range(0, count, PAGE_SIZE):
                records = store.get(include=["documents", "metadatas"],
                                    limit=PAGE_SIZE, offset=offset)
                ids = records.get("ids") or []
                documents = records.get("documents") or []
                metadata_rows = records.get("metadatas")
                if metadata_rows is None:
                    metadata_rows = [None] * len(ids)
                if not (len(ids) == len(documents) == len(metadata_rows)):
                    raise ValueError("知识库返回的数据列未对齐")
                for chunk_id, text, metadata in zip(ids, documents, metadata_rows):
                    if chunk_id in seen:
                        raise ValueError("知识库快照包含重复 chunk ID")
                    seen.add(chunk_id)
                    if not isinstance(metadata, dict) or not is_effective(metadata, today):
                        continue
                    document_id = metadata.get("document_id")
                    if not isinstance(document_id, str) or not document_id:
                        continue
                    if not isinstance(text, str) or not text.strip():
                        continue
                    texts.append(text)
                    metadatas.append(dict(metadata))
                    document_ids.add(document_id)
            if len(seen) != count or collection.count() != count:
                raise ValueError("知识库在读取期间发生变化")
            # 先完整构建，再一起替换；构建失败时绝不以旧版本冒充新版本。
            snapshot = KeywordSnapshot(tuple(texts), tuple(metadatas),
                                       tuple(sorted(document_ids)), BM25Index(texts))
            self._snapshot, self._key = snapshot, key
            self._expires_at = time.monotonic() + CACHE_SECONDS
            return snapshot


keyword_cache = KeywordCache()
