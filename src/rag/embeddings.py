from __future__ import annotations

"""向量库同步：只重建新增、修改或删除的单份知识库文档。"""

import hashlib
import json
import logging
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.rag.sync_state import sync_lock, write_json_atomic

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

if "onnxruntime" not in sys.modules:
    # Chroma 导入时会初始化默认 ONNX embedding，但本项目实际使用 DashScopeEmbeddings。
    stub = types.ModuleType("onnxruntime")
    stub.get_available_providers = lambda: []
    stub.get_all_providers = lambda: []
    sys.modules["onnxruntime"] = stub

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from src.core.config import Settings, settings
from src.rag.documents import DocxIngestor, KnowledgeDocument
from src.rag.engine import build_chroma

logger = logging.getLogger(__name__)
_MANIFEST_NAME = "knowledge_manifest.json"


def _chunk_ids(document_id: str, content_hash: str, count: int) -> list[str]:
    # 不覆盖旧版 ID：新版完整写入后，才移除旧版。
    return [f"{document_id}:{content_hash}:{index}" for index in range(count)]


class VectorStoreManager:
    """单机持久化知识库同步；锁和 pending 清单负责并发及重试。"""

    def __init__(self, config: Settings, embeddings: DashScopeEmbeddings) -> None:
        self.__config = config
        self._embeddings = embeddings

    @property
    def _manifest_path(self) -> Path:
        return self.__config.chroma_dir / _MANIFEST_NAME

    def _scope(self) -> dict:
        model = getattr(self._embeddings, "model", "")
        return {
            "collection": self.__config.chroma_collection,
            "source_dir": str(self.__config.kb_source_dir.resolve()),
            "embedding": f"{type(self._embeddings).__module__}.{type(self._embeddings).__name__}:{model}",
        }

    def sync(self, ingestor: DocxIngestor) -> Chroma:
        """失败抛异常，不把尚未同步成功的 store 返回给检索调用方。"""
        with self.read_context(ingestor) as store:
            return store

    @contextmanager
    def read_context(self, ingestor: DocxIngestor):
        """让检索读期与同步共用同一把锁，读者不看到新旧版本混合。"""
        with sync_lock(self._manifest_path.with_suffix(".lock")):
            yield self._sync_locked(ingestor)

    def _sync_locked(self, ingestor: DocxIngestor) -> Chroma:
        current = {item.id: item for item in ingestor.source_documents()}
        state = self._read_manifest()
        store = build_chroma(
            self.__config.chroma_collection,
            self.__config.chroma_dir,
            embedding_function=self._embeddings,
        )
        existing = self._inventory(store)
        if existing and not self._manifest_path.exists():
            raise RuntimeError("向量库已有数据但同步清单丢失；请恢复匹配的清单或使用新的 CHROMA_DIR")
        known = set(state["documents"]) | set(current) | {state["pending"]}
        if set(existing) - known:
            raise RuntimeError("向量库含清单之外的文档；请检查目录与清单是否匹配")

        # 包含 pending：即使进程上次在数据库写入后退出，也会重新核对该文档。
        removed = (set(state["documents"]) | {state["pending"]}) - set(current) - {None}
        for document_id in sorted(removed):
            state["pending"] = document_id
            self._write_manifest(state)
            store.delete(where={"document_id": document_id})
            if store.get(where={"document_id": document_id}, include=[])["ids"]:
                raise RuntimeError(f"文档删除未完成: {document_id}")
            state["documents"].pop(document_id, None)
            state["pending"] = None
            self._write_manifest(state)

        for document_id, item in sorted(current.items()):
            record = state["documents"].get(document_id, {})
            count = record.get("chunks", 0)
            old_ids = existing.get(document_id, set())
            if (state["pending"] != document_id
                    and record.get("hash") == item.content_hash
                    and count > 0 and count == len(old_ids)
                    and old_ids == set(_chunk_ids(document_id, item.content_hash, count))):
                continue
            self._sync_document(store, ingestor, item, state, old_ids)

        # 空知识库首次同步也创建清单；未变化时不反复写文件。
        if not self._manifest_path.exists():
            self._write_manifest(state)
        store._knowledge_revision = hashlib.sha256(
            json.dumps({"scope": state["scope"], "documents": state["documents"]},
                       sort_keys=True).encode("utf-8")
        ).hexdigest()
        return store

    def _sync_document(
        self, store: Chroma, ingestor: DocxIngestor, item: KnowledgeDocument,
        state: dict, old_ids: set[str],
    ) -> None:
        chunks = ingestor.ingest_document(item)
        if not chunks or any(not chunk.page_content.strip() for chunk in chunks):
            raise ValueError(f"文档未生成有效 chunk，保留原版本并等待重试: {item.path}")
        ingestor.assert_current(item)
        ids = _chunk_ids(item.id, item.content_hash, len(chunks))
        desired = set(ids)
        state["pending"] = item.id
        self._write_manifest(state)

        try:
            # add_documents 内部使用 upsert；相同版本重试不会产生重复 ID。
            for start in range(0, len(chunks), 64):
                store.add_documents(documents=chunks[start:start + 64], ids=ids[start:start + 64])
            written = set(store.get(where={"document_id": item.id}, include=[])["ids"])
            if not desired.issubset(written):
                raise RuntimeError(f"新版本 chunk 写入不完整: {item.path}")
            ingestor.assert_current(item)
        except Exception:
            # 新版准备失败，旧版保持原样；保留 pending 应对清理再次失败。
            try:
                self._delete_ids(store, sorted(desired - old_ids))
            except Exception:
                logger.exception("清理未完成的新版失败，下次同步继续恢复: %s", item.path)
            raise

        self._delete_ids(store, sorted(written - desired))
        actual = set(store.get(where={"document_id": item.id}, include=[])["ids"])
        if actual != desired:
            raise RuntimeError(f"旧版本清理未完成: {item.path}")
        state["documents"][item.id] = {"hash": item.content_hash, "chunks": len(chunks)}
        state["pending"] = None
        self._write_manifest(state)
        logger.info("知识库文档已同步: %s, chunks=%s", item.path.name, len(chunks))

    @staticmethod
    def _delete_ids(store, ids: list[str]) -> None:
        for start in range(0, len(ids), 64):
            store.delete(ids=ids[start:start + 64])

    @staticmethod
    def _inventory(store) -> dict[str, set[str]]:
        # 分页核对 ID 与归属。metadata 含父块文本，故大库仍需评估扫描开销。
        result: dict[str, set[str]] = {}
        offset = 0
        while True:
            page = store.get(include=["metadatas"], limit=1000, offset=offset)
            for chunk_id, metadata in zip(page["ids"], page["metadatas"]):
                document_id = (metadata or {}).get("document_id")
                if not isinstance(document_id, str) or not document_id:
                    raise RuntimeError("集合含未标记 document_id 的记录，请为知识库使用独立集合")
                result.setdefault(document_id, set()).add(chunk_id)
            if len(page["ids"]) < 1000:
                return result
            offset += len(page["ids"])

    def _read_manifest(self) -> dict:
        state = {"schema_version": 2, "scope": self._scope(), "documents": {}, "pending": None}
        if not self._manifest_path.exists():
            return state
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("documents"), dict):
                raise ValueError("清单格式错误")
            if "schema_version" not in data:
                # 原版 {document_id: hash} 清单：保留归属，逐篇迁移至版本化 ID。
                for document_id, content_hash in data["documents"].items():
                    if not document_id or not isinstance(content_hash, str) or not content_hash:
                        raise ValueError("旧版清单记录错误")
                    state["documents"][document_id] = {"hash": content_hash, "chunks": 0}
                return state
            if data.get("schema_version") != 2 or data.get("scope") != self._scope():
                raise ValueError("清单版本、来源目录、集合或 Embedding 模型不匹配，请使用独立 CHROMA_DIR")
            if "pending" not in data or (data["pending"] is not None and
                    (not isinstance(data["pending"], str) or not data["pending"])):
                raise ValueError("清单 pending 格式错误")
            for document_id, record in data["documents"].items():
                if (not document_id or not isinstance(record, dict)
                        or not isinstance(record.get("hash"), str) or not record["hash"]
                        or type(record.get("chunks")) is not int or record["chunks"] < 0):
                    raise ValueError("清单文档记录错误")
            return data
        except (ValueError, OSError) as exc:
            # 不把损坏清单当成空库继续写，避免漏清理、误删和静默重复。
            raise RuntimeError(f"知识库同步清单读取失败: {self._manifest_path}: {exc}") from exc

    def _write_manifest(self, state: dict) -> None:
        write_json_atomic(self._manifest_path, state)


_default_manager: Optional[VectorStoreManager] = None


def _manager() -> VectorStoreManager:
    """惰性单例：避免导入时立刻创建 Chroma 客户端。"""

    global _default_manager
    if _default_manager is None:
        _default_manager = VectorStoreManager(settings, DashScopeEmbeddings())
    return _default_manager


@contextmanager
def knowledge_read_context():
    """检索用入口：两路召回应在同一个上下文内读取向量库。"""
    with _manager().read_context(DocxIngestor(settings)) as store:
        yield store


def get_index_revision(store: Chroma) -> str:
    """仅在 knowledge_read_context 内读取，用于检索缓存失效判断。"""
    return store._knowledge_revision


def _ensure_store() -> Chroma:
    """兼容入口；同步校验后返回 store，完整检索读期请使用 knowledge_read_context。"""

    return _manager().sync(DocxIngestor(settings))


def ingest() -> dict:
    """手动触发一次知识库同步，返回当前 collection 的基本信息。"""

    with knowledge_read_context() as store:
        return {
            "collection": store._collection.name,
            "chunks": store._collection.count(),
            "path": str(settings.chroma_dir),
        }


def ingest_new() -> dict:
    """兼容旧调用方；当前实现会自动增量同步。"""

    return ingest()


