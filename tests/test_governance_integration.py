"""使用临时 Chroma 和本地假向量验证完整入库；零外部 API 调用。"""

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch

# 先导入同步模块，复用项目对 Chroma 默认 ONNX 依赖的处理。
from src.rag.embeddings import VectorStoreManager
from langchain_core.embeddings import Embeddings
from src.core.config import Settings
from src.rag.documents import DocxIngestor
from src.rag.governance import MetadataError, is_effective


class LocalEmbeddings(Embeddings):
    """只用于测试：不加载模型、不请求网络。"""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [byte / 255 for byte in digest[:8]]


class GovernanceIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Windows 的 Chroma 可能暂时保留文件句柄；测试数据仅位于系统临时目录。
        self.directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.config = Settings()
        self.config.kb_source_dir = root / "source"
        self.config.kb_source_dir.mkdir()
        self.config.chroma_dir = root / "chroma"
        self.config.chroma_collection = "governance_integration"
        self.config.chunk_size = 300
        self.config.chunk_overlap = 30
        self.document = self.config.kb_source_dir / "guide.txt"
        self.document.write_text("平台操作：先核实用户信息，再提交申请。\n" * 120, encoding="utf-8")
        self.metadata_path = self.config.kb_source_dir / "knowledge_metadata.json"
        self.write_config("active")
        self.ingestor = DocxIngestor(self.config)
        self.manager = VectorStoreManager(self.config, LocalEmbeddings())

    def write_config(self, status):
        self.metadata_path.write_text(json.dumps({
            "default": {"owner": "test-team", "authority_rank": 80, "version": "1.2"},
            "documents": {"guide.txt": {"knowledge_id": "guide-business-id", "status": status}},
        }), encoding="utf-8")

    def records(self, store):
        return store._collection.get(include=["metadatas", "documents"])

    def test_all_chunks_keep_both_ids_and_document_metadata(self):
        store = self.manager.sync(self.ingestor)
        records = self.records(store)
        self.assertGreater(len(records["ids"]), 1)
        internal_ids = set()
        for metadata in records["metadatas"]:
            internal_ids.add(metadata["document_id"])
            self.assertEqual(metadata["knowledge_id"], "guide-business-id")
            self.assertEqual(metadata["owner"], "test-team")
            self.assertEqual(metadata["authority_rank"], 80)
            self.assertEqual(metadata["version"], "1.2")
            self.assertTrue(is_effective(metadata, date(2026, 9, 27)))
        self.assertEqual(len(internal_ids), 1)
        self.assertNotIn("guide-business-id", internal_ids)

    def test_disable_and_restore_updates_every_chunk(self):
        store = self.manager.sync(self.ingestor)
        initial_count = store._collection.count()
        self.write_config("inactive")
        store = self.manager.sync(self.ingestor)
        disabled = self.records(store)
        self.assertEqual(len(disabled["ids"]), initial_count)
        self.assertTrue(all(m["status"] == "inactive" for m in disabled["metadatas"]))
        self.assertFalse(any(is_effective(m) for m in disabled["metadatas"]))
        self.write_config("active")
        store = self.manager.sync(self.ingestor)
        restored = self.records(store)
        self.assertEqual(len(restored["ids"]), initial_count)
        self.assertTrue(all(is_effective(m) for m in restored["metadatas"]))

    def test_malformed_config_does_not_mutate_existing_index(self):
        self.write_config("inactive")
        store = self.manager.sync(self.ingestor)
        before = self.records(store)
        self.metadata_path.write_text('{"default":', encoding="utf-8")
        with self.assertRaises(MetadataError):
            self.manager.sync(self.ingestor)
        self.assertEqual(self.records(store), before)

    def test_missing_config_does_not_reactivate_disabled_chunks(self):
        self.write_config("inactive")
        store = self.manager.sync(self.ingestor)
        before = self.records(store)
        self.metadata_path.unlink()
        with self.assertRaises(MetadataError):
            self.manager.sync(self.ingestor)
        self.assertEqual(self.records(store), before)

    def test_search_never_returns_disabled_records_after_same_size_update(self):
        from src.rag import search as search_module

        def local_rerank(query, candidates, top_n):
            return [replace(hit, rerank_score=0.9) for hit in candidates][:top_n]

        with (
            patch("src.rag.embeddings._manager", return_value=self.manager),
            patch("src.rag.embeddings.settings", self.config),
            patch.object(search_module, "settings", self.config),
            patch.object(search_module, "_rerank", side_effect=local_rerank),
        ):
            active = search_module.search("平台申请", k=2)
            self.assertTrue(active)
            self.assertTrue(all(hit.governance["knowledge_id"] == "guide-business-id" for hit in active))
            self.write_config("inactive")
            self.assertEqual(search_module.search("平台申请", k=2), [])
            self.write_config("active")
            self.assertTrue(search_module.search("平台申请", k=2))
            self.metadata_path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(MetadataError):
                search_module.search("平台申请", k=2)

    def test_failed_disable_sync_does_not_fall_back_to_old_active_records(self):
        from src.rag import search as search_module

        self.manager.sync(self.ingestor)
        self.write_config("inactive")
        with (
            patch("src.rag.embeddings._manager", return_value=self.manager),
            patch("src.rag.embeddings.settings", self.config),
            patch.object(search_module, "settings", self.config),
            patch.object(search_module, "_rerank") as rerank,
        ):
            with patch.object(self.manager._embeddings, "embed_documents", side_effect=RuntimeError("embedding unavailable")):
                with self.assertRaises(RuntimeError):
                    search_module.search("平台申请", k=2)
            rerank.assert_not_called()
            # 重试同步完成后，仍是停用状态，旧 active Chunk 不进入回答。
            self.assertEqual(search_module.search("平台申请", k=2), [])


if __name__ == "__main__":
    unittest.main()
