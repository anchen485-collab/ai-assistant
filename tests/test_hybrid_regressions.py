"""同一组断言在原实现失败、在加固实现通过，避免只测新代码路径。"""
import importlib
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from src.rag.bm25 import top_bm25_indices

rag = importlib.import_module("src.rag.search")


class HybridRegressionTests(unittest.TestCase):
    def setUp(self):
        self.hit = rag.SearchHit("alpha", "a.md", 1, governance={"document_id": "a", "status": "active"})
        self.patches = [
            patch.object(rag, "knowledge_read_context", return_value=nullcontext(object()), create=True),
            patch.object(rag, "get_index_revision", return_value="test", create=True),
            patch.object(rag, "settings", SimpleNamespace(retrieval_k=5, rerank_fetch_k=10, rerank_threshold=.3)),
            patch.object(rag, "_rerank", return_value=None),
            patch.object(rag, "_mmr_recall", return_value=[self.hit]),
            patch.object(rag, "_bm25_recall", return_value=[self.hit]),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_zero_k_does_not_return_default_count(self):
        self.assertEqual(rag.search("alpha", k=0), [])

    def test_blank_query_does_not_retrieve_documents(self):
        self.assertEqual(rag.search(" "), [])

    def test_keyword_route_survives_embedding_outage(self):
        with patch.object(rag, "_mmr_recall", side_effect=TimeoutError):
            self.assertEqual(rag.search("alpha"), [self.hit])

    def test_negative_bm25_limit_returns_no_documents(self):
        self.assertEqual(top_bm25_indices("alpha", ["alpha", "alpha"], -1), [])


if __name__ == "__main__":
    unittest.main()
