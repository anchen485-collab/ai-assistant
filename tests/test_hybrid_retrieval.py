"""混合检索回归：真实实现 + 故障注入，不调用在线模型。"""
import importlib
import unittest
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from src.rag.keyword_cache import KeywordCache
from test_keyword_cache import FakeStore, row

rag = importlib.import_module("src.rag.search")
TODAY = date(2026, 9, 27)


def hit(text, document="doc-a", parent=1):
    return rag.SearchHit(text, document + ".md", parent,
                         governance={"document_id": document, "status": "active"},
                         parent_id=f"{document}:{parent}")


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore([row(1)])
        self.locked = False
        self.context_entries = 0

        @contextmanager
        def read_context():
            self.context_entries += 1
            self.locked = True
            try:
                yield self.store
            finally:
                self.locked = False

        self.patch("knowledge_read_context", read_context)
        self.patch("get_index_revision", return_value="revision-1")
        self.patch("business_date", return_value=TODAY)
        self.patch("keyword_cache", KeywordCache())
        self.patch("settings", SimpleNamespace(retrieval_k=5, rerank_fetch_k=10,
                                               rerank_threshold=.3, dashscope_api_key=""))

    def patch(self, name, *args, **kwargs):
        patcher = patch.object(rag, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def test_empty_queries_and_nonpositive_k_skip_io(self):
        for query, k in [("", None), (" \n", 3), ("platform", 0), ("platform", -1)]:
            self.assertEqual(rag.search(query, k), [])
        self.assertEqual(self.context_entries, 0)

    def test_invalid_types_and_config_fail_before_io(self):
        with self.assertRaises(TypeError):
            rag.search(None)
        for k in (True, 2.5, "2"):
            with self.assertRaises(ValueError):
                rag.search("query", k)
        rag.settings.rerank_fetch_k = 0
        with self.assertRaises(ValueError):
            rag.search("query")
        self.assertEqual(self.context_entries, 0)

    def test_same_store_revision_date_and_unlock_before_rerank(self):
        def recall(query, **kwargs):
            self.assertTrue(self.locked)
            self.assertIs(kwargs["store"], self.store)
            self.assertEqual(kwargs["revision"], "revision-1")
            self.assertEqual(kwargs["today"], TODAY)
            self.assertEqual(kwargs["limit"], 12)
            return [hit("平台操作")]
        def rerank(query, candidates, top_n):
            self.assertFalse(self.locked)
            return None
        self.patch("_mmr_recall", side_effect=recall)
        self.patch("_bm25_recall", side_effect=recall)
        self.patch("_rerank", side_effect=rerank)
        self.assertEqual(len(rag.search("平台操作", k=12)), 1)
        self.assertEqual(self.context_entries, 1)

    def test_each_single_recall_failure_degrades(self):
        for failed, healthy in [("_mmr_recall", "_bm25_recall"), ("_bm25_recall", "_mmr_recall")]:
            with patch.object(rag, failed, side_effect=TimeoutError), \
                 patch.object(rag, healthy, return_value=[hit("平台")]), \
                 patch.object(rag, "_rerank", return_value=None):
                self.assertEqual(len(rag.search("平台")), 1)

    def test_both_fail_surface_error_and_release_lock(self):
        self.patch("_mmr_recall", side_effect=TimeoutError)
        self.patch("_bm25_recall", side_effect=OSError)
        with self.assertRaisesRegex(RuntimeError, "均失败"):
            rag.search("平台")
        self.assertFalse(self.locked)

    def test_bm25_matches_keep_metadata_alignment(self):
        self.store.rows = [row(1, None), row(2, "GPU 4090", source="gpu.md", chunk=7),
                           row(3, "GPU 4090", status="inactive")]
        found = rag._bm25_recall("4090", limit=2)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].source, found[0].chunk), ("gpu.md", 7))

    def test_mmr_sends_effective_whitelist_before_topk(self):
        self.store.rows = [row(1), row(2, status="inactive"), row(3, effective_to="2020-01-01")]
        def mmr(query, **kwargs):
            self.assertEqual(kwargs["filter"], {"document_id": {"$in": ["doc-1"]}})
            return [Document(page_content="平台", metadata=row(1)[2])]
        self.store.max_marginal_relevance_search = mmr
        self.assertEqual(len(rag._mmr_recall("平台", limit=1)), 1)

    def test_empty_whitelist_skips_vector_query(self):
        self.store.rows = [row(1, status="inactive")]
        self.assertEqual(rag._mmr_recall("平台"), [])

    def test_parent_dedup_before_keyword_limit(self):
        self.store.rows = [row(i, "平台", document_id="doc-a", parent_id="doc-a:1",
                               parent_chunk=1, parent_text="父块 A") for i in range(4)]
        self.store.rows.append(row(9, "平台 操作", document_id="doc-b",
                                   parent_id="doc-b:1", parent_chunk=1, parent_text="父块 B"))
        result = rag._bm25_recall("平台", limit=2)
        self.assertEqual([h.text for h in result], ["父块 A", "父块 B"])

    def test_rrf_duplicate_does_not_boost_and_cross_route_does(self):
        a, b, c = hit("A"), hit("B", "doc-b"), hit("C", "doc-c")
        actual = rag._rrf_merge([a] * 20 + [b], [b, c], limit=3)
        self.assertEqual(actual, [b, a, c])

    def test_identical_text_different_source_and_parent_kept(self):
        a, b, c = hit("相同", "doc-a", 1), hit("相同", "doc-b", 1), hit("相同", "doc-a", 2)
        self.assertEqual(rag._rrf_merge([a, b, c]), [a, b, c])
        self.assertEqual(rag._rrf_merge([a, b, c], limit=0), [])

    def test_rrf_skips_empty_text_and_keeps_ties_stable(self):
        a, b = hit("A"), hit("B", "doc-b")
        self.assertEqual(rag._rrf_merge([hit(" "), a], [b]), [a, b])

    def test_corrupt_revision_refresh_never_serves_old_cache(self):
        rag._bm25_recall("平台")
        self.patch("get_index_revision", return_value="revision-2")
        with patch.object(self.store, "get", side_effect=OSError), self.assertRaises(OSError):
            rag._bm25_recall("平台")


class RealChromaTests(unittest.TestCase):
    def test_real_chroma_get_mmr_bm25_and_cache_update(self):
        from uuid import uuid4
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        from langchain_core.embeddings import Embeddings

        class OfflineEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[float("4090" in text), float("platform" in text), .1] for text in texts]
            def embed_query(self, text):
                return self.embed_documents([text])[0]

        client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        name = "hybrid-test-" + uuid4().hex
        store = Chroma(client=client, collection_name=name, embedding_function=OfflineEmbeddings())
        try:
            store.add_texts(["GPU 4090 power", "platform manual", "GPU 4090 old"],
                            metadatas=[row(1)[2], row(2)[2], row(3, status="inactive")[2]],
                            ids=["1", "2", "3"])
            with patch.object(rag, "keyword_cache", KeywordCache()):
                kwargs = dict(store=store, revision="1", today=TODAY, limit=3)
                keyword = rag._bm25_recall("4090", **kwargs)
                vector = rag._mmr_recall("4090", **kwargs)
                self.assertEqual([h.text for h in keyword], ["GPU 4090 power"])
                self.assertTrue(all(h.governance["status"] == "active" for h in vector))
                store._collection.update(ids=["1"], documents=["GPU 5090 power"], embeddings=[[1., 0., .1]])
                kwargs["revision"] = "2"
                self.assertEqual(rag._bm25_recall("4090", **kwargs), [])
                self.assertEqual(len(rag._bm25_recall("5090", **kwargs)), 1)
        finally:
            client.delete_collection(name)


if __name__ == "__main__":
    unittest.main()
