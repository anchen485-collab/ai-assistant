"""Small-to-Big 回归：真实分割器/临时 Chroma，无线上模型调用。

运行：python -m unittest discover -s tests -p test_small_to_big.py -v
"""
from __future__ import annotations

import importlib
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 与应用一致，从 embeddings 入口初始化 Chroma 依赖。
from src.rag import embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.core.config import Settings
from src.rag.documents import DocxIngestor, _build_child_chunks
from src.rag.engine import build_splitter
from src.rag.governance import business_date

retrieval = importlib.import_module("src.rag.search")


def settings_with(config=None, **updates):
    result = copy.copy(config) if config is not None else Settings()
    for name, value in updates.items():
        setattr(result, name, value)
    return result


def prepare_catalog(root):
    # 治理模块要求显式配置，即使临时知识库为空也提供合法配置。
    (root / "knowledge_metadata.json").write_text(
        '{"default": {"status": "active", "authority_rank": 0}, "documents": {}}',
        encoding="utf-8",
    )


class SplitTests(unittest.TestCase):
    def test_long_unbroken_text_has_hard_limit(self):
        for text in ("汉" * 4001, "x" * 4001, "🙂" * 1200):
            with self.subTest(length=len(text)):
                chunks = build_splitter(73, 0).split_text(text)
                self.assertGreater(len(chunks), 1)
                self.assertLessEqual(max(map(len, chunks)), 73)
                self.assertEqual("".join(chunks), text)

    def test_zero_overlap_is_respected(self):
        splitter = build_splitter(200, 0)
        self.assertEqual(splitter._chunk_overlap, 0)

    def test_mixed_markdown_and_punctuation(self):
        text = ("\n# 标题\n第一章 说明\n\n正文，第二句；第三句。问题？回答！" * 100)
        chunks = build_splitter(100, 10).split_text(text)
        self.assertTrue(chunks)
        self.assertLessEqual(max(map(len, chunks)), 100)

    def test_zero_size_is_not_replaced_with_default(self):
        with self.assertRaises(ValueError):
            build_splitter(0, 0)

    def test_children_preserve_parent_and_governance_without_mutation(self):
        parent = Document(page_content="内容" * 400, metadata={"source": "a.md", "owner": "team"})
        children = _build_child_chunks([parent], document_id="doc-a", parent_size=800, parent_overlap=100)
        self.assertGreater(len(children), 1)
        self.assertLessEqual(max(len(c.page_content) for c in children), 266)
        for child in children:
            self.assertEqual(child.metadata["parent_text"], parent.page_content)
            self.assertEqual(child.metadata["parent_id"], "doc-a:1")
            self.assertEqual(child.metadata["owner"], "team")
            self.assertIsInstance(child.metadata["parent_chunk"], int)
        self.assertNotIn("parent_text", parent.metadata)
        children[0].metadata["owner"] = "changed"
        self.assertEqual(children[1].metadata["owner"], "team")

    def test_small_parent_settings_and_empty_content(self):
        for size in (1, 3, 50, 199, 200, 800):
            with self.subTest(size=size):
                children = _build_child_chunks(
                    [Document(page_content="x" * size)], document_id="a",
                    parent_size=size, parent_overlap=0,
                )
                self.assertLessEqual(max(len(c.page_content) for c in children), size)
        self.assertEqual(_build_child_chunks(
            [Document(page_content=" \n ")], document_id="a", parent_size=50, parent_overlap=0,
        ), [])

    def test_invalid_configuration_fails_before_ingestion(self):
        for size, overlap in ((0, 0), (-1, 0), (100, -1), (100, 100), (100, 101), (True, 0), (1.5, 0)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                DocxIngestor(settings_with(chunk_size=size, chunk_overlap=overlap))

    def test_instance_config_used_and_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare_catalog(root)
            (root / "a.md").write_text("正文" * 2000, encoding="utf-8")
            config = settings_with(kb_source_dir=root, chunk_size=120, chunk_overlap=0)
            ingestor = DocxIngestor(config)
            first = ingestor.source_documents()[0]
            chunks = ingestor.ingest_document(first)
            self.assertTrue(chunks)
            self.assertTrue(all(len(c.metadata["parent_text"]) <= 120 for c in chunks))
            self.assertEqual(first.content_hash, ingestor.source_documents()[0].content_hash)
            for updated in (settings_with(config, chunk_size=121), settings_with(config, chunk_overlap=10)):
                other = DocxIngestor(updated).source_documents()[0]
                self.assertEqual(first.id, other.id)
                self.assertNotEqual(first.content_hash, other.content_hash)
            with patch("src.rag.documents.SMALL_TO_BIG_VERSION", "test-next-version", create=True):
                self.assertNotEqual(first.content_hash, ingestor.source_documents()[0].content_hash)

    def test_empty_knowledge_base_and_whitespace_document(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare_catalog(root)
            ingestor = DocxIngestor(settings_with(kb_source_dir=root))
            self.assertEqual(ingestor.ingest(), [])
            (root / "blank.md").write_text("  \n  ", encoding="utf-8")
            self.assertEqual(ingestor.ingest(), [])

    def test_parent_ids_stable_across_unrelated_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare_catalog(root)
            (root / "z.md").write_text("正文" * 600, encoding="utf-8")
            ingestor = DocxIngestor(settings_with(kb_source_dir=root))
            before = [c.metadata["parent_id"] for c in ingestor.ingest()]
            (root / "a.md").write_text("其他正文", encoding="utf-8")
            after = [c.metadata["parent_id"] for c in ingestor.ingest() if c.metadata["source"] == "z.md"]
            self.assertEqual(before, after)


class RecallTests(unittest.TestCase):
    def hit(self, name, text="same parent"):
        return retrieval._to_hit("child", {
            "parent_id": name, "parent_text": text, "parent_chunk": 3,
            "source_path": f"{name}/a.md", "owner": "team", "status": "active",
            "authority_rank": 0,
        })

    def test_parent_expansion_and_old_index_fallback(self):
        hit = self.hit("a")
        self.assertEqual((hit.text, hit.chunk), ("same parent", 3))
        self.assertEqual(hit.governance["owner"], "team")
        old = retrieval._to_hit("old text", {"source_path": "a.md", "chunk": "4"})
        self.assertEqual((old.text, old.chunk), ("old text", 4))
        self.assertEqual(retrieval._to_hit("old text", {}).text, "old text")

    def test_bad_parent_metadata_falls_back(self):
        for parent in (None, "", "   ", 123, [], {}):
            with self.subTest(parent=parent):
                hit = retrieval._to_hit("child", {"parent_text": parent, "parent_chunk": 9, "chunk": 2})
                self.assertEqual((hit.text, hit.chunk), ("child", 2))
        for chunk in (None, "bad", -1, "-8", [], True, 1.5):
            with self.subTest(chunk=chunk):
                hit = retrieval._to_hit("child", {"parent_text": "parent", "parent_chunk": chunk, "chunk": 2})
                self.assertEqual(hit.chunk, 2)
        self.assertEqual(retrieval._to_hit("child", None).chunk, 0)

    def test_duplicate_siblings_cannot_inflate_rrf_score(self):
        a, b = self.hit("a"), self.hit("b")
        # a 在同一路命中很多次；b 在两路都出现，应该优先 b。
        merged = retrieval._rrf_merge([a] * 10 + [b], [b])
        self.assertEqual(merged, [b, a])

    def test_distinct_sources_with_identical_text_are_preserved(self):
        a, b = self.hit("a"), self.hit("b")
        self.assertEqual(len(retrieval._rrf_merge([a, b], [a])), 2)
        old = [retrieval._to_hit("same", {"source_path": path, "chunk": 1})
               for path in ("one/a.md", "two/a.md")]
        self.assertEqual(len(retrieval._rrf_merge(old)), 2)

    def test_empty_and_top_k_limit(self):
        self.assertEqual(retrieval._rrf_merge([], []), [])
        with patch.object(retrieval.settings, "rerank_fetch_k", 2):
            self.assertEqual(len(retrieval._rrf_merge([self.hit(str(i)) for i in range(5)])), 2)

    def test_public_serialization_remains_compatible(self):
        self.assertEqual(set(retrieval.serialize_hits([self.hit("a")])[0]), {
            "text", "source", "chunk", "distance", "rerank_score", "governance",
        })

    def test_rerank_retains_identity_and_governance(self):
        from dashscope.rerank import TextReRank
        hit = self.hit("a")
        response = SimpleNamespace(status_code=200, output={"results": [{"index": 0, "relevance_score": 0.8}]})
        with patch.object(retrieval.settings, "dashscope_api_key", "test-only"), patch.object(TextReRank, "call", return_value=response):
            ranked = retrieval._rerank("question", [hit], 1)
        self.assertEqual(ranked[0].parent_id, hit.parent_id)
        self.assertEqual(ranked[0].governance, hit.governance)
        self.assertEqual(ranked[0].rerank_score, 0.8)

    def test_existing_quality_gate_and_governance_preserved(self):
        hit = self.hit("a")
        inactive = replace(hit, governance={"status": "inactive"})
        with (
            patch.object(retrieval, "_hybrid_recall", return_value=([hit, inactive], [], [hit, inactive], [hit])),
            patch.object(retrieval, "_rerank", return_value=[replace(hit, rerank_score=0.01)]),
            patch.object(retrieval.settings, "rerank_threshold", 0.3),
        ):
            self.assertEqual(retrieval.search("q"), [])
        self.assertEqual(retrieval._filter_effective([inactive]), [])


class LocalEmbeddings(Embeddings):
    """确定性本地向量，只替代付费 Embedding API；Chroma 和召回代码均真实。"""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count("苹果") + 1), float(text.count("香蕉") + 1), 1.0]


class ChromaIntegrationTests(unittest.TestCase):
    def test_sync_real_chroma_recall_and_config_rebuild(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            kb = root / "kb"
            kb.mkdir()
            prepare_catalog(kb)
            (kb / "a.md").write_text("苹果营养介绍。" * 70, encoding="utf-8")
            (kb / "b.md").write_text("香蕉营养介绍。" * 70, encoding="utf-8")
            config = settings_with(kb_source_dir=kb, chroma_dir=root / "chroma", chroma_collection="small-to-big-test", chunk_size=240, chunk_overlap=0)
            manager = embeddings.VectorStoreManager(config, LocalEmbeddings())
            store = manager.sync(DocxIngestor(config))
            first_count = store._collection.count()
            self.assertGreater(first_count, 2)
            records = store.get(include=["documents", "metadatas"])
            self.assertTrue(all(len(t) <= 200 for t in records["documents"]))
            self.assertTrue(all(len(m["parent_text"]) <= 240 for m in records["metadatas"]))
            store = manager.sync(DocxIngestor(config))
            self.assertEqual(first_count, store._collection.count())
            # 显式绑定临时索引并持有读锁，禁止走生产单例/真实 API。
            with (
                patch.object(embeddings, "_manager", side_effect=AssertionError("test accessed production store")),
                manager.read_context(DocxIngestor(config)) as store,
            ):
                revision = embeddings.get_index_revision(store)
                mmr = retrieval._mmr_recall("苹果", store=store, revision=revision, today=business_date())
                bm25 = retrieval._bm25_recall("苹果", store=store, revision=revision, today=business_date())
            self.assertTrue(mmr)
            self.assertTrue(bm25)
            parents = {m["parent_id"]: m["parent_text"] for m in records["metadatas"]}
            for hit in mmr + bm25:
                self.assertEqual(hit.text, parents[hit.parent_id])
            merged = retrieval._rrf_merge(mmr, bm25)
            self.assertEqual(len(merged), len({h.parent_id for h in merged}))
            # 切分参数改变后，同一文档的新 chunk 取代旧 chunk。
            with (
                patch.object(embeddings, "_manager", side_effect=AssertionError("test accessed production store")),
                manager.read_context(DocxIngestor(settings_with(config, chunk_size=90))) as store,
            ):
                changed = store.get(include=["metadatas"])["metadatas"]
                self.assertTrue(all(len(m["parent_text"]) <= 90 for m in changed))
                new_revision = embeddings.get_index_revision(store)
                self.assertNotEqual(revision, new_revision)
                mmr = retrieval._mmr_recall("苹果", store=store, revision=new_revision, today=business_date())
                bm25 = retrieval._bm25_recall("苹果", store=store, revision=new_revision, today=business_date())
                self.assertTrue(mmr)
                self.assertTrue(bm25)
                self.assertTrue(all(len(hit.text) <= 90 for hit in mmr + bm25))


if __name__ == "__main__":
    unittest.main()
