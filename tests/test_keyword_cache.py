"""缓存的失效、分页、治理筛选及并发测试；不访问真实知识库。"""
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from src.rag.keyword_cache import KeywordCache


class FakeStore:
    def __init__(self, rows=None, name="collection-a"):
        self.rows = rows or []
        self.reads = 0
        self._persist_directory = "test-only"
        self._collection = SimpleNamespace(id=name, count=lambda: len(self.rows))

    def get(self, *, include, limit, offset):
        self.reads += 1
        rows = self.rows[offset:offset + limit]
        return {"ids": [row[0] for row in rows], "documents": [row[1] for row in rows],
                "metadatas": [row[2] for row in rows]}


def row(index, text="平台操作", **metadata):
    return (str(index), text, {"document_id": f"doc-{index}", "status": "active", **metadata})


class KeywordCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = KeywordCache()
        self.today = date(2026, 9, 27)

    def get(self, store, revision="v1", today=None):
        return self.cache.get(store, revision=revision, today=today or self.today)

    def test_empty_collection(self):
        snapshot = self.get(FakeStore())
        self.assertEqual(snapshot.texts, ())
        self.assertEqual(snapshot.document_ids, ())
        self.assertEqual(snapshot.index.top_indices("平台", 3), [])

    def test_revision_and_date_are_required(self):
        for revision, today in [(None, self.today), ("", self.today), ("v1", None)]:
            with self.assertRaises(ValueError):
                self.cache.get(FakeStore(), revision=revision, today=today)

    def test_reuses_warm_index(self):
        store = FakeStore([row(1)])
        first = self.get(store)
        self.assertIs(first, self.get(store))
        self.assertEqual(store.reads, 1)

    def test_same_count_revision_update_is_immediate(self):
        store = FakeStore([row(1, "old")])
        old = self.get(store)
        store.rows = [row(1, "new")]
        new = self.get(store, revision="v2")
        self.assertEqual(new.texts, ("new",))
        self.assertEqual(old.texts, ("old",))

    def test_collection_count_and_ttl_invalidate(self):
        store = FakeStore([row(1)])
        first = self.get(store)
        store._collection.id = "recreated"
        second = self.get(store)
        self.assertIsNot(first, second)
        store.rows.append(row(2))
        self.assertEqual(len(self.get(store).texts), 2)
        self.cache._expires_at = 0
        self.get(store)
        self.assertEqual(store.reads, 4)

    def test_expiration_rechecked_on_next_business_day(self):
        store = FakeStore([row(1, effective_to="2026-09-27")])
        self.assertEqual(len(self.get(store).texts), 1)
        self.assertEqual(self.get(store, today=date(2026, 9, 28)).texts, ())

    def test_filter_before_bm25_statistics_and_ranking(self):
        store = FakeStore([row(1, "平台操作"), row(2, "平台操作", status="inactive"),
                           row(3, "平台操作", effective_from="2027-01-01"),
                           row(4, "平台操作", effective_to="2020-01-01"),
                           ("5", "平台操作", None), row(6, " "), row(7, None)])
        snapshot = self.get(store)
        self.assertEqual(snapshot.document_ids, ("doc-1",))
        self.assertEqual(snapshot.texts, ("平台操作",))

    def test_pages_align_with_metadata(self):
        store = FakeStore([row(i, f"code{i}") for i in range(5)])
        with patch("src.rag.keyword_cache.PAGE_SIZE", 2):
            snapshot = self.get(store)
        self.assertEqual(store.reads, 3)
        self.assertEqual(snapshot.index.top_indices("code4", 1), [4])
        self.assertEqual(snapshot.metadatas[4]["document_id"], "doc-4")

    def test_missing_metadata_is_fail_closed(self):
        store = FakeStore([row(1)])
        with patch.object(store, "get", return_value={"ids": ["1"], "documents": ["平台"], "metadatas": None}):
            self.assertEqual(self.get(store).texts, ())

    def test_bad_columns_do_not_replace_valid_snapshot(self):
        store = FakeStore([row(1)])
        first = self.get(store)
        with patch.object(store, "get", return_value={"ids": ["1"], "documents": []}):
            with self.assertRaises(ValueError):
                self.get(store, revision="bad")
        self.assertIs(self.cache._snapshot, first)
        self.assertIsNot(self.get(store, revision="fixed"), first)

    def test_duplicate_or_missing_records_rejected(self):
        for records in [dict(ids=["1", "1"], documents=["x", "x"], metadatas=[{}, {}]),
                        dict(ids=[], documents=[], metadatas=[])]:
            store = FakeStore([row(1), row(2)])
            with patch.object(store, "get", return_value=records), self.assertRaises(ValueError):
                self.get(store)

    def test_simultaneous_cold_queries_build_once(self):
        store = FakeStore([row(1)])
        with ThreadPoolExecutor(max_workers=8) as pool:
            snapshots = list(pool.map(lambda _: self.get(store), range(40)))
        self.assertEqual(store.reads, 1)
        self.assertTrue(all(snapshot is snapshots[0] for snapshot in snapshots))


if __name__ == "__main__":
    unittest.main()
