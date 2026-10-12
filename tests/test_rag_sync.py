"""离线回归：不读取真实知识库，不调用在线 Embedding。"""
import json
import multiprocessing
import tempfile
import threading
import time
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# 先导入现有 Chroma 初始化模块，保持 Windows 项目导入顺序。
from src.rag import embeddings as module
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.core.config import settings
from src.rag.documents import DocxIngestor
from src.rag.sync_state import sync_lock, write_json_atomic


def replace(config, **values):
    result = copy(config)
    for key, value in values.items():
        setattr(result, key, value)
    return result


class OfflineEmbeddings(Embeddings):
    model = "offline-v1"

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class FakeStore:
    def __init__(self):
        self.rows = {}
        self.add_calls = 0
        self.fail_add = None
        self.fail_delete = False

    def add_documents(self, documents, ids):
        self.add_calls += 1
        for key, doc in zip(ids, documents):
            self.rows[key] = doc
        if self.fail_add:
            raise self.fail_add

    def get(self, where=None, include=None, limit=None, offset=0, ids=None):
        rows = [(key, doc) for key, doc in self.rows.items()
                if (not where or all(doc.metadata.get(k) == v for k, v in where.items()))
                and (ids is None or key in ids)]
        rows = rows[offset:offset + limit if limit else None]
        return {"ids": [key for key, _ in rows], "metadatas": [doc.metadata for _, doc in rows]}

    def delete(self, where=None, ids=None):
        if self.fail_delete:
            raise OSError("injected delete failure")
        for key in self.get(where=where, ids=ids)["ids"]:
            self.rows.pop(key, None)


class FakeIngestor:
    def __init__(self):
        self.docs = {"a": ("v1", 2)}
        self.fail_current = False

    def source_documents(self):
        return [SimpleNamespace(id=key, path=Path(key + ".md"), content_hash=value[0])
                for key, value in self.docs.items()]

    def ingest_document(self, item):
        return [Document(page_content=f"{item.content_hash} content {i}",
                         metadata={"document_id": item.id, "parent_text": "parent", "knowledge_id": "business"})
                for i in range(self.docs[item.id][1])]

    def assert_current(self, item):
        if self.fail_current:
            raise RuntimeError("source changed")


def lock_worker(path, queue):
    try:
        with sync_lock(Path(path), timeout=0.15):
            queue.put("acquired")
    except TimeoutError:
        queue.put("timeout")


def crash_while_locked(path, connection):
    import os
    with sync_lock(Path(path)):
        connection.send("locked")
        os._exit(17)


def real_sync_worker(root, version, connection):
    root = Path(root)
    config = replace(settings, chroma_dir=root, kb_source_dir=root / "kb")
    ingestor = FakeIngestor()
    ingestor.docs = {"a": (version, 1)} if version else {}
    store = module.VectorStoreManager(config, OfflineEmbeddings()).sync(ingestor)
    connection.send(store.get(include=[])["ids"])
    connection.close()


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = replace(settings, chroma_dir=self.root / "db", kb_source_dir=self.root / "kb")
        self.store = FakeStore()
        self.ingestor = FakeIngestor()
        self.manager = module.VectorStoreManager(self.config, OfflineEmbeddings())
        self.patcher = patch.object(module, "build_chroma", return_value=self.store)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def sync(self):
        return self.manager.sync(self.ingestor)

    def state(self):
        return json.loads(self.manager._manifest_path.read_text(encoding="utf-8"))

    def test_add_unchanged_modify_delete(self):
        self.sync()
        self.assertEqual(len(self.store.rows), 2)
        previous = self.manager._manifest_path.stat().st_mtime_ns
        self.sync()
        self.assertEqual(self.store.add_calls, 1)
        self.assertEqual(self.manager._manifest_path.stat().st_mtime_ns, previous)
        self.ingestor.docs["a"] = ("v2", 1)
        self.sync()
        self.assertEqual(set(self.store.rows), {"a:v2:0"})
        self.ingestor.docs = {}
        self.sync()
        self.assertEqual(self.store.rows, {})
        self.assertEqual(self.state()["documents"], {})

    def test_parse_failure_preserves_old_chunks_and_hash(self):
        self.sync()
        old = dict(self.store.rows)
        self.ingestor.docs["a"] = ("v2", 0)
        with self.assertRaises(ValueError):
            self.sync()
        self.assertEqual(self.store.rows, old)
        self.assertEqual(self.state()["documents"]["a"]["hash"], "v1")
        self.ingestor.docs["a"] = ("v2", 1)
        self.sync()
        self.assertEqual(set(self.store.rows), {"a:v2:0"})

    def test_embedding_or_partial_write_failure_preserves_old(self):
        self.sync()
        old = dict(self.store.rows)
        self.ingestor.docs["a"] = ("v2", 70)
        self.store.fail_add = OSError("injected partial add")
        with self.assertRaises(OSError):
            self.sync()
        self.assertEqual(self.store.rows, old)
        self.assertEqual(self.state()["pending"], "a")
        self.store.fail_add = None
        self.sync()
        self.assertEqual(len(self.store.rows), 70)
        self.assertIsNone(self.state()["pending"])

    def test_process_exit_then_source_deleted_cleans_orphans(self):
        self.sync()
        self.ingestor.docs["a"] = ("v2", 2)
        self.store.fail_add = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.sync()
        self.assertEqual(len(self.store.rows), 4)
        self.ingestor.docs = {}
        self.store.fail_add = None
        self.sync()
        self.assertEqual(self.store.rows, {})

    def test_process_exit_then_source_reverted(self):
        self.sync()
        self.ingestor.docs["a"] = ("v2", 2)
        self.store.fail_add = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.sync()
        self.ingestor.docs["a"] = ("v1", 2)
        self.store.fail_add = None
        self.sync()
        self.assertEqual(set(self.store.rows), {"a:v1:0", "a:v1:1"})

    def test_delete_failure_can_retry(self):
        self.sync()
        self.ingestor.docs["a"] = ("v2", 1)
        self.store.fail_delete = True
        with self.assertRaises(OSError):
            self.sync()
        self.assertEqual(self.state()["pending"], "a")
        self.store.fail_delete = False
        self.sync()
        self.assertEqual(set(self.store.rows), {"a:v2:0"})

    def test_manifest_failure_after_db_commit_can_retry(self):
        self.sync()
        self.ingestor.docs["a"] = ("v2", 1)
        real = self.manager._write_manifest
        def fail_commit(state):
            if state["pending"] is None:
                raise OSError("injected manifest failure")
            real(state)
        with patch.object(self.manager, "_write_manifest", side_effect=fail_commit):
            with self.assertRaises(OSError):
                self.sync()
        self.assertEqual(set(self.store.rows), {"a:v2:0"})
        self.assertEqual(self.state()["pending"], "a")
        self.sync()
        self.assertEqual(self.state()["documents"]["a"]["hash"], "v2")

    def test_missing_chunks_are_rebuilt(self):
        self.sync()
        self.store.rows.clear()
        self.sync()
        self.assertEqual(len(self.store.rows), 2)
        self.assertEqual(self.store.add_calls, 2)

    def test_corrupt_and_missing_manifest_fail_closed(self):
        self.sync()
        old = dict(self.store.rows)
        self.manager._manifest_path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.sync()
        self.assertEqual(self.store.rows, old)
        self.manager._manifest_path.unlink()
        with self.assertRaises(RuntimeError):
            self.sync()

    def test_invalid_manifest_shapes_fail_closed(self):
        for data in ([], {"documents": []}, {"documents": {"a": 5}}):
            self.manager._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manager._manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                self.sync()

    def test_scope_mismatch_fails_closed(self):
        self.sync()
        other = module.VectorStoreManager(replace(self.config, chroma_collection="other"), OfflineEmbeddings())
        with self.assertRaises(RuntimeError):
            other.sync(self.ingestor)

    def test_legacy_manifest_migrates(self):
        path = self.manager._manifest_path
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"documents": {"a": "v1"}}), encoding="utf-8")
        self.store.rows["a:1"] = Document(page_content="legacy", metadata={"document_id": "a"})
        self.sync()
        self.assertEqual(set(self.store.rows), {"a:v1:0", "a:v1:1"})
        self.assertEqual(self.state()["schema_version"], 2)

    def test_source_changes_before_commit_keeps_old(self):
        self.sync()
        self.ingestor.docs["a"] = ("v2", 1)
        original = self.store.add_documents
        def mutate(**kwargs):
            original(**kwargs)
            self.ingestor.fail_current = True
        with patch.object(self.store, "add_documents", side_effect=mutate):
            with self.assertRaises(RuntimeError):
                self.sync()
        self.assertEqual(set(self.store.rows), {"a:v1:0", "a:v1:1"})

    def test_revision_stable_and_changes_on_equal_size_update(self):
        first = module.get_index_revision(self.sync())
        self.assertEqual(first, module.get_index_revision(self.sync()))
        self.ingestor.docs["a"] = ("v2", 2)
        self.assertNotEqual(first, module.get_index_revision(self.sync()))

    def test_read_context_blocks_writer_until_reader_finishes(self):
        errors = []
        started = threading.Event()
        done = threading.Event()
        def writer():
            started.set()
            try:
                self.sync()
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()
        with self.manager.read_context(self.ingestor) as store:
            self.ingestor.docs["a"] = ("v2", 1)
            thread = threading.Thread(target=writer)
            thread.start()
            self.assertTrue(started.wait(5))
            self.assertFalse(done.wait(0.2))
            self.assertEqual(set(store.rows), {"a:v1:0", "a:v1:1"})
        thread.join(10)
        self.assertTrue(done.is_set())
        self.assertEqual(errors, [])
        self.assertEqual(set(self.store.rows), {"a:v2:0"})

    def test_governance_failure_does_not_return_old_store(self):
        self.sync()
        old = dict(self.store.rows)
        with patch.object(self.ingestor, "source_documents", side_effect=ValueError("invalid governance")):
            with self.assertRaises(ValueError):
                self.sync()
        self.assertEqual(self.store.rows, old)

    def test_pending_new_document_deleted_after_interrupted_first_sync(self):
        self.store.fail_add = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.sync()
        self.assertEqual(self.state()["documents"], {})
        self.ingestor.docs = {}
        self.store.fail_add = None
        self.sync()
        self.assertEqual(self.store.rows, {})

    def test_empty_kb_initializes_manifest(self):
        self.ingestor.docs = {}
        self.sync()
        self.assertEqual(self.state()["documents"], {})

    def test_two_threads_only_embed_once(self):
        errors = []
        def run():
            try:
                self.sync()
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(self.store.add_calls, 1)


class StateTests(unittest.TestCase):
    def test_atomic_write_failure_keeps_previous_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json_atomic(path, {"old": 1})
            with patch("src.rag.sync_state.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    write_json_atomic(path, {"new": 2})
            self.assertEqual(json.loads(path.read_text()), {"old": 1})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_process_crash_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sync.lock"
            ctx = multiprocessing.get_context("spawn")
            reader, writer = ctx.Pipe(duplex=False)
            process = ctx.Process(target=crash_while_locked, args=(str(path), writer))
            process.start()
            writer.close()
            self.assertTrue(reader.poll(20))
            self.assertEqual(reader.recv(), "locked")
            process.join(20)
            self.assertEqual(process.exitcode, 17)
            reader.close()
            with sync_lock(path, timeout=0.2):
                pass

    def test_process_lock_timeout_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sync.lock"
            ctx = multiprocessing.get_context("spawn")
            queue = ctx.Queue()
            with sync_lock(path):
                process = ctx.Process(target=lock_worker, args=(str(path), queue))
                process.start()
                self.assertEqual(queue.get(timeout=20), "timeout")
                process.join(20)
                self.assertEqual(process.exitcode, 0)
            with sync_lock(path, timeout=0.1):
                pass
            queue.close()


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "knowledge_metadata.json").write_text(json.dumps({"default": {}, "documents": {}}), encoding="utf-8")
        (self.root / "a.md").write_text("第一段内容。\n" * 100, encoding="utf-8")
        self.config = replace(settings, kb_source_dir=self.root, chunk_size=300, chunk_overlap=0)
        self.ingestor = DocxIngestor(self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_hash_changes_with_content_metadata_and_split_config(self):
        original = self.ingestor.source_documents()[0]
        other = DocxIngestor(replace(self.config, chunk_size=500)).source_documents()[0]
        self.assertEqual(original.id, other.id)
        self.assertNotEqual(original.content_hash, other.content_hash)
        (self.root / "a.md").write_text("changed", encoding="utf-8")
        changed = self.ingestor.source_documents()[0]
        self.assertEqual(original.id, changed.id)
        self.assertNotEqual(original.content_hash, changed.content_hash)
        (self.root / "knowledge_metadata.json").write_text(json.dumps({"default": {"owner": "new"}, "documents": {}}), encoding="utf-8")
        self.assertNotEqual(changed.content_hash, self.ingestor.source_documents()[0].content_hash)

    def test_snapshot_and_parent_metadata(self):
        item = self.ingestor.source_documents()[0]
        chunks = self.ingestor.ingest_document(item)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["document_id"], item.id)
            self.assertEqual(chunk.metadata["source_path"], str(item.path))
            self.assertIn("parent_text", chunk.metadata)
        (self.root / "a.md").write_text("changed", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.ingestor.ingest_document(item)

    def test_metadata_changed_during_sync_is_detected(self):
        item = self.ingestor.source_documents()[0]
        (self.root / "knowledge_metadata.json").write_text(json.dumps({"default": {"owner": "changed"}}), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.ingestor.assert_current(item)

    def test_file_instead_of_directory_is_not_empty_knowledge_base(self):
        ingestor = DocxIngestor(replace(self.config, kb_source_dir=self.root / "a.md"))
        with self.assertRaises(FileNotFoundError):
            ingestor.source_documents()

    def test_zero_overlap_respected(self):
        from src.rag.engine import build_splitter
        self.assertEqual(build_splitter(300, 0)._chunk_overlap, 0)


class RealChromaTests(unittest.TestCase):
    def test_real_store_survives_process_restart(self):
        root = Path(tempfile.mkdtemp(prefix="qf-rag-real-test-"))
        ctx = multiprocessing.get_context("spawn")
        for version, expected in (("v1", ["a:v1:0"]), ("v2", ["a:v2:0"]), (None, [])):
            reader, writer = ctx.Pipe(duplex=False)
            process = ctx.Process(target=real_sync_worker, args=(str(root), version, writer))
            process.start()
            writer.close()
            self.assertTrue(reader.poll(25))
            self.assertEqual(reader.recv(), expected)
            process.join(25)
            self.assertEqual(process.exitcode, 0)
            reader.close()
        print(f"REAL_CHROMA_DIR={root}")

    def test_real_store_end_to_end(self):
        # Windows Chroma 持有 SQLite 句柄直到进程退出；由外部测试运行器清理此临时目录。
        root = Path(tempfile.mkdtemp(prefix="qf-rag-real-test-"))
        config = replace(settings, chroma_dir=root, kb_source_dir=root / "kb")
        ingestor = FakeIngestor()
        manager = module.VectorStoreManager(config, OfflineEmbeddings())
        store = manager.sync(ingestor)
        self.assertEqual(len(store.get(include=[])["ids"]), 2)
        ingestor.docs["a"] = ("v2", 1)
        manager.sync(ingestor)
        self.assertEqual(store.get(include=[])["ids"], ["a:v2:0"])
        manager.sync(ingestor)
        ingestor.docs = {}
        manager.sync(ingestor)
        self.assertEqual(store.get(include=[])["ids"], [])
        print(f"REAL_CHROMA_DIR={root}")


if __name__ == "__main__":
    unittest.main()
