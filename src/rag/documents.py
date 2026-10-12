from __future__ import annotations

"""知识库文档加载、切分和文档变更识别。"""

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

from src.core.config import Settings
from src.rag.engine import build_splitter, load_files
from src.rag.governance import load_metadata, metadata_for_document

logger = logging.getLogger(__name__)

# 切分或 metadata 规则变化后自动重新同步一次知识库。
INDEX_VERSION = "dynamic-update-governance-v2"
SMALL_TO_BIG_VERSION = "small-to-big-v2"


@dataclass(frozen=True)
class KnowledgeDocument:
    """知识库中的一份原始文档及其当前内容指纹。"""

    id: str
    path: Path
    content_hash: str
    metadata: dict[str, Any]


class DocxIngestor:
    """加载知识库文档，并为每份文档生成独立的 chunk。"""

    def __init__(self, config: Settings) -> None:
        _validate_chunk_settings(config.chunk_size, config.chunk_overlap)
        self.__config = config

    def source_documents(self, source_dir: Path | None = None) -> list[KnowledgeDocument]:
        """列出当前知识库文档，并计算每份文档的内容 hash。"""

        root = source_dir or self.__config.kb_source_dir
        if not root.is_dir():
            raise FileNotFoundError(f"知识库目录不存在: {root}")

        catalog = load_metadata(root)
        documents: list[KnowledgeDocument] = []
        for path in self._kb_paths(root):
            relative_name = path.relative_to(root).as_posix()
            metadata = metadata_for_document(path, catalog)
            documents.append(
                KnowledgeDocument(
                    id=hashlib.md5(relative_name.encode("utf-8")).hexdigest(),
                    path=path,
                    content_hash=_document_hash(
                        path, metadata,
                        chunk_size=self.__config.chunk_size,
                        chunk_overlap=self.__config.chunk_overlap,
                    ),
                    metadata=metadata,
                )
            )
        return documents

    def kb_hash(self, source_dir: Path | None = None) -> str:
        """保留联合 hash，方便兼容旧调用方和排查知识库版本。"""

        h = hashlib.md5(INDEX_VERSION.encode("utf-8"))
        for item in self.source_documents(source_dir):
            h.update(item.id.encode("utf-8"))
            h.update(item.content_hash.encode("utf-8"))
        return h.hexdigest()

    def ingest(self, source_dir: Path | None = None) -> List[Document]:
        """加载并切分全部知识库文档。"""

        chunks: List[Document] = []
        for item in self.source_documents(source_dir):
            chunks.extend(self.ingest_document(item))
        return chunks

    def ingest_document(self, item: KnowledgeDocument) -> List[Document]:
        """只加载一份文档，供增量更新时重新入库。"""

        # 独立快照保证解析内容与扫描指纹一致，避免读到写入一半的原文件。
        with tempfile.TemporaryDirectory(prefix="qf-kb-document-") as directory:
            snapshot = Path(directory) / item.path.name
            snapshot.write_bytes(item.path.read_bytes())
            if self._hash(snapshot, item.metadata) != item.content_hash:
                raise RuntimeError(f"文档在扫描后发生变化，请重试同步: {item.path}")
            docs = load_files([snapshot])
        if not docs:
            return []

        for doc in docs:
            doc.metadata.update(item.metadata)
            doc.metadata["source_path"] = str(item.path)
        parents = build_splitter(
            self.__config.chunk_size, self.__config.chunk_overlap,
        ).split_documents(docs)
        return _build_child_chunks(
            parents,
            document_id=item.id,
            parent_size=self.__config.chunk_size,
            parent_overlap=self.__config.chunk_overlap,
        )

    def _hash(self, path: Path, metadata: dict[str, Any]) -> str:
        return _document_hash(
            path, metadata, chunk_size=self.__config.chunk_size,
            chunk_overlap=self.__config.chunk_overlap,
        )

    def assert_current(self, item: KnowledgeDocument) -> None:
        """提交前再查一次正文和治理信息；变化时让调用方重试。"""
        metadata = metadata_for_document(item.path, load_metadata(self.__config.kb_source_dir))
        if self._hash(item.path, metadata) != item.content_hash:
            raise RuntimeError(f"文档在同步过程中发生变化，请重试同步: {item.path}")

    def _kb_paths(self, root: Path) -> list[Path]:
        patterns = ["*.txt", "*.md", "*.docx"]
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(root.glob(pattern))
        return sorted(p for p in paths if p.is_file() and not p.name.startswith("~$"))


def _document_hash(
    path: Path, metadata: dict[str, Any], *, chunk_size: int, chunk_overlap: int,
) -> str:
    """内容或治理元数据改变时，都触发该文档的增量重建。"""

    h = hashlib.sha256()
    h.update(INDEX_VERSION.encode("utf-8"))
    # 切分配置也是索引的一部分；变更后交给现有增量同步逻辑重建。
    h.update(json.dumps(
        [SMALL_TO_BIG_VERSION, chunk_size, chunk_overlap],
    ).encode("utf-8"))
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            h.update(block)
    h.update(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _validate_chunk_settings(size: int, overlap: int) -> None:
    if type(size) is not int or size <= 0:
        raise ValueError("chunk_size 必须是正整数")
    if type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("chunk_overlap 必须是非负整数，且小于 chunk_size")


def _build_child_chunks(
    parents: List[Document],
    *,
    document_id: str,
    parent_size: int,
    parent_overlap: int,
) -> List[Document]:
    """小块用于检索，metadata 中保留父块全文供回答时读取。"""

    _validate_chunk_settings(parent_size, parent_overlap)
    # 小于默认尺寸的自定义父块，也要保证子块不会反而更大。
    child_size = min(parent_size, max(200, parent_size // 3))
    child_overlap = min(parent_overlap, child_size // 5)
    child_splitter = build_splitter(child_size, child_overlap)
    children: List[Document] = []

    for parent_index, parent in enumerate(parents, start=1):
        if not parent.page_content.strip():
            continue
        parent_id = f"{document_id}:{parent_index}"
        for child in child_splitter.split_documents([parent]):
            metadata = dict(child.metadata)
            metadata.update({
                "document_id": document_id,
                "parent_id": parent_id,
                "parent_chunk": parent_index,
                "parent_text": parent.page_content,
                "chunk": parent_index,
            })
            children.append(Document(page_content=child.page_content, metadata=metadata))

    return children
