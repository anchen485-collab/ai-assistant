from __future__ import annotations

"""RAG 引擎：统一的文档加载、分割、Chroma 工厂。

知识库（rag/documents.py）和会话临时索引（attachments/session.py）
共用这一套基础能力，各自只需要关心来源和生命周期。
"""

import logging
from pathlib import Path
from typing import List, Optional

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings

logger = logging.getLogger(__name__)

# ── 分割器工厂 ──────────────────────────────────────────────────────


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建项目统一的分割器，按 大结构 → 小结构 → 字数兜底 递归切分。

    RecursiveCharacterTextSplitter 先尝试上一级分隔符，超过 chunk_size
    的片段才会用下一级继续切，因此一个自然段落只要不超过上限就是完整 chunk。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size if chunk_size is None else chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        is_separator_regex=True,
        separators=[
            # Level 1: 章节/标题边界
            r"\n#{1,3}\s",                             # Markdown 标题
            r"\n第[一二三四五六七八九十\d]+[章节条]",      # 第X章/节/条
            r"\n[一二三四五六七八九十\d]+[、．.)]\s",     # 一、二、三、...
            r"\n（[一二三四五六七八九十\d]+）",          # （一）（二）
            # Level 2: 段落边界
            r"\n\n",
            r"\n",
            # Level 3: 句子边界
            r"(?<=。)",
            r"(?<=！)",
            r"(?<=？)",
            # Level 4: 子句边界
            r"(?<=；)",
            r"(?<=，)",
            # Level 5: 兜底 — 按字符硬切
            "",
        ],
    )


# ── Chroma 工厂 ─────────────────────────────────────────────────────


def build_chroma(
    collection_name: str,
    persist_dir: str | Path,
    embedding_function: Embeddings | None = None,
) -> Chroma:
    """创建项目统一的 Chroma 实例，使用余弦相似度。"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function or DashScopeEmbeddings(),
        persist_directory=str(persist_dir),
        client_settings=ChromaSettings(anonymized_telemetry=False),
        collection_metadata={"hnsw:space": "cosine"},
    )


# ── 文档加载器 ──────────────────────────────────────────────────────


def load_file(path: str | Path, *, source_label: str | None = None) -> List[Document]:
    """加载单个文档文件（txt/md/docx），设置来源元数据。

    Args:
        path: 文件路径。
        source_label: 展示给用户的来源名称，默认取文件名。
    """
    path = Path(path)
    label = source_label or path.name
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(
                page_content=text,
                metadata={"source": label, "source_path": str(path)},
            )]
        except Exception:
            logger.exception("加载文本文件失败：%s", path)
            return []

    if suffix == ".docx":
        try:
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = label
                doc.metadata.setdefault("source_path", str(path))
            return docs
        except Exception:
            logger.exception("加载 docx 失败：%s", path)
            return []

    if suffix == ".doc":
        try:
            from src.attachments.document import extract_text
            from src.attachments.models import Attachment

            att = Attachment(
                id=path.stem,
                filename=path.name,
                content_type="application/msword",
                kind="document",
                path=path,
            )
            text = extract_text(att)
            if text:
                return [Document(
                    page_content=text,
                    metadata={"source": label, "source_path": str(path)},
                )]
        except Exception:
            logger.exception("加载 doc 失败：%s", path)
        return []

    logger.warning("不支持的文件类型：%s", suffix)
    return []


def load_files(
    paths: list[Path],
    *,
    skip_temp: bool = True,
) -> List[Document]:
    """批量加载文档文件，跳过 Word 临时锁文件。

    Args:
        paths: 文件路径列表。
        skip_temp: 是否跳过以 ~$ 开头的临时文件。
    """
    documents: List[Document] = []
    for path in sorted(paths):
        if skip_temp and path.name.startswith("~$"):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".md", ".docx"}:
            continue
        documents.extend(load_file(path))
    return documents
