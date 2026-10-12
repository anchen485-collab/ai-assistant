"""轻量 BM25：建索引时计算词权重，查询时只处理命中词。"""

from __future__ import annotations

import heapq
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence


class BM25Index:
    """建好后只读，可被并发请求共享；每次查询独立计算分数。"""

    def __init__(self, texts: Sequence[str]) -> None:
        frequencies = [Counter(_tokens(text)) for text in texts]
        lengths = [sum(words.values()) for words in frequencies]
        average_length = (sum(lengths) / len(lengths) if lengths else 0) or 1
        document_frequency = Counter(word for words in frequencies for word in words)

        # 倒排表：词 -> [(文档下标, 这个词为文档贡献的分数)]。
        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        k1, b = 1.5, 0.75
        for index, words in enumerate(frequencies):
            length_factor = k1 * (1 - b + b * lengths[index] / average_length)
            for word, frequency in words.items():
                count = document_frequency[word]
                idf = math.log1p((len(texts) - count + 0.5) / (count + 0.5))
                weight = idf * frequency * (k1 + 1) / (frequency + length_factor)
                postings[word].append((index, weight))
        self._postings = {word: tuple(rows) for word, rows in postings.items()}

    def top_indices(self, query: str, limit: int) -> list[int]:
        if type(limit) is not int:
            raise ValueError("limit 必须为整数")
        if limit <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        # 固定累加顺序及同分顺序，重复查询结果保持一致。
        for word in sorted(set(_tokens(query))):
            for index, weight in self._postings.get(word, ()):
                scores[index] += weight
        return heapq.nlargest(limit, scores, key=lambda index: (scores[index], -index))


def top_bm25_indices(query: str, texts: list[str], limit: int) -> list[int]:
    """兼容原入口；同一批文本反复搜索时应复用 BM25Index。"""
    if type(limit) is not int:
        raise ValueError("limit 必须为整数")
    if limit <= 0 or not _tokens(query):
        return []
    return BM25Index(texts).top_indices(query, limit)


def _tokens(text: str) -> list[str]:
    """中文双字切分；连续英文/数字为一个词，统一全半角和大小写。"""
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    tokens: list[str] = []
    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        if part.isascii() or len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(part[index:index + 2] for index in range(len(part) - 1))
    return tokens
