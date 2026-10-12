"""无需模型服务的 BM25 回归测试。"""
import math
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.rag.bm25 import BM25Index, _tokens, top_bm25_indices


class BM25Tests(unittest.TestCase):
    def test_original_examples(self):
        texts = ["全发平台操作指南", "项目供应频道介绍", "RTX 4090 显卡功耗参数"]
        self.assertEqual(top_bm25_indices("平台怎么操作", texts, 2), [0])
        self.assertEqual(top_bm25_indices("RTX 4090 功耗", texts, 2), [2])

    def test_empty_and_no_matches(self):
        for query, texts in [("", ["abc"]), ("?!", ["abc"]), ("abc", []),
                             ("abc", ["", " "]), ("unknown", ["abc"])]:
            self.assertEqual(top_bm25_indices(query, texts, 5), [])

    def test_limit(self):
        for limit in (0, -1):
            self.assertEqual(top_bm25_indices("abc", ["abc", "abc"], limit), [])
        for limit in (True, 1.5, "2"):
            with self.assertRaises(ValueError):
                top_bm25_indices("abc", ["abc"], limit)

    def test_case_width_and_repeated_terms(self):
        index = BM25Index(["RTX 4090", "GPU 3090"])
        self.assertEqual(index.top_indices("ｒｔｘ ４０９０", 2), [0])
        self.assertEqual(index.top_indices("RTX RTX", 2), index.top_indices("rtx", 2))

    def test_stable_ties(self):
        self.assertEqual(BM25Index(["abc"] * 20).top_indices("abc", 3), [0, 1, 2])

    def test_matches_reference_formula(self):
        texts = ["alpha alpha beta", "beta gamma", "alpha", "", "gamma " * 10]
        docs = [_tokens(text) for text in texts]
        avg = sum(map(len, docs)) / len(docs)
        df = Counter(word for doc in docs for word in set(doc))
        for query in ("alpha", "beta gamma", "alpha beta gamma"):
            scores = []
            for doc in docs:
                tf = Counter(doc)
                score = 0
                for word in set(_tokens(query)):
                    if tf[word]:
                        idf = math.log1p((len(docs) - df[word] + .5) / (df[word] + .5))
                        score += idf * tf[word] * 2.5 / (tf[word] + 1.5 * (.25 + .75 * len(doc) / avg))
                scores.append(score)
            expected = sorted((i for i, score in enumerate(scores) if score > 0),
                              key=lambda i: (-scores[i], i))
            self.assertEqual(BM25Index(texts).top_indices(query, 10), expected)

    def test_concurrent_queries_do_not_mutate_index(self):
        index = BM25Index(["alpha", "beta", "gamma"])
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(lambda q: index.top_indices(q, 1), ["alpha", "beta"] * 50))
        self.assertEqual(actual, [[0], [1]] * 50)


if __name__ == "__main__":
    unittest.main()
