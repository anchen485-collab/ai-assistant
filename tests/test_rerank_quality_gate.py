"""质量门控回归：真实模块和 SDK 响应，仅替换召回、网络调用与等待。"""
import importlib
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import call, patch

from dashscope.api_entities.dashscope_response import ReRankResponse, ReRankOutput
from dashscope.rerank import TextReRank
from requests import Response
from requests.exceptions import Timeout

rag = importlib.import_module('src.rag.search')
rag_tool = importlib.import_module('src.tools.rag')
trace = importlib.import_module('src.trace')
REAL_RERANK_CALL = TextReRank.call


def response(results, status=200):
    return ReRankResponse(status_code=status, output={'results': results})


def row(index=0, score=0.8):
    return {'index': index, 'relevance_score': score}


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.hits = [
            rag.SearchHit('内部片段 A', 'a.md', 1, distance=0.2, governance={
                'document_id': 'internal-a', 'knowledge_id': 'policy-a',
                'owner': 'test', 'authority_rank': 1, 'version': '1.0',
                'status': 'active', 'published_at': '2020-01-01',
                'effective_from': '2020-01-01', 'effective_to': '',
            }),
            rag.SearchHit('内部片段 B', 'b.md', 2, governance={
                'document_id': 'internal-b', 'knowledge_id': 'policy-b',
                'owner': 'test', 'authority_rank': 1, 'version': '1.0',
                'status': 'active', 'published_at': '2020-01-01',
                'effective_from': '2020-01-01', 'effective_to': '',
            }),
        ]
        # 整体替换配置对象，不读取/发送本机 .env 密钥。
        self.settings = SimpleNamespace(
            retrieval_k=5, rerank_fetch_k=10, rerank_threshold=0.3,
            dashscope_api_key='unit-test-key', rerank_model='unit-test-model',
        )
        self._patch('src.rag.search.settings', self.settings)
        self.api = self._patch('dashscope.rerank.TextReRank.call')
        self.sleep = self._patch('src.rag.search.time.sleep')
        self.mmr = self._patch('src.rag.search._mmr_recall', return_value=self.hits)
        self.bm25 = self._patch('src.rag.search._bm25_recall', return_value=[])
        # 混合召回入口使用一致性读锁；门控测试不访问真实索引。
        # create=True 也兼容加固前的快照，便于同一套测试做前后对照。
        self._patch('src.rag.search._hybrid_recall', side_effect=self._recall, create=True)
        self._patch('src.rag.search.ensure_store', side_effect=AssertionError('测试禁止访问真实知识库'))
        self.api.return_value = response([row(0), row(1, 0.1)])
        self.trace_id = trace.new_trace()
        self.addCleanup(trace.clear_trace)

    def _recall(self, query, limit):
        mmr, bm25 = self.mmr(query), self.bm25(query)
        merged = rag._rrf_merge(mmr, bm25)
        return mmr, bm25, merged, rag._filter_effective(merged)

    def _patch(self, *args, **kwargs):
        patcher = patch(*args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def gate(self):
        steps = trace.get_trace(self.trace_id)['steps']
        return [step for step in steps if step['step'] == 'rag:quality_gate'][-1]

    def test_mixed_scores_accept_only_qualified(self):
        found = rag.search('报销制度')
        self.assertEqual([hit.source for hit in found], ['a.md'])
        self.assertEqual(found[0].rerank_score, 0.8)
        self.assertEqual(self.gate()['decision'], 'accepted')

    def test_all_low_scores_are_blocked(self):
        self.api.return_value = response([row(0, 0.1), row(1, 0.2)])
        self.assertEqual(rag.search('报销制度'), [])
        self.assertEqual(self.gate()['decision'], 'blocked_low_rerank_score')

    def test_empty_success_blocks_instead_of_fallback(self):
        self.api.return_value = response([])
        self.assertEqual(rag.search('报销制度'), [])
        self.assertEqual(self.gate()['decision'], 'blocked_empty_or_invalid_rerank')
        self.assertFalse(self.gate()['ok'])

    def test_empty_ranked_does_not_crash(self):
        with patch.object(rag, '_rerank', return_value=[]):
            self.assertEqual(rag.search('报销制度'), [])

    def test_threshold_is_inclusive_and_zero_is_valid(self):
        for threshold in (0.0, 0.3, 1.0):
            with self.subTest(threshold=threshold):
                self.settings.rerank_threshold = threshold
                self.api.return_value = response([row(0, threshold)])
                self.assertEqual(len(rag.search('报销制度')), 1)

    def test_invalid_threshold_fails_before_recall(self):
        for value in (None, True, '0.3', -0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(threshold=value):
                self.settings.rerank_threshold = value
                with self.assertRaises(ValueError):
                    rag.search('报销制度')
        self.mmr.assert_not_called()
        self.api.assert_not_called()

    def test_invalid_scores_never_become_zero_or_pass(self):
        self.settings.rerank_threshold = 0
        for value in (None, True, False, '0.9', {}, [], -1, 1.1, 10**1000,
                      float('nan'), float('inf'), -float('inf')):
            with self.subTest(score=repr(value)[:30]):
                self.api.return_value = response([row(0, value)])
                self.assertEqual(rag.search('报销制度'), [])

    def test_invalid_indices_never_select_a_document(self):
        for index in (-1, 2, True, False, 0.0, '0', None):
            with self.subTest(index=index):
                self.api.return_value = response([row(index)])
                self.assertEqual(rag.search('报销制度'), [])

    def test_malformed_payloads_block_without_retry(self):
        for output in (None, [], 'bad', {}, {'results': None}, {'results': {}},
                       {'results': [None, [], {}, {'relevance_score': 0.9}]}):
            with self.subTest(output=output):
                self.api.reset_mock()
                self.api.return_value = SimpleNamespace(status_code=200, output=output)
                self.assertEqual(rag.search('报销制度'), [])
                self.api.assert_called_once()
        self.sleep.assert_not_called()

    def test_valid_rows_survive_beside_invalid_rows(self):
        self.api.return_value = response([None, row(-1), row(0), row(1, 'bad')])
        self.assertEqual([hit.source for hit in rag.search('报销制度')], ['a.md'])

    def test_duplicate_index_uses_first_valid_row_once(self):
        self.api.return_value = response([row(0, None), row(0, 0.4), row(0, 0.99)])
        found = rag.search('报销制度')
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].rerank_score, 0.4)

    def test_original_metadata_and_text_preserved(self):
        self.api.return_value = response([{**row(0), 'document': {'text': '服务端文本'}}])
        found = rag.search('报销制度')[0]
        self.assertEqual(found, replace(self.hits[0], rerank_score=0.8))
        self.assertIsNone(self.hits[0].rerank_score)
        self.assertEqual(rag.serialize_hits([found])[0]['governance'], self.hits[0].governance)

    def test_full_candidate_rerank_before_governance_top_k(self):
        self.hits[1] = replace(self.hits[1], governance={**self.hits[1].governance, 'authority_rank': 9})

        def service(**kwargs):
            # 模拟服务按相关性 top_n 截断：权威 B 分数略低，但也过线。
            scores = {'内部片段 A': 0.99, '内部片段 B': 0.4}
            rows = [row(i, scores[text]) for i, text in enumerate(kwargs['documents'])]
            rows.sort(key=lambda item: item['relevance_score'], reverse=True)
            return response(rows[:kwargs['top_n']])

        self.api.side_effect = service
        self.assertEqual([hit.source for hit in rag.search('报销制度', k=1)], ['b.md'])
        self.assertEqual(self.api.call_args.kwargs['top_n'], 2)

    def test_inactive_document_excluded_before_api(self):
        self.hits[1] = replace(self.hits[1], governance={**self.hits[1].governance, 'status': 'inactive'})
        self.api.return_value = response([row(0)])
        self.assertEqual(len(rag.search('报销制度')), 1)
        self.assertEqual(self.api.call_args.kwargs['documents'], ['内部片段 A'])

    def test_no_candidates_skips_api(self):
        self.mmr.return_value = []
        self.assertEqual(rag.search('报销制度'), [])
        self.api.assert_not_called()

    def test_no_key_preserves_explicit_degraded_fallback(self):
        self.settings.dashscope_api_key = ''
        self.assertEqual(rag.search('报销制度', k=1), self.hits[:1])
        self.api.assert_not_called()
        self.assertFalse(self.gate()['ok'])
        self.assertFalse(self.gate()['quality_gate_applied'])

    def test_transient_http_errors_retry_then_recover(self):
        for status in (408, 429, 500, 503):
            with self.subTest(status=status):
                self.api.reset_mock()
                self.sleep.reset_mock()
                self.api.side_effect = [response([], status), response([row(0)])]
                self.assertEqual(len(rag.search('报销制度')), 1)
                self.assertEqual(self.api.call_count, 2)
                self.sleep.assert_called_once_with(0.5)

    def test_retry_exhaustion_is_bounded_and_marked_degraded(self):
        self.api.side_effect = [response([], 503)] * 3
        self.assertEqual(rag.search('报销制度'), self.hits)
        self.assertEqual(self.api.call_count, 3)
        self.assertEqual(self.sleep.call_args_list, [call(0.5), call(1.0)])
        self.assertEqual(self.gate()['decision'], 'rerank_unavailable_use_governed_rrf')
        self.assertFalse(self.gate()['ok'])

    def test_network_timeout_retries_with_request_timeout(self):
        self.api.side_effect = [Timeout('do-not-log-sensitive-value'), response([row(0)])]
        with self.assertLogs(rag.logger, level='WARNING') as logs:
            self.assertEqual(len(rag.search('报销制度')), 1)
        self.assertNotIn('do-not-log-sensitive-value', '\n'.join(logs.output))
        self.assertEqual(self.api.call_count, 2)
        self.assertEqual(self.api.call_args.kwargs['request_timeout'], 10)

    def test_permanent_http_errors_do_not_retry(self):
        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                self.api.reset_mock()
                self.api.return_value = response([], status)
                self.assertEqual(rag.search('报销制度'), self.hits)
                self.api.assert_called_once()
        self.sleep.assert_not_called()

    def test_unexpected_sdk_error_blocks_without_retry_or_logging_body(self):
        self.api.side_effect = ValueError('do-not-log-sensitive-value')
        with self.assertLogs(rag.logger, level='WARNING') as logs:
            self.assertEqual(rag.search('报销制度'), [])
        self.api.assert_called_once()
        self.sleep.assert_not_called()
        self.assertNotIn('do-not-log-sensitive-value', '\n'.join(logs.output))

    def test_invalid_response_status_blocks(self):
        for status in (None, '200', True, -1, 600):
            with self.subTest(status=status):
                self.api.return_value = SimpleNamespace(status_code=status, output={'results': [row(0)]})
                self.assertEqual(rag.search('报销制度'), [])

    def test_real_sdk_nested_output_is_supported(self):
        self.api.return_value = ReRankResponse(status_code=200, output=ReRankOutput(results=[row(0)]))
        self.assertEqual(len(rag.search('报销制度')), 1)

    def test_real_sdk_transport_receives_timeout_and_parses_result(self):
        wire_response = Response()
        wire_response.status_code = 200
        wire_response._content = json.dumps({'output': {'results': [row(0)]}}).encode()
        with patch.object(TextReRank, 'call', side_effect=REAL_RERANK_CALL), \
                patch('requests.Session.post', return_value=wire_response) as post:
            found = rag.search('报销制度')
        self.assertEqual(len(found), 1)
        self.assertEqual(post.call_args.kwargs['timeout'], 10)
        body = json.loads(post.call_args.kwargs['data'])
        self.assertEqual(body['input']['documents'], ['内部片段 A', '内部片段 B'])
        self.assertNotIn('request_timeout', body.get('parameters', {}))

    def test_real_sdk_malformed_success_cannot_bypass_gate(self):
        # SDK 自身解析错误同样要拦截，不允许变成“服务不可用”放行原候选。
        for body in ('not json', '{"output":{"results":[null]}}', '{"output":null}'):
            with self.subTest(body=body):
                wire_response = Response()
                wire_response.status_code = 200
                wire_response._content = body.encode()
                with patch.object(TextReRank, 'call', side_effect=REAL_RERANK_CALL), \
                        patch('requests.Session.post', return_value=wire_response) as post:
                    self.assertEqual(rag.search('报销制度'), [])
                post.assert_called_once()
        self.sleep.assert_not_called()

    def test_tool_returns_no_sources_when_blocked(self):
        self.api.return_value = response([row(0, 0.1)])
        with patch.object(rag_tool, 'has_pronoun', return_value=False):
            result = rag_tool.rag_search.invoke({'query': '报销制度', 'k': 5})
        self.assertEqual(result['sources'], [])
        self.assertIn('没有检索到', result['observation'])
        self.assertNotIn('内部片段', result['observation'])

    def test_gate_trace_contains_no_document_body(self):
        rag.search('报销制度')
        self.assertNotIn('内部片段', str(self.gate()))
        self.assertEqual(self.gate()['selected'][0]['source'], 'a.md')


if __name__ == '__main__':
    unittest.main()
