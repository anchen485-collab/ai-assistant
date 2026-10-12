"""Trace 核心/并发与真实 FastAPI 路由测试，不调用模型或向量库。

app.py 的选定函数通过 AST 原样编译，隔离无关启动导入；因此属于组件测试。
"""
import ast
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import trace


class TraceTests(unittest.TestCase):
    def setUp(self):
        with trace._lock:
            trace._store.clear()
            trace._store_bytes = 0
        trace.clear_trace()

    def tearDown(self):
        trace.clear_trace()

    def test_refs_accept_bad_metadata_and_exclude_body(self):
        refs = trace.source_refs([
            {"source": "a", "chunk": "bad", "text": "private", "distance": float('nan')},
            SimpleNamespace(source="b", chunk=2, rerank_score="0.912345"),
            {"chunk": float('inf'), "rerank_score": "bad"},
        ])
        self.assertEqual(refs[0], {"rank": 1, "source": "a", "chunk": 0})
        self.assertEqual(refs[1]['rerank_score'], 0.9123)
        self.assertEqual(refs[2]['chunk'], 0)
        json.dumps(refs, allow_nan=False)
        self.assertEqual(trace.source_refs(None), [])
        self.assertEqual(len(trace.source_refs([{}] * 1000)), trace._max_sources)

    def test_broken_object_does_not_break_source_refs(self):
        class Broken:
            @property
            def source(self):
                raise RuntimeError('bad metadata')
        self.assertTrue(trace.source_refs([Broken()])[0]['metadata_invalid'])

    def test_finish_is_frozen_and_idempotent(self):
        with patch.object(trace.time, 'perf_counter', return_value=10):
            tid = trace.new_trace()
        with patch.object(trace.time, 'perf_counter', return_value=12):
            trace.finish_trace(tid)
        with patch.object(trace.time, 'perf_counter', return_value=20):
            trace.finish_trace(tid)
            self.assertEqual(trace.get_trace(tid)['total_ms'], 2000)
            self.assertEqual(trace.list_traces()[0]['total_ms'], 2000)
        self.assertIsNone(trace._current_trace_id.get())

    def test_get_and_input_are_independent_snapshots(self):
        tid = trace.new_trace()
        value = {'nested': [1]}
        trace.log_step(tid, 'step', payload=value)
        value['nested'].append(2)
        record = trace.get_trace(tid)
        record['steps'][0]['payload']['nested'].append(3)
        self.assertEqual(trace.get_trace(tid)['steps'][0]['payload'], {'nested': [1]})

    def test_json_sanitization_and_secret_fields(self):
        tid = trace.new_trace()
        cycle = {}; cycle['self'] = cycle
        trace.log_step(tid, 'step', duration_ms=float('inf'), payload=cycle,
                       token='private', nested={'api_key': 'private'}, score=float('nan'), obj=object())
        step = trace.get_trace(tid)['steps'][0]
        self.assertEqual(step['duration_ms'], 0)
        self.assertIsNone(step['score'])
        self.assertEqual(step['token'], '[redacted]')
        self.assertEqual(step['nested']['api_key'], '[redacted]')
        json.dumps(step, allow_nan=False)

    def test_step_and_trace_limits(self):
        tid = trace.new_trace()
        with patch.object(trace, '_max_steps', 2):
            for _ in range(4):
                trace.log_step(tid, 'step')
        record = trace.get_trace(tid)
        self.assertEqual(len(record['steps']), 2)
        self.assertEqual(record['dropped_steps'], 2)
        with patch.object(trace, '_max_trace_bytes', 1):
            trace.log_step(tid, 'step')
        self.assertEqual(trace.get_trace(tid)['dropped_steps'], 3)

    def test_huge_step_has_explicit_truncation(self):
        tid = trace.new_trace()
        trace.log_step(tid, 'step', payload=['x' * 2048] * 64)
        record = trace.get_trace(tid)
        self.assertTrue(record['steps'][0]['payload_truncated'])
        self.assertLess(trace._store_bytes, trace._max_step_bytes)

    def test_global_bytes_and_count_caps(self):
        with patch.object(trace, '_max_store_bytes', 200), patch.object(trace, '_max_traces', 3):
            for _ in range(20):
                tid = trace.new_trace(); trace.log_step(tid, 'step', text='a' * 60)
            self.assertLessEqual(trace._store_bytes, 200)
            self.assertLessEqual(len(trace._store), 3)
            self.assertEqual(trace._store_bytes, sum(r['payload_bytes'] for r in trace._store.values()))

    def test_ttl_cleanup_and_evicted_context(self):
        with patch.object(trace.time, 'perf_counter', return_value=1):
            tid = trace.new_trace(); trace.log_step(tid, 'step')
        with patch.object(trace.time, 'perf_counter', return_value=4000):
            self.assertIsNone(trace.get_trace(tid))
        trace.finish_trace(tid)
        self.assertIsNone(trace._current_trace_id.get())
        self.assertEqual(trace._store_bytes, 0)

    def test_finish_does_not_clear_another_trace(self):
        first = trace.new_trace(); second = trace.new_trace()
        trace.finish_trace(first)
        self.assertEqual(trace._current_trace_id.get(), second)

    def test_zero_limit_and_unknown_trace(self):
        trace.new_trace()
        self.assertEqual(trace.list_traces(0), [])
        self.assertEqual(trace.list_traces(-1), [])
        self.assertIsNone(trace.get_trace('missing'))
        trace.log_step('missing', 'step')
        trace.clear_trace(); trace.log_step_ctx('ignored')

    def test_finished_trace_ignores_late_steps(self):
        tid = trace.new_trace(); trace.finish_trace(tid); trace.log_step(tid, 'late')
        self.assertEqual(trace.get_trace(tid)['steps'], [])

    def test_logger_failure_does_not_break_business(self):
        with patch.object(trace.logger, 'info', side_effect=RuntimeError('disk error')):
            tid = trace.new_trace(); trace.log_step(tid, 'step'); trace.finish_trace(tid)
        self.assertEqual(len(trace.get_trace(tid)['steps']), 1)

    def test_info_logs_do_not_copy_question(self):
        tid = trace.new_trace()
        with self.assertLogs(trace.logger, level='INFO') as logs:
            trace.log_step(tid, 'step', question='secret question')
        self.assertNotIn('secret question', ''.join(logs.output))

    def test_concurrent_eviction_read_write(self):
        def worker(index):
            for _ in range(30):
                tid = trace.new_trace()
                trace.log_step_ctx('worker', owner=index)
                trace.get_trace(tid); trace.list_traces(); trace.finish_trace(tid)
        with patch.object(trace, '_max_traces', 4), ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(16)))
        self.assertLessEqual(len(trace._store), 4)
        self.assertEqual(trace._store_bytes, sum(r['payload_bytes'] for r in trace._store.values()))

    def test_async_context_isolation_and_to_thread(self):
        async def worker(index):
            tid = trace.new_trace()
            await asyncio.sleep(0)
            await asyncio.to_thread(trace.log_step_ctx, 'worker', owner=index)
            trace.finish_trace(tid)
            return index, trace.get_trace(tid)
        async def run():
            return await asyncio.gather(*(worker(i) for i in range(20)))
        for index, record in asyncio.run(run()):
            self.assertEqual(record['steps'][0]['owner'], index)


def load_app_components():
    """只编译实际函数，保留原路由装饰器，mock 模型/附件等外部系统。"""
    import secrets
    import time
    from uuid import uuid4
    from fastapi import FastAPI, Depends, Header, HTTPException, Query, Response, status
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    names = {'ChatRequest', 'chat_stream', 'require_trace_access', 'api_list_traces',
             'api_get_trace', 'stream_chat_events', '_sse'}
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names]
    scope = dict(locals(), asyncio=asyncio, os=os, logging=logging, logger=logging.getLogger('trace-test'),
                 app=FastAPI(), ModelConfig=object, **{n: getattr(trace, n) for n in
                 ('new_trace', 'log_step', 'finish_trace', 'clear_trace', 'set_trace', 'get_trace', 'list_traces')})
    scope.update(validate_mode=lambda mode: mode, normalize_conversation_id=lambda cid: cid,
                 load_attachments=lambda ids: [], resolve_model_for_chat=lambda m: SimpleNamespace(name='test'),
                 _analyze_images=lambda *args: '', _process_document_context=lambda *args: '',
                 has_image_attachments=lambda attachments: False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app.py', 'exec'), scope)
    return scope


# 独立组件组，避免重复执行核心测试。
class AppComponentTests(unittest.TestCase):
    setUp = TraceTests.setUp
    tearDown = TraceTests.tearDown

    def test_trace_routes_access_control_and_limits(self):
        from fastapi.testclient import TestClient
        scope = load_app_components(); client = TestClient(scope['app'])
        tid = trace.new_trace(); token = 't' * 32
        with patch.dict(os.environ, {'TRACE_API_TOKEN': ''}):
            self.assertEqual(client.get('/api/traces').status_code, 404)
            self.assertEqual(client.get('/api/traces/' + tid).status_code, 404)
        with patch.dict(os.environ, {'TRACE_API_TOKEN': 'short'}):
            self.assertEqual(client.get('/api/traces').status_code, 503)
        with patch.dict(os.environ, {'TRACE_API_TOKEN': token}):
            self.assertEqual(client.get('/api/traces').status_code, 401)
            self.assertEqual(client.get('/api/traces/' + tid, headers={'Authorization': 'Bearer wrong'}).status_code, 401)
            headers = {'Authorization': 'Bearer ' + token}
            self.assertEqual(client.get('/api/traces', headers=headers).status_code, 200)
            self.assertEqual(client.get('/api/traces', headers=headers).headers['cache-control'], 'no-store')
            self.assertEqual(client.get('/api/traces/' + tid, headers=headers).json()['trace_id'], tid)
            self.assertEqual(client.get('/api/traces/missing', headers=headers).status_code, 404)
            for limit in ('0', '-1', '101', 'bad'):
                self.assertEqual(client.get('/api/traces?limit=' + limit, headers=headers).status_code, 422)

    def test_http_chat_stream_in_both_modes(self):
        from fastapi.testclient import TestClient
        scope = load_app_components()
        async def answer(*args, **kwargs):
            await asyncio.to_thread(trace.log_step_ctx, 'model:mock')
            yield scope['_sse']({'type': 'delta', 'text': 'hello'})
            yield scope['_sse']({'type': 'done'})
        scope['stream_normal_answer'] = answer
        scope['stream_deep_answer'] = answer
        client = TestClient(scope['app'])
        for mode in ('normal', 'deep'):
            response = client.post('/api/chat/stream', json={'question': 'q', 'mode': mode})
            self.assertEqual(response.status_code, 200)
            frames = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            tid = frames[0]['trace_id']
            self.assertEqual([f['type'] for f in frames], ['meta', 'delta', 'done'])
            self.assertEqual([step['step'] for step in trace.get_trace(tid)['steps']], ['preprocess', 'model:mock'])
            self.assertIn('total_ms', trace._store[tid])

    def test_preprocess_failure_always_finishes(self):
        scope = load_app_components()
        def fail(*args):
            raise ValueError('preprocess failed')
        scope['load_attachments'] = fail
        async def run():
            with self.assertRaises(ValueError):
                await scope['chat_stream'](scope['ChatRequest'](question='question'))
            self.assertIsNone(trace._current_trace_id.get())
        asyncio.run(run())
        record = next(iter(trace._store.values()))
        self.assertIn('total_ms', record)
        self.assertEqual(json.loads(record['steps'][0])['step'], 'preprocess:error')

    def test_stream_binds_context_and_finishes(self):
        scope = load_app_components()
        async def answer(*args, **kwargs):
            trace.log_step_ctx('model:mock')
            yield scope['_sse']({'type': 'done'})
        scope['stream_normal_answer'] = answer
        async def run():
            response = await scope['chat_stream'](scope['ChatRequest'](question='hello'))
            self.assertIsNone(trace._current_trace_id.get())
            return [item async for item in response.body_iterator]
        frames = asyncio.run(run()); tid = json.loads(frames[0][6:])['trace_id']
        self.assertEqual([s['step'] for s in trace.get_trace(tid)['steps']], ['preprocess', 'model:mock'])
        self.assertIn('total_ms', trace._store[tid])

    def test_stream_error_uses_valid_json(self):
        scope = load_app_components()
        async def fail(*args, **kwargs):
            raise ValueError('quote " newline\n')
            yield
        scope['stream_normal_answer'] = fail
        async def run():
            tid = trace.new_trace(); trace.clear_trace()
            frames = [x async for x in scope['stream_chat_events'](tid, 'q', 'normal', object())]
            self.assertIsNone(trace._current_trace_id.get())
            return tid, [json.loads(x[6:]) for x in frames]
        tid, frames = asyncio.run(run())
        self.assertEqual(frames[-2]['type'], 'error'); self.assertEqual(frames[-1]['type'], 'done')
        self.assertIn('total_ms', trace._store[tid])

    def test_stream_close_records_disconnect(self):
        scope = load_app_components()
        async def run():
            tid = trace.new_trace(); trace.clear_trace()
            stream = scope['stream_chat_events'](tid, 'q', 'normal', object())
            await anext(stream); await stream.aclose()
            self.assertIsNone(trace._current_trace_id.get())
            return tid
        tid = asyncio.run(run())
        self.assertEqual(trace.get_trace(tid)['steps'][-1]['step'], 'stream:cancelled')
        self.assertIn('total_ms', trace._store[tid])

    def test_stream_task_cancellation(self):
        scope = load_app_components()
        started = asyncio.Event()
        async def answer(*args, **kwargs):
            started.set(); await asyncio.Event().wait()
            yield
        scope['stream_deep_answer'] = answer
        async def run():
            tid = trace.new_trace(); trace.clear_trace()
            async def consume():
                async for _ in scope['stream_chat_events'](tid, 'q', 'deep', object()):
                    pass
            task = asyncio.create_task(consume())
            await started.wait(); task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return tid
        tid = asyncio.run(run())
        self.assertEqual(trace.get_trace(tid)['steps'][-1]['step'], 'stream:cancelled')
        self.assertIn('total_ms', trace._store[tid])


class RagToolTraceTests(unittest.TestCase):
    setUp = TraceTests.setUp
    tearDown = TraceTests.tearDown

    def load_tool(self, search):
        import time
        from dataclasses import dataclass
        @dataclass
        class ToolResult:
            observation: str
            hits: list
        path = Path(__file__).resolve().parents[1] / 'src/tools/rag.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_rag_search')
        scope = dict(time=time, logger=logging.getLogger('rag-trace-test'), ToolResult=ToolResult,
                     has_pronoun=lambda q: True, rewrite_query=lambda q: SimpleNamespace(rewritten='rewritten'),
                     search=search, log_step_ctx=trace.log_step_ctx, source_refs=trace.source_refs)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        return scope['run_rag_search']

    def test_rewrite_trace_preserves_result(self):
        hit = SimpleNamespace(source='doc', chunk=3, text='answer', distance=None, rerank_score=0.9)
        def search(query, k):
            self.assertEqual(query, 'rewritten'); self.assertEqual(k, 5)
            return [hit]
        run = self.load_tool(search); tid = trace.new_trace()
        result = run('original')
        step = trace.get_trace(tid)['steps'][-1]
        self.assertEqual(result.hits, [hit])
        self.assertEqual(step['original_query'], 'original')
        self.assertEqual(step['query'], 'rewritten')
        self.assertIs(step['query_rewritten'], True)
        self.assertEqual(step['selected_sources'][0]['rerank_score'], 0.9)

    def test_empty_and_failed_search_remain_empty(self):
        for failure in (None, FileNotFoundError('missing'), RuntimeError('failure')):
            def search(query, k):
                if failure is not None:
                    raise failure
                return []
            run = self.load_tool(search); tid = trace.new_trace()
            self.assertEqual(run('question').hits, [])
            self.assertEqual(trace.get_trace(tid)['steps'][-1]['ok'], failure is None)


if __name__ == '__main__':
    unittest.main()
