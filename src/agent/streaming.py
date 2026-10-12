from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agent.deep.prompts import build_user_prompt
from src.agent.deep.react_agent import (
    DeepThinkingAgent,
    build_message_content as build_deep_message_content,
)
from src.agent.langchain_result import extract_sources_from_messages, parse_tool_payload
from src.agent.normal.react_agent import (
    NormalAgent,
    build_message_content as build_normal_message_content,
)
from src.agent.normal.service import recommendations
from src.core.config import ModelConfig, settings
from src.memory.service import build_context, save_turn
from src.trace import log_step_ctx, source_refs

logger = logging.getLogger(__name__)


def _sse(payload: dict) -> str:
    """把字典序列化成 SSE data 事件。"""

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# 遇到自然断点或缓冲满一定长度就 flush，避免前端一直等长句结束。
_NATURAL_BREAKS = frozenset({
    "。", "，", "；", "：", "！", "？",
    "\n", ".", ",", "!", "?", ";", ":", " ",
})


async def process_agent_stream(agent_graph, messages: list[dict], config: dict):
    """实时流式处理 LangGraph 事件，并输出前端可消费的事件对象。"""

    call_buf = ""
    call_has_tc = False
    pending_steps: dict[str, dict] = {}
    completed_steps: list[dict] = []
    step_counter = 0
    final_answer = ""
    final_sources: list[dict] = []

    try:
        async for event in agent_graph.astream_events(
            {"messages": messages}, config=config, version="v2"
        ):
            kind = event.get("event", "")
            data = event.get("data", {})

            if kind == "on_chat_model_start":
                call_buf = ""
                call_has_tc = False

            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", "")
                tc_chunks = getattr(chunk, "tool_call_chunks", None) or []

                if tc_chunks:
                    call_has_tc = True

                if isinstance(content, str) and content:
                    call_buf += content
                    if call_has_tc:
                        continue
                    should_flush = content[-1] in _NATURAL_BREAKS or len(call_buf) >= 4
                    if should_flush:
                        final_answer += call_buf
                        yield {"type": "delta", "text": call_buf}
                        call_buf = ""

            elif kind == "on_chat_model_end":
                output = data.get("output")
                if output is not None and getattr(output, "tool_calls", None):
                    for tc in output.tool_calls:
                        tc_id = str(tc.get("id", ""))
                        pending_steps[tc_id] = {
                            "thought": call_buf.strip(),
                            "action": str(tc.get("name", "")),
                            "action_input": tc.get("args", {}),
                        }
                    call_buf = ""
                elif call_buf:
                    final_answer += call_buf
                    yield {"type": "delta", "text": call_buf}
                    call_buf = ""

            elif kind == "on_tool_end":
                tool_output = data.get("output")
                observation = _find_tool_output(tool_output) if tool_output is not None else ""
                tc_id = ""
                if tool_output is not None and hasattr(tool_output, "tool_call_id"):
                    tc_id = str(tool_output.tool_call_id or "")

                matched = pending_steps.pop(tc_id, None) if tc_id else None
                if matched is None and pending_steps:
                    _, matched = pending_steps.popitem()

                if matched:
                    step_counter += 1
                    step = {**matched, "observation": observation}
                    completed_steps.append(step)
                    yield {
                        "type": "step",
                        "index": step_counter,
                        "step": step,
                    }

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = data.get("output", {})
                if isinstance(output, dict):
                    final_messages = output.get("messages", [])
                    if final_messages:
                        final_sources = extract_sources_from_messages(final_messages)

    except Exception as exc:
        logger.exception("流式处理异常")
        yield {"type": "error", "text": str(exc)}

    yield {
        "type": "end",
        "answer": final_answer,
        "sources": final_sources,
        "steps": completed_steps,
    }


def _find_tool_output(tool_output: Any) -> str:
    """从 LangChain 工具消息中取出可展示的 observation。"""

    content = tool_output
    if hasattr(tool_output, "content"):
        content = tool_output.content
    parsed = parse_tool_payload(content)
    return str(parsed.get("observation", ""))


async def stream_normal_answer(
    question: str,
    model_config: ModelConfig | None = None,
    conversation_id: str | None = None,
    attachment_context: str = "",
    image_urls: list[str] | None = None,
    attachment_ids: list[str] | None = None,
):
    """普通问答 Agent 流式输出，产生 SSE 字符串。"""

    config = model_config or settings.model_config(settings.default_model)
    yield _sse({"type": "status", "text": "普通问答模式启动", "model": config.name})

    cid, context_messages = build_context(conversation_id)

    if not config.api_key:
        yield _sse({
            "type": "error",
            "text": f"模型 {config.name} 缺少 API Key，请在 .env 中配置对应厂商的 API Key。",
        })
        yield _sse({"type": "done"})
        return

    agent = NormalAgent(
        model=config.name,
        api_key=config.api_key,
        base_url=config.base_url,
    )

    messages = list(context_messages) if context_messages else []
    content = build_normal_message_content(question, attachment_context, image_urls or [])
    messages.append({"role": "user", "content": content})

    async for sse in _stream_agent_answer(
        agent=agent,
        messages=messages,
        question=question,
        mode="normal",
        model_name=config.name,
        conversation_id=cid,
        attachment_ids=attachment_ids or [],
        attachment_context=attachment_context,
    ):
        yield sse


async def stream_deep_answer(
    question: str,
    model_config: ModelConfig | None = None,
    conversation_id: str | None = None,
    attachment_context: str = "",
    image_urls: list[str] | None = None,
    attachment_ids: list[str] | None = None,
):
    """深度思考 Agent 流式输出，产生 SSE 字符串。"""

    config = model_config or settings.model_config(settings.default_model)
    yield _sse({"type": "status", "text": "深度思考模式启动", "model": config.name})

    cid, context_messages = build_context(conversation_id)

    if not config.api_key:
        yield _sse({
            "type": "error",
            "text": f"模型 {config.name} 缺少 API Key，请在 .env 中配置对应厂商的 API Key。",
        })
        yield _sse({"type": "done"})
        return

    agent = DeepThinkingAgent(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.name,
        max_steps=settings.deep_agent_max_steps,
        temperature=settings.deep_agent_temperature,
        timeout=settings.deep_agent_timeout,
    )
    agent.reset_reflection()

    messages = list(context_messages) if context_messages else []
    user_prompt = build_user_prompt(question, context_messages)
    if attachment_context:
        user_prompt = f"{user_prompt}\n\n{attachment_context}"
    content = build_deep_message_content(user_prompt, image_urls or [])
    messages.append({"role": "user", "content": content})

    async for sse in _stream_agent_answer(
        agent=agent,
        messages=messages,
        question=question,
        mode="deep",
        model_name=config.name,
        conversation_id=cid,
        attachment_ids=attachment_ids or [],
        attachment_context=attachment_context,
    ):
        yield sse


async def _stream_agent_answer(
    agent: NormalAgent | DeepThinkingAgent,
    messages: list[dict],
    question: str,
    mode: str,
    model_name: str,
    conversation_id: str,
    attachment_ids: list[str],
    attachment_context: str = "",
):
    """复用普通/深度 Agent 的流式收尾逻辑。"""

    graph_config = {"recursion_limit": max(6, agent.max_steps * 4 + 4)}
    final_answer = ""
    t0 = time.perf_counter()

    async for event in process_agent_stream(agent.agent, messages, graph_config):
        if event["type"] == "delta":
            final_answer += event["text"]
            yield _sse(event)
        elif event["type"] == "step":
            yield _sse(event)
        elif event["type"] == "error":
            log_step_ctx("agent:stream", ok=False,
                         duration_ms=(time.perf_counter() - t0) * 1000,
                         agent=mode, model=model_name, error=event["text"])
            yield _sse(event)
            yield _sse({"type": "done"})
            return
        elif event["type"] == "end":
            final_answer = event.get("answer") or final_answer
            log_step_ctx("agent:stream", ok=True,
                         duration_ms=(time.perf_counter() - t0) * 1000,
                         agent=mode, model=model_name,
                         steps=len(event.get("steps", [])),
                         sources=len(event.get("sources", [])),
                         source_refs=source_refs(event.get("sources", [])),
                         answer_len=len(final_answer))
            if final_answer:
                try:
                    save_turn(conversation_id, question, final_answer, attachment_context)
                except Exception:
                    logger.exception("保存对话失败")
            yield _sse({
                "type": "metadata",
                "recommendations": recommendations(question),
                "sources": event.get("sources", []),
                "steps": event.get("steps", []),
                "mode": mode,
                "model": model_name,
                "conversation_id": conversation_id,
                "attachments": attachment_ids,
            })

    yield _sse({"type": "done"})
