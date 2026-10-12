from __future__ import annotations

"""查询重写模块：指代词替换 + 问题扩写。

在用户问题进入检索或 Agent 之前对其重写，提升检索命中率和回答质量。
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 常见中文指代词，用于快速判断是否需要重写
_PRONOUN_RE = re.compile(
    r"他(?:们)?|她(?:们)?|它(?:们)?|其|该(?:公司|平台|项目|企业|产品|服务|人|类|种)?"
)

REWRITE_SYSTEM_PROMPT = """你是一个查询重写助手。对用户问题进行优化，使其更适合知识库检索。

规则：
1. 指代词替换：将问题中的"他"、"她"、"它"、"他们"、"她们"、"它们"、"其"、"该"等指代词，基于对话历史替换为具体指代的对象名称。
2. 问题扩写：如果问题过于简短或隐含上下文，适当补充使其更完整、更具体。保持用户核心意图不变。
3. 仅输出重写后的问题，不加引号、解释或任何前缀标记。
4. 如果问题已经很清晰且无指代词，直接原样输出问题。"""


@dataclass(frozen=True)
class RewriteResult:
    """重写结果。"""

    original: str
    rewritten: str
    changes: list[str] = field(default_factory=list)


def has_pronoun(text: str) -> bool:
    """快速检查文本是否包含需要消解的指代词。"""
    return bool(_PRONOUN_RE.search(text or ""))


def rewrite_query(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> RewriteResult:
    """重写用户问题：替换指代词，适当扩写。

    Args:
        query: 用户原始问题。
        history: 可选的历史对话，用于指代词消解。
                 格式为 [{"role": "user/assistant", "content": "..."}, ...]。
        api_key: LLM API Key，默认从环境变量读取。
        base_url: LLM API 地址，默认使用 OPENAI_COMPATIBLE_BASE_URL。
        model: 模型名，默认用 deepseek-v4-flash（轻量快速适合重写任务）。

    Returns:
        RewriteResult，包含原始问题、重写后问题和改动列表。
    """
    query = (query or "").strip()
    if not query:
        return RewriteResult(original="", rewritten="", changes=[])

    rewritten = _call_llm_rewrite(query, history, api_key, base_url, model)
    if rewritten and rewritten.strip() != query:
        return RewriteResult(
            original=query,
            rewritten=rewritten.strip(),
            changes=["LLM 重写"],
        )

    return RewriteResult(original=query, rewritten=query, changes=[])


def _build_rewrite_messages(
    query: str,
    history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    """构建发送给 LLM 的消息列表，包含系统提示、历史对话和当前问题。"""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT}
    ]
    if history:
        # 只取最近 10 轮防止上下文过长
        messages.extend(history[-20:])
    messages.append(
        {"role": "user", "content": f"请重写以下问题：\n{query}"}
    )
    return messages


def _call_llm_rewrite(
    query: str,
    history: list[dict[str, str]] | None,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> str | None:
    """调用 LLM 执行重写，失败时返回 None 由调用方兜底。"""
    api_key = (
        api_key
        or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        logger.info("重写模块：未配置 API Key，跳过 LLM 重写")
        return None

    base_url = (
        base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL") or "https://api.deepseek.com"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    model = model or os.getenv("REWRITE_MODEL") or "deepseek-v4-flash"

    messages = _build_rewrite_messages(query, history)

    try:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 500,
            "stream": False,
        }
        req = urllib.request.Request(
            url=f"{base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        content: str = data["choices"][0]["message"].get("content", "")
        content = content.strip()
        logger.info("查询重写完成：%s → %s", query[:60], content[:60])
        return content
    except Exception:
        logger.exception("查询重写 LLM 调用失败")
        return None
