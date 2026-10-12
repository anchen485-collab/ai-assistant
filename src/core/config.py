from __future__ import annotations

"""项目配置：从 .env 文件和环境变量加载运行参数。"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# 项目根目录 = 当前文件向上两级 (src/core/config.py -> 项目根)
ROOT_DIR = Path(__file__).resolve().parents[2]
# 启动时加载一次 .env，让本地开发的密钥/路径不进版本库
load_dotenv(ROOT_DIR / ".env")


def env_path(name: str, default: Path) -> Path:
    """读取环境变量中的路径，并展开 ~ 为用户目录。"""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def first_env(*names: str, default: str = "") -> str:
    """按优先级读取环境变量，方便兼容不同 API 服务商的命名。"""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def env_list(name: str, defaults: list[str]) -> list[str]:
    """读取逗号分隔的环境变量列表；配置存在时完全使用配置值。"""

    raw_value = os.getenv(name)
    source = raw_value if raw_value is not None else ",".join(defaults)
    values = [item.strip() for item in source.split(",") if item.strip()]
    merged: list[str] = []
    for item in values:
        if item and item not in merged:
            merged.append(item)
    return merged


MODEL_ALIASES = {
    # DeepSeek 2026-07-24 起模型名变更，保留旧名兼容浏览器缓存和历史 .env。
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-pro",
}


def canonical_model_name(model: str | None) -> str:
    """把历史模型名映射成当前模型服务可识别的模型名。"""

    selected = (model or "").strip()
    return MODEL_ALIASES.get(selected, selected)


def canonical_model_list(models: list[str]) -> list[str]:
    """规范化模型列表并去重，避免旧模型名继续暴露给前端。"""

    normalized: list[str] = []
    for model in models:
        selected = canonical_model_name(model)
        if selected and selected not in normalized:
            normalized.append(selected)
    return normalized


def env_choice(name: str, options: list[str], default: str) -> str:
    """读取一个必须在候选列表中的配置项。"""

    value = canonical_model_name(os.getenv(name, default))
    return value if value in options else options[0]


@dataclass(frozen=True)
class ModelConfig:
    """单个模型的后端配置，不直接返回给前端。"""

    name: str
    api_key: str
    base_url: str


class Settings:
    """集中存放各类运行参数；其它模块统一通过 `settings` 单例读取。"""

    kb_source_dir: Path = env_path(
        "KB_SOURCE_DIR", ROOT_DIR / "src" / "rag" / "data"
    )
    chroma_dir: Path = env_path("CHROMA_DIR", ROOT_DIR / "data" / "chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "qf_knowledge_base")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    rerank_fetch_k: int = int(os.getenv("RERANK_FETCH_K", "10"))
    rerank_threshold: float = float(os.getenv("RERANK_THRESHOLD", "0.3"))
    rerank_model: str = os.getenv("RERANK_MODEL", "qwen3-rerank")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # 深度思考 Agent 使用 OpenAI 兼容接口。
    # DeepSeek V4 thinking 模式可配置为：
    # OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com
    # DEEP_AGENT_MODEL=deepseek-v4-pro
    openai_compatible_api_key: str = first_env(
        "OPENAI_COMPATIBLE_API_KEY",
        "DEEP_AGENT_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    openai_compatible_base_url: str = os.getenv(
        "OPENAI_COMPATIBLE_BASE_URL", "https://api.deepseek.com"
    )
    deep_agent_model: str = canonical_model_name(
        os.getenv("DEEP_AGENT_MODEL", "deepseek-v4-pro")
    )
    deep_agent_temperature: float = float(os.getenv("DEEP_AGENT_TEMPERATURE", "0.2"))
    deep_agent_max_steps: int = int(os.getenv("DEEP_AGENT_MAX_STEPS", "12"))
    deep_agent_timeout: int = int(os.getenv("DEEP_AGENT_TIMEOUT", "90"))

    # 模型选择只维护一份列表；normal/deep 只决定走哪个 Agent。
    model_options: list[str] = canonical_model_list(
        env_list(
            "MODEL_OPTIONS",
            [llm_model, "deepseek-v4-flash", deep_agent_model],
        )
    )
    default_model: str = env_choice("DEFAULT_MODEL", model_options, model_options[0])
    text_model: str = canonical_model_name(os.getenv("TEXT_MODEL", default_model))

    # 用户上传的图片和文件先落到本地，再由聊天接口按附件类型处理。
    upload_dir: Path = env_path("UPLOAD_DIR", ROOT_DIR / "data" / "uploads")
    upload_max_mb: int = int(os.getenv("UPLOAD_MAX_MB", "5"))
    upload_image_max_count: int = int(os.getenv("UPLOAD_IMAGE_MAX_COUNT", "3"))
    upload_total_max_mb: int = int(os.getenv("UPLOAD_TOTAL_MAX_MB", "12"))
    vision_image_max_side: int = int(os.getenv("VISION_IMAGE_MAX_SIDE", "1280"))
    vision_image_jpeg_quality: int = int(os.getenv("VISION_IMAGE_JPEG_QUALITY", "80"))
    vision_payload_max_mb: int = int(os.getenv("VISION_PAYLOAD_MAX_MB", "8"))
    attachment_text_max_chars: int = int(os.getenv("ATTACHMENT_TEXT_MAX_CHARS", "8000"))
    # 超过此阈值的文档附件不直接拼入 prompt，改为临时索引让 Agent 按需检索
    attachment_index_threshold_chars: int = int(os.getenv("ATTACHMENT_INDEX_THRESHOLD_CHARS", "1000"))

    # ── 阿里云 OSS 静态资源存储 ──
    # 只存 agent 相关资源：上传附件、图片生成结果、图片生成历史。
    oss_enabled: bool = os.getenv("OSS_ENABLED", "false").lower() == "true"
    oss_endpoint: str = os.getenv("OSS_ENDPOINT", "")
    oss_bucket: str = os.getenv("OSS_BUCKET", "")
    oss_access_key_id: str = os.getenv("OSS_ACCESS_KEY_ID", "")
    oss_access_key_secret: str = os.getenv("OSS_ACCESS_KEY_SECRET", "")
    oss_uploads_prefix: str = os.getenv("OSS_UPLOADS_PREFIX", "uploads")
    oss_history_prefix: str = os.getenv("OSS_HISTORY_PREFIX", "history")
    oss_image_result_prefix: str = os.getenv("OSS_IMAGE_RESULT_PREFIX", "image-result")

    # ── 多模态 / 视觉模型 ──
    vision_model: str = os.getenv("VISION_MODEL", "qwen-vl-max")
    vision_api_key: str = os.getenv("VISION_API_KEY", dashscope_api_key)
    vision_base_url: str = os.getenv(
        "VISION_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # 视觉分析结果缓存目录（按图片内容哈希存储，相同图片命中缓存）
    analyses_dir: Path = env_path("ANALYSES_DIR", ROOT_DIR / "data" / "analyses")
    # 图片分析最大图片大小（MB）
    image_analysis_max_mb: int = int(os.getenv("IMAGE_ANALYSIS_MAX_MB", "20"))
    # 视频抽帧最大数量
    max_video_frames: int = int(os.getenv("MAX_VIDEO_FRAMES", "10"))

    # ── 图片生成模型 ──
    # DashScope 文生图返回的图片 URL 有有效期，后端会下载到本地目录后再返回给前端。
    image_generation_model: str = os.getenv("IMAGE_GENERATION_MODEL", "wan2.6-t2i")
    image_generation_api_key: str = first_env(
        "IMAGE_GENERATION_API_KEY",
        "DASHSCOPE_API_KEY",
    )
    image_generation_base_url: str = os.getenv("IMAGE_GENERATION_BASE_URL", "")
    image_generation_size: str = os.getenv("IMAGE_GENERATION_SIZE", "1280*1280")
    image_generation_count: int = int(os.getenv("IMAGE_GENERATION_COUNT", "1"))
    image_generation_prompt_extend: bool = os.getenv(
        "IMAGE_GENERATION_PROMPT_EXTEND", "true"
    ).lower() != "false"
    image_generation_watermark: bool = os.getenv(
        "IMAGE_GENERATION_WATERMARK", "false"
    ).lower() == "true"
    generated_image_dir: Path = env_path(
        "GENERATED_IMAGE_DIR", ROOT_DIR / "data" / "generated"
    )
    generated_image_history_dir: Path = env_path(
        "GENERATED_IMAGE_HISTORY_DIR", generated_image_dir / "history"
    )
    # 图片编辑使用 DashScope 图生图/图片编辑模型；未配置 base_url 时只影响“修改上一张图”场景。
    image_edit_model: str = os.getenv("IMAGE_EDIT_MODEL", "wan2.6-image")
    image_edit_base_url: str = os.getenv(
        "IMAGE_EDIT_BASE_URL", image_generation_base_url
    )
    image_edit_prompt_extend: bool = os.getenv(
        "IMAGE_EDIT_PROMPT_EXTEND", "true"
    ).lower() != "false"
    image_edit_watermark: bool = os.getenv(
        "IMAGE_EDIT_WATERMARK", "false"
    ).lower() == "true"

    # ── 图片分类 Prompt（轻量，只输出类型代码） ──
    image_classify_prompt: str = (
        "判断这张图片属于以下哪种类型，只输出一个字母代码，不要输出任何其他文字。\n"
        "A-界面截图（网页/APP/系统界面、表单、弹窗、报错页面）\n"
        "B-文档表格（合同、发票、资质证书、表格、证件扫描件）\n"
        "C-实物照片（产品、设备、场景、人物、物品拍摄）\n"
        "D-图表数据（折线图、柱状图、流程图、架构图、脑图）\n"
        "E-聊天记录（微信/企业微信/钉钉等对话框截图）"
    )

    # ── 图片分析 Prompt（按类型，输出结构化 JSON） ──
    image_analyze_prompts: dict[str, str] = {
        "A": (
            "这是一张界面截图。请仔细识别图片中的所有内容，以 JSON 格式输出。\n"
            "输出结构：\n"
            '{\n'
            '  "image_type": "screenshot",\n'
            '  "topic": "页面主题/标题（简短概括）",\n'
            '  "summary": "用一句话描述这个页面在展示什么、用户在做什么",\n'
            '  "ocr_text": "页面上所有可见文字，逐行提取，保留原文",\n'
            '  "elements": [\n'
            '    {"type": "button/input/select/tab/link/text/label/alert/error/icon",\n'
            '     "label": "元素显示文本",\n'
            '     "status": "enabled/disabled/active/error/empty/filled",\n'
            '     "detail": "补充描述（颜色、位置、异常状态等）"}\n'
            '  ],\n'
            '  "errors": ["页面上出现的错误提示或异常信息列表"],\n'
            '  "user_intent_hint": "推测用户上传这张截图想咨询什么问题"\n'
            '}\n'
            '要求：\n'
            '1. ocr_text 必须逐行提取所有可见文字，不要遗漏\n'
            '2. elements 中每个界面元素都要描述，特别是红色/黄色标注的异常元素\n'
            '3. errors 中列出所有报错信息、必填项为空、格式校验失败等\n'
            '4. 严格只输出 JSON，不要包含 markdown 代码块标记或其他文字'
        ),
        "B": (
            "这是一份文档/表格/证件类图片。请仔细识别图片中的所有内容，以 JSON 格式输出。\n"
            "输出结构：\n"
            '{\n'
            '  "image_type": "document",\n'
            '  "topic": "文档主题/类型（合同/发票/资质/证明/表格/证件/其他）",\n'
            '  "summary": "用一句话概括文档的核心内容",\n'
            '  "ocr_text": "文档上所有可见文字，逐行提取，保留原文格式",\n'
            '  "fields": [\n'
            '    {"key": "字段名（如合同编号/金额/日期/甲方/乙方）",\n'
            '     "value": "字段值",\n'
            '     "note": "补充说明（手写/盖章/模糊/缺失等）"}\n'
            '  ],\n'
            '  "tables": [\n'
            '    {"caption": "表格标题", "headers": ["列1", "列2"], "rows": [["值1", "值2"]]}\n'
            '  ],\n'
            '  "stamps": ["印章/签名信息"],\n'
            '  "user_intent_hint": "推测用户上传这份文档想问什么"\n'
            '}\n'
            '要求：\n'
            '1. fields 要提取所有可见的结构化字段（编号、金额、日期、签署方等）\n'
            '2. 表格内容完整输出，不要省略\n'
            '3. 印章、签名等关键标识单独标注\n'
            '4. 严格只输出 JSON，不要包含 markdown 代码块标记或其他文字'
        ),
        "C": (
            "这是一张实物照片。请仔细观察图片，以 JSON 格式输出。\n"
            "输出结构：\n"
            '{\n'
            '  "image_type": "photo",\n'
            '  "topic": "照片主题（简短概括拍的是什么）",\n'
            '  "summary": "详细描述画面中的内容，包括物体、人物、场景、环境",\n'
            '  "objects": [\n'
            '    {"name": "物体/人物名称", "count": 数量(数字),\n'
            '     "attributes": "颜色、大小、材质、状态等",\n'
            '     "position": "在画面中的位置"}\n'
            '  ],\n'
            '  "visible_text": ["画面上可见的文字、标识、标签"],\n'
            '  "anomalies": ["值得注意的异常细节（损坏、缺失、不规范等）"],\n'
            '  "user_intent_hint": "推测用户拍摄这张照片想咨询什么"\n'
            '}\n'
            '要求：\n'
            '1. 尽可能详细描述物体的状态和特征\n'
            '2. 所有可见文字必须提取\n'
            '3. 异常细节对问题排查很重要，不要遗漏\n'
            '4. 严格只输出 JSON，不要包含 markdown 代码块标记或其他文字'
        ),
        "D": (
            "这是一张图表/数据图片。请仔细分析，以 JSON 格式输出。\n"
            "输出结构：\n"
            '{\n'
            '  "image_type": "chart",\n'
            '  "topic": "图表主题/标题",\n'
            '  "chart_type": "折线图/柱状图/饼图/流程图/架构图/脑图/其他",\n'
            '  "summary": "用自然语言概括图表表达的核心信息和趋势",\n'
            '  "axes": [\n'
            '    {"name": "坐标轴名称（X轴/Y轴/分类轴）", "values": ["刻度值列表"]}\n'
            '  ],\n'
            '  "data_points": [\n'
            '    {"label": "数据点名称", "value": "数值或描述", "trend": "up/down/stable"}\n'
            '  ],\n'
            '  "relationships": ["节点间的关系描述（流程图/架构图专用）"],\n'
            '  "insights": ["从图表中可以得出的关键洞察"],\n'
            '  "ocr_text": "图表中所有文字标注",\n'
            '  "user_intent_hint": "推测用户上传这张图表想咨询什么"\n'
            '}\n'
            '要求：\n'
            '1. 数据趋势和洞察是重点，不要只罗列数据\n'
            '2. 流程图/架构图重点描述节点间关系\n'
            '3. 严格只输出 JSON，不要包含 markdown 代码块标记或其他文字'
        ),
        "E": (
            "这是一张聊天记录截图。请仔细阅读对话内容，以 JSON 格式输出。\n"
            "输出结构：\n"
            '{\n'
            '  "image_type": "chat_log",\n'
            '  "topic": "对话主题（简短概括）",\n'
            '  "platform": "微信/企业微信/钉钉/QQ/短信/其他",\n'
            '  "participants": ["对话参与方"],\n'
            '  "messages": [\n'
            '    {"speaker": "说话人", "content": "消息内容", "time": "时间（如有）"}\n'
            '  ],\n'
            '  "key_conclusions": ["对话中的关键结论或决定"],\n'
            '  "outstanding": ["未解决的问题或待办事项"],\n'
            '  "user_intent_hint": "推测用户分享这段对话想咨询什么"\n'
            '}\n'
            '要求：\n'
            '1. 按时间顺序还原对话，每条消息单独列出\n'
            '2. 区分不同说话人\n'
            '3. 提取关键结论和待办事项\n'
            '4. 严格只输出 JSON，不要包含 markdown 代码块标记或其他文字'
        ),
    }

    def model_config(self, model: str) -> ModelConfig:
        """按模型名选择厂商级 key/base_url，再把模型名原样传给模型服务。"""

        model = canonical_model_name(model)
        api_key = self.provider_api_key(model)
        base_url = self.provider_base_url(model)
        return ModelConfig(name=model, api_key=api_key, base_url=base_url)

    def provider_api_key(self, model: str) -> str:
        """同一厂商的模型共用一套 key；切模型时只切模型名。"""

        if model.startswith("qwen"):
            return self.dashscope_api_key
        return self.openai_compatible_api_key

    def provider_base_url(self, model: str) -> str:
        """同一厂商的模型共用一个 base_url。"""

        if model.startswith("qwen"):
            return os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self.openai_compatible_base_url

    # 短期记忆模块
    memory_enabled: bool = os.getenv("MEMORY_ENABLED", "true").lower() != "false"
    memory_dir: Path = env_path("MEMORY_DIR", ROOT_DIR / "data" / "conversations")
    memory_summary_rounds: int = int(os.getenv("MEMORY_SUMMARY_ROUNDS", "5"))
    memory_recent_rounds: int = int(os.getenv("MEMORY_RECENT_ROUNDS", "5"))


# 全局单例，避免在每个模块重复实例化
settings = Settings()
