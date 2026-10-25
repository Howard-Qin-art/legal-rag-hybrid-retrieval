"""统一配置：全部走环境变量，方便本地开发与 Docker 部署切换。

关键设计：
- 不配 DATABASE_URL 时用内存向量库跑通链路，配了就走真实 pgvector，体现"可切换"。
- 不配 REDIS_URL 时用 fakeredis，配了走真实 Redis。
- EMBED_BACKEND=hash 是离线兜底，仅用于无模型时验证流程，指标不具备参考价值。
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # PostgreSQL + pgvector，例如 postgresql://postgres:pwd@localhost:5432/rag
    database_url: str = os.getenv("DATABASE_URL", "")

    # Redis，例如 redis://localhost:6379/0
    redis_url: str = os.getenv("REDIS_URL", "")

    # 嵌入后端：st=真实句向量；hash=离线兜底
    embed_backend: str = os.getenv("EMBED_BACKEND", "st")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")  # 512 维

    # 重排模型，留空表示不重排；例如 BAAI/bge-reranker-base
    rerank_model: str = os.getenv("RERANK_MODEL", "")

    topk: int = int(os.getenv("TOPK", "5"))

    # 语料模块：data=内置示例语料(20段)；data_law=本地真实法律问答语料(3000条)
    data_module: str = os.getenv("DATA_MODULE", "data")

    # 向量库里已有同样条数的向量时跳过重建，省掉重复的编码与写入
    # 换语料但条数恰好相同时，设 REUSE_INDEX=0 强制重建
    reuse_index: bool = os.getenv("REUSE_INDEX", "1") == "1"

    # ---- 生成层（/ask）----
    # 走 OpenAI 兼容的 chat/completions 接口，各家只要换 base 和模型名
    model_api_base: str = os.getenv(
        "MODEL_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_api_key: str = os.getenv("MODEL_API_KEY", "")  # 留空则 /ask 退化成纯检索
    model_name: str = os.getenv("MODEL_NAME", "qwen-plus")
    model_temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    model_timeout: int = int(os.getenv("MODEL_TIMEOUT", "60"))
    # 送进提示词的每段正文最长多少字，长了截断，省 token 也避免超出上下文
    passage_max_chars: int = int(os.getenv("PASSAGE_MAX_CHARS", "600"))


CONFIG = Config()
