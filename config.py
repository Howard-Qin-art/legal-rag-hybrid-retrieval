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

    # 语料模块：data=内置示例语料(20段)；data_law=本地真实法律问答语料(600条)
    data_module: str = os.getenv("DATA_MODULE", "data")


CONFIG = Config()
