"""统一配置：全部走环境变量，方便本地开发与 Docker 部署切换。

关键设计：
- 不配 DATABASE_URL 时用内存向量库跑通链路，配了就走真实 pgvector，体现"可切换"。
- 不配 REDIS_URL 时用 fakeredis，配了走真实 Redis。
- EMBED_BACKEND=hash 是离线兜底，仅用于无模型时验证流程，指标不具备参考价值。
"""
import os
from dataclasses import dataclass


def _load_dotenv(path=None):
    """把 .env 里的键值对塞进 os.environ，文件不存在就跳过。

    没用 python-dotenv：就是一个 key=value 的解析，为此加一层依赖不划算。
    已经存在的环境变量不覆盖 —— 命令行 export 的优先级应当高于 .env，
    不然想临时换个模型还得去改文件。
    """
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_dotenv()


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
    model_timeout: int = int(os.getenv("MODEL_TIMEOUT", "120"))

    # 思考模式。默认关：实测 qwen3.5-plus 开着的时候，一个 46 字的答案配了
    # 4286 字的思考过程，输出 token 从 30 涨到 1434，耗时从 3.3 秒涨到 28.5 秒。
    # RAG 要的是照着资料归纳，不是推演，这笔开销纯属浪费。
    # 想看模型的推理过程就设 MODEL_ENABLE_THINKING=1
    model_enable_thinking: bool = os.getenv("MODEL_ENABLE_THINKING", "0") == "1"
    # 送进提示词的每段正文最长多少字，长了截断，省 token 也避免超出上下文
    passage_max_chars: int = int(os.getenv("PASSAGE_MAX_CHARS", "600"))

    # ---- 审计日志（main.py 的 /retrieve 与 /ask）----
    # 每次请求落一行 JSONL。法律问答这类场景要求事后能查到"问了什么、答了什么、
    # 依据哪几段"，没有留痕等于没有服务。设 AUDIT_ENABLED=0 关掉。
    audit_enabled: bool = os.getenv("AUDIT_ENABLED", "1") == "1"
    # 留空则写在项目根目录下 audit.jsonl
    audit_log_path: str = os.getenv("AUDIT_LOG_PATH", "")

    # ---- 输入约束（防止一个畸形请求拖死服务）----
    # query 上限：超长输入会让嵌入模型跑很久，甚至把它撑爆，必须在入口拦掉
    max_query_chars: int = int(os.getenv("MAX_QUERY_CHARS", "1000"))
    # topk 上限：返回条数失控会把带宽和响应时间都拖垮
    max_topk: int = int(os.getenv("MAX_TOPK", "20"))


CONFIG = Config()
