"""嵌入模块：把文本转成向量。

两种后端：
- st  ：真实句向量（默认 BAAI/bge-small-zh-v1.5，512 维，中文效果好，需下载模型）
- hash：离线兜底（词/字二元组哈希），只在没有模型时跑通整条链路用，
        语义质量不真实，**不能用来写简历数字**。

面试可讲点：为什么中文要用中文句向量模型、向量维度必须与库表定义一致、
切分后再嵌入（本项目语料已按段落切好）。
"""
import hashlib
import re

import numpy as np

from config import CONFIG


def tokenize(text: str) -> list[str]:
    """简易中文切分：英文/数字按词，中文按字二元组（中文无空格也能切）。"""
    t = text.lower()
    toks = re.findall(r"[a-z0-9]+", t)
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]+", t))
    toks += [cjk[i:i + 2] for i in range(len(cjk) - 1)]
    return toks


class HashEmbedding:
    """离线兜底嵌入：把 token 哈希到固定维度并归一化。"""

    dim = 256

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class STEmbedding:
    """真实句向量（sentence-transformers）。"""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name)
        # 新版 sentence-transformers 已重命名为 get_embedding_dimension，这里做兼容
        getter = getattr(self.m, "get_embedding_dimension", None)
        self.dim = getter() if getter else self.m.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.m.encode(list(texts), normalize_embeddings=True).astype("float32")


def get_embedding():
    if CONFIG.embed_backend == "st":
        try:
            return STEmbedding(CONFIG.embed_model)
        except Exception as e:  # noqa: BLE001
            print(
                f"[warn] 句向量模型加载失败({e})；"
                f"若模型已在本地缓存，可尝试设 HF_HUB_OFFLINE=1 走离线加载。"
                f"本次回退 hash 兜底（仅跑通链路，勿用于真实指标）"
            )
            return HashEmbedding()
    return HashEmbedding()
