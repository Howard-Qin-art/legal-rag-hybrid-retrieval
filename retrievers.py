"""检索器：BM25 稀疏检索 + 向量稠密检索 + RRF 融合 + 可选重排。

要点：
- BM25 吃关键词/专有名词/精确匹配；向量吃语义相似（同义改写）。
- RRF（倒数排名融合）只依赖名次不依赖分数，避免两路分数不可比的问题。
- 重排（Cross-Encoder）精度高但慢，所以只对融合后的少量候选做，是经典的两阶段检索。
"""
import numpy as np
from rank_bm25 import BM25Okapi

from embedding import get_embedding, tokenize
from vectorstore import get_vectorstore


class BM25Retriever:
    def __init__(self, docs):
        self.ids = [d["id"] for d in docs]
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])

    def search(self, query, topk=10):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(-scores)[:topk]
        return [(self.ids[i], float(scores[i])) for i in idx]


class VectorRetriever:
    def __init__(self, vectorstore, embedding):
        self.vs = vectorstore
        self.emb = embedding

    def search(self, query, topk=10):
        qvec = self.emb.encode([query])[0]
        return self.vs.search(qvec, topk=topk)


def rrf_fuse(*ranked_lists, k=60, topk=10):
    """倒数排名融合：score = Σ 1/(k + rank)。"""
    scores: dict = {}
    for lst in ranked_lists:
        for rank, (doc_id, _score) in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])[:topk]
    return [(d, float(s)) for d, s in fused]


def get_reranker():
    """重排模型可选；没配或加载失败就返回 None（自动跳过重排）。"""
    from config import CONFIG
    if not CONFIG.rerank_model:
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(CONFIG.rerank_model)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 重排模型加载失败({e})，本次跳过重排")
        return None


def rerank(reranker, query, candidates, docs_by_id, topk=5):
    if reranker is None:
        return candidates[:topk]
    pairs = [(query, docs_by_id[doc_id]["text"]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(-np.asarray(scores))
    return [(candidates[i][0], float(scores[i])) for i in order[:topk]]


def build_index(corpus):
    """一次性构建：嵌入模型 + 向量库写入 + 两路检索器。"""
    emb = get_embedding()
    vs = get_vectorstore(emb.dim)
    texts = [d["text"] for d in corpus]
    # 把正文也写进库（pgvector 版会存到 content 列，内存版忽略）
    vs.add([d["id"] for d in corpus], emb.encode(texts), texts=texts)
    return emb, vs, BM25Retriever(corpus), VectorRetriever(vs, emb)
