"""检索质量评测：对比 单路BM25 / 单路向量 / 混合RRF / 混合+重排。

指标：
- Recall@5：Top-5 里命中了多少比例的相关段落（看"召没召回得来"）
- MRR：第一个相关结果出现名次的倒数（看"好的结果排得靠不靠前"）

运行：
    EMBED_BACKEND=hash python evaluate.py          # 离线跑通流程（指标不真实）
    python evaluate.py                              # 用真实句向量跑真实指标
"""
import importlib

import numpy as np

from config import CONFIG
from retrievers import build_index, rrf_fuse, get_reranker, rerank

# 语料可切换：DATA_MODULE=data（示例）或 data_law（真实法律语料）
_dm = importlib.import_module(CONFIG.data_module)
CORPUS, EVAL_SET = _dm.CORPUS, _dm.EVAL_SET


def recall_at_k(retrieved, gold, k=5):
    top = [doc_id for doc_id, _ in retrieved[:k]]
    return len(set(top) & set(gold)) / max(1, len(gold))


def mrr(retrieved, gold, k=10):
    for rank, (doc_id, _s) in enumerate(retrieved[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def main(topk=5):
    _emb, _vs, bm25, vec = build_index(CORPUS)
    docs_by_id = {d["id"]: d for d in CORPUS}
    reranker = get_reranker()

    def hybrid(q):
        return rrf_fuse(bm25.search(q, topk=10), vec.search(q, topk=10), topk=10)

    methods = {
        "BM25": lambda q: bm25.search(q, topk=10),
        "向量": lambda q: vec.search(q, topk=10),
        "混合RRF": hybrid,
    }
    if reranker is not None:
        methods["混合+重排"] = lambda q: rerank(reranker, q, hybrid(q), docs_by_id, topk=topk)

    print(f"\n检索质量对比（{len(EVAL_SET)} 条查询 / 语料 {len(CORPUS)} 段）")
    print("-" * 48)
    print(f"{'方法':<12}{'Recall@1':>10}{'Recall@' + str(topk):>14}{'MRR':>12}")
    print("-" * 48)
    rows = []
    for name, fn in methods.items():
        r1, rk, mrrs = [], [], []
        for query, gold in EVAL_SET:
            res = fn(query)
            r1.append(recall_at_k(res, gold, k=1))
            rk.append(recall_at_k(res, gold, k=topk))
            mrrs.append(mrr(res, gold))
        rows.append((name, float(np.mean(r1)), float(np.mean(rk)), float(np.mean(mrrs))))
        print(f"{name:<12}{rows[-1][1]:>10.3f}{rows[-1][2]:>14.3f}{rows[-1][3]:>12.3f}")
    print("-" * 48)
    print(f"嵌入后端：{CONFIG.embed_backend}（hash 为离线兜底，指标不真实）")
    return rows


if __name__ == "__main__":
    main(topk=CONFIG.topk)
