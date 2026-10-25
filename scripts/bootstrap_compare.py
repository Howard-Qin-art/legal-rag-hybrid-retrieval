"""用 bootstrap 看看各个方法之间的差距到底显不显著。

起因是件尴尬事：在全量 150 条上 BM25 比向量高 4.7 个点，但随机抽 30 条出来跑，
结论直接反过来（向量更高）。也就是说这 4.7 个点可能只是样本波动，不是真差距。

单看一个均值很容易被骗，所以这里做 bootstrap：把 150 条查询结果当有放回地重采样
很多次，看差值的分布落在哪。如果差值的 95% 置信区间跨过 0，那就不能说谁更好。

用法：
    DATA_MODULE=data_law python scripts/bootstrap_compare.py
    N_BOOT=5000 python scripts/bootstrap_compare.py
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG  # noqa: E402
from retrievers import build_index, rrf_fuse, get_reranker, rerank  # noqa: E402

N_BOOT = int(os.getenv("N_BOOT", "2000"))
SEED = int(os.getenv("BOOT_SEED", "42"))
TOPK = int(os.getenv("TOPK", CONFIG.topk))
PAIRS = os.getenv("PAIRS", "BM25-向量,混合RRF-向量,混合RRF-BM25")


def recall_at_k(retrieved, gold, k):
    top = {d for d, _ in retrieved[:k]}
    return len(top & set(gold)) / max(1, len(gold))


def mrr(retrieved, gold, k=10):
    for rank, (d, _s) in enumerate(retrieved[:k], start=1):
        if d in set(gold):
            return 1.0 / rank
    return 0.0


def ci(samples, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def main():
    dm = importlib.import_module(CONFIG.data_module)
    corpus, eval_set = dm.CORPUS, dm.EVAL_SET
    docs_by_id = {d["id"]: d for d in corpus}  # rerank 要的是 id -> {"text": ...}

    _emb, _vs, bm25, vec = build_index(corpus)
    reranker = get_reranker()

    def hybrid(q):
        return rrf_fuse(bm25.search(q, topk=10), vec.search(q, topk=10), topk=10)

    methods = {
        "BM25": lambda q: bm25.search(q, topk=10),
        "向量": lambda q: vec.search(q, topk=10),
        "混合RRF": hybrid,
    }
    if reranker is not None:
        methods["混合+重排"] = lambda q: rerank(reranker, q, hybrid(q), docs_by_id, topk=TOPK)

    # 先把每条查询的指标存下来，后面重采样的是这些"每条的结果"而不是重新检索
    per_query = {name: {"r1": [], "rk": [], "mrr": []} for name in methods}
    for q, gold in eval_set:
        for name, fn in methods.items():
            res = fn(q)
            per_query[name]["r1"].append(recall_at_k(res, gold, 1))
            per_query[name]["rk"].append(recall_at_k(res, gold, TOPK))
            per_query[name]["mrr"].append(mrr(res, gold))

    n = len(eval_set)
    rng = np.random.default_rng(SEED)

    print(f"\n各方法在 {n} 条查询上的表现（bootstrap {N_BOOT} 次，95% 置信区间）")
    print("-" * 58)
    print(f"{'方法':<12}{'Recall@1':>10}{'95% CI':>22}")
    print("-" * 58)
    for name in methods:
        arr = np.asarray(per_query[name]["r1"])
        boot = np.array([rng.choice(arr, n, replace=True).mean() for _ in range(N_BOOT)])
        lo, hi = ci(boot)
        print(f"{name:<12}{arr.mean():>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}")
    print("-" * 58)

    print("\n两两差值（正数是前者更好，CI 跨 0 就说明分不出高下）")
    print("-" * 58)
    for pair in PAIRS.split(","):
        a, b = pair.split("-")
        if a not in methods or b not in methods:
            continue
        arr_a = np.asarray(per_query[a]["r1"])
        arr_b = np.asarray(per_query[b]["r1"])
        idx = rng.integers(0, n, (N_BOOT, n))
        diff = arr_a[idx].mean(axis=1) - arr_b[idx].mean(axis=1)
        lo, hi = ci(diff)
        verdict = "显著" if (lo > 0 or hi < 0) else "不显著，别下结论"
        print(f"{a} - {b}: {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    print("-" * 58)
    print("注：用的是配对 bootstrap（两组抽同一批查询序号），比各抽各的更有力。")


if __name__ == "__main__":
    main()
