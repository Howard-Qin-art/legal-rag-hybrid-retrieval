"""RRF 的 k 值消融：k=60 到底合不合适，还是只是抄来的默认值。

k=60 出自 RRF 原始论文，那篇用的是英文数据集。中文法律检索上是不是也这个值最好，
得自己跑一遍才知道。

做法是先把两路候选（各 50 条）算出来存着，之后换 k 只做融合、不再检索，
所以扫 8 个 k 值和扫 1 个 k 值的检索开销是一样的。

用法：
    DATA_MODULE=data_law EVAL_MAX=150 python scripts/ablate_rrf_k.py
    KS=1,10,60,200 CAND_DEPTH=100 python scripts/ablate_rrf_k.py
"""
import importlib
import os
import sys

import numpy as np

# 脚本在 scripts/ 下，把项目根目录加进来才能 import 顶层模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG  # noqa: E402
from retrievers import build_index, rrf_fuse

_dm = importlib.import_module(CONFIG.data_module)
CORPUS, EVAL_SET = _dm.CORPUS, _dm.EVAL_SET

# 候选深度：给融合留多少条。k 很大时名次差异被压缩，深度太浅会看不出区别
CAND_DEPTH = int(os.getenv("CAND_DEPTH", "50"))
KS = [int(x) for x in os.getenv("KS", "1,5,10,20,40,60,100,200").split(",")]


def recall_at_k(retrieved, gold, k):
    top = [d for d, _ in retrieved[:k]]
    return len(set(top) & set(gold)) / max(1, len(gold))


def mrr(retrieved, gold, k=10):
    for rank, (d, _s) in enumerate(retrieved[:k], start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def main():
    _emb, _vs, bm25, vec = build_index(CORPUS)

    prepared = []
    for q, gold in EVAL_SET:
        prepared.append(
            (gold, bm25.search(q, topk=CAND_DEPTH), vec.search(q, topk=CAND_DEPTH))
        )

    print(
        f"\nRRF k 值消融（{len(EVAL_SET)} 条查询 / 语料 {len(CORPUS)} 段 / "
        f"两路各取 {CAND_DEPTH} 条候选）"
    )
    print("-" * 52)
    print(f"{'k':>6}{'Recall@1':>12}{'Recall@5':>12}{'MRR':>10}")
    print("-" * 52)

    rows = []
    best = None
    for k in KS:
        r1, r5, ms = [], [], []
        for gold, b, v in prepared:
            fused = rrf_fuse(b, v, k=k, topk=10)
            r1.append(recall_at_k(fused, gold, 1))
            r5.append(recall_at_k(fused, gold, 5))
            ms.append(mrr(fused, gold))
        row = (k, float(np.mean(r1)), float(np.mean(r5)), float(np.mean(ms)))
        rows.append(row)
        print(f"{k:>6}{row[1]:>12.3f}{row[2]:>12.3f}{row[3]:>10.3f}")
        if best is None or row[1] > best[1]:
            best = row

    print("-" * 52)
    print(f"Recall@1 最高出现在 k={best[0]}（{best[1]:.3f}）")
    return rows


if __name__ == "__main__":
    main()
