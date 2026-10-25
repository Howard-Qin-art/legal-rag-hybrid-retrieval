"""用人工标的多 gold 数据重新评测，跟单 gold 的结果对照。

做这件事是为了回答一个问题：单 gold 的评测方式，有没有夸大 BM25 相对向量的优势？
（问句和它自己的回答天然共享词汇，BM25 可能白捡了分数；而且只认一个答案，
召回了别的正确段落也不给分。）

跑之前得先把 annotation/sheet.md 里的候选勾完。

用法：
    DATA_MODULE=data_law python scripts/eval_multi_gold.py
"""
import importlib
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG  # noqa: E402
from retrievers import build_index, rrf_fuse, get_reranker, rerank  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHEET = os.getenv("SHEET", os.path.join(ROOT, "annotation", "sheet.md"))

RE_Q = re.compile(r"^## \[\d+\] (.+)$")
RE_ITEM = re.compile(r"^- \[(x| )\] \*\*(\d+)\*\*")


def load_sheet(path):
    """解析标注表，返回 [(query, {相关 id}), ...]。没勾任何项的条目会被跳过。"""
    items = []
    cur_q, cur_ids = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = RE_Q.match(line.rstrip("\n"))
            if m:
                if cur_q is not None and cur_ids:
                    items.append((cur_q, set(cur_ids)))
                cur_q, cur_ids = m.group(1), []
                continue
            m = RE_ITEM.match(line)
            if m and m.group(1) == "x":
                cur_ids.append(int(m.group(2)))
    if cur_q is not None and cur_ids:
        items.append((cur_q, set(cur_ids)))
    return items


def recall_at_k(retrieved, golds, k):
    top = {d for d, _ in retrieved[:k]}
    return len(top & golds) / max(1, len(golds))


def mrr(retrieved, golds, k=10):
    for rank, (d, _s) in enumerate(retrieved[:k], start=1):
        if d in golds:
            return 1.0 / rank
    return 0.0


def run(label, eval_pairs, methods, docs, topk):
    print(f"\n{label}（{len(eval_pairs)} 条查询，语料 {len(docs)} 段）")
    print("-" * 46)
    print(f"{'方法':<12}{'Recall@1':>10}{'Recall@' + str(topk):>14}{'MRR':>10}")
    print("-" * 46)
    out = {}
    for name, fn in methods.items():
        r1, rk, ms = [], [], []
        for q, golds in eval_pairs:
            res = fn(q)
            r1.append(recall_at_k(res, golds, 1))
            rk.append(recall_at_k(res, golds, topk))
            ms.append(mrr(res, golds))
        out[name] = (float(np.mean(r1)), float(np.mean(rk)), float(np.mean(ms)))
        print(f"{name:<12}{out[name][0]:>10.3f}{out[name][1]:>14.3f}{out[name][2]:>10.3f}")
    print("-" * 46)
    return out


def main(topk=5):
    if not os.path.exists(SHEET):
        raise SystemExit(f"找不到标注表 {SHEET}，先跑 scripts/build_annotation_sheet.py 并勾选")

    multi = load_sheet(SHEET)
    if not multi:
        raise SystemExit("标注表里没有一条是勾过的，标完再跑")

    dm = importlib.import_module(CONFIG.data_module)
    corpus = dm.CORPUS
    docs = {d["id"]: d for d in corpus}  # rerank 要的是 id -> {"text": ...}
    eval_set = dm.EVAL_SET

    # 同一批评测问句的单 gold 结果，用来对照
    q_to_gold = {q: set(g) for q, g in eval_set}
    single = [(q, q_to_gold[q]) for q, _ in multi if q in q_to_gold]

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
        methods["混合+重排"] = lambda q: rerank(reranker, q, hybrid(q), docs, topk=topk)

    s_out = run("单 gold（原数据集自带）", single, methods, docs, topk)
    m_out = run("多 gold（人工标注）", multi, methods, docs, topk)

    if "BM25" in s_out and "向量" in s_out:
        d_single = s_out["BM25"][0] - s_out["向量"][0]
        d_multi = m_out["BM25"][0] - m_out["向量"][0]
        print("\nBM25 相对向量的 Recall@1 差距")
        print(f"  单 gold 口径：{d_single:+.3f}")
        print(f"  多 gold 口径：{d_multi:+.3f}")
        if d_multi < d_single - 1e-9:
            print(
                "  差距缩小了 —— 单 gold 下 BM25 的优势里有一部分是评测方式给的，"
                "不全是它真本事。"
            )
        elif d_multi > d_single + 1e-9:
            print("  差距反而变大，BM25 的优势在多 gold 下更站得住。")
        else:
            print("  两种口径下差距一致。")


if __name__ == "__main__":
    main(topk=CONFIG.topk)
