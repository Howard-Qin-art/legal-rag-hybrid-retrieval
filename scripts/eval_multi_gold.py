"""用人工标的多 gold 数据重新评测，跟单 gold 的结果对照。

做这件事是为了回答一个问题：单 gold 的评测方式，有没有夸大 BM25 相对向量的优势？
（问句和它自己的回答天然共享词汇，BM25 可能白捡了分数；而且只认一个答案，
召回了别的正确段落也不给分。）

跑之前得先把 annotation/sheet.md 里的候选勾完。

用法：
    DATA_MODULE=data_law python scripts/eval_multi_gold.py
    N_BOOT=5000 python scripts/eval_multi_gold.py

关于指标，这里踩过一个坑，写在最前面免得以后自己看不懂：

多 gold 下不能直接拿 Recall@k 跟单 gold 比。Recall@k 的分母是 gold 总数，
标了 2 个 gold 之后，top-1 最多只能命中 1 个，召回率天然被压到 0.5。
所以表格里多 gold 那一栏 Recall 掉下来，不是方法变差了，是分母变大了。

能跨口径比较的只有两个：
  Hit@1  —— 第一名是不是在 gold 集合里，命中记 1 否则 0，除的是查询数
  MRR    —— 第一个命中的名次取倒数，同样与 gold 数量无关
所以主结论看这两个，Recall@5 只作为多 gold 下"覆盖率"的参考单独列。
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
N_BOOT = int(os.getenv("N_BOOT", "2000"))
SEED = int(os.getenv("BOOT_SEED", "42"))
TOPK = int(os.getenv("TOPK", CONFIG.topk))

RE_Q = re.compile(r"^## \[\d+\] (.+)$")
RE_ITEM = re.compile(r"^- \[(x| )\] \*\*(\d+)\*\*")


def load_sheet_full(path):
    """同上，但连没勾的候选一起返回，用来统计标注情况。"""
    items = []
    cur_q, cur_gold, cur_all = None, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = RE_Q.match(line.rstrip("\n"))
            if m:
                if cur_q is not None:
                    items.append((cur_q, set(cur_gold), cur_all))
                cur_q, cur_gold, cur_all = m.group(1), [], []
                continue
            m = RE_ITEM.match(line)
            if m:
                did = int(m.group(2))
                cur_all.append(did)
                if m.group(1) == "x":
                    cur_gold.append(did)
    if cur_q is not None:
        items.append((cur_q, set(cur_gold), cur_all))
    return items


def hit_at_1(retrieved, golds):
    """第一名是否命中。分母是查询数，跟 gold 有几个无关，所以能跨口径比。"""
    return 1.0 if retrieved and retrieved[0][0] in golds else 0.0


def recall_at_k(retrieved, golds, k):
    """召回了多少比例的 gold。分母是 gold 数，多 gold 下会被稀释，仅作参考。"""
    top = {d for d, _ in retrieved[:k]}
    return len(top & golds) / max(1, len(golds))


def mrr(retrieved, golds, k=10):
    for rank, (d, _s) in enumerate(retrieved[:k], start=1):
        if d in golds:
            return 1.0 / rank
    return 0.0


def boot_ci(arr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(arr, dtype="float64")
    n = len(a)
    boot = np.array([rng.choice(a, n, replace=True).mean() for _ in range(n_boot)])
    return float(boot.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_diff_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """配对 bootstrap：两组抽同一批查询序号，比各抽各的更有力。"""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    n = len(a)
    idx = rng.integers(0, n, (n_boot, n))
    diff = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def run(label, eval_pairs, methods, docs, topk):
    print(f"\n{label}（{len(eval_pairs)} 条查询，语料 {len(docs)} 段）")
    print("-" * 62)
    print(f"{'方法':<12}{'Hit@1':>9}{'95% CI':>20}{'MRR':>8}{'Recall@5':>11}")
    print("-" * 62)
    per_query = {}
    for name, fn in methods.items():
        h, rk, ms = [], [], []
        for q, golds in eval_pairs:
            res = fn(q)
            h.append(hit_at_1(res, golds))
            rk.append(recall_at_k(res, golds, topk))
            ms.append(mrr(res, golds))
        per_query[name] = {"h1": h, "rk": rk, "mrr": ms}
        mean, lo, hi = boot_ci(h)
        print(f"{name:<12}{mean:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
              f"{np.mean(ms):>8.3f}{np.mean(rk):>11.3f}")
    print("-" * 62)
    return per_query


def main(topk=TOPK):
    if not os.path.exists(SHEET):
        raise SystemExit(f"找不到标注表 {SHEET}，先跑 scripts/build_annotation_sheet.py 并勾选")

    full = load_sheet_full(SHEET)
    multi = [(q, g) for q, g, _ in full if g]
    if not multi:
        raise SystemExit("标注表里没有一条是勾过的，标完再跑")

    # 标注情况先交代清楚，不然后面的数没法定量理解
    n_added = sum(1 for q, g, allc in full if len(g) > 1)
    avg_gold = np.mean([len(g) for q, g, allc in full])
    avg_cand = np.mean([len(allc) for q, g, allc in full])
    print(f"\n标注情况：{len(multi)} 条问句，平均每条 {avg_cand:.1f} 个候选、"
          f"{avg_gold:.2f} 个 gold")
    print(f"  其中 {n_added} 条在原 gold 之外又勾了新的（占 {n_added / len(multi):.0%}），"
          f"这些才是这次标注真正新增的信息")

    dm = importlib.import_module(CONFIG.data_module)
    corpus = dm.CORPUS
    docs = {d["id"]: d for d in corpus}  # rerank 要的是 id -> {"text": ...}
    eval_set = dm.EVAL_SET

    # 同一批评测问句的单 gold 结果，用来对照
    q_to_gold = {q: set(g) for q, g in eval_set}
    single = [(q, q_to_gold[q]) for q, _ in multi if q in q_to_gold]
    if len(single) != len(multi):
        print(f"[warn] 有 {len(multi) - len(single)} 条问句在评测集里没匹配上，已跳过")

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

    if len(single) != len(multi):
        print("\n（样本条数不一致，跳过口径对照）")
        return

    # 核心问题：单 gold 有没有夸大 BM25 的优势
    print("\nBM25 相对向量的 Hit@1 差距（配对 bootstrap）")
    print("-" * 62)
    for label, out in (("单 gold", s_out), ("多 gold", m_out)):
        d, lo, hi = paired_diff_ci(out["BM25"]["h1"], out["向量"]["h1"])
        verdict = "显著" if (lo > 0 or hi < 0) else "CI 跨 0，分不出高下"
        print(f"  {label} 口径：{d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    print("-" * 62)

    # 多 gold 口径下 Recall@5 的两两对比是公平的（同一批 query、同一批 gold），
    # 它衡量的是"top-5 能覆盖多少相关段落"，比 Hit@1 更能拉开差距
    print("\n多 gold 口径下的 Recall@5（同一批 gold，口径内部可比）")
    print("-" * 62)
    for pair in (("BM25", "向量"), ("混合RRF", "BM25"), ("混合+重排", "BM25")):
        if pair[0] not in m_out or pair[1] not in m_out:
            continue
        d, lo, hi = paired_diff_ci(m_out[pair[0]]["rk"], m_out[pair[1]]["rk"])
        verdict = "显著" if (lo > 0 or hi < 0) else "CI 跨 0，分不出高下"
        print(f"  {pair[0]} - {pair[1]}：{d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
    print("-" * 62)

    # 更进一步：换口径这个动作本身，有没有显著改变 BM25 的优势
    d_s = np.asarray(s_out["BM25"]["h1"]) - np.asarray(s_out["向量"]["h1"])
    d_m = np.asarray(m_out["BM25"]["h1"]) - np.asarray(m_out["向量"]["h1"])
    shift, slo, shi = boot_ci(d_m - d_s)
    if slo > 0 or shi < 0:
        print(f"换口径带来的变化：{shift:+.3f}  95% CI [{slo:+.3f}, {shi:+.3f}]  显著")
        if shift > 0:
            print("  BM25 的优势在多 gold 下反而变大，单 gold 没有夸大它。")
        else:
            print("  BM25 的优势在多 gold 下缩水，说明单 gold 确实虚高了一部分。")
    else:
        print(f"换口径带来的变化：{shift:+.3f}  95% CI [{slo:+.3f}, {shi:+.3f}]  不显著")
        print("  两种口径下差距没有可辨别的区别 —— 没有证据表明单 gold 夸大了 BM25。")
        print(f"  不过 {len(multi)} 条样本偏少，这个 CI 很宽，只能说明「没看出来」，")
        print("  不等于「确认没有」。要收紧区间得把标注量扩到 100 条上下。")


if __name__ == "__main__":
    main(topk=TOPK)
