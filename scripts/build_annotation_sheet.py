"""生成多 gold 标注表。

原来的评测集每个 query 只有一个 gold，会低估指标（法律问题常常好几段都算答得上），
也可能因为问句和回答共享词汇而让 BM25 占便宜。要搞清楚到底占了多少便宜，
就得有一批人工标的多 gold 数据。

一条一条从 3000 段里找相关段落太费劲，所以这里用检索池（pooling）的办法：
对每个 query，把 BM25、向量、混合三路的前几名合在一起当候选，
人只需要在十来个候选里勾选，不用看全文。

用法：
    DATA_MODULE=data_law python scripts/build_annotation_sheet.py
    ANNOT_N=50 python scripts/build_annotation_sheet.py
"""
import importlib
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG  # noqa: E402
from retrievers import build_index, rrf_fuse  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N = int(os.getenv("ANNOT_N", "30"))          # 抽多少条 query 来标
SEED = int(os.getenv("ANNOT_SEED", "42"))    # 固定种子，保证可复现
PER_METHOD = int(os.getenv("PER_METHOD", "5"))  # 每路取前几条进候选池
SNIPPET = int(os.getenv("SNIPPET", "220"))   # 每段显示多少字，够判断就行
OUT = os.getenv("OUT", os.path.join(ROOT, "annotation", "sheet.md"))


def snippet(text, n=SNIPPET):
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n] + "……"


def main():
    dm = importlib.import_module(CONFIG.data_module)
    corpus, eval_set = dm.CORPUS, dm.EVAL_SET
    docs = {d["id"]: d["text"] for d in corpus}

    rng = random.Random(SEED)
    idxs = sorted(rng.sample(range(len(eval_set)), min(N, len(eval_set))))

    # 标注表要靠检索出候选，所以这里必须用真实句向量，hash 兜底没有意义
    if CONFIG.embed_backend != "st":
        print("[warn] 当前不是 st 后端，候选池质量会很差，请确认 EMBED_BACKEND=st")

    _emb, _vs, bm25, vec = build_index(corpus)

    lines = []
    lines.append("# 多 gold 标注表\n")
    lines.append(
        f"共 {len(idxs)} 条问句，从 {len(eval_set)} 条评测集里随机抽的（种子 {SEED}）。\n"
    )
    lines.append(
        "每条下面列了候选段落，**已经勾上的**是原数据集自带的答案（gold）。\n"
        "你要做的是：看剩下的候选里有没有**内容上也能回答这个问题的**，有就勾上。\n"
        "判断标准只有一个：一个人看到这段，能不能用它回答上面那个问题。\n"
        "别管措辞像不像、别管是不是同一条法条的不同表述，能答上就算。\n"
    )
    lines.append("标完保存，然后跑 `python scripts/eval_multi_gold.py`。\n")
    lines.append("---\n")

    for n, qi in enumerate(idxs):
        q, gold = eval_set[qi]
        gold_ids = set(gold)

        pool = []
        seen = set()
        for lst in (
            bm25.search(q, topk=PER_METHOD),
            vec.search(q, topk=PER_METHOD),
            rrf_fuse(bm25.search(q, topk=10), vec.search(q, topk=10), topk=PER_METHOD),
        ):
            for did, _s in lst:
                if did not in seen:
                    seen.add(did)
                    pool.append(did)

        # gold 必须出现在候选里，否则没法确认它确实相关
        for g in gold_ids:
            if g not in seen:
                seen.add(g)
                pool.append(g)

        lines.append(f"## [{n}] {q}\n")
        lines.append(f"（原 gold：{', '.join(str(g) for g in sorted(gold_ids))}）\n")
        for did in pool:
            mark = "x" if did in gold_ids else " "
            lines.append(f"- [{mark}] **{did}** {snippet(docs[did])}\n")
        lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已生成标注表：{OUT}")
    print(f"  {len(idxs)} 条问句，每条约 {len(pool)} 个候选")
    print("  标完（把能回答问题的候选勾上）后运行：python scripts/eval_multi_gold.py")


if __name__ == "__main__":
    main()
