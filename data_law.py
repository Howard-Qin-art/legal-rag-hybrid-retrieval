"""法律语料与评测集（真实数据，来源见 law_qa_zh_600.json 的 _meta 字段）。

设计思路（面试可讲）：
- 语料 CORPUS：用每条数据的 `output`（法律回答）作为可检索段落，去重后入库。
- 评测 EVAL_SET：用 `instruction`（法律问句）作为 query，该问句对应 output 的语料 id 即
  为 gold（正确答案段落）。这样无需人工标注就得到 query -> gold passage 的检索评测集。
- 问句是自然语言、并非照抄原文关键词，因此能真实区分 BM25（关键词）与向量（语义）的能力。

用法：设环境变量 DATA_MODULE=data_law 即可让 evaluate.py / main.py 使用本语料。
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_HERE, "law_qa_zh_3000.json")


def _load_rows():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)["rows"]


_rows = _load_rows()

# 1) 用 output 构建检索语料（按文本去重）
_text_to_id: dict = {}
CORPUS = []
for r in _rows:
    text = (r.get("output") or "").strip()
    if not text or text in _text_to_id:
        continue
    doc_id = len(CORPUS)
    _text_to_id[text] = doc_id
    CORPUS.append({"id": doc_id, "text": text})

# 2) 用 instruction 构建评测集（gold = 对应 output 的语料 id）
_raw_eval = []
for r in _rows:
    q = (r.get("instruction") or "").strip()
    text = (r.get("output") or "").strip()
    if not q or text not in _text_to_id:
        continue
    _raw_eval.append((q, [_text_to_id[text]]))

# 3) 同一问句只保留一条
_seen_q = set()
EVAL_SET = []
for q, gold in _raw_eval:
    if q in _seen_q:
        continue
    _seen_q.add(q)
    EVAL_SET.append((q, gold))

# 4) 评测规模可用环境变量控制（默认 150 条，rerank 较慢时可调小）
EVAL_SET = EVAL_SET[: int(os.getenv("EVAL_MAX", "150"))]

if __name__ == "__main__":
    print(f"数据文件: {DATA_FILE}")
    print(f"语料段落数: {len(CORPUS)}")
    print(f"评测问句数: {len(EVAL_SET)}")
    print("样例问句:", EVAL_SET[0][0] if EVAL_SET else "无")
