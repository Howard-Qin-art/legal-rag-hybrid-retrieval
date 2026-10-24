"""从 HuggingFace datasets-server 抓取中文法律问答数据，生成本地语料文件。

用法：
    python scripts/fetch_law_data.py                 # 默认抓 3000 条
    TARGET_ROWS=8000 python scripts/fetch_law_data.py

说明：
- 不需要安装 datasets 库，也不需要 git-lfs，只走公开的 datasets-server HTTP 接口。
- 单次请求最多返回 100 行，因此按页循环抓取。
- 输出文件写入项目根目录 law_qa_zh_<N>.json，结构与 data_law.py 的读取逻辑一致：
  {"_meta": {...}, "rows": [{"instruction": 问句, "input": "", "output": 回答, "id": n}, ...]}
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "Brench/chinese_law_data_rag_ft"
CONFIG = "default"
SPLIT = "train"
PAGE_SIZE = 100  # datasets-server 单次上限
TARGET_ROWS = int(os.getenv("TARGET_ROWS", "3000"))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.getenv(
    "OUT", os.path.join(os.path.dirname(HERE), f"law_qa_zh_{TARGET_ROWS}.json")
)

SERVER = "https://datasets-server.huggingface.co"


def _get(url, retries=3):
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:  # noqa: PERF203
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"请求失败: {url}\n{last}")


def total_rows():
    q = urllib.parse.urlencode({"dataset": DATASET})
    info = _get(f"{SERVER}/size?{q}")
    for cfg in info["size"]["splits"]:
        if cfg["config"] == CONFIG and cfg["split"] == SPLIT:
            return cfg["num_rows"]
    return None


def fetch_page(offset, length):
    q = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    return _get(f"{SERVER}/rows?{q}")["rows"]


def main():
    total = total_rows()
    n = min(TARGET_ROWS, total or TARGET_ROWS)
    print(f"数据集 {DATASET} 共 {total} 行，本次抓取 {n} 行")

    rows = []
    while len(rows) < n:
        length = min(PAGE_SIZE, n - len(rows))
        page = fetch_page(len(rows), length)
        if not page:
            print("没有更多数据，提前结束")
            break
        for r in page:
            row = r["row"]
            rows.append(
                {
                    "instruction": (row.get("instruction") or "").strip(),
                    "input": (row.get("input") or "").strip(),
                    "output": (row.get("output") or "").strip(),
                    "id": len(rows) + 1,
                }
            )
        print(f"  已抓取 {len(rows)}/{n}")

    payload = {
        "_meta": {
            "来源": f"HuggingFace 数据集 {DATASET}",
            "主页": f"https://huggingface.co/datasets/{DATASET}",
            "字段说明": {
                "instruction": "法律问句（用作评测 query）",
                "input": "补充上下文（多数为空）",
                "output": "法律回答（用作检索语料，即正确答案段落）",
                "id": "本文件内的序号",
            },
            "条数": len(rows),
            "抓取时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "用途": "output 去重后构成检索语料库；instruction 作评测问句，其对应 output 即 gold 段落",
        },
        "rows": rows,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"已写入 {OUT_PATH}（{len(rows)} 条，约 {os.path.getsize(OUT_PATH) / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
