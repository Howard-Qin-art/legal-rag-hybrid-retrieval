"""审计日志：每次请求落一行 JSONL，事后能回溯也能统计。

为什么需要这一层：法律问答这类场景要求留痕 —— 问了什么、命中哪几段、答案是啥、
有没有拒答、花了多少毫秒，事后都得查得到。没有它，服务出了问题是没法复盘的。

几个取舍：

- 落 JSONL 而不是数据库：一行一个 JSON，tail / grep / jq 就能直接用，
  不用为了看一眼日志先起个客户端。规模真上去了再换列式存储。
- 写盘失败不能拖垮请求：日志是旁路。打不开文件就去 stderr 说一句，只说一次，
  然后让检索照常返回 —— 不能因为日志文件权限不对就把整个服务搞成 500。
- 每次写都 open / close 配一把锁：uvicorn 跑线程池，多线程共享文件句柄容易
  写出半行。串行写牺牲一点 IO 换行完整性，这个量级下非常划算。
"""
import json
import os
import sys
import threading
from datetime import datetime, timezone

from config import CONFIG

_LOCK = threading.Lock()
_WARNED = False

# 日志里记录的 query 长度上限。存原文是为了能回溯，但没必要把一整篇判决书
# 原样重复落一遍，超了就截断并记下真实长度。
QUERY_SNIPPET = 500


def log_path():
    """日志文件位置。没配就用项目根目录下的 audit.jsonl。"""
    if CONFIG.audit_log_path:
        return CONFIG.audit_log_path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.jsonl")


def log_event(endpoint, payload):
    """追加一行日志。任何异常都吃掉，绝不向上抛。"""
    global _WARNED
    if not CONFIG.audit_enabled:
        return False

    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "endpoint": endpoint}
    row.update(payload)

    try:
        line = json.dumps(row, ensure_ascii=False)
    except (TypeError, ValueError) as e:  # 有字段不是 JSON 可序列化的
        if not _WARNED:
            _WARNED = True
            print(f"[audit] 日志字段无法序列化，跳过：{e}", file=sys.stderr)
        return False

    with _LOCK:
        try:
            with open(log_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except OSError as e:
            if not _WARNED:
                _WARNED = True
                print(f"[audit] 写不了日志文件 {log_path()}：{e}，已跳过落盘", file=sys.stderr)
            return False


def read_events():
    """读出全部日志行。坏行跳过，不让它影响统计。"""
    path = log_path()
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 进程被强杀时可能留下半行，跳过就是了
    return rows


def stats(rows=None):
    """按 endpoint 汇总调用量、缓存命中率、延迟分布、拒答与错误数。"""
    rows = read_events() if rows is None else rows
    out = {"total_calls": len(rows), "by_endpoint": {}}
    if not rows:
        return out

    def _pct(vals, p):
        """分位数，取最近邻。样本很少时只能看个大概。"""
        if not vals:
            return None
        s = sorted(vals)
        idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
        return round(s[idx], 2)

    groups = {}
    for r in rows:
        groups.setdefault(r.get("endpoint", "unknown"), []).append(r)

    for ep, items in sorted(groups.items()):
        lat = [r["latency_ms"] for r in items if isinstance(r.get("latency_ms"), (int, float))]
        pt = [r.get("prompt_tokens", 0) or 0 for r in items]
        ct = [r.get("completion_tokens", 0) or 0 for r in items]
        hit = [r for r in items if r.get("cached")]
        out["by_endpoint"][ep] = {
            "calls": len(items),
            "cache_hit_rate": round(len(hit) / len(items), 4),
            "avg_latency_ms": round(sum(lat) / len(lat), 2) if lat else None,
            "p95_latency_ms": _pct(lat, 95),
            "refused": sum(1 for r in items if r.get("refused")),
            "errors": sum(1 for r in items if r.get("error")),
            "prompt_tokens": sum(pt),
            "completion_tokens": sum(ct),
        }
    return out
