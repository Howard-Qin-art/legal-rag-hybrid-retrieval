"""审计日志与输入校验的测试，不联网、不花钱、不需要 Docker。

跑法：
    EMBED_BACKEND=hash DATA_MODULE=data python tests/test_audit_and_validation.py

覆盖两件事：一是每次请求（包括被拒的）有没有落下可回溯的一行，二是畸形输入
会不会把服务搞成 500 裸抛。这两块是同一个目的的上下半身 —— 没有日志就不知道
发生过什么，没有校验就会让一句话把服务拖垮。
"""
import json
import os
import sys

TMP_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_audit.jsonl")
# 指向一个不可能存在的目录，用来验证写盘失败时不会把异常抛给调用方
BAD_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nope", "deep", "x.jsonl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("EMBED_BACKEND", "hash")
os.environ.setdefault("DATA_MODULE", "data")
os.environ["AUDIT_LOG_PATH"] = TMP_LOG
if os.path.exists(TMP_LOG):
    os.remove(TMP_LOG)

import audit  # noqa: E402
from config import CONFIG  # noqa: E402

_fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")
    if not cond:
        _fails.append(name)


def read_rows():
    if not os.path.exists(TMP_LOG):
        return []
    return [json.loads(l) for l in open(TMP_LOG, encoding="utf-8") if l.strip()]


def reset_log():
    if os.path.exists(TMP_LOG):
        os.remove(TMP_LOG)


def test_audit_module():
    """审计模块本身：能写、能读、能汇总，且写盘失败是安静的。"""
    reset_log()
    check("log_event 写入成功返回 True", audit.log_event("demo", {"a": 1}) is True)
    rows = read_rows()
    check("日志里有一条", len(rows) == 1)
    check("字段完整落盘", rows[0]["endpoint"] == "demo" and rows[0]["a"] == 1)
    check("带时间戳", bool(rows[0].get("ts")))

    ok_latencies = [10.0, 20.0, 30.0, 100.0]
    for i, lat in enumerate(ok_latencies):
        audit.log_event("demo", {"latency_ms": lat, "cached": i % 2 == 0})
    s = audit.stats()["by_endpoint"]["demo"]
    check("stats 统计到全部调用", s["calls"] == 5, f"实际 {s['calls']}")
    check("缓存命中率算对", s["cache_hit_rate"] == 0.4, f"实际 {s['cache_hit_rate']}")
    # 第一次那条没带 latency_ms，不参与延迟统计，底下四个的均值是 40
    check("平均延迟算对", s["avg_latency_ms"] == 40.0, f"实际 {s['avg_latency_ms']}")
    check("p95 取到最大值", s["p95_latency_ms"] == 100.0, f"实际 {s['p95_latency_ms']}")

    # 换到一个写不了的路径，必须返回 False 而不是抛异常
    old = CONFIG.audit_log_path
    CONFIG.audit_log_path = BAD_LOG
    try:
        check("写盘失败时不抛异常，返回 False", audit.log_event("demo", {"x": 1}) is False)
    except Exception as e:  # noqa: BLE001
        check("写盘失败时不抛异常，返回 False", False, f"抛了 {type(e).__name__}")
    finally:
        CONFIG.audit_log_path = old

    # 关掉开关后不该再往文件里追加
    before = len(read_rows())
    CONFIG.audit_enabled = False
    audit.log_event("demo", {"x": 2})
    check("关掉开关后不再写盘", len(read_rows()) == before)
    CONFIG.audit_enabled = True

    # 半行坏数据不能让统计崩掉
    with open(TMP_LOG, "a", encoding="utf-8") as f:
        f.write('{"broken":\n')
    check("坏行被跳过而不是炸掉", audit.stats()["total_calls"] == before)


def test_endpoint_behaviour():
    """接口行为：合法请求落审计，非法请求返回统一结构且同样落审计。"""
    reset_log()
    try:
        from fastapi.testclient import TestClient

        import main
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  接口测试（{type(e).__name__}: {e}）")
        return

    # 生成层关掉，避免真去调模型花钱
    old_key, old_lp = CONFIG.model_api_key, CONFIG.audit_log_path
    CONFIG.model_api_key, CONFIG.audit_log_path = "", TMP_LOG
    try:
        with TestClient(main.app, raise_server_exceptions=False) as client:
            r = client.post("/retrieve", json={"query": "退货之后钱多久退回来？", "topk": 3})
            d = r.json()
            check("合法请求返回 200", r.status_code == 200)
            check("合法请求带 ok=True", d.get("ok") is True)
            check("合法请求有结果", len(d.get("results", [])) > 0)

            rows = read_rows()
            hit = [x for x in rows if x.get("endpoint") == "retrieve"]
            check("合法请求落了审计", len(hit) == 1)
            check("审计里有命中段落 id", bool(hit and hit[0].get("hit_ids")))
            check("审计里有 query 原文", hit and hit[0].get("query") == "退货之后钱多久退回来？")
            check("审计里有耗时", isinstance(hit and hit[0].get("latency_ms"), float))

            cases = [
                ("空 query", {"query": "   "}, "invalid_query"),
                ("超长 query", {"query": "退" * 9999}, "invalid_request"),
                ("topk 为 0", {"query": "退货", "topk": 0}, "invalid_request"),
                ("topk 为负", {"query": "退货", "topk": -5}, "invalid_request"),
                ("topk 超上限", {"query": "退货", "topk": 9999}, "invalid_request"),
                ("topk 类型错", {"query": "退货", "topk": "abc"}, "invalid_request"),
            ]
            for name, body, want_code in cases:
                r = client.post("/retrieve", json=body)
                d = r.json()
                ok_code = r.status_code in (400, 422) and d.get("ok") is False
                check(
                    f"{name} 返回结构化错误而非 500",
                    ok_code and d.get("error", {}).get("code") == want_code,
                    f"实际 status={r.status_code} body={json.dumps(d, ensure_ascii=False)[:90]}",
                )

            rejected = [x for x in read_rows() if x.get("endpoint") == "rejected"]
            check("被拒的请求也落了审计", len(rejected) == len(cases), f"实际 {len(rejected)} 条")

            sv = client.get("/stats").json()
            check("/stats 返回 ok", sv.get("ok") is True)
            check("/stats 有 total_calls", sv.get("total_calls", 0) > 0)
            check("/stats 有 retrieve 分组", "retrieve" in sv.get("by_endpoint", {}))

            # 内部异常必须变成结构化响应，而不是 traceback 甩给调用方
            boom = Exception("模拟底层炸了")
            orig = main._do_retrieve

            def _boom(*_a, **_k):
                raise boom

            main._do_retrieve = _boom
            try:
                # 服务内部会 traceback.print_exc() 打堆栈，这是刻意为之（排查需要）。
                # 测试输出里不想看到它，临时把 stderr 收起来。
                import contextlib
                import io

                with contextlib.redirect_stderr(io.StringIO()):
                    r = client.post("/retrieve", json={"query": "退货", "topk": 3})
                d = r.json()
                check("内部异常返回 500 结构化响应", r.status_code == 500 and d.get("ok") is False)
                check("内部异常带 internal_error 码", d.get("error", {}).get("code") == "internal_error")
                err_rows = [x for x in read_rows() if x.get("error")]
                check("失败请求也留下审计记录", len(err_rows) >= 1)
            finally:
                main._do_retrieve = orig
    finally:
        CONFIG.model_api_key, CONFIG.audit_log_path = old_key, old_lp


def main():
    print("=== 审计模块 ===")
    test_audit_module()
    print()
    print("=== 接口行为 ===")
    test_endpoint_behaviour()

    if os.path.exists(TMP_LOG):
        os.remove(TMP_LOG)

    print()
    if _fails:
        print(f"{len(_fails)} 项失败：" + "、".join(_fails))
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
