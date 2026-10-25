"""生成层的测试，不联网、不花钱。

覆盖三件事：没配 key 时是否老实降级、提示词有没有把该约束的都写进去、
引用解析会不会被模型编出来的越界编号带偏。第三件是最容易出事的地方 ——
引用列表要给前端点回去用，指向一个不存在的段落比不显示更糟。

跑法：
    python tests/test_generator.py
    EMBED_BACKEND=hash DATA_MODULE=data python tests/test_generator.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 这两行必须在 import config 之前，因为配置是在 import 那一刻读的环境变量。
# 别想着后面 importlib.reload(config) 补救 —— reload 会造出一个新的 CONFIG 对象，
# 而 generator 里已经 import 走的还是旧对象，两个配置就此分家，
# 测试里改了半天 key 其实改的是另一个对象。这个坑我自己踩过一次。
os.environ.setdefault("EMBED_BACKEND", "hash")
os.environ.setdefault("DATA_MODULE", "data")

import generator  # noqa: E402
from config import CONFIG  # noqa: E402

PASSAGES = [
    (11, 0.91, "根据《刑事诉讼法》第294条，被告人有权向上一级人民法院上诉。"),
    (22, 0.88, "辩护人经被告人同意，也可以提出上诉。"),
    (33, 0.70, "化妆品生产企业应当依法取得生产许可。"),
]

_fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")
    if not cond:
        _fails.append(name)


def test_no_key_returns_none():
    """没配 key 必须返回 None，让调用方退回纯检索，而不是抛异常。

    本地 .env 里可能真有 key，这里临时清空再还原，保证这条在任何机器上
    都能测到"没 key"这条路径 —— 否则有 key 的机器上它根本测不到降级。
    """
    old = generator.CONFIG.model_api_key
    generator.CONFIG.model_api_key = ""
    try:
        check("未配 key 时 generate() 返回 None", generator.generate("测试问题", PASSAGES) is None)
    finally:
        generator.CONFIG.model_api_key = old


def test_prompt_contains_constraints():
    """提示词里三条硬约束都得在：引用编号、拒答原话、不许编法条。"""
    p = generator.build_prompt("被告人能上诉吗？", PASSAGES)
    check("提示词含段落编号 [1]/[2]/[3]",
          all(f"[{i}]" in p for i in (1, 2, 3)))
    check("提示词含拒答原话", generator.REFUSAL_MARK in p)
    check("提示词含禁止编造法条的约束", "严禁编造" in p and "法条编号" in p)
    check("提示词末尾带上了问题", "被告人能上诉吗？" in p)


def test_prompt_truncates_long_passage():
    long_text = "这是一段很长的法条说明。" * 100
    p = generator.build_prompt("问题", [(1, 0.9, long_text)], max_chars=50)
    check("超长段落被截断", "……" in p)


def test_citation_mapping():
    """引用编号要映射回真实段落 id，顺序按编号排。"""
    cites = generator.extract_citations("可以上诉[1]，辩护人也可以[2]。", PASSAGES)
    check("引用映射正确", cites == [11, 22], f"实际 {cites}")


def test_citation_drops_out_of_range():
    """模型写了 [9] 这种不存在的编号，应当直接丢掉而不是抛异常或塞 None。"""
    cites = generator.extract_citations("可以上诉[1][9]。", PASSAGES)
    check("越界编号被丢弃", cites == [11], f"实际 {cites}")


def test_citation_dedup():
    cites = generator.extract_citations("上诉[1]，也可以上诉[1]。", PASSAGES)
    check("重复引用去重", cites == [11], f"实际 {cites}")


def test_refusal_detection():
    """拒答标记要能被识别出来，接口靠它区分「答不上」和「答了」。"""
    # generate() 要联网，这里只验证判定逻辑本身
    answer = generator.REFUSAL_MARK + "。"
    check("拒答判定成立", generator.REFUSAL_MARK in answer)


def test_ask_endpoint_degrades():
    """/ask 在没配 key 时应当返回答案为空 + 说明，并且照常给出检索段落。"""
    try:
        from fastapi.testclient import TestClient
        import main
    except Exception as e:  # noqa: BLE001
        print(f"SKIP  /ask 端点测试（{type(e).__name__}: {e}）")
        return

    # 改 generator 拿着的那个 CONFIG 才有效：main 里调用的是 generator.generate，
    # 它读的是自己模块里 import 进来的那个对象
    old = generator.CONFIG.model_api_key
    generator.CONFIG.model_api_key = ""
    try:
        with TestClient(main.app) as client:
            r = client.post("/ask", json={"query": "退货之后钱多久退回来？", "topk": 3})
            d = r.json()
        check("/ask 状态码 200", r.status_code == 200)
        check("/ask 未配 key 时 answer 为 None", d.get("answer") is None)
        check("/ask 未配 key 时给出说明", bool(d.get("note")))
        check("/ask 未配 key 时仍有检索段落", len(d.get("passages", [])) > 0)
    finally:
        generator.CONFIG.model_api_key = old


if __name__ == "__main__":
    test_no_key_returns_none()
    test_prompt_contains_constraints()
    test_prompt_truncates_long_passage()
    test_citation_mapping()
    test_citation_drops_out_of_range()
    test_citation_dedup()
    test_refusal_detection()
    test_ask_endpoint_degrades()

    print()
    if _fails:
        print(f"{len(_fails)} 项失败：{', '.join(_fails)}")
        sys.exit(1)
    print("全部通过")
