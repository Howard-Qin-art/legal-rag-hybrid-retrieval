"""生成层：把检索到的段落和问句拼成提示词，调大模型出答案。

这是整个项目里唯一调外部 API 的地方，也是唯一会"说瞎话"的地方。

法律场景下模型编法条的倾向很明显：读完几段原文，它可能顺嘴给你造一个
"根据《XX法》第XXX条"，编号是它自己想出来的，看着特别像真的。所以提示词里
花了不少篇幅约束这件事，并且要求每句话都标出依据来自哪一段 —— 标了引用之后
前端能点回去核对，模型编的东西也更容易被一眼看出来。

用法上就一个入口 generate()，返回 None 表示没配 key，调用方自己退回纯检索，
保证这个项目不装模型也能跑完整条链路。

请求走的是 OpenAI 兼容的 chat/completions，用标准库 urllib 发的，没引 httpx
也没引 openai sdk —— 这个接口实在太简单了，为一个 POST 加一层依赖不值得。
"""
import json
import re
import urllib.error
import urllib.request

from config import CONFIG

# 模型拒答时要求原样输出这句话，接口靠它判断这次是"答不上"而不是"答了"
REFUSAL_MARK = "检索到的内容里没有相关规定"

# 答案里出现的引用标记，形如 [1]
RE_CITE = re.compile(r"\[(\d+)\]")

PROMPT_TEMPLATE = """你是法律问答助手。只根据下面提供的资料回答问题，不允许使用资料之外的知识。

资料：
{blocks}

要求：
1. 回答里的每一句话都要用 [编号] 标出依据来自哪一段，例如"你可以向上一级人民法院上诉[1]"
   也可以一句话标多个，例如"上诉期限是十日[1][3]"。编号只能用上面出现过的。
2. 如果资料不足以回答问题，就只回答"{refusal}"，不要推测，不要用常识补充。
3. 严禁编造法条编号、罪名、机关名称、期限、金额。资料里没出现的具体数字和编号一律不许写。
4. 用口语回答，别照抄原文，也别复述资料里跟问题无关的内容。
5. 不要说"根据资料""如上所述"这类废话，直接回答问题。

问题：{query}

回答："""


def build_prompt(query, passages, max_chars=None):
    """passages 是 [(doc_id, score, text), ...]，编号从 1 开始。

    截断放在这里而不是外面：段落太长既烧 token 又容易把真正相关的那句挤到
    上下文边缘，600 字对法律条文的一条说明基本够用。
    """
    n = max_chars or CONFIG.passage_max_chars
    blocks = []
    for i, (_doc_id, _score, text) in enumerate(passages, start=1):
        t = " ".join(text.split())
        if len(t) > n:
            t = t[:n] + "……"
        blocks.append(f"[{i}] {t}")
    return PROMPT_TEMPLATE.format(
        blocks="\n".join(blocks), refusal=REFUSAL_MARK, query=query.strip()
    )


def _post_chat(prompt, timeout):
    url = CONFIG.model_api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": CONFIG.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG.model_temperature,
        # 关掉思考模式。带 reasoning 的模型默认会先吐几千字思考过程再给答案，
        # 实测这一步能让耗时从 28 秒降到 3 秒、输出 token 省掉 98%，
        # 而答案质量没差别 —— 因为这活儿是照着资料归纳，不是推演
        "enable_thinking": CONFIG.model_enable_thinking,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG.model_api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_citations(answer, passages):
    """把答案里的 [编号] 映射回真实的段落 id。

    模型偶尔会引用一个不存在的编号（比如给了 3 段它写 [4]），这里直接丢掉，
    不让它污染引用列表 —— 引用列表是要给前端点回去用的，指向空段落更糟。
    """
    nums = {int(x) for x in RE_CITE.findall(answer)}
    return [passages[i - 1][0] for i in sorted(nums) if 1 <= i <= len(passages)]


def generate(query, passages, timeout=None):
    """调模型出答案。

    passages: [(doc_id, score, text), ...]，一般是检索层给的 top-k。
    返回 dict；没配 key 返回 None；请求失败返回带 error 的 dict。
    """
    if not CONFIG.model_api_key:
        return None

    prompt = build_prompt(query, passages)
    try:
        data = _post_chat(prompt, timeout or CONFIG.model_timeout)
    except urllib.error.HTTPError as e:
        # 4xx 一般是 key 不对或者没开通该模型，把 body 带出来省得猜
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:  # noqa: BLE001 - 超时、断网这些都归到一类
        return {"error": f"{type(e).__name__}: {e}"}

    try:
        msg = data["choices"][0]["message"]
        answer = (msg.get("content") or "").strip()
        reasoning = msg.get("reasoning_content") or ""
    except (KeyError, IndexError):
        return {"error": f"返回结构不认识：{json.dumps(data, ensure_ascii=False)[:300]}"}

    # 推理型模型偶尔会把内容全放进 reasoning 而 content 是空的，这时候不能当成
    # "回答了"，否则前端会显示一片空白还带着 citations
    if not answer:
        return {
            "error": f"模型没给出正文，只有 {len(reasoning)} 字的思考过程。"
            f"试试把 MODEL_ENABLE_THINKING 换个值，或者换一个非推理型模型。"
        }

    usage = dict(data.get("usage", {}))
    if reasoning:
        usage["reasoning_chars"] = len(reasoning)

    return {
        "answer": answer,
        "citations": extract_citations(answer, passages),
        "refused": REFUSAL_MARK in answer,
        "model": CONFIG.model_name,
        "usage": usage,
        "prompt": prompt,  # 留着排查提示词问题用，线上可以去掉
    }
