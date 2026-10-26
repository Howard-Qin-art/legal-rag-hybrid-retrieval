"""RAG 服务（FastAPI）。

两个接口：
- /retrieve 检索层，返回命中的原文段落
- /ask     在上面接一层大模型，返回带引用标记的答案

链路：查 Redis 缓存 -> 命中直接返回；否则 BM25+向量 RRF 融合 -> 可选重排 -> 写缓存。

启动：uvicorn main:app --reload --port 8000
演示页：浏览器打开 http://localhost:8000
"""
import importlib
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import audit
from cache import cache_get, cache_set
from config import CONFIG
from generator import generate
from retrievers import build_index, rrf_fuse, get_reranker, rerank

# 语料可切换：DATA_MODULE=data（示例）或 data_law（真实法律语料）
CORPUS = importlib.import_module(CONFIG.data_module).CORPUS

app = FastAPI(title="RAG 混合检索服务")


class ApiError(Exception):
    """业务层的可预期错误，统一由下面的 handler 转成结构化响应。

    分三个码：invalid_query 是输入不合规，upstream_error 是模型或缓存那一层挂了，
    internal_error 是没预料到的。前端按 code 分支处理，别去解析 message 文本。
    """

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _err(code, message, status=400):
    return {"ok": False, "error": {"code": code, "message": message}}


@app.exception_handler(ApiError)
def _handle_api_error(_req: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status, content=_err(exc.code, exc.message, exc.status))


@app.exception_handler(RequestValidationError)
def _handle_validation_error(request: Request, exc: RequestValidationError):
    """pydantic 校验失败也要走统一结构，不然字段类型不对会返回自己那套格式。"""
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(x) for x in first.get("loc", []))
    msg = first.get("msg", "")

    # 落在中间件之后这些请求就进不到路由函数了，所以得在这里补记一笔。
    # 被拒的请求往往比成功的更值得看：有人反复试探参数边界、上游调用方的字段
    # 格式总是不对，这些信号只藏在成功日志里是发现不了的。
    audit.log_event(
        "rejected",
        {
            "path": request.url.path,
            "field": loc,
            "reason": f"{first.get('type', '')}: {msg}",
            "body_excerpt": str(getattr(exc, "body", ""))[:200],
        },
    )
    return JSONResponse(
        status_code=422, content=_err("invalid_request", f"字段 {loc} 不合规：{msg}")
    )


@app.exception_handler(Exception)
def _handle_unexpected(_req: Request, exc: Exception):
    """兜底：打印堆栈便于排查，但对外只给错误码，别把内部细节抛出去。"""
    traceback.print_exc()
    return JSONResponse(status_code=500, content=_err("internal_error", str(exc)))

_emb, _vs, _bm25, _vec = build_index(CORPUS)
_DOCS = {d["id"]: d for d in CORPUS}
_RERANKER = get_reranker()


class RetrieveRequest(BaseModel):
    query: str = Field(max_length=CONFIG.max_query_chars)
    topk: int = Field(default=CONFIG.topk, ge=1, le=CONFIG.max_topk)
    use_rerank: bool = True


def _check(req: RetrieveRequest, endpoint: str):
    """入口处的输入校验，挡掉会把服务拖垮的畸形请求。

    query 长度和 topk 范围交给 pydantic 的 Field 约束（在进路由函数之前就拦住了，
    见上面的 _handle_validation_error），这里补它做不了的一件：纯空白输入。
    strip 之后为空说明用户什么都没输入，记一笔 rejected 再拒绝——这类请求单独看
    没意义，攒起来是能看出异常流量的。

    顺带把首尾空白去掉再往下游传：不然「退货 」和「退货」会生成两个缓存 key，
    同一句话被重复算一遍。
    """
    query = (req.query or "").strip()
    if not query:
        audit.log_event(
            "rejected", {"path": "/" + endpoint, "topk": req.topk, "reason": "empty_query"}
        )
        raise ApiError("invalid_query", "query 不能为空", 422)
    return query


def _do_retrieve(query, topk, use_rerank):
    """检索层，/retrieve 和 /ask 共用。返回 (results, cached, reranked)。"""
    key = f"rag:{query}:{topk}:{use_rerank}"

    hit = cache_get(key)
    if hit is not None:
        return hit, True, None

    candidates = rrf_fuse(_bm25.search(query, topk=10), _vec.search(query, topk=10), topk=10)
    if use_rerank and _RERANKER is not None:
        final = rerank(_RERANKER, query, candidates, _DOCS, topk=topk)
    else:
        final = candidates[:topk]

    results = [
        {"id": doc_id, "score": round(score, 4), "text": _DOCS[doc_id]["text"]}
        for doc_id, score in final
    ]
    cache_set(key, results)
    return results, False, bool(use_rerank and _RERANKER is not None)


def _finish(endpoint, t0, fields):
    """统一收尾：算耗时、落一行审计。日志写失败不影响这次请求返回。"""
    latency = round((time.time() - t0) * 1000, 2)
    audit.log_event(endpoint, {**fields, "latency_ms": latency})
    return latency


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    query = _check(req, "retrieve")
    t0 = time.time()
    try:
        results, cached, reranked = _do_retrieve(query, req.topk, req.use_rerank)
    except Exception as e:  # noqa: BLE001 - 失败样本也要留痕，不能漏掉就抛上去
        traceback.print_exc()
        _finish("retrieve", t0, {"query": query, "error": f"{type(e).__name__}: {e}"})
        raise ApiError("internal_error", str(e), 500) from e

    latency = _finish(
        "retrieve",
        t0,
        {
            "query": query,
            "topk": req.topk,
            "cached": bool(cached),
            "rerank": bool(reranked),
            "hit_ids": [r["id"] for r in results],
            "hit_count": len(results),
            "top_score": results[0]["score"] if results else None,
        },
    )
    return {
        "ok": True,
        "cached": cached,
        "results": results,
        "latency_ms": latency,
        "rerank": reranked,
    }


@app.post("/ask")
def ask(req: RetrieveRequest):
    """检索 + 生成。没配 MODEL_API_KEY 时退化成纯检索，answer 为 null。"""
    query = _check(req, "ask")
    t0 = time.time()
    try:
        results, cached, reranked = _do_retrieve(query, req.topk, req.use_rerank)
        passages = [(r["id"], r["score"], r["text"]) for r in results]
        gen = generate(query, passages)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        _finish("ask", t0, {"query": query, "error": f"{type(e).__name__}: {e}"})
        raise ApiError("internal_error", str(e), 500) from e

    out = {
        "ok": True,
        "query": query,
        "passages": [
            {"id": r["id"], "score": r["score"], "text": r["text"]} for r in results
        ],
        "cached": cached,
        "rerank": reranked,
    }

    if gen is None:
        out.update(
            answer=None,
            citations=[],
            refused=False,
            note="未配置 MODEL_API_KEY，只返回检索结果。配了 key 才有生成环节。",
        )
    elif "error" in gen:
        out.update(answer=None, citations=[], refused=False, error=gen["error"])
    else:
        out.update(
            answer=gen["answer"],
            citations=gen["citations"],
            refused=gen["refused"],
            model=gen["model"],
            usage=gen["usage"],
        )

    usage = gen.get("usage", {}) if isinstance(gen, dict) else {}
    out["latency_ms"] = _finish(
        "ask",
        t0,
        {
            "query": query,
            "topk": req.topk,
            "cached": bool(cached),
            "rerank": bool(reranked),
            "hit_ids": [r["id"] for r in results],
            "cited_ids": out.get("citations", []),
            "refused": bool(out.get("refused")),
            "error": out.get("error"),
            "model": out.get("model"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
    )
    return out


@app.get("/stats")
def get_stats():
    """审计日志的汇总视图：调用量、缓存命中率、延迟 p95、拒答与错误数。"""
    return {"ok": True, "log_path": audit.log_path(), **audit.stats()}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_backend": CONFIG.embed_backend,
        "rerank": bool(CONFIG.rerank_model),
        "llm": bool(CONFIG.model_api_key),
        "model": CONFIG.model_name if CONFIG.model_api_key else None,
        "audit": CONFIG.audit_enabled,
        "max_query_chars": CONFIG.max_query_chars,
        "max_topk": CONFIG.max_topk,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html lang="zh"><head><meta charset="utf-8"><title>RAG 混合检索演示</title>
    <style>body{font-family:sans-serif;margin:24px;max-width:820px}input{width:340px;padding:6px}
    button{padding:6px 14px;margin-left:6px}
    .item{border:1px solid #ddd;border-radius:8px;padding:10px;margin:8px 0}
    .ans{background:#f6f8fa;border-left:4px solid #4a90d9;padding:12px;border-radius:6px;white-space:pre-wrap}
    .cite{color:#888;font-size:13px}</style></head>
    <body><h2>RAG 检索 + 生成演示</h2>
    <input id="q" placeholder="输入问题，如：如果被告人不服判决，有什么权利？">
    <button onclick="go('retrieve')">只检索</button>
    <button onclick="go('ask')">检索并生成答案</button>
    <div id="out"></div>
    <script>
    function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
    async function go(mode){
      const q=document.getElementById('q').value;
      const el=document.getElementById('out');
      el.innerHTML='<p class="cite">请求中…</p>';
      const r=await fetch('/'+mode,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query:q,topk:3})});
      const d=await r.json();
      let s='';
      if(mode==='ask'){
        if(d.answer){
          s+='<div class="ans">'+esc(d.answer)+'</div>';
          s+='<p class="cite">引用段落 id: '+(d.citations||[]).join(', ')
            +' | 模型: '+esc(d.model)+' | 拒答: '+d.refused+'</p>';
        }else{
          s+='<p class="cite">'+(d.error?('调用失败：'+esc(d.error)):esc(d.note))+'</p>';
        }
      }
      s+='<p class="cite">缓存命中: '+d.cached+' | 耗时: '+d.latency_ms+'ms</p>';
      (d.passages||d.results||[]).forEach((x,i)=>{
        s+='<div class="item"><b>'+(i+1)+'. [id='+x.id+', score='+x.score+']</b><br>'+esc(x.text)+'</div>';});
      el.innerHTML=s;
    }
    </script></body></html>
    """
