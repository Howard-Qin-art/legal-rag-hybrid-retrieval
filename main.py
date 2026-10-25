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

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from cache import cache_get, cache_set
from config import CONFIG
from generator import generate
from retrievers import build_index, rrf_fuse, get_reranker, rerank

# 语料可切换：DATA_MODULE=data（示例）或 data_law（真实法律语料）
CORPUS = importlib.import_module(CONFIG.data_module).CORPUS

app = FastAPI(title="RAG 混合检索服务")

_emb, _vs, _bm25, _vec = build_index(CORPUS)
_DOCS = {d["id"]: d for d in CORPUS}
_RERANKER = get_reranker()


class RetrieveRequest(BaseModel):
    query: str
    topk: int = 5
    use_rerank: bool = True


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


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    t0 = time.time()
    results, cached, reranked = _do_retrieve(req.query, req.topk, req.use_rerank)
    return {
        "cached": cached,
        "results": results,
        "latency_ms": round((time.time() - t0) * 1000, 2),
        "rerank": reranked,
    }


@app.post("/ask")
def ask(req: RetrieveRequest):
    """检索 + 生成。没配 MODEL_API_KEY 时退化成纯检索，answer 为 null。"""
    t0 = time.time()
    results, cached, reranked = _do_retrieve(req.query, req.topk, req.use_rerank)

    passages = [(r["id"], r["score"], r["text"]) for r in results]
    gen = generate(req.query, passages)

    out = {
        "query": req.query,
        "passages": [
            {"id": r["id"], "score": r["score"], "text": r["text"]} for r in results
        ],
        "cached": cached,
        "rerank": reranked,
        "latency_ms": round((time.time() - t0) * 1000, 2),
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
    return out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_backend": CONFIG.embed_backend,
        "rerank": bool(CONFIG.rerank_model),
        "llm": bool(CONFIG.model_api_key),
        "model": CONFIG.model_name if CONFIG.model_api_key else None,
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
