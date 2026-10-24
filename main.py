"""RAG 混合检索服务（FastAPI）。

链路：/retrieve -> 查 Redis 缓存 -> 命中直接返回；否则 BM25+向量 RRF 融合 -> 可选重排 -> 写缓存。

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


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    t0 = time.time()
    key = f"rag:{req.query}:{req.topk}:{req.use_rerank}"

    # 1) 缓存命中直接返回（省掉 embedding 与检索开销）
    hit = cache_get(key)
    if hit is not None:
        return {"cached": True, "results": hit, "latency_ms": 0.0}

    # 2) 两路检索 + RRF 融合
    candidates = rrf_fuse(_bm25.search(req.query, topk=10), _vec.search(req.query, topk=10), topk=10)

    # 3) 可选重排（只对融合后的少量候选做）
    if req.use_rerank and _RERANKER is not None:
        final = rerank(_RERANKER, req.query, candidates, _DOCS, topk=req.topk)
    else:
        final = candidates[: req.topk]

    results = [
        {"id": doc_id, "score": round(score, 4), "text": _DOCS[doc_id]["text"]}
        for doc_id, score in final
    ]
    cache_set(key, results)
    return {
        "cached": False,
        "results": results,
        "latency_ms": round((time.time() - t0) * 1000, 2),
        "rerank": bool(req.use_rerank and _RERANKER is not None),
    }


@app.get("/health")
def health():
    return {"status": "ok", "embed_backend": CONFIG.embed_backend}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html lang="zh"><head><meta charset="utf-8"><title>RAG 混合检索演示</title>
    <style>body{font-family:sans-serif;margin:24px}input{width:320px;padding:6px}
    .item{border:1px solid #ddd;border-radius:8px;padding:10px;margin:8px 0}</style></head>
    <body><h2>RAG 混合检索演示</h2>
    <input id="q" placeholder="输入问题，如：退货之后钱多久退回来？">
    <button onclick="go()">检索</button><pre id="out"></pre>
    <script>
    async function go(){
      const q=document.getElementById('q').value;
      const r=await fetch('/retrieve',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query:q,topk:3})});
      const d=await r.json();
      let s='缓存命中: '+d.cached+' | 耗时: '+d.latency_ms+'ms\\n\\n';
      (d.results||[]).forEach((x,i)=>{s+=(i+1)+'. [id='+x.id+', score='+x.score+'] '+x.text+'\\n\\n';});
      document.getElementById('out').textContent=s;
    }
    </script></body></html>
    """
