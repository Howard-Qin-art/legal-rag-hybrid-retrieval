# Legal RAG Hybrid Retrieval · 法律领域混合检索服务

> 一个**带真实评测数据**的 RAG 检索后端：BM25 稀疏检索 + 向量稠密检索 → RRF 融合 → Cross-Encoder 重排，
> 底层接 pgvector，上层是 FastAPI 服务 + Redis 缓存。
>
> 核心结论（不是拍脑袋，是跑出来的）：**在法律文本上，BM25 单路的 Recall@1 比纯向量高 4.7 个点；
> 混合 + 重排比纯向量高 6.0 个点。**

---

## 一、这个项目的目标

| 维度 | 目标 | 完成情况 |
| --- | --- | --- |
| 算法 | 搞清楚"单路向量检索够不够用"，用数据回答 | ✅ 4 种方案横向对比，有 Recall@1 / Recall@5 / MRR |
| 工程 | 不是 notebook demo，是能起服务、能切真实数据库的后端 | ✅ FastAPI + pgvector + Redis + Docker |
| 数据 | 语料和评测集必须真实、可复现，不能是 20 段手写玩具数据 | ✅ 3000 段真实法律语料 + 150 条真实问句 |
| 可信 | 简历上每个数字都能当场复现 | ✅ 一条命令重跑出下表 |

**为什么选法律场景**：法律文本里法条编号、罪名、程序名是强精确匹配信号（"第 294 条"不能被语义泛化），
这类场景最能暴露纯向量检索的短板——事实也确实如此（见效果表）。

---

## 二、效果（真实跑出来的数据）

语料 **3000 段**，评测问句 **150 条**，嵌入 `BAAI/bge-small-zh-v1.5`，重排 `BAAI/bge-reranker-base`：

| 方法 | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| BM25（稀疏） | 0.900 | 0.980 | 0.938 |
| 向量（稠密） | 0.853 | 0.953 | 0.905 |
| 混合 RRF | 0.907 | 0.987 | 0.942 |
| **混合 + 重排** | **0.913** | **0.987** | **0.947** |

结论（面试讲这三句就够）：

1. **法律场景下 BM25(0.900) 强于纯向量(0.853)** —— 术语和法条编号靠精确匹配，语义向量反而被稀释。
2. **RRF 融合把两者拉到 0.907**，说明两路的错误是互补的，不是重叠的。
3. **重排再涨到 0.913**，但涨幅收窄（+0.6），因为 Recall@5 已经 0.987，天花板在"召回"而不在"排序"。

关于"天花板效应"的教训写在[踩坑记录](#七踩坑记录)里——这是我认为本项目最有价值的部分。

---

## 三、整体流程

```mermaid
flowchart TD
    A["① 数据获取<br/>HuggingFace datasets-server<br/>Brench/chinese_law_data_rag_ft<br/>281,142 行"]
    A -->|"scripts/fetch_law_data.py<br/>取 3000 条"| B["law_qa_zh_3000.json"]
    B --> C["② 语料/评测集构建<br/>output → CORPUS（去重后作为可检索段落）<br/>instruction → EVAL_SET（问句）<br/>问句对应的 output 即 gold 段落"]
    C --> D["③ 索引构建<br/>bge-small-zh 编码 3000 段 → 512 维<br/>写入 pgvector rag_chunks<br/>同时构建 BM25 倒排"]
    D --> E["④ 在线检索<br/>BM25 Top-10 ＋ 向量 Top-10"]
    E --> F["RRF 融合<br/>score = Σ 1/(60 + rank)"]
    F --> G["Cross-Encoder 重排<br/>只对 10 条候选做"]
    G --> H["Top-K 返回 + Redis 缓存"]
    C --> I["⑤ 离线评测<br/>evaluate.py 跑 4 种方案<br/>输出 Recall@1/@5、MRR"]
```

**检索链路（请求视角）**

```
POST /retrieve
  ├─ Redis 命中？ → 直接返回（cached=true）
  ├─ BM25.search(q, 10)  ─┐
  ├─ Vector.search(q, 10)─┴→ rrf_fuse(k=60) → 10 条候选
  ├─ 可选：CrossEncoder 重排 → Top-K
  └─ 写回 Redis（TTL 600s）
```

---

## 四、快速开始

### 方式 A：零依赖跑通（不需要模型、不需要数据库）

```bash
pip install -r requirements.txt
EMBED_BACKEND=hash DATA_MODULE=data python evaluate.py
```

用的是哈希兜底嵌入 + 内置 20 段示例语料，**只验证链路，指标没有意义**。

### 方式 B：本地真实效果（推荐，出上表的方式）

```bash
# 1) 起中间件（pgvector + redis）
docker compose up -d db redis

# 2) 装模型依赖（torch 较大，约 2GB）
pip install sentence-transformers

# 3) 跑评测（Windows PowerShell）
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"
$env:DATA_MODULE  = "data_law"
$env:RERANK_MODEL = "BAAI/bge-reranker-base"
python evaluate.py

# 4) 起服务，浏览器打开 http://localhost:8000
uvicorn main:app --reload --port 8000
```

> `init_db.sql` 会在 db 容器首次启动时自动执行（已挂载到 `docker-entrypoint-initdb.d`），
> 无需手动建表。换嵌入模型时必须同步改 `init_db.sql` 里的 `vector(512)` 维度。

### 方式 C：全 Docker

```bash
docker compose up --build
```

`api` 容器内需要 torch，默认不装；需要时：

```bash
docker compose build --build-arg WITH_TORCH=true api
```

---

## 五、环境变量

复制 `.env.example` 为 `.env` 后按需修改。全部变量都有降级默认值，不配也能跑。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | 空 | 空 → 内存向量库；填 → 真实 pgvector |
| `REDIS_URL` | 空 | 空 → fakeredis；填 → 真实 Redis |
| `EMBED_BACKEND` | `st` | `st` 真实句向量 / `hash` 离线兜底 |
| `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 512 维，换模型要同步改表结构 |
| `RERANK_MODEL` | 空 | 空 → 跳过重排；`BAAI/bge-reranker-base` |
| `DATA_MODULE` | `data` | `data_law` 法律语料 / `data` 示例语料 |
| `TOPK` / `EVAL_MAX` | `5` / `150` | 返回条数 / 评测问句条数 |

Windows 必看：`HF_HUB_DISABLE_SYMLINKS=1`（见踩坑记录）。

---

## 六、目录结构

```
├── config.py              统一配置，全走环境变量
├── embedding.py           嵌入层：sentence-transformers / 哈希兜底，含中文切分
├── vectorstore.py         向量库：pgvector（psycopg3 + register_vector）/ 内存版
├── retrievers.py          BM25 + 向量 + RRF 融合 + Cross-Encoder 重排
├── cache.py               Redis 查询缓存（key = query:topk:use_rerank，TTL 600s）
├── main.py                FastAPI 服务：/retrieve、/health、演示页
├── evaluate.py            四种方案横向评测，输出 Recall@1/@5、MRR
├── data_law.py            法律语料与评测集构建（output→语料，instruction→问句）
├── data.py                内置 20 段示例语料（零依赖演示用）
├── law_qa_zh_3000.json    真实语料（含 _meta 标注来源与字段含义）
├── scripts/fetch_law_data.py   从 HuggingFace 重新抓取语料，保证数据可复现
├── init_db.sql            pgvector 建表（vector(512) + HNSW 索引建议）
├── Dockerfile / docker-compose.yml
└── .env.example
```

---

## 六、设计权衡（面试可聊的点）

**1. 为什么用 RRF 而不是加权求和？**
BM25 分数是 TF-IDF 量纲（可能几十），向量相似度是 0~1，两者不可比。加权求和要先归一化，
而归一化参数很脆。RRF 只用名次 `1/(60+rank)`，天然跨量纲，且不需要调参——工程上更省心。

**2. 为什么要重排，但不能全程重排？**
Cross-Encoder 把 query 和 doc 拼在一起过一遍 BERT，精度高但要算 N 次前向。
对 3000 段全量重排 = 3000 次前向，不可接受。所以是**两阶段**：先用便宜的方式（BM25+向量）召回 10 条，
再用贵的模型精排 10 条。这是工业界标准做法。

**3. 为什么评测集不用人工标注？**
数据集本身是问答对，于是：**回答 = 语料段落**，**问句 = 评测 query**，
问句对应回答的段落 id 天然就是 gold。零标注成本拿到 3000 条 query-passage 对。
前提是问句是自然语言（不是照抄原文关键词），否则两路检索都满分、失去区分度。

**4. 缓存 key 为什么带 topk 和 use_rerank？**
`rag:{query}:{topk}:{use_rerank}`。少任何一个字段都会把不同参数的结果串味。
TTL 只设 600 秒，因为知识库更新后旧结果要尽快失效——缓存穿透的成本远低于返回过期法条。

**5. 为什么要做降级设计？**
没配 `DATABASE_URL` 用内存向量库、没配 `REDIS_URL` 用 fakeredis、模型加载失败回退哈希嵌入。
好处是任何人在任何机器上 `pip install -r requirements.txt` 都能立刻跑通，不被环境卡死。
代价是**可能静默降级出假指标**，所以 `evaluate.py` 每次都会打印当前嵌入后端，提醒别拿 hash 结果当真。

---

## 七、踩坑记录

都是实际遇到并解决的，面试聊这些比聊概念有说服力。

**① Windows + HuggingFace 符号链接 → `config.json` 0 字节**
报错 `OSError: config file is not a valid JSON`。排查后不是网络问题：HF 用 symlink 把 `snapshots/` 指向 `blobs/`，
Windows 没开开发者模式时创建失败会退化成空文件。
解决：`rm -rf` 损坏缓存 + `HF_HUB_DISABLE_SYMLINKS=1` 重新下载（改为复制而非链接）。

**② sentence-transformers 静默降级**
模型其实在本地缓存里，但默认会联网校验，网络不通就报 502 然后悄悄回退哈希嵌入，
指标看着正常其实是假的。解决：设 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`。

**③ psycopg3 写向量报类型错误**
直接把 Python list 传给 `vector` 列会失败。解决：优先 `pgvector.psycopg.register_vector`（可直接传 numpy），
没装该包时退化成 `'[v1,v2,...]'::vector` 字符串强转，两种方式都验证过。

**④ 指标天花板效应（最有价值的一条）**
语料 20 段时四种方案**全是 1.000**；扩到 600 段仍然 0.98+，毫无区分度。原因是语料太少 +
问句几乎是原文关键词，两路都能秒中。
解决：语料扩到 3000 段 + 加 **Recall@1**（比 Recall@5 更难，能拉开差距）+ 用真实自然语言问句，
才得到有区分度的结果。**教训：评测做不出来和做不出来效果，是两回事——先看天花板在哪。**

**⑤ 换嵌入模型要同步改表结构**
`bge-small-zh-v1.5` 是 512 维，`init_db.sql` 里必须写 `vector(512)`。换 `bge-base-zh`（768）不改表会直接报错。

---

## 八、数据来源与许可

- 数据集：[Brench/chinese_law_data_rag_ft](https://huggingface.co/datasets/Brench/chinese_law_data_rag_ft)（281,142 条中文法律问答）
- 本仓库取前 3000 条，存为 `law_qa_zh_3000.json`（含 `_meta` 标注来源与字段含义）
- 重新抓取：`python scripts/fetch_law_data.py`（可调 `TARGET_ROWS`）
- 嵌入模型：`BAAI/bge-small-zh-v1.5`；重排模型：`BAAI/bge-reranker-base`

## 九、还没做的（诚实列出来，别让面试官替你发现）

- 语料按段落切分：目前直接用整段回答，长文本场景需要 chunk + overlap
- HNSW 索引未启用（`init_db.sql` 里有注释），3000 段规模下全表扫描更快
- 没有接 LLM 生成，只做到检索层（RAG 的 "G" 未包含）
- 多租户隔离（tenant_id + RLS）未做

## 十、License

[MIT](LICENSE)
