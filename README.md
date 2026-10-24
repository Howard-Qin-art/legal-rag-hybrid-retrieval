# Legal RAG Hybrid Retrieval · 法律领域 RAG 混合检索服务

一个面向中文法律文本的**检索后端**：BM25 稀疏检索 + 向量稠密检索 → RRF 融合 → Cross-Encoder 重排，
底层接 pgvector，上层是 FastAPI 服务 + Redis 查询缓存，并附带一套可复现的离线评测。

> 核心结论：**在法律文本上 BM25 单路的 Recall@1 比纯向量高 4.7 个点；
> 混合 + 重排比纯向量高 6.0 个点。** 纯向量检索并不是万能解，本项目用数据量化了这个差距。

---

## 一、项目要解决什么

通用 RAG 方案通常默认"向量检索就够了"，但在法律这类专业文本上，这个假设并不成立：
法条编号、罪名、程序名是**强精确匹配**信号（"第二百九十四条"不该被语义泛化成"某条规定"），
而向量检索倾向于语义泛化，反而会把精确信号稀释掉。

本项目做的是把四种检索方案放在同一份真实语料、同一批评测问句上横向对比，
看清楚每种方案的边界，并把效果最好的一条链路做成可调用的 HTTP 服务。

| 目标 | 说明 | 状态 |
| --- | --- | --- |
| 量化方案差异 | 4 种检索方案对比，指标用 Recall@1 / Recall@5 / MRR | ✅ |
| 工程可用 | 不是 notebook demo，能起服务、能切真实向量数据库 | ✅ FastAPI + pgvector + Redis + Docker |
| 数据可信 | 语料与评测集来自真实数据集，可一键复现 | ✅ 3000 段语料 + 150 条真实问句 |
| 结果可复现 | 报告中的每个数字都能用一条命令重跑出来 | ✅ `python evaluate.py` |

---

## 二、效果

语料 **3000 段**，评测问句 **150 条**，嵌入 `BAAI/bge-small-zh-v1.5`（512 维），重排 `BAAI/bge-reranker-base`：

| 方法 | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| BM25（稀疏） | 0.900 | 0.980 | 0.938 |
| 向量（稠密） | 0.853 | 0.953 | 0.905 |
| 混合 RRF | 0.907 | 0.987 | 0.942 |
| **混合 + 重排** | **0.913** | **0.987** | **0.947** |

从数据里读出来的三点：

1. **法律场景下 BM25（0.900）强于纯向量（0.853）** —— 术语与法条编号依赖精确匹配，语义向量在此反而是劣势。
2. **RRF 融合提升到 0.907** —— 说明两路检索的命中的错误是互补的，不是重叠的，融合确实拿到了增量。
3. **重排再提升到 0.913，但涨幅收窄（+0.6）** —— 因为 Recall@5 已达 0.987，瓶颈在"召回"而非"排序"；
   继续优化应该往召回侧投入，而不是堆更贵的重排模型。

指标撞天花板的问题及处理方式记录在[已知问题](#七已知问题与处理记录)第 ④ 条。

---

## 三、整体流程

```mermaid
flowchart TD
    A["① 数据获取<br/>HuggingFace datasets-server<br/>Brench/chinese_law_data_rag_ft<br/>281,142 行"]
    A -->|"scripts/fetch_law_data.py<br/>取 3000 条"| B["law_qa_zh_3000.json"]
    B --> C["② 语料与评测集构建<br/>output → CORPUS（去重后作为可检索段落）<br/>instruction → EVAL_SET（问句）<br/>问句对应 output 的段落 id 即 gold"]
    C --> D["③ 索引构建<br/>bge-small-zh 编码 3000 段 → 512 维<br/>写入 pgvector rag_chunks<br/>同时构建 BM25 倒排"]
    D --> E["④ 在线检索<br/>BM25 Top-10 ＋ 向量 Top-10"]
    E --> F["RRF 融合<br/>score = Σ 1/(60 + rank)"]
    F --> G["Cross-Encoder 重排<br/>只对 10 条候选做"]
    G --> H["Top-K 返回 + 写 Redis 缓存"]
    C --> I["⑤ 离线评测<br/>evaluate.py 跑 4 种方案<br/>输出 Recall@1 / Recall@5 / MRR"]
```

**请求视角的检索链路**

```
POST /retrieve
  ├─ Redis 命中？ → 直接返回（cached=true）
  ├─ BM25.search(q, 10)   ─┐
  ├─ Vector.search(q, 10) ─┴→ rrf_fuse(k=60) → 10 条候选
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

使用哈希兜底嵌入 + 内置 20 段示例语料，**仅用于验证链路连通性，指标不具备参考价值**。

### 方式 B：本地完整效果（产出上表的方式）

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

> `init_db.sql` 会在 db 容器首次启动时自动执行（已挂载到 `docker-entrypoint-initdb.d`），无需手动建表。
> 更换嵌入模型时必须同步修改 `init_db.sql` 中的 `vector(512)` 维度。

### 方式 C：全 Docker

```bash
docker compose up --build
```

`api` 镜像默认不含 torch，需要真实句向量时重新构建：

```bash
docker compose build --build-arg WITH_TORCH=true api
```

### 接口

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "如果被告人不服判决，有什么权利？", "topk": 3, "use_rerank": true}'
```

返回：

```json
{
  "cached": false,
  "results": [{"id": 0, "score": 0.9981, "text": "根据《刑事诉讼法》第294条……"}],
  "latency_ms": 123.45,
  "rerank": true
}
```

---

## 五、环境变量

复制 `.env.example` 为 `.env` 后按需修改。所有变量都有降级默认值，不配置也能运行。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | 空 | 空 → 内存向量库；填 → 真实 pgvector |
| `REDIS_URL` | 空 | 空 → fakeredis；填 → 真实 Redis |
| `EMBED_BACKEND` | `st` | `st` 真实句向量 / `hash` 离线兜底 |
| `EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 512 维，换模型需同步改表结构 |
| `RERANK_MODEL` | 空 | 空 → 跳过重排；`BAAI/bge-reranker-base` |
| `DATA_MODULE` | `data` | `data_law` 法律语料 / `data` 示例语料 |
| `TOPK` / `EVAL_MAX` | `5` / `150` | 返回条数 / 评测问句条数 |

Windows 环境请额外设置 `HF_HUB_DISABLE_SYMLINKS=1`，原因见[已知问题](#七已知问题与处理记录)第 ① 条。

---

## 六、目录结构

```
├── config.py              统一配置，全部走环境变量
├── embedding.py           嵌入层：sentence-transformers / 哈希兜底，含中文切分
├── vectorstore.py         向量库：pgvector（psycopg3 + register_vector）/ 内存版
├── retrievers.py          BM25 + 向量 + RRF 融合 + Cross-Encoder 重排
├── cache.py               Redis 查询缓存（key = query:topk:use_rerank，TTL 600s）
├── main.py                FastAPI 服务：/retrieve、/health、演示页
├── evaluate.py            四种方案横向评测，输出 Recall@1 / Recall@5 / MRR
├── data_law.py            法律语料与评测集构建（output→语料，instruction→问句）
├── data.py                内置 20 段示例语料（零依赖演示用）
├── law_qa_zh_3000.json    真实语料（含 _meta 标注来源与字段含义）
├── scripts/fetch_law_data.py   从 HuggingFace 重新抓取语料，保证数据可复现
├── init_db.sql            pgvector 建表（vector(512)，含 HNSW 索引建议）
├── Dockerfile / docker-compose.yml
└── .env.example
```

---

## 七、关键设计决策

**1. 融合为什么用 RRF 而不是加权求和**

BM25 分数是 TF-IDF 量纲（数值可能到几十），向量相似度是 0~1，两者不可直接比较。
加权求和需要先做归一化，而归一化系数对语料分布敏感，换一批数据就要重新调。
RRF 只使用名次：`score = Σ 1/(60 + rank)`，天然跨量纲，无需调参，工程上更稳定。

**2. 重排为什么只做第二阶段**

Cross-Encoder 把 query 与 doc 拼接后完整过一遍 BERT，精度高但每次都要一次前向计算。
对 3000 段全量重排等于 3000 次前向，延迟不可接受。
因此采用两阶段：先用低成本方式（BM25 + 向量）召回 10 条，再用高成本模型精排这 10 条。

**3. 评测集如何在零标注成本下构造**

数据集本身是问答对，于是：**回答 = 语料段落**，**问句 = 评测 query**，
问句所对应回答的段落 id 天然就是 gold。由此零成本得到 query-passage 评测对。
前提是该问句必须是自然语言表述而非原文关键词拷贝，否则两路检索都会满分、失去区分度。

**4. 缓存 key 的设计**

`rag:{query}:{topk}:{use_rerank}` —— 三个字段缺一不可，否则不同参数的检索结果会互相串味。
TTL 设为 600 秒：知识库更新后旧结果要尽快失效，返回过期内容的代价远高于多算一次检索。

**5. 为什么要做降级设计**

未配置 `DATABASE_URL` 时用内存向量库，未配置 `REDIS_URL` 时用 fakeredis，模型加载失败时回退哈希嵌入。
好处是任何机器上 `pip install -r requirements.txt` 后都能立刻跑通，不被环境卡住。
代价是可能出现"静默降级出假指标"，因此 `evaluate.py` 每次运行都会打印当前嵌入后端，提示结果是否可采信。

---

## 八、已知问题与处理记录

均为实际开发中遇到并已解决的问题，记录于此便于复现和维护。

**① Windows + HuggingFace 符号链接导致 `config.json` 0 字节**

报错 `OSError: config file is not a valid JSON`。排查确认不是网络问题：
HF 用 symlink 将 `snapshots/` 指向 `blobs/`，Windows 未开启开发者模式时创建失败，会退化成空文件。
处理：删除损坏缓存后设置 `HF_HUB_DISABLE_SYMLINKS=1` 重新下载（改为复制而非链接）。

**② sentence-transformers 静默降级**

模型已在本地缓存，但默认会联网校验；网络不通时报 502 后自动回退哈希嵌入，
此时指标看似正常实际无效。处理：设置 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` 走离线加载。

**③ psycopg3 写入向量报类型错误**

直接把 Python list 传给 `vector` 列会失败。处理：优先使用 `pgvector.psycopg.register_vector`
（可直接传 numpy 数组）；未安装该包时退化为 `'[v1,v2,...]'::vector` 字符串强转，两种方式均已验证可用。

**④ 指标天花板效应（影响最大的一条）**

语料 20 段时四种方案指标**全部为 1.000**；扩到 600 段仍普遍 0.98+，毫无区分度。
原因是语料规模太小，且评测问句接近原文关键词，两路检索都能轻易命中。
处理：语料扩至 3000 段 + 增加 **Recall@1**（比 Recall@5 更难，能拉开差距）+ 使用真实自然语言问句，
最终得到有区分度的结果。**经验：指标全满分时，先检查天花板，而不是急着调模型。**

**⑤ 更换嵌入模型需同步修改表结构**

`bge-small-zh-v1.5` 输出 512 维，`init_db.sql` 中必须对应 `vector(512)`。
换成 `bge-base-zh`（768 维）不改表结构会直接报错。

---

## 九、数据来源

- 数据集：[Brench/chinese_law_data_rag_ft](https://huggingface.co/datasets/Brench/chinese_law_data_rag_ft)（281,142 条中文法律问答）
- 本仓库取前 3000 条，存为 `law_qa_zh_3000.json`（文件内 `_meta` 字段标注了来源、字段含义与用途）
- 重新抓取：`python scripts/fetch_law_data.py`（可用 `TARGET_ROWS` 调整条数）
- 嵌入模型：`BAAI/bge-small-zh-v1.5`；重排模型：`BAAI/bge-reranker-base`

---

## 十、后续计划

当前检索链路已完整可用，以下部分尚未实现：

- **文本切分**：目前直接使用整段回答作为检索单元，长文档场景需要 chunk + overlap
- **向量索引**：HNSW 索引未启用（`init_db.sql` 中留有注释语句），3000 段规模下全表扫描更快
- **生成环节**：目前只做到检索层，未接入 LLM 生成最终答案
- **多租户隔离**：未实现 tenant_id + 行级安全策略

---

## License

[MIT](LICENSE)
