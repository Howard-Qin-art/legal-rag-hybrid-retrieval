# legal-rag-hybrid-retrieval

中文法律文本的混合检索后端。BM25 和向量各召回一路，用 RRF 合一下，再用 Cross-Encoder 把前几条候选重排一遍。向量存 pgvector，服务是 FastAPI，查询结果丢 Redis 缓存。

写这个的起因有点简单：我看过的 RAG 项目和教程里，检索这一层基本都是直接上一套向量检索就结束了，很少有人真的把关键词检索和向量检索放在同一份语料、同一批问题上比过。大家默认"向量能理解语义，所以一定比关键词强"，这个默认我一直不太信，于是想自己跑一遍看看差多少。

结果是：至少在法律文本上，真不是这么回事。BM25 单路的 Recall@1 比纯向量高了 4.7 个点。下面表格里的数都是真跑出来的，`python evaluate.py` 一条命令能复现，不用光信我说的。

## 结果

3000 段语料，150 条问句，嵌入用 `BAAI/bge-small-zh-v1.5`，重排用 `BAAI/bge-reranker-base`：

|              | Recall@1 | Recall@5 | MRR   |
| ------------ | -------: | -------: | ----: |
| BM25         |    0.900 |    0.980 | 0.938 |
| 向量          |    0.853 |    0.953 | 0.905 |
| 混合 RRF      |    0.907 |    0.987 | 0.942 |
| 混合 + 重排    |    0.913 |    0.987 | 0.947 |

几个值得一说的点：

法律文本里 BM25 反而赢过纯向量，一开始有点意外，想想也合理 —— 法条编号、罪名、程序名这些东西要的是精确匹配，语义模型擅长泛化，在这儿等于把信号稀释了。

两路融合能涨到 0.907，说明它俩犯的错不是同一类，确实是互补的。真要是一路强一路弱到没贡献，融合只会拖后腿。

重排只多了 0.6 个点，比预期低。原因在 Recall@5 上：已经 0.987 了，说明该召回的基本都召回来了，瓶颈压根不在排序。想继续提得动召回侧，换个更强的重排模型收益很小。这个结论是我改方向之前唯一觉得亏的地方 —— 重排模型 1.1GB，性价比一般。

## 怎么跑起来

不装模型不装数据库也能先把链路跑通，用的是哈希兜底嵌入加 20 段示例语料，**这个模式下的指标没有参考价值**，只是验证代码没坏：

```bash
pip install -r requirements.txt
EMBED_BACKEND=hash DATA_MODULE=data python evaluate.py
```

要复现上面那张表，需要的东西比第一种模式多三样：

- **pgvector 数据库**（5432）：存 3000 段语料的 512 维向量，用 Docker 起
- **嵌入和重排模型**：本地能加载就行，可以现下，也可以像我这样提前下好放本地
- **Redis**（6379）：只有跑 HTTP 服务时才用得上，单纯跑评测其实可以不起

```bash
# 1. 起 pgvector（db，5432）和 Redis（redis，6379）
#    只跑评测的话，后面那个 redis 可以不加
docker compose up -d db redis

# 2. 装嵌入 + 重排模型依赖，会带上 torch，2GB 左右
pip install sentence-transformers

# 3. 告诉程序连哪个库、用哪份语料（Windows PowerShell）
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rag"
$env:DATA_MODULE  = "data_law"
$env:RERANK_MODEL = "BAAI/bge-reranker-base"
python evaluate.py

# 4. 起 FastAPI 服务（8000）。要用到缓存就再把 Redis 指过去
$env:REDIS_URL = "redis://localhost:6379/0"
uvicorn main:app --reload --port 8000
```

macOS / Linux 把 `$env:XXX = "..."` 换成 `export XXX="..."` 就行。

`init_db.sql` 挂在 `docker-entrypoint-initdb.d` 上，db 容器第一次起来会自动建表，不用手动执行。但要注意：换嵌入模型一定得同步改 `init_db.sql` 里的 `vector(512)`，维度对不上会直接报错。

调接口：

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "如果被告人不服判决，有什么权利？", "topk": 3, "use_rerank": true}'
```

```json
{
  "cached": false,
  "results": [{"id": 0, "score": 0.9981, "text": "根据《刑事诉讼法》第294条……"}],
  "latency_ms": 123.45,
  "rerank": true
}
```

浏览器打开 `http://localhost:8000` 有个能直接输问题的演示页。

全部 Docker 化是 `docker compose up --build`，但 api 镜像默认不含 torch，需要真嵌入时得 `docker compose build --build-arg WITH_TORCH=true api`，镜像会大不少。

## 数据是哪来的

语料来自 HuggingFace 上的 [Brench/chinese_law_data_rag_ft](https://huggingface.co/datasets/Brench/chinese_law_data_rag_ft)，28 万多条中文法律问答，我取了前 3000 条存成 `law_qa_zh_3000.json`，文件里的 `_meta` 字段写了来源和字段含义。想重新抓或者换数量：

```bash
python scripts/fetch_law_data.py              # 默认 3000 条
TARGET_ROWS=10000 python scripts/fetch_law_data.py
```

脚本走的是 datasets-server 的 HTTP 接口，不用装 datasets 也不用 git-lfs，比较省事。

评测集这块我一条都没自己标。数据本身就是问答对，那就换个用法：**回答当作语料段落**，**问句当作 query**，这条问句对应的那个回答，它的段落 id 天然就是 gold。于是一分钱不花就有了 query-passage 对。

要说清楚的是：标注成本不是消失了，是数据集作者已经付过了，我只是把问答对换了个用法。好处是我不用干这活，坏处是指标上限取决于那份数据本身的质量，我没法逐条验证每个问答是不是真的对得上。

这个做法还有两个绕不开的局限：

一是每个 query 只有一个 gold。但法律问题常常是好几个法条都算答得上，只认一个的话，把另一个召回来也不给分，算出来的数偏保守。

二是问句和它自己的回答天生会共享一些词（毕竟讲的是同一件事），BM25 在这上面多少占了点便宜。上面那个 0.900 里面，有一部分可能就是这个原因，不全是 BM25 真的更强。

要严谨的话，得另外人工标一批多 gold 的评测集来交叉验证。我没做这个，所以这里的结果当成趋势看就好，别当成定论。

## 几个设计上的选择

**融合为什么用 RRF。** BM25 的分数是 TF-IDF 量纲，能到几十；向量相似度是 0~1。两边加起来之前得先归一化，而归一化系数对语料分布特别敏感，换批数据就得重调。RRF 只看名次，`score = Σ 1/(60 + rank)`，天生跨量纲，也不用调参。省事且稳。

**重排为什么放在第二阶段。** Cross-Encoder 要把 query 和 doc 拼一起完整过一遍 BERT，精度是高，但每条候选都要一次前向。3000 段全量重排就是 3000 次前向，延迟没法看。所以先用便宜的方式召回 10 条，再用贵的模型排这 10 条。也就是召回和精排分开，做搜索的基本都这么拆，没什么新鲜的，但确实有效。

**缓存 key 带上 topk 和 use_rerank。** 拼成 `rag:{query}:{topk}:{use_rerank}`。少任何一个字段，不同参数的检索结果就会互相串味。TTL 设的是 600 秒，知识库更新之后旧结果得尽快失效 —— 多算一次检索的成本，比返回过期法条低得多。

**到处留了降级。** 没配 `DATABASE_URL` 就用内存向量库，没配 `REDIS_URL` 就用 fakeredis，模型加载失败就回退哈希嵌入。好处是换台机器装上 requirements 就能跑，不会被环境卡住。坏处是会**静默降级出一版假指标**，所以 `evaluate.py` 每次都会打印当前用的是哪种嵌入后端，看到 `hash` 就别当真。

## 踩过的坑

**Windows 下 HF 缓存把 config.json 弄成 0 字节。** 报 `OSError: config file is not a valid JSON`，我一开始以为是网不好，重试好几遍。实际是符号链接：HF 用 symlink 把 `snapshots/` 指到 `blobs/`，Windows 没开开发者模式时创建失败，退化成空文件。解决办法是删掉坏缓存，然后 `HF_HUB_DISABLE_SYMLINKS=1` 重新下载（改成复制而非链接）。Windows 用户建议直接把这条写进环境变量。

**模型在本地却加载不出来。** 同上联网校验的问题，网络不通时报 502，代码 catch 住然后悄悄回退哈希嵌入 —— 指标看着挺正常，其实是假的。设 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 走离线加载。

**psycopg3 写向量报类型错误。** Python list 不能直接丢给 `vector` 列。优先用 `pgvector.psycopg.register_vector`（能直接吃 numpy 数组），没装这个包就退化成拼 `'[v1,v2,...]'` 字符串再 `::vector` 强转，两条路都验过。

**指标天花板，这个是坑里最深的。** 语料 20 段的时候，四种方案 Recall 全是 1.000。扩到 600 段，还是普遍 0.98+，完全分不出高下。查下来是两个原因：语料太少，以及评测问句写得太像原文关键词，等于给两路都送分。最后把语料加到 3000 段、指标换成更难的 Recall@1、问句用真实自然语言，才拿到有区分度的数。**教训是：指标全满分时先怀疑天花板，别急着调模型。**

**换嵌入模型忘了改表。** `bge-small-zh-v1.5` 是 512 维，`init_db.sql` 里就得写 `vector(512)`。换 `bge-base-zh` 是 768 维，不改表直接报错。

## 环境变量

复制 `.env.example` 改成 `.env` 就行，全部都有降级默认值，一个不配也能跑。

| 变量                | 默认                     | 说明                                  |
| ------------------- | ------------------------ | ------------------------------------- |
| `DATABASE_URL`      | 空                       | 空则内存向量库，填了走真实 pgvector    |
| `REDIS_URL`         | 空                       | 空则 fakeredis，填了走真实 Redis       |
| `EMBED_BACKEND`     | `st`                     | `st` 真实句向量 / `hash` 离线兜底      |
| `EMBED_MODEL`       | `BAAI/bge-small-zh-v1.5` | 512 维，换模型要同步改表               |
| `RERANK_MODEL`      | 空                       | 空则跳过重排                           |
| `DATA_MODULE`       | `data`                   | `data_law` 法律语料 / `data` 示例语料  |
| `TOPK` / `EVAL_MAX` | `5` / `150`              | 返回条数 / 评测问句条数                |

## 代码结构

```
config.py              配置，全走环境变量
embedding.py           嵌入层，含中文切分和 hash 兜底
vectorstore.py         pgvector / 内存，两种向量库
retrievers.py          BM25、向量、RRF 融合、重排
cache.py               Redis 查询缓存
main.py                FastAPI：/retrieve、/health、演示页
evaluate.py            四方案评测，输出 Recall@1/@5、MRR
data_law.py            法律语料与评测集构建
data.py                20 段示例语料
scripts/fetch_law_data.py   重新抓语料
init_db.sql            建表语句
```

## 还没做

- 文本切分：现在直接拿整段回答当检索单元，长文档得上 chunk + overlap
- 向量索引：HNSW 没开，语句在 `init_db.sql` 注释里。3000 段这规模全表扫描更快，开了反而慢
- 生成环节：只做到检索层，没接 LLM 出答案，严格说现在是 R 不是完整的 RAG
- 多租户：没做隔离

## License

MIT
