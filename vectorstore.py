"""向量库：真实用 pgvector；没配 DATABASE_URL 时用内存版跑通链路。

pgvector 要点：
- 表结构见 init_db.sql：embedding vector(512)，维度要与嵌入模型一致。
- 检索用余弦距离运算符 <=>，相似度 = 1 - 距离。
- 建议对 embedding 列建 ivfflat / hnsw 索引（数据量上来后必做）。
"""
import numpy as np

from config import CONFIG


class MemoryVS:
    """内存版：向量已归一化，直接做点积即余弦相似度。仅本地开发/测试。"""

    def __init__(self, dim):
        self.dim = dim
        self.ids: list = []
        self.mat = None

    def add(self, ids, vecs, texts=None):
        self.ids = list(ids)
        self.mat = np.asarray(vecs, dtype="float32")

    def search(self, qvec, topk=10):
        if self.mat is None:
            return []
        sims = self.mat @ np.asarray(qvec, dtype="float32")
        idx = np.argsort(-sims)[:topk]
        return [(self.ids[i], float(sims[i])) for i in idx]


class PgVectorVS:
    """pgvector 版：依赖 psycopg 与已创建的表 rag_chunks（见 init_db.sql）。

    向量入参处理（关键）：
    - 优先用 pgvector 官方适配器 register_vector，可直接传 numpy 数组；
    - 若没装 pgvector 包，则退化成把向量序列化成 '[v1,v2,...]' 字符串再 ::vector 强转，
      两种方式都能正确写入/检索，避免 psycopg 把 list 当成普通数组报错。
    """

    def __init__(self, dsn, dim):
        import psycopg
        self.dim = dim
        self.conn = psycopg.connect(dsn)
        try:
            from pgvector.psycopg import register_vector
            register_vector(self.conn)
            self._native = True
        except Exception:  # noqa: BLE001
            self._native = False

    def _param(self, vec):
        arr = np.asarray(vec, dtype="float32")
        if self._native:
            return arr
        return "[" + ",".join(f"{x:.6f}" for x in arr) + "]"

    def add(self, ids, vecs, texts=None):
        rows = [
            (
                int(i),
                texts[k] if texts is not None else "",
                self._param(v),
            )
            for k, (i, v) in enumerate(zip(ids, vecs))
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO rag_chunks(id, content, embedding) VALUES(%s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, "
                "embedding = EXCLUDED.embedding",
                rows,
            )
        self.conn.commit()

    def search(self, qvec, topk=10):
        p = self._param(qvec)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, 1 - (embedding <=> %s::vector) AS score "
                "FROM rag_chunks ORDER BY embedding <=> %s::vector LIMIT %s",
                (p, p, topk),
            )
            return [(r[0], float(r[1])) for r in cur.fetchall()]


def get_vectorstore(dim):
    if CONFIG.database_url:
        return PgVectorVS(CONFIG.database_url, dim)
    return MemoryVS(dim)
