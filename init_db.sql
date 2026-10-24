-- pgvector 初始化：启用扩展 + 建表
-- 注意：vector(512) 的维度要与嵌入模型一致（bge-small-zh-v1.5 = 512）
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id        INTEGER PRIMARY KEY,
    content   TEXT,
    embedding VECTOR(512)
);

-- 数据量上来后建议建索引（示例：HNSW，余弦距离）
-- CREATE INDEX ON rag_chunks USING hnsw (embedding vector_cosine_ops);
