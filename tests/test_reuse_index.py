"""索引复用逻辑的测试。

不连真实数据库，用 mock 顶掉 psycopg.connect，只验证「库里已有 N 行时跳过重建」这个判断。
这样在没有 Docker 的机器上也能跑：

    python tests/test_reuse_index.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from config import CONFIG  # noqa: E402
from vectorstore import MemoryVS, PgVectorVS  # noqa: E402


def _fake_pgvector(count):
    """造一个连着假数据库的 PgVectorVS，count() 会返回指定行数。"""
    with mock.patch("psycopg.connect") as connect:
        conn = connect.return_value
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = (count,)
        vs = PgVectorVS("postgresql://user:pwd@localhost:5432/rag", 512)
    return vs


def test_pgvector_skips_when_row_count_matches():
    vs = _fake_pgvector(3000)
    assert vs.count() == 3000
    assert vs.needs_build(3000) is False, "条数一致时应该跳过重建"
    assert vs.needs_build(2999) is True, "条数不一致时必须重建"


def test_reuse_can_be_disabled():
    old = CONFIG.reuse_index
    try:
        CONFIG.reuse_index = False
        vs = _fake_pgvector(3000)
        assert vs.needs_build(3000) is True, "关掉复用后应当总是重建"
    finally:
        CONFIG.reuse_index = old


def test_memory_store_always_builds():
    vs = MemoryVS(512)
    assert vs.needs_build(3000) is True, "内存版每次都得重建，没有可复用的东西"


def test_build_index_skips_encode_when_reusing():
    """库里已有数据时，连嵌入编码都不该做——这才是真正省下的开销。"""
    from retrievers import build_index

    calls = {"encode": 0}

    class FakeEmb:
        dim = 512

        def encode(self, texts):
            calls["encode"] += 1
            return np.zeros((len(texts), 512), dtype="float32")

    vs = _fake_pgvector(20)
    corpus = [{"id": i, "text": f"段落{i}"} for i in range(20)]

    with mock.patch("retrievers.get_embedding", return_value=FakeEmb()), \
            mock.patch("retrievers.get_vectorstore", return_value=vs):
        build_index(corpus)

    assert calls["encode"] == 0, f"复用索引时不该再编码，实际调了 {calls['encode']} 次"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} 项全部通过")


if __name__ == "__main__":
    main()
