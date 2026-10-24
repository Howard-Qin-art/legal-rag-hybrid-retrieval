"""Redis 缓存：相同查询直接复用检索结果，省掉 embedding 与检索开销。

面试可讲点：缓存 key 怎么设计（查询+topk+是否重排）、TTL 为什么设短、
未配 Redis 时用 fakeredis 兜底保证可运行。
"""
import json

import redis as redis_lib

from config import CONFIG

if CONFIG.redis_url:
    # 配了 REDIS_URL 就走真实 Redis（例如 Docker 起的 redis:7-alpine）
    _r = redis_lib.Redis.from_url(CONFIG.redis_url, decode_responses=True)
else:
    # 没配才用 fakeredis 兜底，保证离线也能跑通
    from fakeredis import FakeStrictRedis

    _r = FakeStrictRedis(decode_responses=True)

TTL = 600  # 10 分钟，知识库更新后能较快生效


def cache_get(key: str):
    val = _r.get(key)
    return json.loads(val) if val else None


def cache_set(key: str, value) -> None:
    _r.setex(key, TTL, json.dumps(value, ensure_ascii=False))
