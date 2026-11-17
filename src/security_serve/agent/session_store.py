"""session_store.py —— H4 会话外置：会话表从进程内存迁到 Redis。

原 _sessions 是服务进程堆里的 dict：进程私有（多副本不共享）、随进程生死（重启失忆）、
无时效管理。外置后三者同时解决：Redis 独立进程（服务重启会话还在）、任意实例按
session_id 共享读写、写时 EXPIRE 由引擎自动过期。

取舍（学习版）：整个会话 JSON 序列化成一个 string（session:{id}），全量读写——
语义最清晰；生产优化是按消息增量写（list 结构 RPUSH）+ 本地缓存，会话变长后的演进台阶。

key 设计：session:{id} 前缀命名空间隔离——Redis 库里将来放缓存/其他数据时互不踩踏。
"""
import json

import redis

from security_serve.agent.config import REDIS_URL, SESSION_TTL
from security_serve.agent.safety import Budget

# 连接池：TCP 连接复用（生产纪律——每请求新建连接要过三次握手 + auth，池化后纳秒级取用）
_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)

_KEY_PREFIX = "session:"


def _conn() -> redis.Redis:
    return redis.Redis(connection_pool=_pool)


def save_session(session_id: str, sess: dict) -> None:
    """整个会话（messages + Budget）原子写一个 key，顺手设 TTL——写即续命。"""
    data = {"messages": sess["messages"], "budget": sess["budget"].to_dict()}
    _conn().set(_KEY_PREFIX + session_id, json.dumps(data, ensure_ascii=False), ex=SESSION_TTL)


def load_session(session_id: str) -> dict | None:
    """按 id 取会话，不存在（或已过期）返回 None——调用方从新会话开始。Budget 还原成实例。"""
    raw = _conn().get(_KEY_PREFIX + session_id)
    if raw is None:
        return None
    data = json.loads(raw)
    data["budget"] = Budget.from_dict(data["budget"])
    return data


def count_sessions() -> int:
    """活跃会话数（health 探针用）。SCAN 渐进遍历——KEYS 全量扫会阻塞 Redis（生产纪律）。"""
    n, cursor = 0, 0
    conn = _conn()
    while True:
        # cursor：一个整数游标
        # keys：命中的key名字符串列表，形如["session:h4-check", "session:h4-cross"]
        cursor, keys = conn.scan(cursor, match=_KEY_PREFIX + "*", count=100)
        n += len(keys)
        if cursor == 0:
            return n
