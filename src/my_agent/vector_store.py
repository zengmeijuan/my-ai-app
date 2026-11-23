"""vector_store.py —— pgvector 防腐层：全项目唯一 psycopg 依赖点。

防腐层（anti-corruption layer）：业务代码与外部存储系统（SQL 协议、pgvector
类型、DSN 连接串）交界处的一层翻译——把外部世界的协议翻译成内部模型
（search 返回 [{source, text, score}]，不是 cursor / 行元组），把外部依赖
限制在本文件之内。rag.py / tools.py 不出现 psycopg 字样；将来换向量库
（Milvus / Qdrant）或加缓存只动这一层。

不是 ORM、不是通用封装：只有项目需要的操作——建表 / 全量灌库 / 增量同步 / 检索。

表 doc_chunks 与 rag_index.json 的对应关系：
  JSON {"chunks": [{source, text}], "vectors": [[768 floats]]}
    → 表 doc_chunks(id, source, text, embedding vector(768), content_hash)，行级一行一 chunk。
  换库不换模型：向量仍由 jina 模型产出，pgvector 只收现成向量（存 + 检索）。
  H6-2 增 content_hash 列：文档级 sha256 指纹（切块前全文算一次），同文档所有
  chunk 行共用——增量同步拿它 diff，文档不变则零计算跳过。

运行前：docker compose up -d pgvector（表结构与数据可再生，sync-pg 一条命令重建）
"""
import os
from contextlib import contextmanager

import psycopg

# DSN 与 REDIS_URL 同一套注入思想：地址是环境问题，注入解决。
# 本地进程形态默认 localhost（compose 端口映射出来）；容器形态由 compose 注入
PG_DSN = os.environ.get("PG_DSN", "postgresql://rag:rag@localhost:5432/ragdb")

# 维度 = jina 模型输出维度（EMBED_MODEL 768 维）——向量化模型与向量库的唯一耦合点
DIM = 768

# 任意可迭代的浮点序列（numpy 数组或 list）翻译成 pgvector 约定的文本协议，'[v1,v2,...]'
def to_pgvec(vec) -> str:
    """pgvector 的输入协议：方括号 + 逗号分隔数字，插入/比较时 PG 自动 cast 成 vector。text → vector"""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"

# 把一个带 yield 的生成器函数，自动包装成上下文管理器类，省掉手写 __enter__/__exit__
@contextmanager
def _connect():
    """真·连接上下文：psycopg 的 with connect() 只是事务上下文（退出 commit 不关
    连接），常驻进程里会泄漏——这里显式 try/finally close 补上生命周期管理。"""
    # connect_timeout=5：链路故障（如 WSL 转发掉线）时 5 秒报错退出——快速失败优于无限挂起
    conn = psycopg.connect(PG_DSN, autocommit=True, connect_timeout=5)  # yield 前 = __enter__（进入时执行）
    try:
        yield conn  # yield 本身 = 把值交给 as 变量，暂停在这里，控制权交给 with 块内的代码
    finally:
        conn.close()    # yield 后 = __exit__（退出时必执行）


def ensure_table():
    """建扩展与表（幂等）：doc_chunks 一行一个 chunk，维度锁死在列定义。"""
    with _connect() as conn:
        # conn.execute(sql, params) 把 SQL 发给 PG 执行，返回一个 cursor（可 .fetchone()/.fetchall() 取结果行）
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector") # 建表前先装插件
        conn.execute(
            "CREATE TABLE IF NOT EXISTS doc_chunks ("   # 建表
            " id serial PRIMARY KEY,"                   # 自增整数主键id
            " source text NOT NULL,"
            " text text NOT NULL,"
            f" embedding vector({DIM}) NOT NULL,"       # 维度锁死在列定义
            " content_hash text)"                       # H6-2：文档级 sha256 指纹，同文档所有 chunk 共用
        )
        # 老表幂等升级：H6-1 灌过数据的表没有该列，ADD COLUMN IF NOT EXISTS 对已有列是空操作
        conn.execute("ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS content_hash text")


def sync_all(rows: list[tuple[str, str, list[float], str]]):
    """全量灌库（sync-pg --full）：清表 + 批量插入。

    rows: [(source, text, vector_list, content_hash)]——embedding 计算与指纹
    计算都留在调用方（rag.py），本层只管存取：向量化模型与向量库是两个工种。
    增量路径走 insert_chunks / delete_by_source，本函数只在 --full 时被调。
    """
    ensure_table()
    with _connect() as conn:
        conn.execute("TRUNCATE doc_chunks")     # 清空全表，直接丢弃数据页，比逐行delete快
        _insert_rows(conn, rows)
        n = conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
    print(f"PG 向量库全量重建完成：{n} 个 chunk → doc_chunks（TRUNCATE 重灌）")


def _insert_rows(conn, rows):
    """批量插入公共执行体：sync_all 与 insert_chunks 共用同一条 INSERT 语句。"""
    # 批量操作，需要显式创建游标，用来执行SQL和取结果的句柄。
    with conn.cursor() as cur:
        # executemany 把 N 组参数复用同一条 INSERT 语句，51 个 chunk 就是 51 组 (source, text, '[...]', hash)
        cur.executemany(
            "INSERT INTO doc_chunks (source, text, embedding, content_hash) VALUES (%s, %s, %s, %s)",
            [(src, text, to_pgvec(vec), h) for src, text, vec, h in rows],
        )


def insert_chunks(rows: list[tuple[str, str, list[float], str]]):
    """增量插入（H6-2）：只插 diff 出来的新增/变更 chunk，其他文档的行原封不动。"""
    with _connect() as conn:
        _insert_rows(conn, rows)


def fetch_hashes() -> dict[str, str | None]:
    """库内指纹对照面：{source: content_hash}，增量 diff 的另一半。

    同文档所有 chunk 行共用同一指纹，多行同 source 在 dict 推导里自然折叠
    （值相同，覆盖无损）。H6-1 灌的旧行该列为 NULL——NULL 与任何新指纹都不
    相等，首次增量同步会自动把旧行判成「变更」完成升级，无需手工迁移。
    """
    with _connect() as conn:
        rows = conn.execute("SELECT source, content_hash FROM doc_chunks").fetchall()
    return {src: h for src, h in rows}


def delete_by_source(source: str) -> int:
    """删除一个文档的全部 chunk 行（变更替换与文档下线共用），返回删除行数。"""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM doc_chunks WHERE source = %s", (source,))
        return cur.rowcount  # cursor.rowcount：本条语句影响（删掉）的行数


def search(query_vec, top_k: int) -> list[dict]:
    """向量检索：ORDER BY 余弦距离取 top_k，score = 1 - dist 还原为相似度。

    <=> 返回余弦距离（= 1 - 相似度），升序取最小 = 最相似——score 还原后
    与 rag.retrieve 的 json 后端同口径（float4 存储带来的微差 <1e-3，
    排名逐位一致，见 step10a 实验 4/5）。
    返回内部模型 [{"source", "text", "score"}]——外部协议（行元组）不出本层。
    """
    v = to_pgvec(query_vec) # ① 查询向量同样走字面量协议
    with _connect() as conn:
        # embedding <=> v，<=> 算余弦距离，越小越相似，1 - dist 还原成相似度口径，越大越相似
        rows = conn.execute(
            "SELECT source, text, 1 - (embedding <=> %s) AS score"
            " FROM doc_chunks ORDER BY embedding <=> %s LIMIT %s",
            (v, v, top_k),  # 三占位符按序绑参
        ).fetchall()    # 结果行全部取回，每行形如 ('退货政策.md', '## 退款时效\n原路退回…', 0.5344)
    return [{"source": r[0], "text": r[1], "score": float(r[2])} for r in rows]
