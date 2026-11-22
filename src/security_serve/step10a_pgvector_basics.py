"""step10a_pgvector_basics.py —— H6-0 验收：pgvector 裸 SQL 六实验（叙事驱动）

在接业务之前，纯 SQL 层把 pgvector 的原语亲手过一遍——每个实验是一条
「动作 → 现象 → 结论」链，扫输出即得知识点；断言收拢为每步一个 bool，
由 main 收尾汇总。验收口径不变：6/6 通过才算 H6-0 收口。

六个实验的知识链：
  [1] 扩展机制   pgvector 是 PG 的插件，不是内核（CREATE EXTENSION + pg_extension 登记）
  [2] 建表       维度锁死在列定义——768 是模型输出维度，模型与库的唯一耦合点
  [3] 协议与精度  输入协议 '[v1,v2,...]' 自动 cast；float4 约 7 位有效十进制，存进去 ≠ 原样存
  [4] 余弦检索   <=> 返回余弦距离（= 1-相似度），升序取最小 = 最相似；与 numpy float64 手算逐位一致
  [5] 精度容差   排名稳、score 微差 < 1e-3——H6-1 验收口径的实测依据
  [6] 维度约束   767 维被拒且坏行不落库——约束做在数据层，不靠调用者自觉

跑法（项目根）：
  uv run python src/security_serve/step10a_pgvector_basics.py
前置：pgvector 容器在跑——docker compose up -d pgvector（宿主验收走 localhost:5432 端口映射）
"""
import os
import sys

import numpy as np
import psycopg

# DSN = Data Source Name：数据库连接串的行业术语。五段：协议://用户:密码@主机:端口/库名。
# 本地进程形态默认 localhost（compose 端口映射出来）；容器形态由 compose 注入 pgvector:5432——
# 与 REDIS_URL / BASE_URL 同一套「地址是环境问题，注入解决」的思想
DSN = os.environ.get("PG_DSN", "postgresql://rag:rag@localhost:5432/ragdb")


def to_pgvec(vec) -> str:
    """pgvector 的输入协议：方括号 + 逗号分隔数字，插入 vector 列时 PG 自动 cast。"""
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


def _banner(n: int, title: str):
    print(f"\n实验 {n}｜{title}")
    print("-" * 56)


def experiment_1_extension(conn) -> bool:
    """证明：pgvector 是 PG 的扩展（extension），不是内核功能。"""
    _banner(1, "扩展机制：pgvector 是 PG 的插件，不是内核")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    name, ver = conn.execute(
        "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    print(f"CREATE EXTENSION vector → pg_extension 登记：{name} v{ver}")
    print("→ 官方 postgres 镜像不含它；compose 用的 pgvector/pgvector:pg17 是")
    print("  pgvector 项目发的整合镜像，本质仍是 PG 17，psycopg 连它无差别")
    return name == "vector"


def experiment_2_create_table(conn) -> bool:
    """证明：维度是列定义的一部分——768 来自模型输出。"""
    _banner(2, "建表：维度锁死在列定义")
    conn.execute("DROP TABLE IF EXISTS vec_probe")
    conn.execute(
        "CREATE TABLE vec_probe ("
        " id serial PRIMARY KEY,"
        " tag text NOT NULL,"
        " embedding vector(768) NOT NULL)"
    )
    print("CREATE TABLE vec_probe (embedding vector(768)) 成功")
    print("→ vector(768) 的 768 = jina 模型输出维度——向量化模型与向量库的")
    print("  唯一耦合点；H6 换库不换模型，语义空间不变（无退化验收的前提）")
    return True


def experiment_3_literal_and_float4(conn, vecs) -> bool:
    """证明：字面量协议 '[v1,v2,...]' 可用；float4 存储必然截断。"""
    _banner(3, "字面量协议 + float4 截断：存进去 ≠ 原样存")
    for i, v in enumerate(vecs, 1):
        conn.execute(
            "INSERT INTO vec_probe (tag, embedding) VALUES (%s, %s)",
            (f"v{i}", to_pgvec(v)),
        )
    print("5 条候选向量插入成功（手写字面量协议，自动 cast）")
    probe = 0.1234567890123456789      # float64 才能表达的高精度数
    conn.execute(
        "INSERT INTO vec_probe (tag, embedding) VALUES (%s, %s)",
        ("probe", to_pgvec([probe] + [0.0] * 767)),
    )
    back = float(
        conn.execute(
            "SELECT embedding::text FROM vec_probe WHERE tag = 'probe'"
        ).fetchone()[0].split(",")[0][1:]
    )
    print(f"截断探针：插入 {probe!r} → 查回 {back!r}")
    conn.execute("DELETE FROM vec_probe WHERE tag = 'probe'")   # 探针行用完即清
    print("→ vector 列是 float4 单精度存储，有效位约 7 位十进制——")
    print("  JSON 版索引是 float64，第 4 位小数起必然微差")
    return back != probe and abs(back - probe) < 1e-6


def experiment_4_cosine(conn, vecs, query) -> bool:
    """证明：<=> 检索与 numpy float64 手算排名逐位一致。"""
    _banner(4, "余弦检索：<=> 与 numpy 手算同构")
    rows = conn.execute(
        "SELECT tag, embedding <=> %s AS dist FROM vec_probe ORDER BY dist LIMIT 3",
        (to_pgvec(query),),
    ).fetchall()
    for tag, dist in rows:
        print(f"  {tag}: dist={dist:.6f} → score=1-dist={1 - dist:.6f}")
    # numpy float64 手算同一道题：余弦 = 点积 / (模长 × 模长)，与 rag.retrieve 公式逐字一致
    scores = (vecs @ query) / (
        np.linalg.norm(vecs, axis=1) * np.linalg.norm(query)
    )
    np_top3 = [f"v{i + 1}" for i in np.argsort(scores)[::-1][:3]]
    print(f"pgvector  top-3 = {[r[0] for r in rows]}")
    print(f"numpy 手算 top-3 = {np_top3}")
    print("→ <=> 返回余弦距离（= 1-相似度），升序取最小即最相似；")
    print("  检索语义进 SQL——「库 vs 文件」的分界线（H6-3 元数据过滤的根基）")
    return [r[0] for r in rows] == np_top3 and 0 < 1 - rows[0][1] <= 1


def experiment_5_tolerance(conn, vecs, query) -> bool:
    """证明：排名确定、score 受 float4 影响有量级可预期的微差。"""
    _banner(5, "精度容差：排名稳、分数有微差")
    scores = (vecs @ query) / (
        np.linalg.norm(vecs, axis=1) * np.linalg.norm(query)
    )                                                   # 同实验 4 的手算公式
    rows = conn.execute(
        "SELECT tag, embedding <=> %s AS dist FROM vec_probe",
        (to_pgvec(query),),
    ).fetchall()
    diffs = [abs((1 - dist) - scores[int(tag[1:]) - 1]) for tag, dist in rows]
    max_diff = max(diffs)
    print(f"全 {len(rows)} 条 |pg_score - float64_score| 最大 = {max_diff:.2e}")
    print("→ H6-1 验收口径由此而定：断言排名全等、score 容差 < 1e-3，")
    print("  不断言逐位一致")
    return max_diff < 1e-3


def experiment_6_dimension(conn) -> bool:
    """证明：错维数据进不了列，且坏行不落库。"""
    _banner(6, "维度约束：767 维被拒，坏行不落库")
    reject_msg = None
    try:
        conn.execute(
            "INSERT INTO vec_probe (tag, embedding) VALUES (%s, %s)",
            ("bad", to_pgvec([0.1] * 767)),
        )
    except psycopg.DataError as e:                     # pgvector 报 data exception
        reject_msg = str(e).strip()
    n_bad = conn.execute(
        "SELECT COUNT(*) FROM vec_probe WHERE tag = 'bad'"
    ).fetchone()[0]
    print(f"767 维插入 → {reject_msg}")
    print(f"坏行落库数 = {n_bad}")
    print("→ 维度是类型的一部分：约束做在数据层，不靠调用者自觉（G1 同源）；")
    print("  JSON 文件版没有这道防线，什么维度都收")
    return (
        reject_msg is not None
        and ("768" in reject_msg or "dimension" in reject_msg.lower())
        and n_bad == 0
    )


if __name__ == "__main__":
    print("H6-0 验收：pgvector 裸 SQL 六实验（扩展 / 维度 / 协议 / 检索 / 精度 / 约束）")
    conn = psycopg.connect(DSN, autocommit=True)
    try:
        rng = np.random.RandomState(42)   # 伪随机数生成器对象
        vecs = rng.randn(5, 768)          # 5 条向量、每条 768 维
        query = rng.randn(768)            # 一条 768 维查询向量
        results = [
            experiment_1_extension(conn),
            experiment_2_create_table(conn),
            experiment_3_literal_and_float4(conn, vecs),
            experiment_4_cosine(conn, vecs, query),
            experiment_5_tolerance(conn, vecs, query),
            experiment_6_dimension(conn),
        ]
    finally:
        # 试验表整体丢弃；vector 扩展保留——H6-1 的正式表 doc_chunks 还要用它
        conn.execute("DROP TABLE IF EXISTS vec_probe")
        conn.close()

    passed = sum(results)
    print("\n" + "=" * 56)
    print(f"验收汇总：{passed}/6 通过")
    if passed == 6:
        print("H6-0 收口——扩展机制、维度锁定、字面量协议、<=> 检索、")
        print("float4 精度、维度约束全部成立")
    else:
        sys.exit(1)
