"""db_seed.py —— H1 示例数据库：星辰商城业务库（users / products / orders）。

生产形态是「真实业务库 + 只读账号连接」，学习版用 SQLite 造一个小型业务库当
替身：表结构与查询语义同生产，写权限由 tools.py 的双防线拦截（白名单预检 +
URI 只读模式连接）。

种子数据固定且可断言（step6 验收的确定性来源）：
  金牌会员 3 人；库存最低商品唯一（星辰蓝牙音箱 8 件）；
  2026-10 订单 10 条；已退款订单 2 条。

跑法：build() 幂等重建（每次删旧建新，种子固定，结果可复现）。
"""
import sqlite3

from security_serve.agent.config import DB_PATH

# 建库用普通连接（写场景）——种子数据是本模块写死的受控内容，不是模型可控输入；
# 查询侧的只读防线在 tools.py（预检 + mode=ro），两处分工不同
_DDL = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    level         TEXT NOT NULL,      -- 普通 / 银牌 / 金牌
    registered_at TEXT NOT NULL       -- YYYY-MM-DD
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL,          -- 数码 / 家居 / 食品
    price    REAL NOT NULL,
    stock    INTEGER NOT NULL
);

CREATE TABLE orders (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    amount     REAL NOT NULL,        -- price × quantity
    status     TEXT NOT NULL,        -- 已完成 / 已发货 / 待发货 / 已退款
    created_at TEXT NOT NULL
);
"""

_SEED = """
INSERT INTO users (id, name, level, registered_at) VALUES
    (1,  '张伟', '金牌', '2025-03-12'),
    (2,  '李娜', '银牌', '2025-06-08'),
    (3,  '王芳', '普通', '2025-08-01'),
    (4,  '刘强', '金牌', '2025-01-25'),
    (5,  '陈静', '普通', '2025-09-15'),
    (6,  '杨洋', '银牌', '2025-11-30'),
    (7,  '赵敏', '金牌', '2025-02-14'),
    (8,  '孙丽', '普通', '2026-01-10'),
    (9,  '周杰', '银牌', '2026-02-22'),
    (10, '吴磊', '普通', '2026-04-05'),
    (11, '郑爽', '银牌', '2026-05-18'),
    (12, '冯绍', '普通', '2026-06-01');

INSERT INTO products (id, name, category, price, stock) VALUES
    (1,  '星辰无线耳机',   '数码', 299.0,  45),
    (2,  '星辰智能手表',   '数码', 1299.0, 20),
    (3,  '星辰蓝牙音箱',   '数码', 399.0,  8),
    (4,  '星辰充电宝',     '数码', 149.0,  100),
    (5,  '星辰保温杯',     '家居', 79.0,   60),
    (6,  '星辰加湿器',     '家居', 199.0,  15),
    (7,  '星辰腰靠',       '家居', 129.0,  30),
    (8,  '星辰坚果礼盒',   '食品', 99.0,   80),
    (9,  '星辰咖啡豆',     '食品', 69.0,   50),
    (10, '星辰数据线',     '数码', 39.0,   25);

INSERT INTO orders (id, user_id, product_id, quantity, amount, status, created_at) VALUES
    (1,  1,   1,  1, 299.0,  '已完成', '2026-10-03'),
    (2,  4,   2,  1, 1299.0, '已完成', '2026-10-05'),
    (3,  7,   3,  2, 798.0,  '已完成', '2026-10-08'),
    (4,  2,   5,  3, 237.0,  '已退款', '2026-10-10'),
    (5,  1,   4,  2, 298.0,  '已完成', '2026-10-12'),
    (6,  9,   8,  1, 99.0,   '已发货', '2026-10-15'),
    (7,  3,   10, 5, 195.0,  '已完成', '2026-10-18'),
    (8,  7,   1,  1, 299.0,  '已发货', '2026-10-21'),
    (9,  5,   6,  1, 199.0,  '待发货', '2026-10-25'),
    (10, 4,   9,  2, 138.0,  '已完成', '2026-10-28'),
    (11, 11,  2,  1, 1299.0, '已退款', '2026-11-01'),
    (12, 1,   3,  1, 399.0,  '已完成', '2026-11-03'),
    (13, 12,  7,  2, 258.0,  '已发货', '2026-11-05'),
    (14, 7,   4,  1, 149.0,  '已完成', '2026-11-07'),
    (15, 6,   1,  2, 598.0,  '待发货', '2026-11-08'),
    (16, 10,  5,  1, 79.0,   '已完成', '2026-11-09'),
    (17, 2,   10, 3, 117.0,  '已发货', '2026-11-09'),
    (18, 8,   8,  2, 198.0,  '待发货', '2026-11-10');
"""


def build(path=DB_PATH) -> None:
    """建库 + 灌种子数据。幂等：先删旧文件再建，多次执行结果一致。"""
    path.unlink(missing_ok=True)    # unlink是pathlib.Path的方法，先删旧db文件，missing_ok=True 文件不存在也不抛FileNotFoundError
    conn = sqlite3.connect(path)    # 连库（文件不存在会自动创建），不会校验文件是否合法，第一次执行SQL时才会读文件头验证
    try:
        conn.executescript(_DDL)   # executescript 多语句一次跑完（建表 + 清表）
        conn.executescript(_SEED)   # 灌种子数据
        conn.commit()               # 提交
    finally:
        conn.close()                # 无论如何都关连接


if __name__ == "__main__":
    build()
    print(f"示例库已重建: {DB_PATH}")
