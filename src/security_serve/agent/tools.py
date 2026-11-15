"""tools.py —— 工具注册表 + 12 个工具实现。

一处注册同时产出实现与 schema（补上 A-3 跳过的注册表重构：旧版 TOOL_FUNCTIONS
与 TOOLS 两处手工维护容易漂移）。加新工具只改一处：写实现 + 挂 @tool 注册。
"""
import json
import os
import re
import sqlite3
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from my_agent import rag

from security_serve.agent.config import (
    BROWSER_USER_AGENT,
    DB_MAX_CELL_CHARS,
    DB_MAX_ROWS,
    DB_PATH,
    FETCH_MAX_CHARS,
    FETCH_TIMEOUT,
    MAX_GREP_FILE_BYTES,
    MAX_MATCHES,
    MAX_READ_CHARS,
    MAX_READ_HARD_LIMIT,
    MEMORY_PATH,
    SEARCH_MAX_RESULTS,
    SEARCH_TIMEOUT,
    SKIP_DIRS,
)
from security_serve.agent.safety import check_path, safe_eval
from security_serve.agent.sandbox import run_code

# ============== 注册表：一处定义，实现与 schema 同时产出 ==============

TOOL_REGISTRY: dict[str, dict] = {}

# 装饰器工厂
def tool(name: str, description: str, required: tuple = (), **params):
    """把裸函数注册为工具：实现存 TOOL_REGISTRY[name]["func"]，
    schema 从 name/description/params 自动组装为 OpenAI function calling 格式。
    params 是参数名 -> JSON Schema 片段，required 列出必填参数名。"""
    def deco(fn):
        TOOL_REGISTRY[name] = {
            "func": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": list(required),
                    },
                },
            },
        }
        return fn
    return deco


# ============== 工具实现（与旧版 agent.py 逐字一致）==============
# @tool 装饰器语法糖，等价于：
# deco = tool("get_weather", "获取...", required=("city",), city={...}) # deco是一个闭包，一个函数对象，以及外层函数的局部变量（四个参数，因为被内层函数引用了），此时注册表还没被写入
# get_weather = deco(get_weather)     # deco 拿到第 1 步的函数对象，执行函数体。此时注册表被写入，并且返回原函数
# @语法糖把 deco 的返回值赋回给名字 get_weather，get_weather 仍指向原函数——照常可以调用、docstring 完好
# 装饰器在这里不改函数，只借 def 语句的执行时机干了一件私活：写注册表

@tool(
    "get_weather",
    "获取指定城市的天气信息。当用户问城市天气时调用。",
    required=("city",),
    city={"type": "string", "description": "城市名，如 北京/上海/广州/深圳"},
)
def get_weather(city: str) -> str:
    """返回指定城市的天气（mock 数据）。"""
    weather_db = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 60},
        "上海": {"temp": 28, "condition": "多云", "humidity": 70},
        "广州": {"temp": 30, "condition": "雷阵雨", "humidity": 85},
        "深圳": {"temp": 29, "condition": "晴", "humidity": 75},
    }
    data = weather_db.get(city, {"temp": 22, "condition": "未知", "humidity": 50})
    return json.dumps(data, ensure_ascii=False)


@tool(
    "calculate",
    "计算数学表达式。当用户问数学计算时调用。",
    required=("expression",),
    expression={"type": "string", "description": "数学表达式，如 (25+5)*2"},
)
def calculate(expression: str) -> str:
    """返回数学表达式的计算结果。

    边界：只接受纯算术表达式（ast 白名单求值器，见 safety.safe_eval），
    函数调用/属性访问/变量名一律拒绝。
    """
    ok, info = safe_eval(expression)
    if not ok:
        return f"错误：{info}"
    return info


@tool(
    "read_file",
    "读取指定路径文本文件的内容。当用户想查看文件内容、读取代码、分析文档时调用。",
    required=("path",),
    path={"type": "string", "description": "文件路径，相对或绝对，如 README.md 或 E:/test/a.txt"},
)
def read_file(path: str) -> str:
    """返回文本文件内容（utf-8 编码）。

    边界：路径仅限允许目录内（safety.check_path 白名单）；超过 10MB 字节硬上限
    直接报错；超过 3万字符软上限只读前 N 字符并提示。
    """
    ok, info = check_path(path)
    if not ok:
        return f"错误：{info}"
    try:
        size = os.path.getsize(path)
        if size > MAX_READ_HARD_LIMIT:
            return f"错误：文件过大（{size} 字节 > 硬上限 {MAX_READ_HARD_LIMIT} 字节），不读取。请用 grep 定位或换更小范围的文件"
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(MAX_READ_CHARS + 1)   # 多读 1 个字符用来判断是否截断
        if len(content) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + f"\n\n... [已截断到 {MAX_READ_CHARS} 字符]"
        return content
    except FileNotFoundError:
        return f"错误：文件不存在: {path}"
    except UnicodeDecodeError:
        return f"错误：无法以 utf-8 解码（可能是二进制或非 utf-8 编码）: {path}"
    except Exception as e:
        return f"读取错误: {e}"


@tool(
    "list_dir",
    "列出指定目录下的文件和子目录（只列一层，不递归）。当用户想看目录结构、确认路径是否存在时调用（找特定文件用 grep）。",
    required=("path",),
    path={"type": "string", "description": "目录路径，相对或绝对，如 . 或 E:/workspace/studying/my-ai-app"},
)
def list_dir(path: str) -> str:
    """返回指定目录下的一层文件和子目录列表（不递归）。"""
    ok, info = check_path(path)
    if not ok:
        return f"错误：{info}"
    try:
        if not os.path.isdir(path):
            return f"错误：目录不存在（或传入的是文件路径）: {path}"
        names = sorted(os.listdir(path))
        if not names:
            return f"目录为空: {path}"
        lines = []
        for name in names:
            tag = "[DIR]" if os.path.isdir(os.path.join(path, name)) else "[FILE]"
            lines.append(f"{tag} {name}")
        return f"{path}（{len(lines)} 项）:\n" + "\n".join(lines)
    except Exception as e:
        return f"列目录错误: {e}"


@tool(
    "grep",
    "在目录下递归搜索包含指定文本/正则的行，返回 文件:行号:内容。当用户想知道「哪个文件里有某段代码或文字」时调用。",
    required=("pattern",),
    pattern={"type": "string", "description": "要搜索的文本或正则表达式，如 TOOL_FUNCTIONS"},
    path={"type": "string", "description": "要搜索的目录，默认为当前工作目录"},
)
def grep(pattern: str, path: str = ".") -> str:
    """返回目录下匹配正则的行（格式：文件:行号:内容）。"""
    try:
        re.compile(pattern)
    except re.error as e:
        return f"错误：无效的正则表达式: {e}"

    ok, info = check_path(path)
    if not ok:
        return f"错误：{info}"

    matches = []
    skipped_large = []  # 被跳过的大文件列表，"未找到"时告知用户可能漏匹配
    # os.walk(path) 遍历目录树，返回一个生成器，每次迭代会得到一个三元组
    # root：当前正在遍历的目录路径，字符串
    # dirs：当前目录下所有子目录名的列表
    # files：当前目录下所有非目录文件的名称列表
    for root, dirs, files in os.walk(path):
        # 原地过滤 dirs，os.walk 就不会进入被排除的目录
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                # 单文件过大直接跳过，防止遍历巨型文件挂死
                if os.path.getsize(fpath) > MAX_GREP_FILE_BYTES:
                    skipped_large.append(fpath)
                    continue
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    # 前 1024 字节含 \x00 视为二进制文件，跳过（ripgrep 同款判据）
                    if "\x00" in f.read(1024):
                        continue
                    # 把文件读写位置重置到文件开头
                    f.seek(0)
                    # 逐行遍历文件对象f，lineno是行号（从1开始），line是这一行的内容
                    for lineno, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            # 单行截断 200 字符，防 minified 文件一行几十 KB
                            matches.append(f"{fpath}:{lineno}: {line.strip()[:200]}")
                            if len(matches) >= MAX_MATCHES:
                                return (
                                    f"已找到 {MAX_MATCHES} 处匹配（达到上限，可能还有更多）:\n"
                                    + "\n".join(matches)
                                )
            except OSError:
                continue  # 无权限或打不开的文件直接跳过

    if not matches:
        msg = f"未找到匹配 '{pattern}'（目录: {path}）"
        if skipped_large:
            msg += f"\n提示：{len(skipped_large)} 个大文件（>1MB）被跳过，可能包含匹配内容。如确需搜索，可先用 list_dir 定位，再 read_file 直接读这些文件。"
        return msg
    return f"共 {len(matches)} 处匹配:\n" + "\n".join(matches)


# 知识库索引缓存：与 rag.py 的 _model 同款懒加载单例——工具第一次被调用才读盘
_knowledge_index = None


@tool(
    "search_knowledge",
    "在「星辰商城」客服知识库中检索资料，返回最相关的知识片段及来源。当用户咨询星辰商城的退货、退款、配送、运费、会员、积分、支付、发票、账号、密码等问题时调用。注意：查本项目的代码文件应使用 read_file/grep，不要用本工具。",
    required=("query",),
    query={"type": "string", "description": "检索问题，用完整的自然语言句子，如「退款多久到账」"},
)
def search_knowledge(query: str) -> str:
    """返回知识库中与问题最相关的 top-3 资料（带来源文件名+相似度）。"""
    global _knowledge_index
    if _knowledge_index is None:
        if not os.path.exists(rag.INDEX_PATH):
            return "错误：知识库索引不存在，请让用户先运行: uv run python rag.py build"
        with open(rag.INDEX_PATH, encoding="utf-8") as f:
            _knowledge_index = json.load(f)
        if _knowledge_index["model"] != rag.EMBED_MODEL:
            _knowledge_index = None  # 不缓存错误索引，避免后续调用反复用它出错
            return "错误：索引由其他 embedding 模型构建，请让用户重新运行 rag.py build"

    # 核心复用：rag.retrieve 用同一个 embedding 模型把问题向量化，算余弦取 top-3
    results = rag.retrieve(query, _knowledge_index)
    blocks = [
        f"资料[{i}]（来源：{r['source']}，相似度 {r['score']:.4f}）\n{r['text']}"
        for i, r in enumerate(results, 1)
    ]
    return "\n\n".join(blocks)


@tool(
    "remember",
    "把重要信息写入长期记忆，跨会话永久保留。当用户说「记住……」，或出现明显值得长期保留的信息（用户偏好、重要事实、关键结论）时调用。只记简短的结论性陈述，不要记聊天过程。",
    required=("content",),
    content={"type": "string", "description": "要记住的信息，一条简短的结论性陈述，如「主人对花生过敏」"},
)
def remember(content: str) -> str:
    """追加一条结论性信息到长期记忆文件（带时间戳）。"""
    line = f"- [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {content.strip()}"
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return f"已写入长期记忆（下次会话或 clear 后注入 system prompt）：{content.strip()}"


# ============== H1：数据库只读工具 ==============
# 生产形态 = 真实业务库 + 只读账号连接；学习版映射：SQLite URI 只读模式（mode=ro）
# 即"只读账号"——写不进去由数据库内核保证。只读双防线：
#   第一道 _check_readonly 白名单预检（对应 G2：白名单即能力集，只认好词不枚举坏词），
#   给模型友好错误串供其自行调整；
#   第二道 mode=ro 连接（对应 G1：闸门在连接层），预检被绕过时硬兜底。
# 拒绝语义同 G3：错误串当工具结果回传，循环不断，模型看到原因向用户解释。

_READONLY_FIRST_WORDS = ("SELECT", "WITH")
# WITH 放行的边界：CTE 查询（WITH ... SELECT）是合法读库形态；但 SQLite 也允许
# WITH 后跟写语句（WITH t AS (...) DELETE FROM ...）——这类输入预检层放行，由
# mode=ro 连接在内核层硬拦（纵深防御分工，step6 [5] 有实证）。预检管友好拒绝，
# 连接层是安全底线，两者覆盖面不等同。


def _open_readonly() -> sqlite3.Connection:
    """只读连接：mode=ro 下任何写语句（含 PRAGMA 写配置、ATTACH 外挂库）都被
    SQLite 内核拒绝（attempt to write a readonly database）——不靠 prompt 自觉。"""
    # sqlite3.connect(path) 默认读写可创建
    # connect("file:...?mode=ro", uri=True) ：SQLite 的 URI 语法，mode=ro = read only，返回只读链接，无法写入数据
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _check_readonly(sql: str) -> str | None:
    """白名单预检：放行返回 None，拒绝返回原因串。

    先剥注释再取首词——"/*x*/ DELETE FROM ..." 这类把写语句藏在注释后的输入
    会在剥注释后现出原形。中间分号 = 多语句（一次一条，execute 也只执行单条）。
    误报从宽（如字符串常量里含分号会被拦）：预检层宁误拒不误放，硬兜底在 ro 层。
    """
    # 把行注释和块注释统一替换成空格再去首尾空白
    no_comment = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip()
    # 取 SQL 剥注释后的"首词"，统一大写，空串兜底
    first = no_comment.split(None, 1)[0].upper() if no_comment else ""
    # 不在白名单返回拒绝串、在则查中间分号（多语句）、都干净返回None
    if first not in _READONLY_FIRST_WORDS:
        return (
            f"拒绝：本工具只允许只读查询（SELECT 或 WITH 开头），收到的是 {first or '空'} 语句。"
            f"数据库连接为只读，即使执行也无法写入；如需修改数据请告知用户由人工在数据库端操作"
        )
    if ";" in no_comment.rstrip(";"):
        return "拒绝：一次只能执行一条语句（检测到中间分号），请拆成多次查询"
    return None


@tool(
    "db_schema",
    "查看星辰商城业务数据库的结构。两步用法：先不带参数调用看有哪些表，再传表名看该表的字段定义和行数。写 SQL 前先看结构，避免猜字段名。",
    required=(),
    table={"type": "string", "description": "表名，如 orders。留空则返回表清单"},
)
def db_schema(table: str = "") -> str:
    """不带参数返回表清单；带表名返回 CREATE 语句（字段定义）+ 约行数。

    边界：表名参数同样来自模型（不可信输入），用参数化查询 sqlite_master 验证
    存在性后才拼接计数 SQL——"先验证再使用"，与 G1 路径校验同源。
    """
    # 每次调用新开连接、用完就关，不开全局常驻连接
    conn = _open_readonly()
    try:
        # 不带参数，列出所有表
        if not table:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            names = "、".join(r[0] for r in rows)
            return f"数据库共 {len(rows)} 张表：{names}。用 db_schema(表名) 查看具体字段"
        # 带表名，先验证再使用
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not row:
            return f"错误：表 {table!r} 不存在。先用 db_schema()（不带参数）查看有哪些表"
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        return f"{row[0]}\n（约 {count} 行）"
    except sqlite3.Error as e:
        return f"错误：查询结构失败: {e}"
    finally:
        conn.close()


@tool(
    "db_query",
    "对星辰商城业务数据库执行只读 SQL 查询（仅 SELECT / WITH 开头），返回表格文本。查订单/会员/商品的统计与明细时调用。先 db_schema 看表结构再写 SQL。数据库只读：INSERT/UPDATE/DELETE/DROP 等写操作会被拒绝。",
    required=("sql",),
    sql={"type": "string", "description": "一条完整的 SELECT 语句，如 SELECT level, COUNT(*) FROM users GROUP BY level"},
)
def db_query(sql: str) -> str:
    """执行只读 SQL，返回 markdown 风格表格文本（行数/单元格宽度有上限）。

    边界：白名单预检 + mode=ro 连接双防线（见模块注释）；fetchmany 取
    DB_MAX_ROWS+1 行，多出的 1 行只用来判断"还有更多"，提示模型加 LIMIT。
    """
    reason = _check_readonly(sql)
    if reason:
        return reason
    try:
        conn = _open_readonly()
        try:
            # execute() 把 SQL 发给 SQLite 执行，返回 cursor（游标），可以理解为结果集的读取手柄
            cur = conn.execute(sql)
            # fetch 从已执行的结果集里取行，fetchmany(n)：最多取 n 行
            rows = cur.fetchmany(DB_MAX_ROWS + 1)
            cols = [d[0] for d in cur.description] if cur.description else []
        finally:
            conn.close()
    except sqlite3.Error as e:
        return f"SQL 执行出错: {e}（可先 db_schema 查看表结构，检查字段名与表名）"

    if not rows:
        return "查询结果为空（0 行）。可换个条件或先用 db_schema 确认字段"
    truncated = len(rows) > DB_MAX_ROWS
    rows = rows[:DB_MAX_ROWS]

    def cell(v) -> str:
        s = "NULL" if v is None else str(v)
        return s if len(s) <= DB_MAX_CELL_CHARS else s[:DB_MAX_CELL_CHARS] + "…"

    # 表头
    table = " | ".join(cols) + "\n"
    # 每行数据
    table += "\n".join(" | ".join(cell(v) for v in row) for row in rows)
    footer = f"\n（显示 {len(rows)} 行" + ("，结果未完，请在 SQL 里加 LIMIT 缩小范围）" if truncated else "）")
    return table + footer


# ============== H3：代码执行沙箱工具 ==============
# 与只读工具的"填空题"不同，执行类工具的行为由模型现场定义（整段代码），
# 白名单管不住任意代码，防线换形态：docker 隔离环境兜底（实现见 sandbox.py）


@tool(
    "run_python",
    "在隔离沙箱（一次性 Docker 容器，无网络/无本地文件/限时 10 秒）里执行一段完整的 Python 代码，返回 print 输出。用于统计/占比/排序/分组等数据计算（先 db_query 查出数据，把结果作为字面量写进代码再算）、批量文本处理、需要循环或标准库的逻辑。代码里必须用 print() 输出结果。纯算术直接用 calculate，不要用本工具。",
    required=("code",),
    code={"type": "string", "description": "完整的 Python 脚本（可多行多语句），数据以字面量写死在代码里，结果必须 print() 出来"},
)
def run_python(code: str) -> str:
    """把模型写的代码送进沙箱执行（实现见 sandbox.py）。

    边界：五道 docker 围栏 + 双层超时；代码超长直接拒绝；输出截断；
    所有失败（超时/OOM/异常/docker 故障）都以错误串返回不抛异常——
    模型拿得到错误信息，agent 循环不至于中断。
    """
    return run_code(code)


# ============== 联网工具：web_search + web_fetch ==============
# 学习版用 DuckDuckGo HTML 接口（无需 key、零成本），是 agent 加联网能力的最短路径
# 生产环境会换成 MCP browser server（Stage 5）或付费搜索 API（Stage 8）


@tool(
    "web_search",
    "用 Bing 中国版联网搜索关键词，返回最多 5 条结果的标题、摘要和 URL。调用时机：search_knowledge 知识库中找不到相关信息时（兜底）、需要时效性/外部/公开信息时（如新闻、最新技术、权威来源）。不要用此工具查本地项目代码（用 read_file/grep）或星辰商城业务问题（用 search_knowledge）。",
    required=("query",),
    query={"type": "string", "description": "搜索关键词或完整问题，如「Python 3.13 新特性」"},
)
def web_search(query: str) -> str:
    """返回最多 5 条 Bing 搜索结果的标题+摘要+URL。"""
    try:
        resp = httpx.get(
            "https://cn.bing.com/search",
            params={"q": query},
            headers={
                "User-Agent": BROWSER_USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            timeout=SEARCH_TIMEOUT,
            follow_redirects=True, # 告诉 HTTP 客户端: 遇到 3xx 时,要不要自动跟着跳
        )
        resp.raise_for_status() # 检查 HTTP 响应状态码,如果不是 2xx 就抛异常
    except httpx.TimeoutException:
        return f"错误：搜索超时（{SEARCH_TIMEOUT} 秒），请检查网络或换个关键词重试"
    except httpx.HTTPStatusError as e:
        return f"错误：搜索请求失败 HTTP {e.response.status_code}"
    except httpx.RequestError as e: # 网络层异常，就算没有resp.raise_for_status()，也会抛出
        return f"错误：连不上搜索引擎: {e}"

    # Bing 中国版的结果容器：li.b_algo（普通结果）/ li.b_ad（广告结果，过滤）
    # 标题在 h2 > a，摘要在 .b_caption > p（有时直接 p，要兼容两种）

    # resp.text：完整的 HTML 字符串
    # html.parser：告诉 BeautifulSoup 用哪个解析器来把字符串变成树
    # BeautifulSoup:将html变成一棵对象树,每个标签都是一个 Python 对象
    soup = BeautifulSoup(resp.text, "html.parser")
    items = soup.select("li.b_algo") # 查找所有 class 为 b_algo 的 li 元素
    if not items:
        # 极少见：Bing 偶发返回反爬验证码页，特征是没有 b_algo
        return f"未找到与 '{query}' 相关的搜索结果（搜索引擎可能返回了验证码页，请稍后重试）"

    blocks = []
    for i, item in enumerate(items[:SEARCH_MAX_RESULTS], 1):
        h2 = item.select_one("h2") # 查找第一个
        a = h2.find("a") if h2 else None
        if not a:
            continue
        title = a.get_text(strip=True) # 返回标签内部的所有纯文本（剥掉 HTML 标签）
        href = a.get("href", "").strip()
        # 摘要：优先 .b_caption > p，兼容直接的 p 标签
        cap = item.select_one(".b_caption")
        snippet_tag = (cap.select_one("p") if cap else None) or item.select_one("p")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        blocks.append(f"[{i}] {title}\n    URL: {href}\n    摘要: {snippet}")

    if not blocks:
        return f"未找到与 '{query}' 相关的搜索结果"

    return f"关于 '{query}' 的搜索结果（共 {len(blocks)} 条，来源 cn.bing.com）:\n\n" + "\n\n".join(blocks)


@tool(
    "web_fetch",
    "抓取指定 URL 的网页正文（去除脚本/样式/导航等噪音），返回清理后的文本。调用时机：拿到 web_search 返回的 URL 后想读详情时，或用户直接给了一个 URL 想看内容时。仅用于公开网页，不要用于登录态、敏感账号或绕过平台限制。",
    required=("url",),
    url={"type": "string", "description": "要抓取的完整 URL，如 https://docs.python.org/3.13/whatsnew/"},
)
def web_fetch(url: str) -> str:
    """抓取 URL 网页正文，返回清理后的纯文本（去 script/style/nav 等噪音）。"""
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        return f"错误：抓取超时（{FETCH_TIMEOUT} 秒）: {url}"
    except httpx.HTTPStatusError as e:
        return f"错误：抓取失败 HTTP {e.response.status_code}: {url}"
    except httpx.RequestError as e:
        return f"错误：连不上目标网站: {e} ({url})"

    # 用 BeautifulSoup 解析 HTML，移除 script/style 等噪音标签，再提取文本
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav"]):
        tag.decompose() # 删除对应的标签
    text = soup.get_text(separator="\n", strip=True) # 标签之间插入换行符
    # 把连续空行压缩成一个（脏 HTML 里常见）
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return f"错误：页面解析后为空（可能是 SPA 渲染，需要浏览器工具）: {url}"

    truncated = len(text) > FETCH_MAX_CHARS
    if truncated:
        text = text[:FETCH_MAX_CHARS] + f"\n\n... [内容过长，已截断到 {FETCH_MAX_CHARS} 字符]"

    return f"URL: {url}\n\n{text}"


# ============== 导出：loop 层用的两个视图 ==============

TOOL_FUNCTIONS = {name: entry["func"] for name, entry in TOOL_REGISTRY.items()}     # 给执行器，按名取函数
TOOLS = [entry["schema"] for entry in TOOL_REGISTRY.values()]                       # 给LLM API，挂报文
