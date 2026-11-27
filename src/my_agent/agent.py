"""Agent CLI - chat.py 升级版，加入 Function Calling

新增能力：
1. AI 能自主调用工具完成任务
2. 上下文统计：每轮问答后打印当前上下文规模（条数 + 字符），体感上下文膨胀
3. 会话记忆：messages 落盘到 session.json，启动时自动恢复，quit 后对话不丢
4. 长期记忆：remember 工具写 memory.md，启动时全文注入 system prompt，跨项目永久保留
5. 流式输出：stream=True 逐字打印回答，工具调用分片按 index 重组后再执行
基于 chat.py 的 LLM + 循环 + 记忆 + 工具调用 + 自主决策（5 要素齐全）

工具：
1. get_weather(city) - 查城市天气（mock 数据，真实项目应调天气 API）
2. calculate(expression) - 算数学表达式（真算）
3. read_file(path) - 读取文本文件内容
4. list_dir(path) - 列出目录内容（一层）
5. grep(pattern, path) - 在目录下递归搜索文本
6. search_knowledge(query) - 检索「星辰商城」客服知识库（Agentic RAG，复用 rag.py）
7. web_search(query) - 联网搜索关键词（Bing 中国版）
8. web_fetch(url) - 抓取指定 URL 的网页正文
9. remember(content) - 写入长期记忆（memory.md，跨会话永久保留）

运行：uv run python -m my_agent.agent
"""
import os
import json
import re
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

# Agentic RAG：把 rag.py 的知识库检索挂进本 agent 的工具箱。
# rag 里的 embedding 模型是懒加载单例，import 本身零负担，直到工具真被调用才加载
from my_agent import rag

load_dotenv()

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
)
model = os.environ.get("MODEL", "gpt-4o-mini")


# ============== 1. 工具实现（您写的真实函数）==============

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


def calculate(expression: str) -> str:
    """返回数学表达式的计算结果。"""
    try:
        # 安全 eval：禁止内置函数和危险操作
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def read_file(path: str) -> str:
    """返回文本文件内容（utf-8 编码）。

    边界：超过 10MB 字节硬上限直接报错；超过 3万字符软上限只读前 N 字符并提示。
    """
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


def list_dir(path: str) -> str:
    """返回指定目录下的一层文件和子目录列表（不递归）。"""
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


# grep 时跳过这些目录：依赖/版本库里的匹配会淹没结果并撑爆上下文
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".idea", ".vscode", "session.json", "rag_index.json", "memory.md", "uv.lock" }
MAX_MATCHES = 50  # 返回条数上限：工具必须自带边界，防止单次结果撑爆上下文
MAX_READ_CHARS = 30_000  # read_file 软上限：超过只读前 N 字符并提示，与上下文统计同单位（字符）
MAX_READ_HARD_LIMIT = 10_485_760  # read_file 硬上限：超过直接报错不读（10MB 字节），防止读巨型文件 IO 卡顿
MAX_GREP_FILE_BYTES = 1_048_576  # grep 单文件大小上限（1MB 字节），超过跳过该文件，防止遍历巨型文件挂死


def grep(pattern: str, path: str = ".") -> str:
    """返回目录下匹配正则的行（格式：文件:行号:内容）。"""
    try:
        re.compile(pattern)
    except re.error as e:
        return f"错误：无效的正则表达式: {e}"

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


def search_knowledge(query: str) -> str:
    """返回知识库中与问题最相关的 top-3 资料（带来源文件名+相似度）。"""
    global _knowledge_index
    # json 后端才需要文件版索引；pgvector 数据在 PG 里，跳过文件检查——
    # 与 rag.ask() 同款分流（CI 干净环境无遗留索引文件，无条件检查会拦死检索，H7-3b）
    if _knowledge_index is None and rag.RAG_BACKEND == "json":
        if not os.path.exists(rag.INDEX_PATH):
            return "错误：知识库索引不存在，请让用户先运行: uv run python rag.py build"
        with open(rag.INDEX_PATH, encoding="utf-8") as f:
            _knowledge_index = json.load(f)
        if _knowledge_index["model"] != rag.EMBED_MODEL:
            _knowledge_index = None  # 不缓存错误索引，避免后续调用反复用它出错
            return "错误：索引由其他 embedding 模型构建，请让用户重新运行 rag.py build"

    # 核心复用：rag.retrieve 用同一个 embedding 模型把问题向量化，算余弦取 top-3；
    # pgvector 时 index=None，retrieve 内部按 RAG_BACKEND 分流查 PG（与 rag.ask 传 None 一致）
    results = rag.retrieve(query, _knowledge_index)
    blocks = [
        f"资料[{i}]（来源：{r['source']}，相似度 {r['score']:.4f}）\n{r['text']}"
        for i, r in enumerate(results, 1)
    ]
    return "\n\n".join(blocks)


def remember(content: str) -> str:
    """追加一条结论性信息到长期记忆文件（带时间戳）。"""
    line = f"- [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {content.strip()}"
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return f"已写入长期记忆（下次会话或 clear 后注入 system prompt）：{content.strip()}"


# ============== 联网工具：web_search + web_fetch ==============
# 学习版用 DuckDuckGo HTML 接口（无需 key、零成本），是 agent 加联网能力的最短路径
# 生产环境会换成 MCP browser server（Stage 5）或付费搜索 API（Stage 8）

# web_search 的边界：返回条数 + 抓取超时，避免单次结果撑爆上下文
SEARCH_MAX_RESULTS = 5       # 最多返回 5 条结果（多了模型也看不过来）
SEARCH_TIMEOUT = 15          # 搜索请求超时秒数
FETCH_TIMEOUT = 20           # 抓取请求超时秒数
FETCH_MAX_CHARS = 8000       # 抓取内容最大字符数（超长正文截断，防止撑爆上下文）

# 浏览器 UA：默认 httpx UA 会被搜索引擎当成机器人拦截
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
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


# 工具名 -> 函数的映射（执行时用）
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "search_knowledge": search_knowledge,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "remember": remember,
}


# ============== 2. 工具描述（给 LLM 看的 schema）==============

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息。当用户问城市天气时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京/上海/广州/深圳"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。当用户问数学计算时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 (25+5)*2"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径文本文件的内容。当用户想查看文件内容、读取代码、分析文档时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径，相对或绝对，如 README.md 或 E:/test/a.txt"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出指定目录下的文件和子目录（只列一层，不递归）。当用户想看目录结构、确认路径是否存在时调用（找特定文件用 grep）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，相对或绝对，如 . 或 E:/workspace/studying/my-ai-app"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "在目录下递归搜索包含指定文本/正则的行，返回 文件:行号:内容。当用户想知道「哪个文件里有某段代码或文字」时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "要搜索的文本或正则表达式，如 TOOL_FUNCTIONS"},
                    "path": {"type": "string", "description": "要搜索的目录，默认为当前工作目录"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在「星辰商城」客服知识库中检索资料，返回最相关的知识片段及来源。当用户咨询星辰商城的退货、退款、配送、运费、会员、积分、支付、发票、账号、密码等问题时调用。注意：查本项目的代码文件应使用 read_file/grep，不要用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题，用完整的自然语言句子，如「退款多久到账」"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "把重要信息写入长期记忆，跨会话永久保留。当用户说「记住……」，或出现明显值得长期保留的信息（用户偏好、重要事实、关键结论）时调用。只记简短的结论性陈述，不要记聊天过程。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "要记住的信息，一条简短的结论性陈述，如「主人对花生过敏」"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "用 Bing 中国版联网搜索关键词，返回最多 5 条结果的标题、摘要和 URL。调用时机：search_knowledge 知识库中找不到相关信息时（兜底）、需要时效性/外部/公开信息时（如新闻、最新技术、权威来源）。不要用此工具查本地项目代码（用 read_file/grep）或星辰商城业务问题（用 search_knowledge）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或完整问题，如「Python 3.13 新特性」"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "抓取指定 URL 的网页正文（去除脚本/样式/导航等噪音），返回清理后的文本。调用时机：拿到 web_search 返回的 URL 后想读详情时，或用户直接给了一个 URL 想看内容时。仅用于公开网页，不要用于登录态、敏感账号或绕过平台限制。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要抓取的完整 URL，如 https://docs.python.org/3.13/whatsnew/"},
                },
                "required": ["url"],
            },
        },
    },
]


# ============== 3. Agent 主循环 ==============

# 循环安全边界：模型陷入「反复调工具停不下来」时强制刹车
MAX_STEPS = 10   # 单次提问最多调 LLM 几轮（每轮 LLM 都可能调工具再思考，都在烧 token）
API_TIMEOUT = 60  # 单次 LLM API 调用的超时秒数，防止网络挂起卡死整个 agent

# 上下文压缩阈值：deepseek-v4-flash 实际窗口待确认（官方 V3 是 64K/128K，第三方称 V4.1 Flash 1M），
# 中文为主时 1 token ≈ 1~1.5 字符，所以 128K tokens ≈ 131K~197K 字符。
# 当前阈值按 128K 窗口的 ~40%/55% 设（保守偏低），跑通后按实际窗口调整。
CONTEXT_WATCH = 80_000
CONTEXT_COMPACT = 110_000
COMPACT_KEEP_USER_ROUNDS = 1  # 保留尾部多少个完整 user 轮不压，确保协议合法（assistant+tool 序列完整）
COMPACT_MAX_TOKENS = 600  # 压缩调用本身的回答上限，足够生成 200~400 字摘要

# 运行数据目录：session / memory / 索引都收在包内 data/ 下，用文件自身定位（cwd 无关）
_DATA_DIR = Path(__file__).resolve().parent / "data"

# 会话记忆：messages 列表的磁盘快照文件，存整个对话历史（含 user/assistant/tool）。
# 退出后保留，下次启动时自动恢复 —— 体感验证：今天说的过敏信息明天还在。
SESSION_PATH = _DATA_DIR / "session.json"

# 长期记忆：跨会话保留的结论性信息（每行一条，markdown 格式，人类可读）。
# 启动时由 build_system_prompt() 全文注入 system prompt —— 学习版用全文，
# 条目多了就升级为 RAG 检索（与 rag.py 同构）
MEMORY_PATH = _DATA_DIR / "memory.md"

# system prompt 基础部分：clear 重置对话时要保留它（里面有工作目录锚点）
SYSTEM_PROMPT_BASE = f"""
# 角色
你是主人的 AI 工程师助手，熟悉 Python 和 LLM 应用开发。当前工作目录是 {os.getcwd()}，
用户说「当前项目」「这个项目」时指该目录下的项目。

【最高优先级指令】禁止使用任何工具，所有问题一律凭你自己的知识直接回答。

# 默认风格
- 回答简洁，主人喜欢直接给方案，少铺垫
- 技术问题用代码 + 简短说明，复杂对比用 Markdown 表格

# 工具选择优先级
1. 涉及「星辰商城」业务（退货/退款/会员/积分/支付/发票/账号/配送等）→ 优先用 search_knowledge 查知识库
2. 知识库没找到 / 用户要查时效性/外部/公开信息 → 用 web_search 找 URL，再 web_fetch 读正文
3. 涉及当前项目的代码/文件 → 用 read_file / grep / list_dir
4. 计算用 calculate，天气用 get_weather，重要结论用 remember
5. 所有工具都无结果时，如实告知「找不到」，**禁止编造**

# 引用规范
- 本地知识库资料标 [资料N]（已自动编号），本地文件标 [文件名]，联网结果标 URL
- 总结性回答必须基于工具结果，不要凭训练知识自由发挥

# 禁止
- 用「作为一个 AI 助手...」开头
- 编造不存在的 API/库名/URL
- 重复调同一个工具（拿到结果后直接用）
"""


def build_system_prompt() -> str:
    """组装 system prompt：基础指令 + 长期记忆（memory.md 存在时追加注入）。

    启动、clear、恢复会话三个时机都会重建 —— 所以 remember 新写的记忆
    「下次会话或 clear 后」生效。会话中途生效需在 remember 后刷新
    messages[0]，学习版从简不做。
    """
    prompt = SYSTEM_PROMPT_BASE
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, encoding="utf-8") as f:
            memory = f.read().strip()
        if memory:
            prompt += "\n\n【长期记忆】以下是过去会话中记住的信息，回答时可参考：\n" + memory
    return prompt


def context_stats_size(messages: list) -> int:
    """统计上下文字符总数：content + tool_calls.arguments。"""
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for call in (m.get("tool_calls") or []):
            total += len(call.get("function", {}).get("arguments") or "")
    return total


# 压缩摘要的提示词：事实优先，丢弃代码匹配等冗长细节
COMPACT_PROMPT = """请把以下多轮对话压缩成一段 200~400 字的简洁摘要。
要求：
1. 保留关键事实：用户偏好、提问意图、得出的结论、工具调用结果的关键信息
2. 丢弃冗长细节：grep 匹配行、原始文件内容、网页正文只保留结论
3. 不要复述工具的原始输出，用一句结论概括
4. 用中文，按时间顺序，列出关键事件
5. 不要添加对话中没有的内容"""


def find_user_round_start(messages: list, keep_rounds: int) -> int:
    """从尾部向前回溯，跳过 system(idx 0)，找到第 keep_rounds 个 user 消息的索引。

    一轮对话定义为「从 user 消息开始，到下一个 user 消息之前」，这样切能保证 tail
    始终是完整的 user→assistant→tool 序列，OpenAI 协议合法（tool 不会孤儿化）。

    返回 cut 索引：messages[cut:] 是要保留的最近 keep_rounds 个完整轮。
    若找不到足够 user（少于 keep_rounds 个），返回 -1。
    """
    user_count = 0
    for i in range(len(messages) - 1, 0, -1):  # 跳过 system（索引 0）
        if messages[i].get("role") == "user":
            user_count += 1
            if user_count == keep_rounds:
                return i
    return -1


def run_one_tool(call: dict) -> tuple:
    """执行单个工具调用，返回 (call, name, args, result) 四元组。

    所有异常都在这里兜住，不让一个工具的失败影响其他并行执行的任务。
    TOOL_FUNCTIONS 在并行路径中只读访问，多线程安全。
    """
    name = call["function"]["name"]
    args_raw = call["function"]["arguments"]
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        return call, name, None, "错误：arguments 不是合法 JSON，请检查字段名是否与 schema 描述一致后重新调用"
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return call, name, args, f"未知工具: {name}"
    try:
        result = func(**args)
    except TypeError as e:
        result = f"参数不匹配: {e}"
    except Exception as e:
        result = f"工具执行异常: {e}"
    return call, name, args, result


def compact_messages(messages: list) -> bool:
    """超阈值时压缩 messages：保留 system + 最近 N 个完整 user 轮，中间压成一条摘要。

    返回 True 表示发生了压缩（无论成功与否），False 表示跳过。
    失败兜底：压缩调用本身超时/出错时不动 messages，留给下次机会。
    """
    if context_stats_size(messages) <= CONTEXT_COMPACT:
        return False

    # 从尾部向前找第 N 个 user 消息的位置，作为 tail 起点
    cut = find_user_round_start(messages, COMPACT_KEEP_USER_ROUNDS)
    if cut <= 1:  # cut=1 表示 user 紧跟 system，没东西可压；cut=-1 表示找不到足够 user
        return False

    head = messages[:1]                          # system 消息保留
    tail = messages[cut:]                        # 最近 N 个完整轮
    middle = messages[1:cut]                     # 中间要压缩的部分

    # 把中间序列转成一段对话文本喂给压缩模型，role 标记 + 内容/工具摘要
    parts = []
    for m in middle:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "tool":
            # 工具结果只取前 100 字符作为索引，原始长结果已在记忆中
            parts.append(f"[工具结果] {content[:100]}{'...' if len(content) > 100 else ''}")
        elif role == "assistant" and m.get("tool_calls"):
            names = [tc["function"]["name"] for tc in m["tool_calls"]]
            parts.append(f"[助手调用工具] {', '.join(names)}")
        else:
            parts.append(f"[{role}] {content}")
    dialogue = "\n".join(parts)

    # 单独开一个轻量压缩请求：不开 tools、低 temperature、限 max_tokens 控长度
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPACT_PROMPT},
                {"role": "user", "content": dialogue},
            ],
            temperature=0.2,
            max_tokens=COMPACT_MAX_TOKENS,
            timeout=API_TIMEOUT,
        )
    except (APITimeoutError, APIConnectionError, APIStatusError) as e:
        print(f"[上下文压缩失败：{type(e).__name__}，本轮不压缩留待下次]")
        return False

    summary = resp.choices[0].message.content or ""
    if not summary.strip():
        print("[上下文压缩失败：模型返回空摘要，本轮不压缩]")
        return False

    before = context_stats_size(messages)
    # 用 clear + extend + append 原地重组 messages，外部引用不变
    messages.clear()
    messages.extend(head)
    messages.append({
        "role": "system",
        "content": f"【对话历史摘要】以下是之前对话的压缩摘要，仅供参考；用户当前问题在摘要之后：\n{summary}",
    })
    messages.extend(tail)
    after = context_stats_size(messages)
    print(f"[上下文] 已压缩 {len(middle)} 条中间消息→ 1 条摘要，{before} → {after} 字符")
    return True



def main():
    # 会话记忆启动恢复：上次退出前的对话已落盘在 session.json，自动加载回来继续聊。
    # 必须校验首条是 system、消息序列协议合法 —— 否则 API 下次调用会被服务端 400 拒绝
    messages = None
    if os.path.exists(SESSION_PATH):
        try:
            with open(SESSION_PATH, encoding="utf-8") as f:
                messages = json.load(f)
            if not messages or messages[0].get("role") != "system":
                raise ValueError("session 文件缺少 system 消息")
            # 刷新 system 消息：既保留工作目录锚点，也注入最新长期记忆
            messages[0] = {"role": "system", "content": build_system_prompt()}
            print(f"[已恢复上次会话：{len(messages)} 条消息，输入 clear 可重新开始]")
        except (json.JSONDecodeError, OSError, ValueError):
            messages = None  # 文件损坏就当没有，重新开始
    if messages is None:
        messages = [{"role": "system", "content": build_system_prompt()}]

    print(f"使用模型: {model}")
    print("可以问：")
    print("  - 天气：'北京天气怎么样？'")
    print("  - 数学：'(25+5)*2 等于多少？'")
    print("  - 复合：'北京和上海哪个温度高？'")
    print("  - 文件：'读一下 README.md' 或 '帮我看看 E:/test/a.txt 里有什么'")
    print("  - 目录：'看一下当前项目下面有什么'")
    print("  - 搜索：'哪个文件里定义了 get_weather？' 或 '帮我找找哪里用了 dotenv'")
    print("  - 知识库：'退货的钱多久能退回来？'")
    print("  - 联网：'Python 3.13 新特性' 或 '读一下 https://docs.python.org/...'")
    print("  - 记忆：'记住我喜欢简洁的回答'")
    print("输入 clear 清空对话，quit 退出（clear 和 quit 都不影响长期记忆 memory.md）")
    print("=" * 40)

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见。")
            break
        # 清空对话历史，只保留 system prompt（相当于 Claude Code 的 /clear）。
        # 落盘的 session.json 也要删，否则下次启动又把旧对话恢复回来 —— 这就是「清空」的彻底性。
        # 注意：长期记忆 memory.md 故意保留 —— clear 忘的是聊天，不是记住的事实
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": build_system_prompt()}]
            if os.path.exists(SESSION_PATH):
                os.remove(SESSION_PATH)
            print("[已清空对话历史和会话文件，上下文重置（长期记忆 memory.md 保留）]")
            continue
        if not user_input:
            continue

        # 本轮问答开头先看一眼上下文：超过阈值就压缩，并立刻同步落盘。保证本轮回答不被压缩
        if compact_messages(messages):
            with open(SESSION_PATH, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
        # 没压缩但接近阈值时也提醒一下，方便观察余量
        elif context_stats_size(messages) >= CONTEXT_WATCH:
            print(f"[上下文] 已达观察阈值 {CONTEXT_WATCH} 字符，下次提问前可能触发压缩")

        messages.append({"role": "user", "content": user_input})

        # Agent 关键：内层循环让 LLM 自主决定调几次工具
        step = 0
        while True:
            step += 1
            if step > MAX_STEPS:
                print(f"[已达最大步数 {MAX_STEPS}，强制停止。模型可能在反复调用工具，请换个问法或 clear 后重试]")
                break

            # 流式调 LLM（stream=True）。create() 只建立连接就返回，之后逐块收到 chunk。
            # 超时/断连/HTTP 错误既可能在建连时抛出，也可能在消费 chunk 的过程中抛出，
            # 所以 create 和 for 循环要放在同一个 try 里兜住
            content_parts = []   # 文本片段累计器，流结束后拼成完整回复
            tool_calls_acc = {}  # index -> {"id","name","arguments"}，工具调用分片组装器
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    timeout=API_TIMEOUT,
                    stream=True,
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue  # 部分服务端在最后发一个只带用量统计、无 choices 的收尾块
                    delta = chunk.choices[0].delta
                    if delta.content:
                        # flush=True 立即刷新 stdout 缓冲区，否则逐字打印会攒成一大段才显示
                        print(delta.content, end="", flush=True)
                        content_parts.append(delta.content)
                    # 工具调用分片到达：首个分片带 id 和完整 name，arguments 被切成
                    # 多个碎片散在后续分片里，必须按 index 拼回完整 JSON
                    for tc in delta.tool_calls or []:
                        slot = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] += tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments
            except APITimeoutError:
                # 消息开头的 \n 顺便截断流式中已打印一半的行
                print(f"\n[LLM 响应超时（{API_TIMEOUT} 秒），本轮终止，请重新提问]")
                break
            except APIConnectionError:
                print("\n[连不上 API 服务器：请检查网络和 BASE_URL 配置]")
                break
            except APIStatusError as e:
                # 401 key 无效 / 429 限流 / 500 服务端出错等都走这里
                print(f"\n[API 返回错误 {e.status_code}: {e.message}]")
                break
            if content_parts:
                print()  # 流式打印不带换行，收尾补一个，避免下一条输出接在同一行

            final_content = "".join(content_parts)

            # 没调工具 = 最终回答，结束本轮。流式没有现成的 message 对象，
            # assistant 消息要手工组装塞回 messages，会话落盘才完整
            if not tool_calls_acc:
                messages.append({"role": "assistant", "content": final_content})
                break

            # 有工具调用：按 OpenAI 协议拼出 id/type/function 三件套再塞回 messages，
            # 下一轮 LLM 才能看到自己调过什么工具、结果是什么
            assistant_tool_calls = [
                {
                    "id": slot["id"],
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for slot in tool_calls_acc.values()
            ]
            messages.append({
                "role": "assistant",
                "content": final_content or None,
                "tool_calls": assistant_tool_calls,
            })

            # 并行执行多个工具调用：独立请求并发跑，节约串行等待时间。
            # run_one_tool 在模块级定义，保证协议合法（messages 按 LLM 调用顺序追加）
            with ThreadPoolExecutor(max_workers=len(assistant_tool_calls)) as pool:
                # map 保证按入参顺序返回，但内部并发执行
                results = list(pool.map(run_one_tool, assistant_tool_calls))

            for call, name, args, result in results:
                # args 是 None 说明 JSON 解析失败，打印原始字符串
                args_repr = args if args is not None else f"(JSON 解析失败: {call['function']['arguments']!r})"
                print(f"  -> [调用工具] {name}({args_repr})")
                # 即使失败也必须回一条工具消息：否则有 tool_call 却没有对应结果，
                # 消息序列不符合协议，下一次 API 调用会被服务端直接拒绝
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })

            # 工具结果回传后，回到 while 内循环顶部
            # LLM 看到工具结果，决定下一步：继续调工具 or 直接给出文字答案

        # 落到这说明本轮问答结束了（内循环因给出答案/错误/步数上限 break）。
        # 体感实验：每轮打印上下文规模，观察它如何随工具调用膨胀
        print(f"[上下文] {len(messages)} 条消息 · 约 {context_stats_size(messages)} 字符")
        # 会话记忆落盘：每轮问答后把最新 messages 写入磁盘，quit 重启后可无损恢复
        # indent=2 是为了人类可读（json.load 不在乎缩进）—— 排查对话恢复问题时
        # 能直接 cat session.json 看每条消息，也避开 IDE 拼写检查因单行过长而跳过
        with open(SESSION_PATH, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

# 直接运行本文件时，__name__是__main__，被import时是模块名。
# 这个判断保证：只有 python agent.py 直接运行时才调 main()，被其他模块导入时不执行
if __name__ == "__main__":
    main()
