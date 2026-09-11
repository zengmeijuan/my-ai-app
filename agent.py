"""Agent CLI - chat.py 升级版，加入 Function Calling

新增能力：AI 能自主调用工具完成任务
基于 chat.py 的 LLM + 循环 + 记忆 + 工具调用 + 自主决策（5 要素齐全）

工具：
1. get_weather(city) - 查城市天气（mock 数据，真实项目应调天气 API）
2. calculate(expression) - 算数学表达式（真算）
3. read_file(path) - 读取文本文件内容
4. list_dir(path) - 列出目录内容（一层）
5. grep(pattern, path) - 在目录下递归搜索文本

运行：uv run python agent.py
"""
import os
import json
import re

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

load_dotenv()

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
)
model = os.environ.get("MODEL", "gpt-4o-mini")


# ============== 1. 工具实现（您写的真实函数）==============

def get_weather(city: str) -> str:
    """获取指定城市的天气（mock 数据，真实项目应调天气 API）"""
    weather_db = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 60},
        "上海": {"temp": 28, "condition": "多云", "humidity": 70},
        "广州": {"temp": 30, "condition": "雷阵雨", "humidity": 85},
        "深圳": {"temp": 29, "condition": "晴", "humidity": 75},
    }
    data = weather_db.get(city, {"temp": 22, "condition": "未知", "humidity": 50})
    return json.dumps(data, ensure_ascii=False)


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 安全 eval：禁止内置函数和危险操作
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def read_file(path: str) -> str:
    """读取指定路径的文本文件内容"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误：文件不存在: {path}"
    except UnicodeDecodeError:
        return f"错误：无法以 utf-8 解码（可能是二进制或非 utf-8 编码）: {path}"
    except Exception as e:
        return f"读取错误: {e}"


def list_dir(path: str) -> str:
    """列出指定目录下的文件和子目录（只列一层，不递归）"""
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
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".idea", ".vscode"}
MAX_MATCHES = 50  # 返回条数上限：工具必须自带边界，防止单次结果撑爆上下文


def grep(pattern: str, path: str = ".") -> str:
    """在目录下递归搜索包含指定文本/正则的行，返回 文件:行号:内容"""
    try:
        re.compile(pattern)
    except re.error as e:
        return f"错误：无效的正则表达式: {e}"

    matches = []
    for root, dirs, files in os.walk(path):
        # 原地过滤 dirs，os.walk 就不会进入被排除的目录
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    # 前 1024 字节含 \x00 视为二进制文件，跳过（ripgrep 同款判据）
                    if "\x00" in f.read(1024):
                        continue
                    f.seek(0)
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
        return f"未找到匹配 '{pattern}'（目录: {path}）"
    return f"共 {len(matches)} 处匹配:\n" + "\n".join(matches)


# 工具名 -> 函数的映射（执行时用）
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
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
            "description": "列出指定目录下的文件和子目录（只列一层，不递归）。当用户想看目录结构、确认路径是否存在、找文件在哪时调用。",
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
]


# ============== 3. Agent 主循环 ==============

# 循环安全边界：模型陷入「反复调工具停不下来」时强制刹车
MAX_STEPS = 10   # 单次提问最多调 LLM 几轮（每轮 LLM 都可能调工具再思考，都在烧 token）
API_TIMEOUT = 60  # 单次 LLM API 调用的超时秒数，防止网络挂起卡死整个 agent

# system prompt 提成常量：clear 重置对话时要保留它（里面有工作目录锚点）
SYSTEM_PROMPT = (
    f"你是一个智能助手。当前工作目录是 {os.getcwd()}，用户说「当前项目」「这个项目」时指该目录下的项目。"
    "如果用户的问题需要调用工具才能回答，请调用合适的工具；否则直接回答。"
    "调用工具后请基于工具结果给用户简洁的回复。"
)


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"使用模型: {model}")
    print("可以问：")
    print("  - 天气：'北京天气怎么样？'")
    print("  - 数学：'(25+5)*2 等于多少？'")
    print("  - 复合：'北京和上海哪个温度高？高多少？'（AI 会连续调用 2 次工具）")
    print("  - 文件：'读一下 README.md' 或 '帮我看看 E:/test/a.txt 里有什么'")
    print("  - 目录：'看一下 E:/workspace/studying/my-ai-app 下面有什么'")
    print("  - 搜索：'哪个文件里定义了 get_weather？' 或 '帮我找找哪里用了 dotenv'")
    print("输入 clear 清空对话（切换话题/项目前用），输入 quit 退出")
    print("=" * 40)

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见。")
            break
        # 清空对话历史，只保留 system prompt（相当于 Claude Code 的 /clear）
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[已清空对话历史，上下文重置，模型忘记之前聊过的一切]")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent 关键：内层循环让 LLM 自主决定调几次工具
        step = 0
        while True:
            step += 1
            if step > MAX_STEPS:
                print(f"[已达最大步数 {MAX_STEPS}，强制停止。模型可能在反复调用工具，请换个问法或 clear 后重试]")
                break

            # 调 LLM（带 tools 参数）。网络请求必须设超时，并兜住三类 API 异常
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    timeout=API_TIMEOUT,
                )
            except APITimeoutError:
                print(f"[LLM 响应超时（{API_TIMEOUT} 秒），本轮终止，请重新提问]")
                break
            except APIConnectionError:
                print("[连不上 API 服务器：请检查网络和 BASE_URL 配置]")
                break
            except APIStatusError as e:
                # 401 key 无效 / 429 限流 / 500 服务端出错等都走这里
                print(f"[API 返回错误 {e.status_code}: {e.message}]")
                break

            msg = resp.choices[0].message

            # 把 LLM 的回复塞回 messages（即使是工具调用也要塞，下一轮 LLM 才能看到自己调过什么）
            # model_dump(exclude_none=True) 把 SDK 对象转 dict，排除空字段
            messages.append(msg.model_dump(exclude_none=True))

            # 如果 LLM 没调工具，说明给了最终答案，退出内循环
            if not msg.tool_calls:
                print(f"AI: {msg.content}")
                break

            # 执行 LLM 决定调用的所有工具
            for call in msg.tool_calls:
                name = call.function.name

                # arguments 是模型生成的 JSON 字符串，可能不合法
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    print(f"  -> [调用工具] {name}(参数 JSON 解析失败: {call.function.arguments!r})")
                    # 即使解析失败也必须回一条工具消息：否则有 tool_call 却没有对应结果，
                    # 消息序列不符合协议，下一次 API 调用会被服务端直接拒绝
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "错误：arguments 不是合法 JSON，请修正后重新调用该工具",
                    })
                    continue

                print(f"  -> [调用工具] {name}({args})")

                # 执行真实函数。工具内部已自行兜错，这里再兜一层调用层意外（如参数名不匹配）
                func = TOOL_FUNCTIONS.get(name)
                if not func:
                    result = f"未知工具: {name}"
                else:
                    try:
                        result = func(**args)
                    except TypeError as e:
                        result = f"参数不匹配: {e}"
                    except Exception as e:
                        result = f"工具执行异常: {e}"

                # 把工具结果塞回 messages（特殊 role="tool"）
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

            # 工具结果回传后，回到 while 内循环顶部
            # LLM 看到工具结果，决定下一步：继续调工具 or 给最终答案

# 直接运行本文件时，__name__是__main__，被import时是模块名。
# 这个判断保证：只有 python agent.py 直接运行时才调 main()，被其他模块导入时不执行
if __name__ == "__main__":
    main()
