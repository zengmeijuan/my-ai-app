"""Agent CLI - chat.py 升级版，加入 Function Calling

新增能力：AI 能自主调用工具完成任务
基于 chat.py 的 LLM + 循环 + 记忆 + 工具调用 + 自主决策（5 要素齐全）

工具：
1. get_weather(city) - 查城市天气（mock 数据，真实项目应调天气 API）
2. calculate(expression) - 算数学表达式（真算）

运行：uv run python agent.py
"""
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

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


# 工具名 -> 函数的映射（执行时用）
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
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
]


# ============== 3. Agent 主循环 ==============

def main():
    messages = [
        {"role": "system", "content":
            "你是一个智能助手。如果用户的问题需要调用工具才能回答，请调用合适的工具；否则直接回答。"
            "调用工具后请基于工具结果给用户简洁的回复。"
        },
    ]

    print(f"使用模型: {model}")
    print("可以问：")
    print("  - 天气：'北京天气怎么样？'")
    print("  - 数学：'(25+5)*2 等于多少？'")
    print("  - 复合：'北京和上海哪个温度高？高多少？'（AI 会连续调用 2 次工具）")
    print("输入 quit 退出")
    print("=" * 40)

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见。")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent 关键：内层循环让 LLM 自主决定调几次工具
        while True:
            # 调 LLM（带 tools 参数）
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
            )
            msg = resp.choices[0].message

            print(f"\n[出参 finish_reason] {resp.choices[0].finish_reason}")
            print(f"[出参 msg] {msg.model_dump(exclude_none=True)}")
            print(f"[token 用量] {resp.usage}")

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
                args = json.loads(call.function.arguments)

                print(f"  -> [调用工具] {name}({args})")

                # 执行真实函数
                func = TOOL_FUNCTIONS.get(name)
                if func:
                    result = func(**args)
                else:
                    result = f"未知工具: {name}"

                print(f"  <- [工具结果] {result}")

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
