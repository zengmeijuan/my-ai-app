"""多轮流式聊天 CLI

基于 main.py 升级，4 个关键改动：
1. input()  从终端读用户输入（提问不再硬编码）
2. while True  循环持续对话（输入 quit 退出）
3. messages.append()  累积历史（让 AI 记得上下文）
4. stream=True  流式输出（边生成边显示）

运行：uv run python chat.py
退出：输入 quit / exit / q
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
)
model = os.environ.get("MODEL", "gpt-4o-mini")

# 对话历史（在整个会话期间累积，AI 靠这个"记得"上下文）
messages = [
    {"role": "system", "content": "你是一个简洁的 AI 助手，回答尽量精炼。"},
]

print(f"使用模型: {model}")
print("输入 quit / exit / q 退出")
print("=" * 40)

while True:
    # 1. 从终端读用户输入
    user_input = input("\n你: ").strip()

    # 2. 退出判断
    if user_input.lower() in ("quit", "exit", "q"):
        print("再见。")
        break
    if not user_input:
        continue

    # 3. 把用户消息加进历史
    messages.append({"role": "user", "content": user_input})

    # 4. 调 LLM，开流式
    print("AI: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    # 5. 边收 chunk 边打印
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta

    # 6. 把 AI 回答加进历史（下一轮 AI 才能记得自己说过什么）
    messages.append({"role": "assistant", "content": full_response})
