"""第一个 AI 程序：LLM Hello World

支持任意 OpenAI 协议兼容的服务（OpenAI / DeepSeek / 智谱 GLM / Kimi 等），
通过 .env 文件配置切换 provider，无需改代码。

运行：uv run python main.py
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 自动把 .env 里的变量加载到 os.environ

# 通过环境变量配置，可任意切换 provider
client = OpenAI(
    api_key=os.environ["API_KEY"],                                       # 必填
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),    # 可选
)
model = os.environ.get("MODEL", "gpt-4o-mini")

print(f"使用模型: {model}")
print("-" * 40)
print("提问: 用一句话介绍 Python\n")

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是一个简洁的 AI 助手，回答尽量精炼。"},
        {"role": "user", "content": "用一句话介绍 Python"},
    ],
)

print("AI:")
print(resp.choices[0].message.content)
print()
print(
    f"Token 用量: 输入={resp.usage.prompt_tokens}, "
    f"输出={resp.usage.completion_tokens}, 合计={resp.usage.total_tokens}"
)
