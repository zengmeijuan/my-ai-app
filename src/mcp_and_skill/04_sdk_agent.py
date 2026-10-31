"""04_sdk_agent.py —— 自建 agent 接入 MCP：LLM 决策 + MCP 执行

agent 的决策层与执行层在此分工：MCP 工具表套上 OpenAI tools 外壳交给模型，
模型自主产出 tool_calls（调什么、传什么参数）；执行层把每次调用经 MCP 协议
（call_tool）转发给 server 子进程，结果按 tool_call_id 回填消息轨迹，
循环到模型给出文字回答。与 B 阶段 agent.py 逐行同构，唯一差异：
工具不在本地，执行走 MCP 协议（run_one_tool 换成 await session.call_tool）。

跑法（项目根）：
  uv run python src/mcp_and_skill/04_sdk_agent.py            # 内置两条验证问题
  uv run python src/mcp_and_skill/04_sdk_agent.py "你的问题"   # 自定问题
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # 绝对路径：脚本被谁拉起 cwd 不保证是项目根（D2 的坑）

from openai import AsyncOpenAI  # noqa: E402
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SERVER_PATH = str(Path(__file__).resolve().parent / "01_mcp_server.py")
SYSTEM_PROMPT = ("你是「星辰商城」客服助手。金额/数量计算必须用 calculate，"
                 "政策类问题必须先 search_knowledge，依据工具结果简洁作答。")

# AsyncOpenAI = openai 库的原生异步客户端，接口与 agent.py 的 OpenAI 同形，
# 每个 create 加 await——同步世界的 client.chat.completions.create 换了个 awaits 版
client = AsyncOpenAI(api_key=os.environ["API_KEY"],
                     base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"))
model = os.environ.get("MODEL", "gpt-4o-mini")


async def run_one(session: ClientSession, openai_tools: list, question: str):
    """一轮完整问答：决策循环（模型）+ 执行（MCP）"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    for _ in range(10):  # 循环上限：防模型失控连环调用（B 阶段同款护栏）
        resp = await client.chat.completions.create(model=model, messages=messages,
                                                    tools=openai_tools)
        msg = resp.choices[0].message
        if not msg.tool_calls:  # 不再要工具 = 模型认为可以作答了，循环出口
            print(f"回答: {msg.content}")
            return
        # 模型要工具：先把它的决策原样记入轨迹（assistant 消息带 tool_calls）
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:  # 决策 → 执行：这里才轮到 MCP 出场
            args = json.loads(tc.function.arguments)
            result = await session.call_tool(tc.function.name, args)
            text = result.content[0].text
            print(f"  [MCP] {tc.function.name}({args}) -> {text[:60]}...")
            # 执行结果按 tool_call_id 回填轨迹（对照 agent.py 的 role=tool 消息）
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
    print("回答: （连续 10 轮都在调工具，触发护栏停下）")


async def main():
    # 启动配置：启动server子进程的参数
    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])
    # 通过启用配方params创建子进程，并把stdin/stdout两条接管管子包装成对象流对象read、write
    async with stdio_client(params) as (read, write):
        # ClientSession(read, write)：通过两条裸流，创建一个懂MCP协议的会话对象
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 工具表拉一次、两个问题共用（宿主也是启动时拉一次表，不是每问都拉）
            tools = await session.list_tools()
            # MCP 的 inputSchema 本身就是 JSON Schema，套上 OpenAI function 外壳即成 tools 参数
            openai_tools = [{"type": "function",
                             "function": {"name": t.name, "description": t.description,
                                          "parameters": t.input_schema}}
                            for t in tools.tools]
            questions = sys.argv[1:] or [
                "台灯 45 元，小风扇 39 元，两件一共多少钱？",
                "七天无理由退货的条件是什么？",
            ]
            for q in questions:
                print(f"\n===== 问: {q}")
                await run_one(session, openai_tools, q)


if __name__ == "__main__":
    asyncio.run(main())
