"""05_http_client.py —— SDK 客户端的 HTTP 传输形态：连常驻服务而不是拉起子进程（D3b 教学物）

与 03_sdk_client.py 唯一差异在入口函数：stdio_client（拉起子进程接管管子）
换成 streamable_http_client（按 url 连常驻服务）。initialize / list_tools /
call_tool 协议三步一字不变——传输层换皮，协议层不换药。

前置：服务已常驻运行（项目根） uv run python src/mcp_and_skill/01_mcp_server.py http
跑法（项目根）：uv run python src/mcp_and_skill/05_http_client.py
"""
import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client  # 2.x 名（1.x 叫 streamablehttp_client）

sys.stdout.reconfigure(encoding="utf-8")

URL = "http://127.0.0.1:8000/mcp"


async def main():
    # 连远程服务：url 寻址（对比 stdio 的"配方拉起"）；本版本 yield 二元组 (read, write)
    async with streamable_http_client(URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("===== 握手完成（HTTP 传输） =====")

            tools = await session.list_tools()
            print(f"===== 服务暴露 {len(tools.tools)} 个工具: {[t.name for t in tools.tools]} =====")

            result = await session.call_tool("calculate", {"expression": "45+39"})
            print(f"===== calculate 结果: {result.content[0].text} =====")


if __name__ == "__main__":
    asyncio.run(main())
