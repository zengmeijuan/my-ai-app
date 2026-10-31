"""03_sdk_client.py —— 官方 mcp SDK 的客户端侧：连自家 server 走协议三步（D3a 教学物）

02_raw_client.py 手写报文是"拆开看变速箱"，本脚本是"开自动挡"：同一套协议三步
（initialize 握手 → list_tools 查表 → call_tool 调用），SDK 全部包成方法调用。
自建 agent 想用 MCP 工具，走的就是这条路——init 后拿 schema 塞给 bind_tools，
模型出 tool_calls 就 call_tool 执行回填（B 阶段循环不变，只换执行层）。

跑法（项目根）：uv run python src/mcp_and_skill/03_sdk_client.py
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台显示 UTF-8（D2 同款坑）

SERVER_PATH = str(Path(__file__).resolve().parent / "01_mcp_server.py")  # __file__ 钉死位置，不信 cwd


async def main():
    # 配方：怎么拉起 server——与 raw_client 的 SERVER 列表同构（第 0 个是程序，其余是参数）
    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])

    # stdio_client：拉起 server 子进程，接管它的 stdin/stdout 两条管子（等价于 Popen+PIPE），read 读 server 回话，write 发我们的请求
    # async with ... as ... ——异步版 with，退出时内层先关（结束会话）外层再关（server 进程退场）
    async with stdio_client(params) as (read, write):
        # ClientSession：在管子上建一个"会话"对象，之后所有协议交互都喊它的方法
        async with ClientSession(read, write) as session:
            await session.initialize()  # 协议第一步：握手（含 notifications/initialized，SDK 自动补）
            print("===== 握手完成 =====")

            # 协议第二步：tools/list——server 工具表全在这
            tools = await session.list_tools()
            print(f"===== server 暴露 {len(tools.tools)} 个工具 =====")
            for t in tools.tools:
                print(f"- {t.name}：{t.description[:50]}...")
                print(f"  参数 schema: {t.input_schema}")  # 协议报文键是 inputSchema，SDK 对象属性是 snake_case

            # 协议第三步：tools/call——name + arguments，回 content 内容块列表
            result = await session.call_tool("calculate", {"expression": "45+39"})
            print(f"===== calculate 结果: {result.content[0].text} =====")


if __name__ == "__main__":
    asyncio.run(main())  # 同步世界进异步世界的门：启动事件循环并驱动 main 跑完
