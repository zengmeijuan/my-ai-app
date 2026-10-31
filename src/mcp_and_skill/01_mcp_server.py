"""01_mcp_server.py —— 里程碑 D2：把星辰商城工具包装成 MCP server（官方 mcp SDK 的 FastMCP，stdio 传输）

MCP server 是协议外壳，不复制任何业务逻辑——calculate / search_knowledge 直接
复用 my_agent.agent 的现有实现（与被测图同一份代码），只把「工具描述」重写成
面向模型的说法（schema 的 description 是模型选工具的唯一依据，怎么写有讲究）。

跑法（项目根）：
  uv run python src/mcp_and_skill/01_mcp_server.py        # stdio 裸跑（被宿主拉起时就是这个形态）
  uv run python src/mcp_and_skill/02_raw_client.py        # 手写 JSON-RPC 裸客户端，逐帧看报文
  npx @modelcontextprotocol/inspector uv run python src/mcp_and_skill/01_mcp_server.py
                                                       # 官方 inspector 交互调试（浏览器 UI）
接入 Claude Code（D3）：claude mcp add xingchen-mall -- uv run python src/mcp_and_skill/01_mcp_server.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # 项目根（本文件在 src/mcp_and_skill/ 下）

# 不需要 sys.path.insert：uv 把本项目以 editable 模式装进 .venv 时，落地物
# site-packages/my_ai_app.pth 写入了 src 绝对路径，任何用本 .venv 解释器跑的
# 代码都能 import my_agent（与脚本位置、cwd 无关）。

# 生产对照点 1：stdio server 会被宿主从「任意 cwd」拉起。agent.py 顶部的
# load_dotenv() 无参调用只从 cwd 向上找 .env，宿主启动时 cwd 不是项目根就找不到，
# 接着 import 时创建 OpenAI client 直接 KeyError。所以这里必须用绝对路径先加载 .env。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from mcp.server.mcpserver import MCPServer  # noqa: E402

from my_agent.agent import calculate as _calculate, search_knowledge as _search_knowledge  # noqa: E402

# 官方 SDK 2.x：FastMCP 已改名 MCPServer（1.x 的 from mcp.server.fastmcp import FastMCP 报错
# 会提示迁移指南）。构造参数里 auth/auth_server_provider=OAuth 认证、middleware、
# resource_security（默认拒路径穿越）——生产化配置的现成入口，学习版全走默认。
mcp = MCPServer("xingchen-mall")  # server 名字，宿主列表里标识它

# 读函数的类型注解和 docstring，自动生成 JSON Schema，登记进这台 server 的工具表
@mcp.tool()
def calculate(expression: str) -> str:
    """计算数学表达式并返回结果，如 45+39、1200*1。任何金额/数量计算必须调用本工具，禁止心算。"""
    return _calculate(expression)


@mcp.tool()
def search_knowledge(query: str) -> str:
    """检索「星辰商城」客服知识库（含退货政策/配送说明/支付方式/会员权益/账号安全），
    返回与问题最相关的 top-3 资料原文（带来源与相似度）。回答商城政策类问题前必须先调用本工具。"""
    return _search_knowledge(query)


if __name__ == "__main__":
    # 传输选择：默认 stdio（被宿主 Popen 拉起的子进程形态）；传 http 参数则起常驻 HTTP 服务
    # （同一份工具表，两种传输只是"字节怎么搬运"之差，协议层一字不变）
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="streamable-http")  # 监听 127.0.0.1:8000，endpoint 路径 /mcp
    else:
        mcp.run()  # stdio：stdin/stdout 上逐行 JSON-RPC 2.0
