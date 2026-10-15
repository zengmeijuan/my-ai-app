"""Step 0：B2 环境自检 —— import 路径 → 网关连通 → bind_tools 冒烟

三段递进，每段都是后续步骤的地基：
1. import 段：B2 要用的 9 个关键 API 逐个 import + 打印包版本
   （验证 LangGraph 1.x 的 API 路径，讲课时给的不算数，实测才算）
2. 网关段：ChatOpenAI 复刻 agent.py 的接入方式（.env 三件套），发一句极短对话
3. bind_tools 段：@tool 定义迷你 calculate 挂上模型，检查 tool_calls 产出
   ——这是 B2 的核心机制，它通了才有资格写图

跑法（项目根）：
  uv run python src/langGraph_agent/step0_check.py
"""
import os
from importlib.metadata import version

from dotenv import load_dotenv

load_dotenv()  # 读项目根 .env：API_KEY / BASE_URL / MODEL（与 agent.py 同款配置）

print("=" * 55)
print("[1/3] import 检查：LangGraph 1.x 关键 API 路径")
print("=" * 55)

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

print(f"langgraph        = {version('langgraph')}")
print(f"langchain-core   = {version('langchain-core')}")
print(f"langchain-openai = {version('langchain-openai')}")
print(f"openai           = {version('openai')}")
print("全部 import 通过：StateGraph / MessagesState / InMemorySaver / SqliteSaver /")
print("interrupt / Command / ToolNode / @tool / ChatOpenAI")

print()
print("=" * 55)
print("[2/3] 网关连通：ChatOpenAI 复刻 agent.py 的接入方式")
print("=" * 55)

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=60,
)
resp = llm.invoke("只回复两个字：正常")
print(f"model = {llm.model_name}")
print(f"reply = {resp.content}")

print()
print("=" * 55)
print("[3/3] bind_tools 冒烟：工具 schema 挂载 + tool_calls 产出")
print("=" * 55)


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，返回结果。"""
    return str(eval(expression, {"__builtins__": {}}, {}))


tool_llm = llm.bind_tools([calculate])
resp2 = tool_llm.invoke("帮我算一下 45+39 等于多少？必须调用工具计算，不要心算。")
print(f"tool_calls = {resp2.tool_calls}")
if not resp2.tool_calls:
    raise RuntimeError("模型没有产出 tool_calls —— bind_tools 链路异常，先排查再动手")
print("bind_tools 冒烟通过：模型正确请求调用 calculate")

print()
print("=== Step 0 全部通过，B2 环境就绪 ===")
