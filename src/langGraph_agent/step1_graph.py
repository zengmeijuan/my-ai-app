"""step1_graph.py —— Step 1-2：最小 ReAct 图 + 功能对齐验收（B2 的地基，后续所有步骤文件复用它）

图结构（复刻 agent.py 主循环 while + if 的骨架）：

    START → agent ──(有 tool_calls)→ tools → agent（循环）
             │
             └─(无 tool_calls)→ END

工具复用：@tool 包装 my_agent 的 3 个客服场景核心函数，名字与原版一致；
MAX_STEPS 由框架 recursion_limit 接管（invoke 的 config），不手写计步；
记忆注入：system prompt 每轮动态拼装，memory.md 更新后下一轮即生效

跑法（项目根）：
  uv run python src/langGraph_agent/step1_graph.py "自定义问题"
  uv run python src/langGraph_agent/step1_graph.py  # 默认星辰商城工单题
"""
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from my_agent.agent import (
    MEMORY_PATH,
    calculate as _calculate,
    remember as _remember,
    search_knowledge as _search_knowledge,
)

load_dotenv()


# ============== 1. 工具：@tool 包装，实现与 schema 同源 ==============

# @tool 将裸函数变成工具对象，产出schema元数据（name/description/parameters）+ 保留原函数实现
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，返回结果。用于金额、差额等算术。"""
    return _calculate(expression)


@tool
def search_knowledge(query: str) -> str:
    """星辰商城知识库检索：返回与问题最相关的 top-3 资料（带来源文件名和相似度）。"""
    return _search_knowledge(query)


@tool
def remember(content: str) -> str:
    """把一条结论性信息追加到长期记忆，下次会话可召回。"""
    return _remember(content)


tools = [calculate, search_knowledge, remember]


# ============== 2. 模型：ChatOpenAI 接网关 + bind_tools ==============

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=60,
    max_retries=2,  # http请求重试次数
)
# bind_tools 将工具对象的schema翻译成OpenAI function calling 格式，挂在模型上，每次请求都会携带这些schema。它返回新的模型对象
llm_with_tools = llm.bind_tools(tools)


# ============== 3. system prompt：每轮动态拼装（memory 更新下轮即生效）==============

SYSTEM_PROMPT_BASE = """# 角色
你是星辰商城的客服助手。所有政策条款（运费/退货/退款/会员/积分）必须以
search_knowledge 返回的资料为准，禁止编造；知识库没有的如实告知。

# 工作方式
- 业务问题先 search_knowledge，涉及金额用 calculate，最终给顾客的结论用 remember 存一条。
- 按顾客问题逐条回答，给出完整可发送给顾客的答复文本。"""


def build_system_prompt() -> str:
    """组装 system prompt：基础指令 + 长期记忆（memory.md 存在时追加注入）"""
    prompt = SYSTEM_PROMPT_BASE
    if MEMORY_PATH.exists():
        memory = MEMORY_PATH.read_text(encoding="utf-8").strip()
        if memory:
            prompt += "\n\n【长期记忆】以下是过去会话中记住的信息，回答时可参考：\n" + memory
    return prompt


# ============== 4. 节点与图：while 循环的图式表达 ==============

# call_model节点：从state读消息列表、喂给绑定了工具的模型、返回的AIMessage追加回messages（靠state的reducer 归约器机制）
def call_model(state: MessagesState):
    # 每轮 LLM 调用前都获取最新消息
    messages = [SystemMessage(content=build_system_prompt())] + list(state["messages"])
    return {"messages": [llm_with_tools.invoke(messages)]}


# 路由函数：返回下一个节点的名字
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


# 创建"图纸"（builder对象）
builder = StateGraph(MessagesState)
# agent节点
builder.add_node("agent", call_model)
# ToolNode(tools) 是框架自带的工具执行节点
builder.add_node("tools", ToolNode(tools))
# 入口，从agent节点开始
builder.add_edge(START, "agent")
# 有条件边，出口判断，相当于循环里面的if
builder.add_conditional_edges("agent", should_continue, ["tools", END])
# 无条件边：工具结果回填后回到 agent 节点，相当于while循环本体
builder.add_edge("tools", "agent")
# 编译：将图纸冻结成可执行程序
graph = builder.compile()


# ============== 5. 入口：跑星辰商城工单题 ==============

TICKET_QUESTION = (
    "顾客买了 45 元的台灯和 39 元的小风扇，用微信支付。请回答："
    "① 这单收多少运费？"
    "② 台灯无理由退货的话，寄回运费谁承担、退款几天到账？"
)

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else TICKET_QUESTION
    result = graph.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 10},  # 框架版 MAX_STEPS：步数刹车
    )
    print("\n=== 最终回答 ===")
    print(result["messages"][-1].content)
    print(f"\n[共 {len(result['messages'])} 条消息]")
    for m in result["messages"]:
        print(type(m).__name__, ":", str(m.content)[:60].replace("\n", " "))
