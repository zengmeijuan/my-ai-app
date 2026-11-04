"""step2_research_tools.py —— E2：researcher agent 化（web_search + web_fetch 的 ReAct 子图）

在 step1_supervisor.py 基础上只升级一件事：researcher 从"靠模型自带知识的工序"升级为
**带联网工具的 ReAct 子图**——系统出现第 1 个真正的 agent（自主决定搜什么、抓哪页、
搜几轮、何时收手）。supervisor 与 synthesizer 一字未改，对 researcher 内部的工具循环
零感知——"藏"（B2 step5 定下的准绳：调用方只要结果，就把它藏进节点函数）。

图结构：
    主图（与 step1 相同）：supervisor ⇄ researcher / synthesizer → END
    researcher 内部（新增）：agent（带 web_search / web_fetch 决策）↔ tools → END

与 B2 的对照：
    B2 step5 把检索子图藏进 @tool 里（工具内 invoke 子图）；
    E2 把 ReAct 子图藏进 researcher 节点函数里（节点内 invoke 子图）——
    两处同一准绳：跨 State 边界（MessagesState ↔ TeamState 不同构）时，
    调用方只要结果，就在边界处 invoke 子图、取最终产出、写回主图 State。

与 step1 的对照验收：同一道默认题，E1 材料引用权威结论但无来源；
E2 材料带实时来源 URL（搜索与抓取页面的）——工具循环的直接产出差异。

跑法（项目根）：
  uv run python src/multi_agent/step2_research_tools.py "自定义研究题目"
  uv run python src/multi_agent/step2_research_tools.py  # 默认题目（与 step1 同题对照）
"""
import os
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from my_agent.agent import web_fetch as _web_fetch, web_search as _web_search

load_dotenv()


# ============== 1. 共享 State（与 step1 完全相同）==============

class TeamState(TypedDict):
    topic: str            # 研究题目（入口输入）
    supervisor_note: str  # supervisor 给下一个 worker 的当前指令
    research: str         # researcher 产出的研究材料（重派时覆盖追加）
    answer: str           # synthesizer 产出的答案
    round_count: int      # 已执行的研究轮次（熔断计数器）
    next: str             # supervisor 决定的下一节点（路由信号）


# ============== 2. 模型 ==============

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=120,
    max_retries=2,
)


# ============== 3. 联网工具：@tool 包装 my_agent 现有函数，只挂给 researcher ==============

@tool
def web_search(query: str) -> str:
    """搜索网页，返回最多 5 条结果的标题、URL 和摘要。用于获取研究题目的资料来源。"""
    return _web_search(query)


@tool
def web_fetch(url: str) -> str:
    """抓取 URL 网页正文，返回纯文本（超长自动截断）。用于读取搜索结果中值得深挖的页面。"""
    return _web_fetch(url)


tools = [web_search, web_fetch]

# @tool 将裸函数变成工具对象：产出 schema 元数据（name/description/parameters）+ 保留原实现。
# bind_tools 把工具 schema 翻译成 function calling 格式挂在模型上，每次请求都携带。
llm_with_tools = llm.bind_tools(tools)


# ============== 4. researcher 的 ReAct 子图（藏进节点函数）==============

RESEARCHER_SYSTEM = """你是研究员。按给定的调研指令收集资料并产出研究材料：
1. 先用 web_search 搜索指令指定的维度；摘要不足以支撑结论时，用 web_fetch 读原文页面；
2. 紧扣调研指令的范围，指令没要求的不要展开；
3. 工具调用预算共 10 次（搜索与抓取合计），用完即止；
4. 收集完成后产出要点式研究材料：5-8 条要点，每条一句话讲清一个事实或论点，
   事实性要点在句末标注来源 URL（来自搜索结果或抓取的页面）。"""

MAX_TOOL_CALLS = 10  # 工具调用预算：prompt 软引导 + should_continue 硬熔断双层兜底


def _count_tool_calls(messages) -> int:
    """统计消息历史中已发起的 tool_calls 总数。"""
    return sum(len(m.tool_calls) for m in messages if getattr(m, "tool_calls", None))


def call_model(state: MessagesState):
    """子图的 agent 节点：带工具决策。打印 tool_calls 让"自主决定搜什么"可观察。"""
    resp = llm_with_tools.invoke(
        [SystemMessage(content=RESEARCHER_SYSTEM)] + list(state["messages"])
    )
    if resp.tool_calls:
        for tc in resp.tool_calls:
            print(f"  [researcher·agent] 发起工具调用：{tc['name']}({tc['args']})")
    else:
        print("  [researcher·agent] 不再调工具，产出最终研究材料")
    return {"messages": [resp]}


def finalize(state: MessagesState):
    """熔断收尾节点：不给工具，让模型基于已收集的资料产出研究材料。
    agent 的"自主收手"是概率行为——达预算上限时由结构接管（机制约束优于格式约束）。
    收尾指令追加在消息历史末尾（HumanMessage）：模型注意力对末尾最强，
    指令贴近生成位置，产出即为目标格式。"""
    resp = llm.invoke(
        list(state["messages"]) + [HumanMessage(content=(
            "工具调用预算已用完，不能再调用任何工具。请立刻基于上面已收集的全部资料"
            "输出最终研究材料：5-8 条要点，每条一句话讲清一个事实或论点，"
            "事实性要点在句末标注来源 URL。直接输出要点正文，"
            "禁止输出下一步计划、研究意向或任何与材料正文无关的内容。"
        ))]
    )
    return {"messages": [resp]}


def should_continue(state: MessagesState) -> str:
    """子图的条件边：没有 tool_calls → END（模型自主收手）；
    有 tool_calls 且未达预算 → tools；已达预算 → finalize（结构强制收手）。"""
    if not state["messages"][-1].tool_calls:
        return END
    if _count_tool_calls(state["messages"]) >= MAX_TOOL_CALLS:
        print(f"  [researcher·熔断] 工具调用已达预算 {MAX_TOOL_CALLS} 次，强制进入收尾")
        return "finalize"
    return "tools"


def _build_researcher_graph():
    """组装 researcher 的小 ReAct 图：agent ↔ tools，熔断 → finalize → END。"""
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", "finalize", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    return builder.compile()


researcher_graph = _build_researcher_graph()  # 模块级编译一次，反复 invoke


def researcher(state: TeamState) -> dict:
    """研究员 agent 节点：内部自主循环对主图透明——invoke 子图，取最终产出写回主图 State。
    MessagesState（子图消息列表）↔ TeamState（主图多字段）的 schema 翻译在此边界完成。"""
    result = researcher_graph.invoke(
        {"messages": [
            SystemMessage(content=RESEARCHER_SYSTEM),
            ("user", f"研究题目：{state['topic']}\n\n调研指令：{state['supervisor_note']}"),
        ]},
        config={"recursion_limit": 30},  # 子图步数上限：覆盖满预算 10 次工具调用（约 22 步）+ 余量
    )
    material = result["messages"][-1].content
    count = state.get("round_count", 0) + 1
    print(f"\n===== [researcher] 第 {count} 轮研究（含工具循环） =====")
    print(material)
    return {"research": material, "round_count": count}


# ============== 5. synthesizer（与 step1 完全相同：无搜索工具，只依据材料）==============

def synthesizer(state: TeamState) -> dict:
    """综合者：只依据研究材料作答，无搜索工具——从机制上杜绝编造来源。"""
    resp = llm.invoke([
        {"role": "system", "content": (
            "你是综合分析者。只依据提供的研究材料回答题目，禁止使用材料之外的事实，"
            "材料没覆盖的角度如实说明局限。按指令的要求组织答案。"
        )},
        {"role": "user", "content": (
            f"题目：{state['topic']}\n\n研究材料：\n{state['research']}"
            f"\n\n综合要求：{state['supervisor_note']}"
        )},
    ])
    print("\n===== [synthesizer] 答案 =====")
    print(resp.content)
    return {"answer": resp.content}


# ============== 6. supervisor（与 step1 完全相同：三次内容函数路由）==============

MAX_ROUNDS = 2  # 补查轮次熔断上限：达到后 supervisor 不得再派 researcher


class RouteDecision(BaseModel):
    """supervisor 的结构化输出 schema：
    next 是路由信号，directive 是随内容变化的给 worker 的指令，reason 留给运行日志。"""
    next: Literal["researcher", "synthesizer", "END"]
    reason: str = Field(description="一句话说明这次路由判断了什么、结论是什么")
    directive: str = Field(description=(
        "给下一个 worker 的执行指令。拆题时写调研指令（查什么维度、什么算齐了）；"
        "补查时写改写后的指令（依据材料内容写明补什么）；"
        "派综合时写综合要求；打回时写修改意见；END 时写无"
    ))


def supervisor(state: TeamState) -> dict:
    """决策脑：按当前所处阶段做内容判断（阶段标记是字段判断，阶段内是内容判断）。"""
    has_research = bool(state.get("research"))
    has_answer = bool(state.get("answer"))
    count = state.get("round_count", 0)

    if not has_research and not has_answer:
        stage = (
            "【阶段A·拆题】当前还没有任何研究产出。读懂题目，写出给 researcher 的调研指令："
            "调研哪几个维度、每个维度要什么信息、什么情况算资料齐了。"
        )
    elif not has_answer:
        stage = (
            f"【阶段B·判材料】已研究 {count}/{MAX_ROUNDS} 轮。读题目、调研指令和研究材料，"
            "判断材料是否足以回答题目、是否紧扣调研指令：\n"
            f"- 不足或偏题：若 {count} < {MAX_ROUNDS}，改写调研指令重派 researcher"
            "（指令必须依据你刚读到的材料内容，写明缺哪块、哪个维度要换角度）；\n"
            f"- 若 {count} >= {MAX_ROUNDS}，禁止再派 researcher，派 synthesizer 并在指令中"
            "注明材料局限，让综合者带局限作答；\n"
            "- 足够且切题：派 synthesizer，指令写明综合要求（结构、侧重点）。"
        )
    else:
        stage = (
            "【阶段C·终审】读题目和答案，判断答案是否直接回答了题目、"
            "是否忠于研究材料：\n"
            "- 到点：END；\n"
            "- 没答到点上或游离材料：打回 synthesizer，指令写明哪里没答到、怎么改。"
        )

    router = llm.with_structured_output(RouteDecision)
    decision = router.invoke([
        {"role": "system", "content": (
            "你是深度研究团队的 supervisor，只做决策不产出内容。你的每次路由都必须"
            "基于对内容的判断（题目/材料/答案写了什么），不是看字段有没有值。"
            "在 directive 中写清你基于内容做出的判断和给 worker 的具体指令。\n"
            + stage
        )},
        {"role": "user", "content": (
            f"题目：{state['topic']}\n"
            f"当前调研指令：{state.get('supervisor_note') or '（尚未拆题）'}\n"
            f"研究材料：\n{state.get('research') or '（还没有）'}\n"
            f"当前答案：\n{state.get('answer') or '（还没有）'}\n"
            f"已完成研究轮次：{count}/{MAX_ROUNDS}"
        )},
    ])
    print(f"\n>>> supervisor 决策：→ {decision.next}（{decision.reason}）")
    return {"next": decision.next, "supervisor_note": decision.directive}


def route(state: TeamState) -> str:
    """条件边函数：读路由信号分发。"""
    return state["next"]


# ============== 7. 组图（与 step1 相同：supervisor 居中，worker 回边）==============

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "supervisor")
builder.add_edge("researcher", "supervisor")
builder.add_edge("synthesizer", "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route,
    {"researcher": "researcher", "synthesizer": "synthesizer", "END": END},
)

graph = builder.compile()


# ============== 8. 运行 ==============

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "LangGraph 与 CrewAI 做多 agent 原型，选哪个更合适"
    print(f"研究题目：{topic}")
    result = graph.invoke(
        {"topic": topic, "round_count": 0},
        config={"recursion_limit": 25},  # 主图兜底；researcher 子图内部还有自己的 30
    )
    print("\n" + "=" * 60)
    print(f"最终 round_count：{result['round_count']}")
    print("=" * 60)
    print("\n[最终答案]\n")
    print(result["answer"])
