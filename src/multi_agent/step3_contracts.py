"""step3_contracts.py —— E3：成员契约固化 + 停止条件代码层固化 + Send API 并行多实例

在 step2 基础上升级三件事：
①成员契约集中化：每个成员的职责边界、输入输出 schema、停止条件集中定义（§0 契约块）——
  supervisor 按契约写工单、worker 按契约交产出，成员之间只通过 schema 交互；
②停止条件代码层固化：批次熔断从"写进 prompt 请求 LLM 遵守"升级为"决策返回后代码强制
  改写路由"——轮次比较是纯字段判断，能确定的交给代码，不信任概率性的 prompt 约束；
③Send API 并行多实例：supervisor 拆题产出 2 份维度工单，条件边用 Send fan-out 出
  2 个 researcher 实例并行执行——各自独立的 ReAct 循环、上下文、工具调用历史。
  agent 实例数 ≥2，系统自此达标 multi-agent（数实例，不数角色）。

并行带来的 State 适配（新语法的必然后果）：
    research: str → Annotated[list[str], merge_research]——并行实例同时写同一字段，
    默认覆盖语义会互相吞掉，累加 reducer 是并行写的前提；
    round_count：单个 researcher 自增 → supervisor 派发批次时自增（N 个实例算一批）。

图结构：
    主图：supervisor ──route──▶ [Send×2] researcher（并行） ─▶ supervisor ─▶ … ─▶ END
          （补查时 route 返回 [Send×1]，单实例补查 = 长度为 1 的工单列表）
    researcher 内部（与 step2 相同）：agent ↔ tools，预算熔断 → finalize → END

与 step2 的对照验收：同一道默认题，step2 单 researcher 串行调研（一份材料）；
step3 拆题后 2 个 researcher 并行各查一个维度，supervisor 判材料面对的是合并后的材料册。

跑法（项目根）：
  uv run python src/multi_agent/step3_contracts.py "自定义研究题目"
  uv run python src/multi_agent/step3_contracts.py  # 默认题目（与 step1/step2 同题对照）
"""
import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from pydantic import BaseModel, Field

from my_agent.agent import web_fetch as _web_fetch, web_search as _web_search

load_dotenv()


# ============== 0. 成员契约：职责边界 / 输入输出 schema / 停止条件 ==============
#
# 成员契约集中定义在此，节点函数是实现，必须与契约一致——契约先行是协调的前提：
# supervisor 按契约写工单，worker 按契约交产出，成员之间只通过 schema 交互。
#
# supervisor（编排者，1 个实例）
#   职责：拆题 / 判材料 / 终审，只决策不产出内容
#   输入：TeamState 只读
#   输出：next + plan（派发时的工单列表）+ supervisor_note（单 worker 指令）+ round_count
#   停止条件：批次熔断 MAX_ROUNDS=2——代码层强制（见 supervisor 内注释），prompt 规则只作引导
#
# researcher（agent，并行 N 个实例）
#   职责：按工单调研一个维度（内部 ReAct 子图与 step2 相同）
#   输入：工单 dict {topic, dimension, instruction}——自包含，不读主图 State
#   输出：{"research": [material]}（单元素列表，经 merge_research 累加合并）
#   停止条件：工具预算 MAX_TOOL_CALLS=10（prompt 软引导 + should_continue 硬熔断 → finalize）
#
# synthesizer（工序，1 个实例）
#   职责：只依据材料册作答，无搜索工具（机制上杜绝编造来源）
#   输入：TeamState 只读（research 列表拼成材料册）
#   输出：{"answer": str}
#
# 框架兜底：主图 recursion_limit 与子图 recursion_limit（各层独立计数）


# ============== 1. 共享 State：research 改累加 reducer，适配并行写 ==============

def merge_research(current: list[str] | None, update: list[str]) -> list[str]:
    """research 字段的累加 reducer：并行实例的产出按到达顺序合并而非覆盖。
    reducer 签名 (current, update) -> new；current 为 None 时从空列表起步。"""
    return (current or []) + update

# Annotated 是类型旁注，StateGraph(schema) 建图时，会把Annotated的第二个参数注册为该字段的reducer
# 图State的每个字段都有reducer，没有Annotated标注的字段，用的是隐式默认reducer覆盖（new = update，丢弃旧值）
# 每次节点return的dict会按照reducer规则，逐字段合并进State。如果是并行任务，更新需要经过reducer累积，否则前面任务的结果将会丢失
class TeamState(TypedDict):
    topic: str            # 研究题目（入口输入）
    plan: list[dict]      # supervisor 写的工单列表（每份工单派生一个 researcher 实例）
    supervisor_note: str  # supervisor 给单 worker 的当前指令（判材料/综合/打回用）
    research: Annotated[list[str], merge_research]  # 各实例产出的材料（并行累加合并）
    answer: str           # synthesizer 产出的答案
    round_count: int      # 已派发的研究批次（熔断计数器，supervisor 派发时自增）
    next: str             # supervisor 决定的下一节点（路由信号）


# ============== 2. 模型：复用 B2 的 ChatOpenAI 接法 ==============

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=120,
    max_retries=2,
)


# ============== 3. 联网工具：@tool 包装，只挂给 researcher ==============

@tool
def web_search(query: str) -> str:
    """搜索网页，返回最多 5 条结果的标题、URL 和摘要。用于获取研究题目的资料来源。"""
    return _web_search(query)


@tool
def web_fetch(url: str) -> str:
    """抓取 URL 网页正文，返回纯文本（超长自动截断）。用于读取搜索结果中值得深挖的页面。"""
    return _web_fetch(url)


tools = [web_search, web_fetch]
llm_with_tools = llm.bind_tools(tools)


# ============== 4. researcher 的 ReAct 子图（与 step2 完全相同）==============

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


# ============== 5. researcher 节点：Send 派生的并行实例入口 ==============

def researcher(task: dict) -> dict:
    """研究员 agent 节点：每个 Send 工单派生一个独立实例（独立循环/上下文/工具历史）。
    输入不是 TeamState，而是 Send 携带的工单（契约见 §0）——工单自包含，
    实例不读主图 State。返回单元素列表，经 merge_research 与其他实例累加合并。"""
    result = researcher_graph.invoke(
        {"messages": [
            SystemMessage(content=RESEARCHER_SYSTEM),
            ("user", f"研究题目：{task['topic']}\n\n"
                     f"调研维度：{task['dimension']}\n调研指令：{task['instruction']}"),
        ]},
        config={"recursion_limit": 30},  # 子图步数上限：覆盖满预算 10 次工具调用（约 22 步）+ 余量
    )
    material = result["messages"][-1].content
    print(f"\n===== [researcher·{task['dimension']}] 产出材料 =====")
    print(material)
    return {"research": [material]}


# ============== 6. synthesizer（工序，读材料册）==============

def _materials_text(research: list[str]) -> str:
    """材料册：并行实例的产出列表拼成带编号的文本。"""
    return "\n\n".join(f"【材料 {i}】\n{m}" for i, m in enumerate(research, 1))


def synthesizer(state: TeamState) -> dict:
    """综合者：只依据材料册作答，无搜索工具——从机制上杜绝编造来源。
    材料分属不同调研维度，综合时按维度组织、跨维度提炼。"""
    resp = llm.invoke([
        {"role": "system", "content": (
            "你是综合分析者。只依据提供的研究材料回答题目，禁止使用材料之外的事实，"
            "材料没覆盖的角度如实说明局限。材料分属不同调研维度，"
            "综合时按维度组织、跨维度提炼。按指令的要求组织答案。"
        )},
        {"role": "user", "content": (
            f"题目：{state['topic']}\n\n研究材料：\n{_materials_text(state['research'])}"
            f"\n\n综合要求：{state.get('supervisor_note') or '（无）'}"
        )},
    ])
    print("\n===== [synthesizer] 答案 =====")
    print(resp.content)
    return {"answer": resp.content}


# ============== 7. supervisor：契约输出 + 批次熔断的代码层固化 ==============

MAX_ROUNDS = 2  # 批次熔断上限：达到后代码强制禁止再派 researcher（不再依赖 LLM 自觉）


class DimensionTask(BaseModel):
    """researcher 的派发工单：一个并行实例的完整输入（自包含）。"""
    dimension: str = Field(description="维度名（两三个词概括该维度查什么）")
    instruction: str = Field(description="该维度的调研指令：查什么、什么算资料齐")


class PlanDecision(BaseModel):
    """拆题阶段的结构化输出 schema：把题目拆成 2 份并行工单。"""
    reason: str = Field(description="一句话说明拆题思路")
    tasks: list[DimensionTask] = Field(description=(
        "恰好 2 个调研维度的工单：两维度互不重叠，合并后覆盖回答题目所需的信息面"
    ))


class RouteDecision(BaseModel):
    """判材料/终审阶段的结构化输出 schema（与 step1/step2 相同）。"""
    next: Literal["researcher", "synthesizer", "END"]
    reason: str = Field(description="一句话说明这次路由判断了什么、结论是什么")
    directive: str = Field(description=(
        "给下一个 worker 的执行指令。补查时写改写后的指令（依据材料内容写明补什么）；"
        "派综合时写综合要求；打回时写修改意见；END 时写无"
    ))


def supervisor(state: TeamState) -> dict:
    """编排者：阶段判断是字段判断（代码 if/else），阶段内的决策是内容判断（LLM）。
    阶段A拆题输出工单列表（并行派发的数据前提）；阶段B/C 用 RouteDecision 路由；
    批次熔断在 LLM 决策返回后由代码强制执行（见下）。"""
    has_research = bool(state.get("research"))
    has_answer = bool(state.get("answer"))
    count = state.get("round_count", 0)

    if not has_research and not has_answer:
        # ---- 阶段A·拆题：结构化输出 2 份维度工单 ----
        planner = llm.with_structured_output(PlanDecision)
        plan = planner.invoke([
            {"role": "system", "content": (
                "你是深度研究团队的 supervisor，正在拆题。把题目拆成 2 个可并行调研的维度："
                "两维度互不重叠，合起来覆盖回答题目所需的信息面。每个维度写一份独立工单。"
            )},
            {"role": "user", "content": f"题目：{state['topic']}"},
        ])
        # 工单自包含：topic 由代码补齐，实例不依赖主图 State
        tickets = [
            {"topic": state["topic"], "dimension": t.dimension, "instruction": t.instruction}
            for t in plan.tasks
        ]
        print(f">>> supervisor 拆题：{[t['dimension'] for t in tickets]}（{plan.reason}）")
        # 派发即计一批次：N 个并行实例算一批，round_count 由 supervisor 统一维护
        return {"next": "researcher", "plan": tickets, "round_count": count + 1}

    # ---- 阶段B·判材料 / 阶段C·终审：LLM 路由决策 ----
    if not has_answer:
        stage = (
            f"【阶段B·判材料】已研究 {count}/{MAX_ROUNDS} 批。读题目和各维度材料，"
            "判断合并后的材料是否足以回答题目：\n"
            f"- 不足：若 {count} < {MAX_ROUNDS}，派 researcher 补查"
            "（directive 依据材料内容写明缺哪块、要补什么角度）；\n"
            f"- 若 {count} >= {MAX_ROUNDS}，派 synthesizer 并在指令中注明材料局限；\n"
            "- 足够：派 synthesizer，指令写明综合要求（结构、侧重点）。"
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
            f"当前指令：{state.get('supervisor_note') or '（无）'}\n"
            f"研究材料：\n{_materials_text(state.get('research') or [])}\n"
            f"当前答案：\n{state.get('answer') or '（还没有）'}\n"
            f"已完成研究批次：{count}/{MAX_ROUNDS}"
        )},
    ])

    # ---- 停止条件代码层固化 ----
    # 轮次比较是纯字段判断（"能确定的交给代码"）：达到上限后禁止再派 researcher，
    # 由代码直接改写路由——prompt 规则是概率性约束，代码改写是确定性防线。
    if decision.next == "researcher" and count >= MAX_ROUNDS:
        print(f">>> [批次熔断] 已达 {MAX_ROUNDS} 批上限，代码强制改派 synthesizer（带局限收尾）")
        return {
            "next": "synthesizer",
            "supervisor_note": (
                "研究批次已达上限，材料可能存在缺口。请基于现有材料综合作答，"
                "材料没覆盖的角度如实说明局限，不得编造。"
            ),
        }

    print(f">>> supervisor 决策：→ {decision.next}（{decision.reason}）")

    updates: dict = {"next": decision.next, "supervisor_note": decision.directive}
    if decision.next == "researcher":
        # 补查派发：改写后的指令组装成单份工单（统一派发协议，工单列表长度为 1）
        updates["plan"] = [{
            "topic": state["topic"],
            "dimension": "补充调研",
            "instruction": decision.directive,
        }]
        updates["round_count"] = count + 1
    return updates


# ============== 8. route 条件边：researcher 派发转 Send 列表 ==============

def route(state: TeamState):
    """条件边：next 为 researcher 时把工单列表转成 Send 列表（fan-out 并行），
    LangGraph 为每个 Send 起一个独立实例、同一批次并行执行；
    其余路由信号原样返回，按 mapping 查表分发。"""
    if state["next"] == "researcher":
        # Send(node_name, payload) 给调度器的派工指令，解决的是运行时动态派生并行任务。以 node_name 为目标节点、payload 为目标节点函数的输入，起一个独立任务
        # Send派发方式，只改变输入，不改变输出。实例的返回值仍然走标准路径-- return {"research": [material]}，按字段的reducer合并进主图State
        return [Send("researcher", task) for task in state["plan"]]
    return state["next"]


# ============== 9. 组图：supervisor 居中，worker 回边 ==============

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


# ============== 10. 运行 ==============

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "LangGraph 与 CrewAI 做多 agent 原型，选哪个更合适"
    print(f"研究题目：{topic}")
    result = graph.invoke(
        {"topic": topic, "round_count": 0},
        config={"recursion_limit": 25},  # 主图兜底；researcher 子图内部还有自己的 30
    )
    print("\n" + "=" * 60)
    print(f"最终研究批次：{result['round_count']}，材料册 {len(result['research'])} 份")
    print("=" * 60)
    print("\n[最终答案]\n")
    print(result["answer"])
