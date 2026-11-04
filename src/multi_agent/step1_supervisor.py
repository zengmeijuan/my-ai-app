"""step1_supervisor.py —— E1：最小动态深度研究系统（supervisor 内容函数路由）

场景：深度研究是多 agent 三条件（可并行 / 上下文可切分 / 结果易验证）全中的任务。
与 step0_fixed_workflow.py 的本质区别：supervisor 的路由决策是**内容的函数**——
拆题要读懂题目、判材料要读懂资料、终审要读懂答案，if/else 写不出这些判断。
执行路径不预先可穷举：有的题一查就够，有的题要改写指令补查——路径由内容走出来。

三角色（按"产出的东西"分，不按"流程步骤"分）：
    supervisor   产决策：拆搜索指令 / 判材料够不够（改写指令重派，上限 2 轮）/ 判答案到点没
    researcher   产材料：按 supervisor 的指令调研（E1 无工具，E2 接入 web_search/web_fetch）
    synthesizer  产答案：只依据研究材料综合作答（无搜索工具，防编造的机制设计）

图结构（supervisor 居中，worker 回边）：

                 ┌──────────────────────────────────┐
                 │  supervisor（读内容做三次路由决策）    │
                 └──────────────────────────────────┘
                   ↓ 派指令              ↑ 收材料/收答案
             researcher ──────────── synthesizer
             （按指令查资料）          （按指令综合）

supervisor 的载体：路由决策 = 选节点 + 写指令（directive 随内容变）——
指令字段就是"传话筒 supervisor"升级为"决策脑 supervisor"的机制。

跑法（项目根）：
  uv run python src/multi_agent/step1_supervisor.py "自定义研究题目"
  uv run python src/multi_agent/step1_supervisor.py  # 默认题目
"""
import os
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


# ============== 1. 共享 State：节点间唯一通信介质 ==============

class TeamState(TypedDict):
    topic: str            # 研究题目（入口输入）
    supervisor_note: str  # supervisor 给下一个 worker 的当前指令（拆题指令/补查指令/综合要求/打回意见）
    research: str         # researcher 产出的研究材料（重派时覆盖追加）
    answer: str           # synthesizer 产出的答案
    round_count: int      # 已执行的研究轮次（熔断计数器）
    next: str             # supervisor 决定的下一节点（路由信号）


# ============== 2. 模型：复用 B2 的 ChatOpenAI 接法 ==============

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=120,
    max_retries=2,
)


# ============== 3. 结构化 schema ==============

class RouteDecision(BaseModel):
    """supervisor 的结构化输出 schema：
    next 是路由信号，directive 是随内容变化的给 worker 的指令（决策脑的载体），
    reason 留给运行日志（让人看懂每次路由为什么这么走）。"""
    next: Literal["researcher", "synthesizer", "END"]
    reason: str = Field(description="一句话说明这次路由判断了什么、结论是什么")
    directive: str = Field(description=(
        "给下一个 worker 的执行指令。拆题时写调研指令（查什么维度、什么算齐了）；"
        "补查时写改写后的指令（依据材料内容写明补什么）；"
        "派综合时写综合要求；打回时写修改意见；END 时写无"
    ))


# ============== 4. 两个 worker 节点 ==============

def researcher(state: TeamState) -> dict:
    """研究员：按 supervisor 的指令调研。E1 无工具，靠模型自带知识（E2 接入联网工具）。"""
    resp = llm.invoke([
        {"role": "system", "content": (
            "你是研究员。按给定的调研指令产出要点式研究材料：5-8 条要点，"
            "每条一句话讲清一个事实或论点。紧扣调研指令的范围，指令没要求的不要展开。"
            "只输出要点本身，不写客套话。"
        )},
        {"role": "user", "content": (
            f"研究题目：{state['topic']}\n\n调研指令：{state['supervisor_note']}"
        )},
    ])
    count = state.get("round_count", 0) + 1
    print(f"\n===== [researcher] 第 {count} 轮研究 =====")
    print(resp.content)
    return {"research": resp.content, "round_count": count}


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


# ============== 5. supervisor 节点：三次内容函数路由 ==============

MAX_ROUNDS = 2  # 补查轮次熔断上限：达到后 supervisor 不得再派 researcher


def supervisor(state: TeamState) -> dict:
    """决策脑：按当前所处阶段做内容判断。阶段由 State 已有字段标记（字段判断），
    阶段内的决策是内容判断（读懂题目/材料/答案才能做）：
      A 拆题：无 research 无 answer → 读题写调研指令 → researcher
      B 判材料：有 research 无 answer → 读材料判够不够、偏没偏
         不够/偏题且未到熔断上限 → 改写指令重派 researcher
         够 → 派 synthesizer（附综合要求）
         不够但已到熔断上限 → 仍派 synthesizer，指令注明材料局限（带局限收尾）
      C 终审：有 answer → 读答案判是否直接回答了题目
         到点 → END；没到点 → 打回 synthesizer（附修改意见）
    """
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


# ============== 6. 组图：supervisor 居中，worker 回边 ==============

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


# ============== 7. 运行 ==============

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "LangGraph 与 CrewAI 做多 agent 原型，选哪个更合适"
    print(f"研究题目：{topic}")
    result = graph.invoke(
        {"topic": topic, "round_count": 0},
        config={"recursion_limit": 25},  # 双保险：LLM 决策失控时由框架兜底
    )
    print("\n" + "=" * 60)
    print(f"最终 round_count：{result['round_count']}")
    print("=" * 60)
    print("\n[最终答案]\n")
    print(result["answer"])
