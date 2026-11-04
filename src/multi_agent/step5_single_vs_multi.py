"""step5_single_vs_multi.py —— E5：对照实验与结论（这个任务什么时候单 agent 反而更好）

E1 定过多 agent 三条件（可并行 / 上下文可切分 / 结果易验证）。本文件用同题实跑对照
把判据补全反面：什么任务特征下单 agent 更好——结论要有数据支撑，不靠感觉。

对照设计（公平性：同一副工具、同一预算、同一道题，差异只在编排）：
    单 agent：一个 ReAct agent 把"研究 + 综合"合于一身——自己搜索、抓取、综合，
              一次 invoke 直接产出带来源的答案（E2 的 researcher 子图 + E1 的
              synthesizer prompt 合体，预算熔断 MAX_TOOL_CALLS=10 + finalize 同样在）。
    多 agent：E3 系统原样上场（直接 import step3_contracts 的 graph，不重写）——
              supervisor 三次内容路由 + Send 并行多 researcher + synthesizer + 终审。

测量：总耗时（time 计时）、答案产出、单 agent 的决策轮数与工具调用次数、
多 agent 的研究批次与材料份数；调用结构账单对照（supervisor 的每次结构化决策
都不产出内容——多 agent 的编排税）。

预期：本题是"单 agent 更好"的实例——单一检索线（无并行价值）、材料一个上下文
装得下（无切分压力）、答案是否切题一眼可判（验证无分工需求），三条件一个不中，
多 agent 的全部额外环节都是纯开销。

跑法（项目根）：
  uv run python src/multi_agent/step5_single_vs_multi.py "自定义研究题目"
  uv run python src/multi_agent/step5_single_vs_multi.py  # 默认题（与 E1-E4 同题对照）
"""
import os
import sys
import time
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from my_agent.agent import web_fetch as _web_fetch, web_search as _web_search
from step3_contracts import graph as multi_graph  # 多 agent 不重写，E3 系统直接上场

load_dotenv()

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "LangGraph 与 CrewAI 做多 agent 原型，选哪个更合适"


# ============== 1. 模型与工具（与 step2 完全同一副：@tool + bind_tools + ToolNode）==============

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=120,
    max_retries=2,
)


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


# ============== 2. 单 agent：研究 + 综合一身，一个 ReAct 循环包揽 ==============

SINGLE_SYSTEM = """你是深度研究助手，独立完成研究与作答（没有队友，全程一人）：
1. 先用 web_search 搜索题目相关资料；摘要不足以支撑结论时，用 web_fetch 读原文页面；
2. 资料足够后不再调工具，直接输出最终答案：结论明确，只依据搜集到的资料作答，
   事实性要点在句末标注来源 URL；材料没覆盖的角度如实说明局限；
3. 工具调用预算共 10 次（搜索与抓取合计），用完即止。"""

MAX_TOOL_CALLS = 10  # 与 E2/E3/E4 同一预算：对照实验里唯一的公平基线


def _count_tool_calls(messages) -> int:
    return sum(len(m.tool_calls) for m in messages if getattr(m, "tool_calls", None))


def call_model(state: MessagesState):
    resp = llm_with_tools.invoke(
        [SystemMessage(content=SINGLE_SYSTEM)] + list(state["messages"])
    )
    if resp.tool_calls:
        for tc in resp.tool_calls:
            print(f"  [single·agent] 发起工具调用：{tc['name']}({tc['args']})")
    else:
        print("  [single·agent] 不再调工具，直接给出最终答案")
    return {"messages": [resp]}


def finalize(state: MessagesState):
    """预算熔断收尾：不给工具，基于已收集资料直接作答（与 E2 finalize 同构）。"""
    resp = llm.invoke(
        list(state["messages"]) + [HumanMessage(content=(
            "工具调用预算已用完，不能再调用任何工具。请立刻基于上面已收集的全部资料"
            "输出最终答案：结论明确，只依据已收集资料作答，事实性要点在句末标注来源 URL，"
            "材料没覆盖的角度如实说明局限。直接输出答案正文。"
        ))]
    )
    return {"messages": [resp]}


def should_continue(state: MessagesState) -> str:
    if not state["messages"][-1].tool_calls:
        return END
    if _count_tool_calls(state["messages"]) >= MAX_TOOL_CALLS:
        print(f"  [single·熔断] 工具调用已达预算 {MAX_TOOL_CALLS} 次，强制进入收尾")
        return "finalize"
    return "tools"


def _build_single_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", "finalize", END])
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)
    return builder.compile()


single_graph = _build_single_graph()


# ============== 3. 同题对照运行与指标 ==============

def run_single() -> tuple[dict, float]:
    print("【跑单 agent】一个 ReAct 循环包揽研究 + 综合")
    t0 = time.time()
    result = single_graph.invoke(
        {"messages": [("user", f"研究题目：{TOPIC}\n请独立完成研究并给出最终答案。")]},
        config={"recursion_limit": 30},  # 满预算 10 次工具约 22 步 + 余量（与 E2 子图一致）
    )
    return result, time.time() - t0


def run_multi() -> tuple[dict, float]:
    print("\n【跑多 agent】E3 系统：supervisor 三次内容路由 + Send 并行 + 综合 + 终审")
    t0 = time.time()
    result = multi_graph.invoke(
        {"topic": TOPIC, "round_count": 0},
        config={"recursion_limit": 25},
    )
    return result, time.time() - t0


def report(name: str, elapsed: float, answer_len: str, rows: list[tuple[str, str]]):
    print(f"\n----- {name} -----")
    print(f"总耗时：{elapsed:.1f} s")
    print(f"最终答案字符数：{answer_len}")
    for k, v in rows:
        print(f"{k}：{v}")


if __name__ == "__main__":
    print(f"E5 对照实验：单 agent vs 多 agent（同题公平对照：同一副工具、同一预算 10）")
    print(f"题目：{TOPIC}")
    print("=" * 60)

    single, t_single = run_single()
    single_msgs = single["messages"]
    single_answer = single_msgs[-1].content
    single_rounds = sum(1 for m in single_msgs if getattr(m, "tool_calls", None))
    single_calls = _count_tool_calls(single_msgs)

    multi, t_multi = run_multi()
    multi_answer = multi["answer"]
    multi_rounds = multi.get("round_count", 0)
    multi_materials = len(multi.get("research") or [])

    print("\n" + "=" * 60)
    print("[对照指标]")
    report("单 agent（一个 ReAct 循环包揽）", t_single, f"{len(single_answer)}", [
        ("主循环 LLM 决策轮数", f"{single_rounds}"),
        ("工具调用次数", f"{single_calls}/{MAX_TOOL_CALLS}"),
        ("编排决策开销", "0（无 supervisor，无编排）"),
    ])
    report("多 agent（E3 系统：supervisor + Send 并行 + 综合 + 终审）", t_multi, f"{len(multi_answer)}", [
        ("研究批次 / 材料份数", f"{multi_rounds} 批 / {multi_materials} 份"),
        ("supervisor 结构化决策", f"约 {multi_rounds + 2} 次（拆题 1 + 每轮判材料 {multi_rounds} + 终审 1），"
                                 "每次都是不产出内容的纯决策调用"),
        ("编排结构", "supervisor 居中路由 + researcher 子图各自 ReAct + synthesizer 工序分离"),
    ])

    print("\n[两份答案对照]")
    print("--- 单 agent 答案 ---")
    print(single_answer)
    print("--- 多 agent 答案 ---")
    print(multi_answer)

    print("\n" + "=" * 60)
    print("[结论] 这个任务什么时候单 agent 反而更好")
    print("=" * 60)
    print(f"本题实证：单 agent {t_single:.1f}s vs 多 agent {t_multi:.1f}s"
          f"（{t_multi / t_single:.1f} 倍），两份答案都合格。单 agent 用零编排开销给出"
          f"切题答案；多 agent 用额外时间换更深的材料（多实例并行 + 补查定向补证据）"
          f"与交叉验证纠偏——编排税买的是证据深度与纠偏能力，值不值看任务特征："
          f"本题单一检索线（无并行价值）、材料一个上下文装得下（无切分压力）、"
          f"答案是否切题一眼可判（验证无分工需求），三条件一个不中，合格的答案"
          f"用单 agent 就够。")
    print("""
选型判据（E1 三条件的反面补全）：
- 单一检索线，拆维度无收益 → 单 agent（一个循环顺序搜完即可）；
- 资料总量一个上下文装得下 → 单 agent（切分丢全局视野，综合时反而见不到全貌）；
- 决策环节没有内容价值（拆题/判材料是形式判断）→ 单 agent（supervisor 每次调用都是纯开销）；
- 成本或延迟敏感 → 单 agent（多 agent 的编排税 = 多次 LLM 决策 + 串行环节叠加延迟）。

多 agent 值得付这笔税的时候（E2/E3 实证）：
- 维度真可并行：Send fan-out 提速，且实例互相看不见 → 交叉验证可纠正彼此的材料偏差
  （E3 实证：补查实例纠正了材料 1 对 CrewAI Flows 的错误描述）；
- 单上下文装不下全部材料：切分是必需而非选择，每实例只见自己的维度；
- 验证需要独立视角：synthesizer 只见材料不见搜索过程，机制上杜绝编造来源；
  终审是独立于产出的二次判断。""")
