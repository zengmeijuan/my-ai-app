"""step4_failure_lab.py —— E4：协调问题实验（循环故障的注入 → 观察 → 修复对照）

多 agent 的协调问题在正常流程里看不见——成员靠 prompt 规则与路由信号配合，
配合失灵是概率行为，跑了才知道。本文件用**故障注入**把概率故障变成确定性故障，
完整演示"现象 → 根因 → 机制修复 → 对照验证"的处理过程。

实验对象：supervisor（终审）⇄ synthesizer（重写）这条边——step3 真实存在的熔断
缺口：阶段C的打回没有次数上限（MAX_ROUNDS 只熔断研究批次，不管打回次数），唯一
防线是框架 recursion_limit。本文件裁剪至这条边，聚焦循环这一个协调故障。

实验设计（单一变量对照）：
    共同点：故障注入器——苛刻终审：无论终审 LLM 判什么，代码强制改写为打回，
            supervisor ↔ synthesizer 循环被确定性触发；
    唯一变量：有无打回熔断（review_count 计数 + 代码层强制 END）。

    故障组（默认）：注入 + 无熔断 → 无限打回循环 → recursion_limit 兜底
                    → GraphRecursionError 被捕获展示（框架是最后防线；
                      兜底被触发 = 业务熔断设计有漏洞）
    修复组（--fixed）：同一注入 + 打回熔断 MAX_REVISIONS=2 → 2 次打回后代码
                    强制 END，答案保留、带保留意见收尾（优雅降级）

注入的中间态（State 可从任意中间态启动——手工模拟 checkpoint 恢复的场景）：
    跳过研究阶段，直接注入 topic + 一份材料 + 一份偏题答案，round_count=2，
    supervisor 第一步就落在终审。

观测：每次调用打印输入字符数与打回轮次——主图 State 是覆盖语义不累积消息，
循环中输入长度近似恒定；真正的上下文膨胀发生在 ReAct 子图的消息历史
（工具结果逐轮累积）与长会话场景，各有对策（预算熔断 / 上下文压缩）。

跑法（项目根）：
  uv run python src/multi_agent/step4_failure_lab.py            # 故障组
  uv run python src/multi_agent/step4_failure_lab.py --fixed    # 修复组
"""
import os
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

FIXED = "--fixed" in sys.argv  # 修复组开关：--fixed 打开打回熔断；默认故障组


# ============== 0. 成员契约（E4 简版：循环实验只涉及两个成员）==============
#
# supervisor（终审）：读答案判是否到点。本实验中终审被注入为"苛刻终审"，
#   由代码强制打回——注入器与熔断正交：注入提供必然打回，熔断提供停止条件。
# synthesizer（工序）：按打回意见重写答案。输入 State 只读，输出 {"answer": str}。
# 停止条件：故障组仅框架兜底 recursion_limit；修复组加打回熔断 MAX_REVISIONS=2
#   （打回计数是字段判断——能确定的交给代码，不信任 LLM 遵守）。


# ============== 1. State（step3 全字段 + review_count）==============

def merge_research(current: list[str] | None, update: list[str]) -> list[str]:
    """research 字段的累加 reducer（本实验不再更新它，保留以兼容中间态注入）。"""
    return (current or []) + update


class TeamState(TypedDict):
    topic: str            # 研究题目
    plan: list[dict]      # 工单列表（保留字段，本实验不使用）
    supervisor_note: str  # supervisor 给 worker 的当前指令（打回意见）
    research: Annotated[list[str], merge_research]  # 材料册（中间态注入 1 份）
    answer: str           # synthesizer 产出的答案（重写时覆盖）
    round_count: int      # 研究批次计数（中间态注入为 2，跳过研究阶段）
    review_count: int     # 终审打回计数（E4 新增：循环实验的熔断计数器）
    next: str             # supervisor 决定的下一节点（路由信号）


# ============== 2. 模型：复用 B2 的 ChatOpenAI 接法 ==============

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=120,
    max_retries=2,
)


# ============== 3. synthesizer：循环的另一半 ==============

def _materials_text(research: list[str]) -> str:
    """材料册：列表拼成带编号的文本。"""
    return "\n\n".join(f"【材料 {i}】\n{m}" for i, m in enumerate(research, 1))


def synthesizer(state: TeamState) -> dict:
    """重写工序：按打回意见修改答案。它自身没有停止手段——停止权在 supervisor
    （业务熔断）与框架（recursion_limit），这是循环故障的结构根源。"""
    user_content = (
        f"题目：{state['topic']}\n\n"
        f"研究材料：\n{_materials_text(state.get('research') or [])}"
        f"\n\n当前答案：\n{state['answer']}"
        f"\n\n修改意见：{state['supervisor_note']}"
    )
    print(f">>> [观测] synthesizer 输入长度：{len(user_content)} 字符")
    resp = llm.invoke([
        {"role": "system", "content": (
            "你是综合分析者，正在按终审意见修改答案。只依据研究材料回答题目，"
            "禁止使用材料之外的事实。直接输出修改后的完整答案。"
        )},
        {"role": "user", "content": user_content},
    ])
    print(f"\n===== [synthesizer] 第 {state.get('review_count', 0)} 次重写 =====")
    print(resp.content)
    return {"answer": resp.content}


# ============== 4. supervisor：终审 LLM + 故障注入器 + 打回熔断 ==============

MAX_REVISIONS = 2  # 打回熔断上限（修复组）：达到后代码强制 END，不再依赖 LLM 自觉

HARSH_DIRECTIVE = (
    "答案不合格：论证不充分、与题目契合度低。必须重写答案，"
    "直接回答题目，依据研究材料给出明确结论与理由。"
)


class RouteDecision(BaseModel):
    """终审的结构化输出 schema：候选目标与图拓扑一致（本图没有 researcher）。"""
    next: Literal["synthesizer", "END"]
    reason: str = Field(description="一句话说明这次判断了什么、结论是什么")
    directive: str = Field(description="打回时写修改意见（哪里没答到、怎么改）；END 时写无")


def supervisor(state: TeamState) -> dict:
    """终审者。三个层次在此叠加：
    ①LLM 真实终审——判断与意见可观察（决策先产生）；
    ②故障注入器——苛刻终审：无论 LLM 判什么，代码强制改写为打回
      （概率故障变确定性故障，精确控制路由行为）；
    ③打回熔断（修复组）——计数达上限后代码强制 END，业务层接管收尾
      （决策后代码做最后裁决，与 E3 批次熔断同构）。"""
    router = llm.with_structured_output(RouteDecision)
    decision = router.invoke([
        {"role": "system", "content": (
            "你是深度研究团队的 supervisor，正在终审。读题目和答案，"
            "判断答案是否直接回答了题目、是否忠于研究材料：\n"
            "- 到点：END；\n"
            "- 没答到点上或游离材料：打回 synthesizer，指令写明哪里没答到、怎么改。"
        )},
        {"role": "user", "content": (
            f"题目：{state['topic']}\n"
            f"研究材料：\n{_materials_text(state.get('research') or [])}\n"
            f"当前答案：\n{state['answer']}\n"
            f"已打回次数：{state.get('review_count', 0)}/{MAX_REVISIONS}"
        )},
    ])
    print(f"\n>>> 终审 LLM 判定 → {decision.next}（{decision.reason}）")

    # ---- 故障注入器（两组共用）：苛刻终审，强制打回 ----
    rc = state.get("review_count", 0) + 1
    if FIXED and rc > MAX_REVISIONS:
        # ---- 打回熔断（修复组专属）：业务层接管收尾 ----
        print(f">>> [打回熔断] 打回已达上限 {MAX_REVISIONS} 次，代码强制 END（带保留意见收尾）")
        return {"next": "END", "review_count": rc}
    print(f">>> [故障注入] 强制改写为打回（第 {rc} 次）——概率故障变确定性故障")
    return {"next": "synthesizer", "supervisor_note": HARSH_DIRECTIVE, "review_count": rc}


def route(state: TeamState) -> str:
    """条件边函数：读路由信号分发。"""
    return state["next"]


# ============== 5. 组图：supervisor ⇄ synthesizer 循环 ==============

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "supervisor")
builder.add_edge("synthesizer", "supervisor")
builder.add_conditional_edges("supervisor", route, {"synthesizer": "synthesizer", "END": END})

graph = builder.compile()


# ============== 6. 运行：中间态注入 + 两组对照 ==============

if __name__ == "__main__":
    mode = "修复组（打回熔断 MAX_REVISIONS=2 生效）" if FIXED else "故障组（无打回熔断，仅框架兜底）"
    print(f"E4 协调问题实验——循环注入 | 模式：{mode}")
    print("=" * 60)

    # 中间态注入：跳过研究阶段，supervisor 第一步即落终审。等价于手工模拟checkpoint恢复
    init_state = {
        "topic": "LangGraph 与 CrewAI 做多 agent 原型，选哪个更合适",
        "plan": [],
        "supervisor_note": "",
        "research": ["【注入材料】LangGraph 是低层级图编排框架，以状态图为中心，"
                     "控制粒度细、学习成本高；CrewAI 是角色化多 agent 框架，"
                     "上手快、抽象少。两者均活跃维护、生态成熟。"],
        "answer": "【注入答案】今天天气不错，适合外出散步。多agent框架的选择取决于团队喜好，建议多尝试几个。",
        "round_count": 2,
        "review_count": 0,
        "next": "",
    }
    print("注入中间态：材料 1 份 + 偏题答案，round_count=2 → supervisor 直接进入终审")

    # recursion_limit: LangGraph的运行时步数上线，耗尽则抛GraphRecursionError
    try:
        result = graph.invoke(init_state, config={"recursion_limit": 25})
    except GraphRecursionError:
        print("\n" + "=" * 60)
        print(">>> 框架兜底被触发：GraphRecursionError（recursion_limit=25 耗尽）")
        print(">>> 现象：supervisor ↔ synthesizer 无限打回循环，25 步内未到 END。")
        print(">>> 根因：终审打回无业务熔断（MAX_ROUNDS 只管研究批次），唯一防线是框架。")
        print(">>> 生产含义：兜底层被触发 = 业务熔断设计有漏洞——对照 --fixed 修复组。")
        print("=" * 60)
    else:
        # try块没有任何异常时执行，和直接把代码写在try末尾的区别在于：else里的代码抛异常不会被同一个except捕获
        print("\n" + "=" * 60)
        rc = result.get("review_count", 0)
        print(f">>> 正常结束：next={result['next']}，终审 {rc} 轮 = {rc - 1} 次打回重写 + 第 {rc} 次打回被熔断拦截")
        print(">>> [熔断收尾] 打回已达上限，答案保留——用户拿到结果，且知道它的成色"
              "（终审未通过，带保留意见）。")
        print("=" * 60)
        print("\n[最终答案（带保留意见收尾）]\n")
        print(result["answer"])
