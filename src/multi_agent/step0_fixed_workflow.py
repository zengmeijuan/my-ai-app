"""step0_fixed_workflow.py —— E1 前置基线：固定工作流版研究写作系统（对照样本）

定位：research → write → review → revise 是路径可穷举的确定性流程，
supervisor 的六条路由规则全部是 State 字段的函数（有无 / 布尔 / 计数）——
if/elif 即可实现，LLM 路由在这里是可被代码完全替代的形态样本。
保留它作两个对照的实物基线：
①E5 对照矩阵的「固定工作流」栏：简单流水线任务不需要 supervisor；
②supervisor 价值的反衬：只有当路由决策必须"读懂内容"（见 step1_supervisor.py）
  时，LLM 路由才不可替代。

本样本 researcher 不带工具（联网工具的 agent 化见 step1 / step2）。

图结构（supervisor 是中枢路由器，所有 worker 干完活都回到它，由它决定下一步）：

                 ┌────────────────────────────┐
                 │     supervisor（只路由不干活） │
                 └────────────────────────────┘
                   ↓        ↓        ↓        ↓
             researcher  writer  reviewer  revise
                   └────────┴────┴────────┘
                        （干完活回 supervisor）

与 B2 单图 ReAct 的对照：B2 是一个节点内部自循环（agent ↔ tools 到 END），
step0 是多个节点接力 + 一个大脑路由。共享 State（topic/research/draft/review）
是节点间唯一通信介质——每个节点是一次独立的 LLM 调用，这就是多 agent 上下文隔离的来源。

跑法（项目根）：
  uv run python src/multi_agent/step0_fixed_workflow.py "自定义研究题目"
  uv run python src/multi_agent/step0_fixed_workflow.py  # 默认题目
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
    topic: str           # 研究题目（入口输入）
    research: str        # researcher 产出的研究材料
    draft: str           # writer 产出的稿子（初稿或返工稿）
    review: str          # reviewer 的评审意见
    approved: bool       # reviewer 是否通过
    revision_count: int  # 已返工次数（熔断计数器）
    next: str            # supervisor 决定的下一节点（路由信号）


# ============== 2. 模型：复用 B2 的 ChatOpenAI 接法 ==============

llm = ChatOpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    model=os.environ.get("MODEL", "gpt-4o-mini"),
    timeout=120,
    max_retries=2,
)


# ============== 3. 结构化 schema：节点间不传自由文本的地方用 schema 约束 ==============

class ReviewVerdict(BaseModel):
    """reviewer 的结构化输出 schema：approved 是路由依据，feedback 是返工输入。"""
    approved: bool = Field(description="是否通过评审")
    feedback: str = Field(description="评审意见：通过时说明理由；打回时给出具体修改点")


class RouteDecision(BaseModel):
    """supervisor 的结构化输出 schema：next 是路由信号，reason 留给调试。"""
    next: Literal["researcher", "writer", "revise", "reviewer", "END"]
    reason: str = Field(description="一句话说明为什么路由到这个节点")


# ============== 4. 四个 worker 节点：各管一段，只写自己的字段 ==============

def researcher(state: TeamState) -> dict:
    """研究员：围绕题目产出要点式研究材料。本基线不含工具（联网工具的 agent 化见 step1 / step2），靠模型自带知识。"""
    resp = llm.invoke([
        {"role": "system", "content": (
            "你是研究员。针对给定题目产出要点式研究材料：5-8 条要点，"
            "每条一句话讲清一个事实或论点。只输出要点本身，不写客套话。"
        )},
        {"role": "user", "content": f"研究题目：{state['topic']}"},
    ])
    print("\n===== [researcher] 研究材料 =====")
    print(resp.content)
    return {"research": resp.content}


def writer(state: TeamState) -> dict:
    """撰稿人：只依据研究材料写作，禁止自行发挥（编造事实的机制防线）。"""
    resp = llm.invoke([
        {"role": "system", "content": (
            "你是撰稿人。只依据提供的研究材料写一篇 300 字左右的短文，"
            "禁止使用研究材料之外的事实。材料没覆盖的角度如实不写。"
        )},
        {"role": "user", "content": (
            f"题目：{state['topic']}\n\n研究材料：\n{state['research']}"
        )},
    ])
    print("\n===== [writer] 稿子 =====")
    print(resp.content)
    return {"draft": resp.content}


def reviewer(state: TeamState) -> dict:
    """审稿人：对抗视角独立评审，结构化输出结论（这是全图唯一的 quality gate）。"""
    # method 默认 json_schema：schema 约束在解码层强制生效，输出合法性由 API 保证，
    # 是官方推荐方式；评审调用输入最长、生成 JSON 参数最慢，timeout 须给足（见 llm 配置）
    verdict_cls = llm.with_structured_output(ReviewVerdict)
    verdict = verdict_cls.invoke([
        {"role": "system", "content": (
            "你是审稿人。从三个维度评审稿子：①每条事实是否都能在研究材料中找到依据；"
            "②结构是否清晰；③题目是否被回答。有任何一条不满足就打回，"
            "并给出具体的修改意见（指出哪一段有什么问题、怎么改）。"
        )},
        {"role": "user", "content": (
            f"题目：{state['topic']}\n\n研究材料：\n{state['research']}"
            f"\n\n待审稿子：\n{state['draft']}"
        )},
    ])
    print(f"\n===== [reviewer] 评审：{'✅ 通过' if verdict.approved else '❌ 打回'} =====")
    print(verdict.feedback)
    return {"review": verdict.feedback, "approved": verdict.approved}


def revise(state: TeamState) -> dict:
    """返工：writer 的返工形态——对着评审意见改稿，同时累加熔断计数器。"""
    resp = llm.invoke([
        {"role": "system", "content": (
            "你是撰稿人，正在按审稿意见返工。只依据研究材料和审稿意见修改稿子，"
            "仍然禁止编造研究材料之外的事实。"
            "输出格式：直接给出修改后的完整正文，禁止输出任何回应审稿意见的说明文字"
            "（返工者的交付物是正文本身，过程性说明属于 State 之外的内容）。"
        )},
        {"role": "user", "content": (
            f"题目：{state['topic']}\n\n研究材料：\n{state['research']}"
            f"\n\n原稿：\n{state['draft']}\n\n审稿意见：\n{state['review']}"
        )},
    ])
    count = state.get("revision_count", 0) + 1
    print(f"\n===== [revise] 第 {count} 次返工 =====")
    print(resp.content)
    return {"draft": resp.content, "revision_count": count}


# ============== 5. supervisor 节点：只读状态摘要做路由决策 ==============

MAX_REVISIONS = 2  # 熔断上限：返工超过 2 次强制收尾（E3 固化为显式契约）


def supervisor(state: TeamState) -> dict:
    """路由器：把当前进度压成状态摘要，让 LLM 按状态机规则选下一节点。
    路由规则（穷举状态，不留未定义分支）：
      1. 无 research            → researcher
      2. 有 research 无 draft    → writer
      3. 有 draft 无 review      → reviewer
      4. approved=True           → END
      5. 不通过且未到熔断上限      → revise
      6. 不通过且已到熔断上限      → END（带保留意见收尾）
    """
    has_research = bool(state.get("research"))
    has_draft = bool(state.get("draft"))
    has_review = bool(state.get("review"))
    count = state.get("revision_count", 0)
    summary = (
        f"进度：research={'已有' if has_research else '无'}，"
        f"draft={'已有' if has_draft else '无'}，"
        f"review={'已有' if has_review else '无'}，"
        f"approved={state.get('approved')}，revision_count={count}/{MAX_REVISIONS}。"
    )
    router = llm.with_structured_output(RouteDecision)
    decision = router.invoke([
        {"role": "system", "content": (
            "你是多 agent 团队的 supervisor，只负责路由，不产出内容。"
            "按以下状态机规则选择 next：\n"
            "1. research 无 → researcher（先研究）\n"
            "2. research 有、draft 无 → writer（写初稿）\n"
            "3. draft 有、review 无 → reviewer（送审）\n"
            "4. review 有且 approved=True → END（收工）\n"
            f"5. review 有且打回且 revision_count < {MAX_REVISIONS} → revise（返工）\n"
            f"6. review 有且打回且 revision_count >= {MAX_REVISIONS} → END（熔断收尾）"
        )},
        {"role": "user", "content": summary},
    ])
    print(f"\n>>> supervisor 路由：{decision.next}（{decision.reason}）")
    return {"next": decision.next}


def route(state: TeamState) -> str:
    """条件边函数：读路由信号分发。"""
    return state["next"]


# ============== 6. 组图：supervisor 居中，worker 全部回边到它 ==============

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
builder.add_node("revise", revise)

builder.add_edge(START, "supervisor")
# 所有 worker 干完活都回到 supervisor（中心路由的拓扑特征）
for worker in ("researcher", "writer", "reviewer", "revise"):
    builder.add_edge(worker, "supervisor")
# supervisor 的条件边：next 信号 → 对应节点，"END" 特殊映射到 END
builder.add_conditional_edges(
    "supervisor",
    route,
    {"researcher": "researcher", "writer": "writer",
     "reviewer": "reviewer", "revise": "revise", "END": END},
)

graph = builder.compile()


# ============== 7. 运行：一道题走完四阶段 ==============

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "supervisor 和 固定流程节点 的区别"
    print(f"研究题目：{topic}")
    result = graph.invoke(
        {"topic": topic, "revision_count": 0},
        config={"recursion_limit": 25},  # 双保险：LLM 路由失控时由框架兜底
    )
    print("\n" + "=" * 60)
    print(f"最终 revision_count：{result['revision_count']}，approved：{result['approved']}")
    print("=" * 60)
    print("\n[最终稿]\n")
    print(result["draft"])
