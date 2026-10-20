"""step5_subgraph.py —— Step 5：subgraph（给 search_knowledge 套检索子图）

前置：step1_graph.py 已完成（复用其 llm 与建图手法）。

目标：把 search_knowledge 从「一次余弦检索」升级为一条自我纠错的检索流水线：
    查询改写(LLM) → 检索(rag) → 相关性判断(LLM) ──(不够且重试<2)──→ 改写重查
                                     └─(足够 或 到上限)──→ 返回 top-3
大图视角（完整路线）：主图里它仍是普通工具节点，内部多步流程对 agent 完全透明——封装价值。
    用户问题
      → 主图 agent 发起 search_knowledge(query)——query 参数由模型填写，是第一道改写
      → ToolNode 执行工具 → 工具内 invoke 检索子图：
            rewrite → retrieve → judge ──(不够且重试<2)──→ rewrite 重查
            judge 判"够"或到上限 → 子图 END，invoke 返回子图最终 state
      → 工具把 results 格式化成字符串 → ToolMessage 回到主图
      → agent 作答（也可能自主再发起一次检索——查几轮由主图 agent 决定）

设计决策（与骨架 TODO 3 的偏离）：
- 骨架字面：子图 compile 后 add_node 挂进主图（图嵌套）
- 实际采用：子图藏进 @tool search_knowledge 的实现里（工具内 invoke 子图）
- 理由：① 子图 state(query/results/retries) 与主图 state(messages) 不同构，嵌套需
  手写翻译层（解析 tool_calls → invoke → 构造 ToolMessage），等于手工重造 ToolNode；
  ② 路由要按工具名分流，撞上混合批次边界；③ 违背「对 agent 完全透明」的封装目标
- 准绳：主图只要子图的「结果」→ 封装进工具；主图要干预子图「过程」（暂停/重派）
  → 才升格为节点。图嵌套的正场是里程碑 E 的 supervisor（子图即 agent，messages 兼容）

验证：
①对照跑通精确问法（确认子图没把好事改坏——这正是 rewrite 的风险点）
②观察rewrite 前后的查询词和 judge 的判断过程可打印、可解释
③留一道“有无子图”的对比实验挂给里程碑 C

跑法（项目根）：
  uv run python src/langGraph_agent/step5_subgraph.py "钱多久能回来"
  uv run python src/langGraph_agent/step5_subgraph.py "台灯无理由退货谁出运费"   # 对照：精确问法
"""

# TODO(Step 5):
#   1. 定义子图自己的 State：RetrievalState（query/results/retries，覆盖语义）
#   2. 子图节点：rewrite（LLM 改写查询）、retrieve（复用 my_agent.rag.retrieve）、
#      judge（LLM 判断相关性是否足够），条件边控制重查（上限 2 次）
#   3. 子图 compile 后藏进 @tool search_knowledge 的实现（工具内 invoke，
#      不走 add_node 图嵌套——理由见 docstring「设计决策」）
#   4. 验证：精确问法对照 + 过程可观察；对比实验挂里程碑 C

import json
import sys

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from typing import TypedDict
from pydantic import BaseModel, Field

from step1_graph import llm, build_system_prompt
import my_agent.rag as rag

# 自定义子图State,默认覆盖语义，与 MessagesState 的追加语义不同
class RetrievalState(TypedDict):
    query: str              # 当前查询词--rewrite会改写它（覆盖）
    results: list[dict]     # 检索结果：[{"source", "text", "score"}, ...]
    retries: int            # 已重查次数--judge判不够时+1，封顶2

def rewrite(state: RetrievalState):
    """把口语查询规范成适合检索的表述；已规范的查询原样保留"""
    resp = llm.invoke(f"""
        目标：将下面的用户查询改写成适合检索的规范表述
        要求：原查询里的具体信息（商品名、条件、数量词）不许丢；如果原查询已清晰规范，一字不改原样返回
        输出：只输出改写后的查询本身，禁止解释、禁止引导、禁止任何多余文字
        用户查询：{state["query"]}
    """)
    new_query = resp.content.strip()
    # !r 保留不可见字符（空格、换行、引号），而不是默认的str()
    print(f"[rewrite] {state['query']!r} → {new_query!r}")
    return {"query": new_query}

_index_cache = None     # 模块级缓存：索引文件只在首次检索时读盘
def retrieve(state: RetrievalState):
    """知识库检索"""
    global _index_cache
    if _index_cache is None:
        with open(rag.INDEX_PATH, encoding="utf-8") as f:
            _index_cache = json.load(f)
    results = rag.retrieve(state["query"], _index_cache)
    print(f"[retrieve] {state['query']!r} → top{len(results)}: {[r['source'] for r in results]}")
    return {"results": results}

# BaseModel是pydantic的基类，继承它，类字段就带上了“声明即校验”的能力
# Field是字段元数据工具：Field(description=...) 里的 description 会被翻译进 JSON Schema发给模型--相当于给模型“这个字段怎么填”的说明书
# Verdict在实例化时会强制校验，类型不对抛ValidationError，适合数据来源外部（LLM/API/用户）。而typing标准库是纸面约定，适用于数据来源为信任的内部代码
class Verdict(BaseModel):
    enough: bool = Field(description="检索到的资料是否足以回答查询")
    reason: str = Field(description="一句话判断理由")

# with_structured_output(Verdict)，机制是将Verdict这个schema翻译成function calling 约束发给模型，输出形状由框架强制保证，不再需要prompt描述输出格式
# judge_llm.invoke(...)的返回值直接是Verdict实例，Verdict.enough就是布尔值，零解析代码
judge_llm = llm.with_structured_output(Verdict)

def judge(state: RetrievalState) -> Command:
    """LLM 判断资源相关性是否足够"""
    context = "\n\n".join(
        f"[{i}]（来源：{r['source']}，相似度 {r['score']:.3f}）\n{r['text']}"
            for i, r in enumerate(state["results"], 1)
    )
    verdict = judge_llm.invoke(f"""
        目标：判断以下检索到的资料，是否足够，并且给出一句话判断理由
        判断标准：资料中包含能直接回答查询的信息才算足够；仅相似但回答不了（如问时效却只命中流程说明）= 不够
        用户查询：{state["query"]}
        检索资料：{context}
    """)
    print(f"[judge] enough={verdict.enough} ({verdict.reason})")
    if verdict.enough:
        # Command(goto=END)：给调度器的指令单，宣布结束
        return Command(goto=END)
    if state["retries"] >= 2:
        return Command(goto=END)
    return Command(goto="rewrite", update={"retries": state["retries"] + 1})

# 组装检索子图，所有节点读写的是同一份state,这是数据流转的前提。节点返回的key必须在state里声明过；两张schema不同的图不能直接连线，需要翻译
retrieval_builder = StateGraph(RetrievalState)
retrieval_builder.add_node("rewrite", rewrite)
retrieval_builder.add_node("retrieve", retrieve)
retrieval_builder.add_node("judge", judge)
retrieval_builder.add_edge(START, "rewrite")
retrieval_builder.add_edge("rewrite", "retrieve")
retrieval_builder.add_edge("retrieve", "judge")
retrieval_graph = retrieval_builder.compile()

@tool
def search_knowledge(query: str) -> str:
    """星辰商城知识库检索：返回与问题最相关的资料（带来源文件名和相似度）。"""
    # 图结束后，invoke函数调用正常返回，返回值就是图最终的state字典
    final = retrieval_graph.invoke({"query": query, "results": [], "retries": 0})
    return "\n\n".join(
        f"资料[{i}]（来源：{r['source']}，相似度 {r['score']:.3f}）\n{r['text']}"
        for i, r in enumerate(final["results"], 1)
    )
tools = [search_knowledge]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):
    return {
        "messages": [llm_with_tools.invoke(
            [SystemMessage(content=build_system_prompt())] + list(state["messages"])
        )]
    }

def should_continue(state: MessagesState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")
graph = builder.compile()

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "钱多久能回来"
    result = graph.invoke({"messages": [("user", question)]}, config={"recursion_limit": 15})
    print("\n=== 最终回答 ===")
    print(result["messages"][-1].content)
    print(f"\n[共 {len(result['messages'])} 条消息]")
    for m in result["messages"]:
        print(type(m).__name__, ":", str(m.content)[:60].replace("\n", " "))
