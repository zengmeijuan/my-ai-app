"""step4_interrupt.py —— Step 4：interrupt（危险工具人工确认，即 permission gate）

前置：step1_graph.py 已完成；interrupt 依赖 checkpointer（暂停要能恢复，状态必须先有地方存），
所以本步同时用上 Step 3 的 checkpointer。

目标：在危险工具（read_file / python 执行类）执行前插入确认节点：
    interrupt(payload) → 图当场冻结，payload 抛给调用方
    人在终端输入 y/n → Command(resume=决定) → 图从暂停点继续
这是原版 agent.py 和 hello-agents 两边都缺的能力（对比实验结论之一）。

演示场景：问一个会触发危险工具的问题（如"读一下 data/memory.md 的内容"），
程序中途暂停打印"即将执行 read_file(...)，放行吗？[y/n]"，按输入决定继续或拒绝。

跑法（项目根）：
  uv run python src/langGraph_agent/step4_interrupt.py

预期交互：
  [interrupt] 即将执行 read_file("...memory.md")，放行吗？[y/n]: _
  （输入 y → 工具执行、流程继续；输入 n → 工具被拒绝，agent 改口如实告知）
"""
import uuid

# TODO(Step 4):
#   1. 定义 human_approval 节点：检查下一站是否危险工具，危险则 interrupt(payload)
#   2. 主流程接到确认节点：agent → human_approval → tools
#   3. 调用侧捕获 __interrupt__ 结果，input() 收 y/n，Command(resume=...) 恢复
#   4. 验证：暂停、终端输入、继续/拒绝三条路径都走通

from langchain_core.messages import SystemMessage,ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


from my_agent.agent import read_file as _read_file
from step1_graph import build_system_prompt, llm

@tool
def read_file(path: str) -> str:
    """读取指定路径的文本文件，返回内容。"""
    return _read_file(path)

tools = [read_file]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MessagesState):
    messages = [SystemMessage(content=build_system_prompt())] + list(state["messages"])
    return {"messages": [llm_with_tools.invoke(messages)]}

def should_continue(state: MessagesState) -> str:
    return "human_approval" if state["messages"][-1].tool_calls else END

DANGEROUS_TOOLS = {"read_file"}   # 危险工具名单：闸门只拦名单内的

# Command：节点自带的路由权，Command(goto="tools") 让节点在返回的同时决定下一步去哪个节点，还可以update顺便更新状态
# Command是节点运行时的路由函数，而add_conditional_edges是图定义层的静态函数，只读 state、返回节点名，不能改状态、不能暂停
def human_approval(state: MessagesState) -> Command:
    """闸门节点：检查即将执行的工具调用，危险的先 interrupt 要人工批准"""
    last = state["messages"][-1]
    tool_calls = last.tool_calls
    if not any(tc["name"] in DANGEROUS_TOOLS for tc in tool_calls):
        # 无危险工具，直接放行
        return Command(goto="tools")
    dangerous = [tc for tc in tool_calls if tc["name"] in DANGEROUS_TOOLS]
    # 图在这里冻结，恢复时interrupt的返回值就是调用方传进来的决定
    # 入参payload=问题本身，它会被框架打包至Interrupt.value，调用方（审批UI）一般会根据它来理解发生了什么，要决定什么
    # 返回值 = 人类的回答（通过resume传送回来）
    answer = interrupt({"tool_calls": dangerous})

    if answer == "y":
        return Command(goto="tools")
    else:
        rejected = [
            ToolMessage(content="用户拒绝执行该工具", tool_call_id=tc["id"])
            for tc in tool_calls # 整批工具都拒绝,部分执行方案实现更复杂,下一轮推理输入的语义模糊
        ]
        # update 可以增量更新state schema里的任意key
        return Command(goto="agent", update={"messages": rejected})


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_approval", human_approval)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["human_approval", END])
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    config = {"recursion_limit": 20, "configurable": {"thread_id": "approval-demo"}}

    while True:
        question = input("\n顾客提问（q退出）：").strip()
        if not question or question.lower() == "q":
            break
        if question == ":new":
            config["configurable"]["thread_id"] = f"session-{uuid.uuid4().hex[:8]}"
            print(f"[已切换到新会话 {config['configurable']['thread_id']}]")
            continue

        # 正常跑完：result就是最终state--{"messages":[]}
        # 冻结在中断点：result里的messages是到冻结点为止的累积消息（user + 那条带 tool_calls 的 AIMessage），另外多出一个“__interrupt__”键，值是 Interrupt 对象的元组
        # invoke时框架会按thread_id从checkpointer恢复该会话的全部历史，再将新消息追加到后面，然后将整个列表喂给模型
        result = graph.invoke({"messages": [("user", question)]}, config)

        while "__interrupt__" in result:
            for tc in result["__interrupt__"][0].value["tool_calls"]:
                print(f"[interrupt] 即将执行 {tc['name']}({tc['args']})")
            # 提示文字，原样打印在终端光标前等待输入
            answer = input("放行吗？[y/n]").strip().lower()
            # Command(resume=answer) 会把resume的值交给interrupt作为返回值，resume必须配置同一个config(thread_id),框架按id找到冻结的那行代码，从 answer = interrupt(...) 的下一行原地继续
            result = graph.invoke(Command(resume=answer), config)

        print("[回答]", result["messages"][-1].content)
        print("\n=== 全部消息 ===")
        for m in result["messages"]:
            print(type(m).__name__, ":", str(m.content)[:70].replace("\n", " "))

