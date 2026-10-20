"""step6_streaming.py —— Step 6：streaming（三种流式粒度演示）

前置：step1_graph.py 已完成。

目标：graph.stream(input, config, stream_mode=...) 的三种模式各出一个演示，
对照原版 agent.py 714-809 行手搓的 delta 重组逻辑（90 行手写 → 一个参数）：

1. stream_mode="messages"：LLM token 级 chunk 流 —— 对应原版 stream=True 逐字打印，
   ToolMessage 会作为工具结果出现在流里
2. stream_mode="updates"：每个节点跑完吐一次状态增量 —— 实时看到
   "agent 节点产出了一条带 tool_calls 的消息"这类事件
3. stream_mode="values"：每步吐完整状态快照 —— 调试用

验证（B2 验收标准）：终端实时逐字出字 + 能看到节点级事件流，三种模式切换可演示。

跑法（项目根）：
  uv run python src/langGraph_agent/step6_streaming.py                # 默认 messages：逐字输出
  uv run python src/langGraph_agent/step6_streaming.py --mode updates # 节点事件流
  uv run python src/langGraph_agent/step6_streaming.py --mode values  # 状态快照
"""

# TODO(Step 6):
#   1. argparse 支持 --mode {messages,updates,values}
#   2. for chunk in graph.stream(..., stream_mode=mode) 按模式分别解析 chunk 结构
#   3. messages 模式：区分 AIMessageChunk（逐字打印）与 ToolMessage（打印工具结果摘要）

import argparse
from langchain_core.messages import AIMessageChunk, ToolMessage
from step1_graph import TICKET_QUESTION, graph



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="step6 streaming 演示")
    # parser.add_argument("--recall", action="store_true", help="...") 是布尔开关
    # parser.add_argument("--mode", choices=[], default="...", help="...") --mode 后面必须跟一个值，且值只能从三个里挑
    # parser.add_argument的第一个参数是“登记名”，如果以-开头，代表是可选参数，语法本质是“键->值”配对；没有-的裸词代表是位置参数，只在argparse内部登记用，命令行不出现
    parser.add_argument(
        "--mode",
        choices=["messages", "updates", "values"],  # 枚举：传别的值直接报错，并列出合法值
        default="messages",                         # 不带 --mode 时的缺省
        help="流式模式：messages=逐字出字 / updates=节点事件 / values=状态快照"  # --help时显示的说明文字
    )
    # nargs="?" 表示这个位置参数可有可无；不传就是 None
    parser.add_argument("question", nargs="?", default=None)
    args = parser.parse_args()
    question = args.question or TICKET_QUESTION

    graph_input = {"messages": [("user", question)]}
    config = {"recursion_limit": 15}

    # graph.stream和invoke的关系：两者入参完全一样（input、config）,差别在返回物：invoke跑完一次性返回最终state;stream返回一个迭代器--图每走一步（每个节点跑完）就吐一个chunk
    if args.mode == "values":
        # 1、values模式：跑完一个节点，生成一个chunk。chunk = 完整状态快照, 每个chunk就是invoke最终返回值的一个中间版本
        for snapshot in graph.stream(graph_input, config, stream_mode="values"):
            print(f"\n-- 快照：共{len(snapshot['messages'])}条消息")
            for m in snapshot["messages"]:
                print(type(m).__name__, ":", str(m.content)[:60].replace("\n", " "))
    elif args.mode == "updates":
        # 2、updates模式：跑完一个节点，生成一个chunk。chunk=单节点增量，每个chunk形如{节点名：该节点返回的state更新}--只含“这一步新产出的”，不含累计
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                print(f"[节点 {node} 完成] 新增 {len(update['messages'])} 条消息")
    else:
        # 3、messages模式，LLM 生成过程中每产出几个字就推一个 AIMessageChunk；工具函数 return 时整条推一个ToolMessage。迭代粒度是token,是模型吐字的最小单位，而不是节点，得到的chunk是一个二元组（msg,meta）
        # msg:消息片段,两种类型：AIMessageChunk（其中的content是一小片文字）和ToolMessage（工具结果，不分片、整条出现）
        # meta:dict元数据，里面有langgraph_node、模型名等
        for msg, meta in graph.stream(graph_input, config, stream_mode="messages"):
            if isinstance(msg, AIMessageChunk):
                if msg.content:
                    print(msg.content, end="", flush=True)
                elif msg.tool_calls and msg.tool_calls[0]["name"]:
                    names = [tc["name"] for tc in msg.tool_calls]
                    print(f"\n[发起工具调用 {names}]")
            elif isinstance(msg, ToolMessage):
                print(f"\n[工具结果·{msg.name}] {str(msg.content)[:60]}…")
        print()


