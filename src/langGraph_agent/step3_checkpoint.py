"""step3_checkpoint.py —— Step 3：checkpoint（替代 session.json 手工落盘）

前置：step1_graph.py 已完成（本文件 import 它的图）。

目标：给 compile() 挂 checkpointer，体验「框架在每个节点边界自动存状态快照」，
对照原版 agent.py 的手工 dump/load。

两段演示：
1. InMemorySaver（langgraph 内置）：同进程两次 invoke 用同一 thread_id，
   第二次问"刚才算了多少钱"，验证图记得上文（对应原版会话内记忆）
2. SqliteSaver（langgraph-checkpoint-sqlite）：状态落 SQLite 文件，
   新起进程带 --recall 问"上次台灯顾客的咨询，我们最后是怎么答复的？"，
   验证跨进程召回（对应原版 session.json + --recall 场景，B2 验收标准）

跑法（项目根）：
  uv run python src/langGraph_agent/step3_checkpoint.py             # 第一次：工单题，写入 checkpoint
  uv run python src/langGraph_agent/step3_checkpoint.py --recall    # 新进程：验证跨进程召回

产出：src/langGraph_agent/checkpoints.db（演示用 SQLite 文件）
"""

# TODO(Step 3):
#   1. from langgraph.checkpoint.memory import InMemorySaver；同进程两轮对话演示
#   2. from langgraph.checkpoint.sqlite import SqliteSaver；SqliteSaver.from_conn_string(...)
#   3. graph = builder.compile(checkpointer=saver)；invoke 带 config={"configurable": {"thread_id": ...}}
#   4. --recall 分支：新进程直连同一 SQLite，问召回问题，验证历史状态被恢复

import argparse
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from step1_graph import builder,TICKET_QUESTION


def demo_inmemory() -> None:
    """同进程记忆demo"""
    # InMemorySaver，编译出带记忆的运行时，在每次节点跑完后自动存快照，不能跨进程
    graph = builder.compile(checkpointer=InMemorySaver())
    # configurable.thread_id，存档槽位号，checkpointer 按 thread_id 分档保存快照，同一个id共享对话历史，不同id互相看不到
    config = {"recursion_limit": 10, "configurable": {"thread_id": "mem-demo"}}

    r1 = graph.invoke(
        {"messages": [("user", "帮我算一下 45 + 39 等于多少？必须调用工具计算，不要心算。")]},
        config
    )
    print("[第一轮]", r1["messages"][-1].content)

    r2 = graph.invoke(
        {"messages": [("user", "刚才算出多少钱？只回答数字。")]},
        config,
    )
    print("[第二轮]", r2["messages"][-1].content)
    print(f"第一轮 {len(r1['messages'])} 条 → 第二轮 {len(r2['messages'])} 条")

# __file__的值为当前脚本文件路径，resolve()转为绝对路径，.parent取所在目录。合起来即本脚本所在的文件夹
DB_PATH = Path(__file__).resolve().parent / "checkpoints.db"
THREAD_ID = "customer-1"

def demo_sqlite_write() -> None:
    """跨进程磁盘记忆demo"""
    # SqliteSaver 状态写入本地磁盘上的 SQLite 数据库文件，可以跨进程存活
    # from_conn_string SqliteSaver 的类方法，给它一个文件路径，它负责创建（或打开）一个 SQLite 连接
    # with ... as saver Python的上下文管理器语法
    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = builder.compile(checkpointer=saver)
        config = {"recursion_limit": 10, "configurable": {"thread_id": THREAD_ID}}
        result = graph.invoke({"messages": [("user", TICKET_QUESTION)]}, config)
        print("[演示2·工单题]", result["messages"][-1].content)

        # 按 config 里的 thread_id，去 checkpointer 取该档最新快照
        snap = graph.get_state(config)
        print(f"[演示2] 快照里有 {len(snap.values['messages'])} 条消息，下一待执行节点 {snap.next}")

def recall() -> None:
    """召回磁盘记忆demo"""
    # 重建①：连同一个 db 文件
    with SqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        graph = builder.compile(checkpointer=saver)
        # 重建②：同一个槽位号
        config = {"recursion_limit": 10, "configurable": {"thread_id": THREAD_ID}}
        result = graph.invoke(
            {"messages": [("user", "上次台灯顾客的咨询，我们最后是怎么答复的？简要复述。")]},
            config,
        )
        print("[召回]", result["messages"][-1].content)

if __name__ == "__main__":
    # argparse.ArgumentParser 构造参数解析器对象
    parser = argparse.ArgumentParser(description="step3 checkpoint 演示")
    # 注册一个名为 --recall 的可选开关,action="store_true"含义是命令行见到这个词就把args.recall存成True,help是--help时显示的说明文字
    parser.add_argument("--recall", action="store_true", help="新进程验证跨进程召回")
    # 真正解析命令行，返回一个对象，属性名就是注册的参数名（去掉--）
    args = parser.parse_args()

    # 使用argparse参数解析对象，就不用自己截取sys.args进行判断处理
    if args.recall:
        recall()
    else:
        demo_inmemory()
        demo_sqlite_write()
