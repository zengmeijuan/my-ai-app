"""cli.py —— 入口层：组装五层（config/memory/loop/tools），跑用户对话外循环。

跑法（项目根）：
  PYTHONPATH=src uv run python -m security_serve.agent.cli
"""
from security_serve.agent.config import CONTEXT_WATCH, model
from security_serve.agent.memory import (
    build_system_prompt,
    clear_session,
    compact_messages,
    context_stats_size,
    load_session,
    near_context_watch,
    save_session,
)
from security_serve.agent.loop import run_agent_turn
from security_serve.agent.safety import Budget


def main():
    # 会话级成本预算：整个 CLI 进程一个账本，clear 也不清零（熔断管的是真金白银）
    budget = Budget()
    # 会话记忆启动恢复：上次退出前的对话已落盘在 session.json，自动加载回来继续聊
    messages = load_session()
    if messages is not None:
        print(f"[已恢复上次会话：{len(messages)} 条消息，输入 clear 可重新开始]")
    if messages is None:
        messages = [{"role": "system", "content": build_system_prompt()}]

    print(f"使用模型: {model}")
    print("可以问：")
    print("  - 天气：'北京天气怎么样？'")
    print("  - 数学：'(25+5)*2 等于多少？'")
    print("  - 复合：'北京和上海哪个温度高？'")
    print("  - 文件：'读一下 README.md' 或 '帮我看看 pyproject.toml 里有什么'")
    print("  - 目录：'看一下当前项目下面有什么'")
    print("  - 搜索：'哪个文件里定义了 get_weather？' 或 '帮我找找哪里用了 dotenv'")
    print("  - 知识库：'退货的钱多久能退回来？'")
    print("  - 联网：'Python 3.13 新特性' 或 '读一下 https://docs.python.org/...'")
    print("  - 记忆：'记住我喜欢简洁的回答'")
    print("输入 clear 清空对话，quit 退出（clear 和 quit 都不影响长期记忆 memory.md）")
    print("=" * 40)

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见。")
            break
        # 清空对话历史，只保留 system prompt（相当于 Claude Code 的 /clear）。
        # 落盘的 session.json 也要删，否则下次启动又把旧对话恢复回来 —— 这就是「清空」的彻底性。
        # 注意：长期记忆 memory.md 故意保留 —— clear 忘的是聊天，不是记住的事实
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": build_system_prompt()}]
            clear_session()
            print("[已清空对话历史和会话文件，上下文重置（长期记忆 memory.md 保留）]")
            continue
        if not user_input:
            continue

        # 本轮问答开头先看一眼上下文：超过阈值就压缩，并立刻同步落盘。保证本轮回答不被压缩
        if compact_messages(messages):
            save_session(messages)
        # 没压缩但接近阈值时也提醒一下，方便观察余量
        elif near_context_watch(messages):
            print(f"[上下文] 已达观察阈值 {CONTEXT_WATCH} 字符，下次提问前可能触发压缩")

        messages.append({"role": "user", "content": user_input})

        # Agent 内循环：LLM 自主决定调几次工具，直到给出最终回答
        run_agent_turn(messages, budget)

        # 本轮问答结束。体感实验：每轮打印上下文规模，观察它如何随工具调用膨胀
        print(f"[上下文] {len(messages)} 条消息 · 约 {context_stats_size(messages)} 字符 · 已耗约 {budget.used} tokens")
        # 会话记忆落盘：每轮问答后把最新 messages 写入磁盘，quit 重启后可无损恢复
        save_session(messages)


if __name__ == "__main__":
    main()
