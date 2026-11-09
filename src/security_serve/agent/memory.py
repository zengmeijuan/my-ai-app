"""memory.py —— 记忆与上下文层：会话落盘/恢复/清空、长期记忆注入、上下文统计与压缩。

对应 B 阶段精读 harness 识别的四组件之一：context compaction / session store。
"""
import json
import os

from openai import APITimeoutError, APIConnectionError, APIStatusError

from security_serve.agent.config import (
    API_TIMEOUT,
    COMPACT_KEEP_USER_ROUNDS,
    COMPACT_MAX_TOKENS,
    CONTEXT_COMPACT,
    CONTEXT_WATCH,
    MEMORY_PATH,
    SESSION_PATH,
    SYSTEM_PROMPT_BASE,
    client,
    model,
)


# ============== system prompt 组装 ==============

def build_system_prompt() -> str:
    """组装 system prompt：基础指令 + 长期记忆（memory.md 存在时追加注入）。

    启动、clear、恢复会话三个时机都会重建 —— 所以 remember 新写的记忆
    「下次会话或 clear 后」生效。会话中途生效需在 remember 后刷新
    messages[0]，学习版从简不做。
    """
    prompt = SYSTEM_PROMPT_BASE
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, encoding="utf-8") as f:
            memory = f.read().strip()
        if memory:
            prompt += "\n\n【长期记忆】以下是过去会话中记住的信息，回答时可参考：\n" + memory
    return prompt


# ============== 会话落盘 / 恢复 / 清空 ==============

def load_session() -> list | None:
    """启动恢复：上次退出前的对话已落盘在 session.json，自动加载回来继续聊。

    必须校验首条是 system、消息序列协议合法 —— 否则 API 下次调用会被服务端 400 拒绝。
    恢复成功时刷新 system 消息：既保留工作目录锚点，也注入最新长期记忆。
    文件不存在或损坏时返回 None（调用方从新会话开始）。
    """
    messages = None
    if os.path.exists(SESSION_PATH):
        try:
            with open(SESSION_PATH, encoding="utf-8") as f:
                messages = json.load(f)
            if not messages or messages[0].get("role") != "system":
                raise ValueError("session 文件缺少 system 消息")
            messages[0] = {"role": "system", "content": build_system_prompt()}
        except (json.JSONDecodeError, OSError, ValueError):
            messages = None  # 文件损坏就当没有，重新开始
    return messages


def save_session(messages: list) -> None:
    """会话记忆落盘：把最新 messages 写入磁盘，quit 重启后可无损恢复。
    indent=2 是为了人类可读——排查对话恢复问题时能直接看每条消息。"""
    with open(SESSION_PATH, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def clear_session() -> None:
    """清空会话文件（长期记忆 memory.md 故意保留——clear 忘的是聊天，不是记住的事实）。"""
    if os.path.exists(SESSION_PATH):
        os.remove(SESSION_PATH)


# ============== 上下文统计与压缩 ==============

def context_stats_size(messages: list) -> int:
    """统计上下文字符总数：content + tool_calls.arguments。"""
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for call in (m.get("tool_calls") or []):
            total += len(call.get("function", {}).get("arguments") or "")
    return total


# 压缩摘要的提示词：事实优先，丢弃代码匹配等冗长细节
COMPACT_PROMPT = """请把以下多轮对话压缩成一段 200~400 字的简洁摘要。
要求：
1. 保留关键事实：用户偏好、提问意图、得出的结论、工具调用结果的关键信息
2. 丢弃冗长细节：grep 匹配行、原始文件内容、网页正文只保留结论
3. 不要复述工具的原始输出，用一句结论概括
4. 用中文，按时间顺序，列出关键事件
5. 不要添加对话中没有的内容"""


def find_user_round_start(messages: list, keep_rounds: int) -> int:
    """从尾部向前回溯，跳过 system(idx 0)，找到第 keep_rounds 个 user 消息的索引。

    一轮对话定义为「从 user 消息开始，到下一个 user 消息之前」，这样切能保证 tail
    始终是完整的 user→assistant→tool 序列，OpenAI 协议合法（tool 不会孤儿化）。

    返回 cut 索引：messages[cut:] 是要保留的最近 keep_rounds 个完整轮。
    若找不到足够 user（少于 keep_rounds 个），返回 -1。
    """
    user_count = 0
    for i in range(len(messages) - 1, 0, -1):  # 跳过 system（索引 0）
        if messages[i].get("role") == "user":
            user_count += 1
            if user_count == keep_rounds:
                return i
    return -1


def compact_messages(messages: list) -> bool:
    """超阈值时压缩 messages：保留 system + 最近 N 个完整 user 轮，中间压成一条摘要。

    返回 True 表示发生了压缩（无论成功与否），False 表示跳过。
    失败兜底：压缩调用本身超时/出错时不动 messages，留给下次机会。
    """
    if context_stats_size(messages) <= CONTEXT_COMPACT:
        return False

    # 从尾部向前找第 N 个 user 消息的位置，作为 tail 起点
    cut = find_user_round_start(messages, COMPACT_KEEP_USER_ROUNDS)
    if cut <= 1:  # cut=1 表示 user 紧跟 system，没东西可压；cut=-1 表示找不到足够 user
        return False

    head = messages[:1]                          # system 消息保留
    tail = messages[cut:]                        # 最近 N 个完整轮
    middle = messages[1:cut]                     # 中间要压缩的部分

    # 把中间序列转成一段对话文本喂给压缩模型，role 标记 + 内容/工具摘要
    parts = []
    for m in middle:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "tool":
            # 工具结果只取前 100 字符作为索引，原始长结果已在记忆中
            parts.append(f"[工具结果] {content[:100]}{'...' if len(content) > 100 else ''}")
        elif role == "assistant" and m.get("tool_calls"):
            names = [tc["function"]["name"] for tc in m["tool_calls"]]
            parts.append(f"[助手调用工具] {', '.join(names)}")
        else:
            parts.append(f"[{role}] {content}")
    dialogue = "\n".join(parts)

    # 单独开一个轻量压缩请求：不开 tools、低 temperature、限 max_tokens 控长度
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": COMPACT_PROMPT},
                {"role": "user", "content": dialogue},
            ],
            temperature=0.2,
            max_tokens=COMPACT_MAX_TOKENS,
            timeout=API_TIMEOUT,
        )
    except (APITimeoutError, APIConnectionError, APIStatusError) as e:
        print(f"[上下文压缩失败：{type(e).__name__}，本轮不压缩留待下次]")
        return False

    summary = resp.choices[0].message.content or ""
    if not summary.strip():
        print("[上下文压缩失败：模型返回空摘要，本轮不压缩]")
        return False

    before = context_stats_size(messages)
    # 用 clear + extend + append 原地重组 messages，外部引用不变
    messages.clear()
    messages.extend(head)
    messages.append({
        "role": "system",
        "content": f"【对话历史摘要】以下是之前对话的压缩摘要，仅供参考；用户当前问题在摘要之后：\n{summary}",
    })
    messages.extend(tail)
    after = context_stats_size(messages)
    print(f"[上下文] 已压缩 {len(middle)} 条中间消息→ 1 条摘要，{before} → {after} 字符")
    return True


def near_context_watch(messages: list) -> bool:
    """接近观察阈值（未到压缩线）时提醒，方便观察余量。"""
    return context_stats_size(messages) >= CONTEXT_WATCH
