"""loop.py —— agent 循环层：LLM 流式决策 → 工具并行执行 → 结果回传，直到给出最终答案。

run_agent_turn 是单次提问的完整内循环（原地更新 messages，返回最终回答字符串）；
G3 防线已接入：permission gate 在 run_one_tool 执行点（危险工具先过人工确认），
成本预算熔断在内循环步进处（每轮 LLM 调用前查线、调用后记账）。
G5 服务化适配：echo 开关区分输出去向——CLI（echo=True）打印到 stdout 给终端
用户看，HTTP 服务（echo=False）转 logging 并用返回值交出回答。
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from openai import APITimeoutError, APIConnectionError, APIStatusError

from security_serve.agent.config import API_TIMEOUT, MAX_STEPS, client, model
from security_serve.agent.memory import context_stats_size
from security_serve.agent.safety import Budget, ask_user, needs_confirm, wrap_untrusted
from security_serve.agent.tools import TOOL_FUNCTIONS, TOOLS

log = logging.getLogger(__name__)


def _emit(msg: str, echo: bool, warning: bool = False, extra: dict | None = None) -> None:
    """输出去向开关：CLI 打 stdout（终端用户看），服务打日志（排障者看）。

    logging 的分级/落盘/可检索是 print 没有的；服务形态下 root 由 logsetup
    配置（H5-1），info 进 JSON 文件。extra 业务字段只在日志分支生效，
    CLI 的 print 分支不带（事件字段化不影响终端人读输出）。
    """
    if echo:
        print(msg)
    elif warning:
        log.warning(msg, extra=extra)
    else:
        log.info(msg, extra=extra)


def run_one_tool(call: dict) -> tuple:
    """执行单个工具调用，返回 (call, name, args, result) 四元组。

    所有异常都在这里兜住，不让一个工具的失败影响其他并行执行的任务。
    TOOL_FUNCTIONS 在并行路径中只读访问，多线程安全。
    """
    name = call["function"]["name"]
    args_raw = call["function"]["arguments"]
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        return call, name, None, "错误：arguments 不是合法 JSON，请检查字段名是否与 schema 描述一致后重新调用"
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return call, name, args, f"未知工具: {name}"
    # G3 permission gate：危险工具执行前过人工确认。拒绝不走异常，而是把拒绝
    # 当作一次正常的工具结果回传给 LLM（它看到原因后自行调整下一步）；
    # 放行路径一行不改。gate 在 loop 执行点而非 tools 层——直接调工具函数不受影响
    if needs_confirm(name) and not ask_user(name, str(args)):
        return call, name, args, "用户拒绝执行此工具"
    try:
        result = func(**args)
    except TypeError as e:
        result = f"参数不匹配: {e}"
    except Exception as e:
        result = f"工具执行异常: {e}"
    return call, name, args, result


def run_agent_turn(messages: list, budget: Budget, echo: bool = True) -> str:
    """单次提问的 agent 内循环：LLM 自主决定调几次工具，直到给出最终回答。

    原地更新 messages（append assistant/tool 消息），返回最终回答字符串
    （错误路径返回空串或部分内容，原因在日志里）。echo=True 时流式打印到
    stdout（CLI 体验），echo=False 时不打印、状态走日志（服务场景）。
    循环安全边界有两道：模型陷入「反复调工具停不下来」时由 MAX_STEPS 强制刹车
    （管单次提问的轮数）；会话累计消耗达到预算线时由 budget 熔断（管整个会话
    的成本，每轮 LLM 调用前查线、调用后记账）。
    """
    final_content = ""  # 函数级初始化：首轮就熔断/超步数时 break 出来也有值可返
    step = 0
    while True:
        step += 1
        if step > MAX_STEPS:
            _emit(f"[已达最大步数 {MAX_STEPS}，强制停止。模型可能在反复调用工具，请换个问法或 clear 后重试]", echo, warning=True)
            break
        # G3 预算熔断：发起 LLM 请求之前查线——请求发出去就要花钱，超线就不再发
        if budget.exceeded():
            _emit(f"[预算熔断：本次会话已耗约 {budget.used} tokens，达到上限 {budget.limit}，停止调用 LLM]",
                  echo, warning=True, extra={"event": "budget_breach", "used_tokens": budget.used, "budget_limit": budget.limit})
            break

        # 流式调 LLM（stream=True）。create() 只建立连接就返回，之后逐块收到 chunk。
        # 超时/断连/HTTP 错误既可能在建连时抛出，也可能在消费 chunk 的过程中抛出，
        # 所以 create 和 for 循环要放在同一个 try 里兜住
        content_parts = []   # 文本片段累计器，流结束后拼成完整回复
        tool_calls_acc = {}  # index -> {"id","name","arguments"}，工具调用分片组装器
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                timeout=API_TIMEOUT,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue  # 部分服务端在最后发一个只带用量统计、无 choices 的收尾块
                delta = chunk.choices[0].delta
                if delta.content:
                    # flush=True 立即刷新 stdout 缓冲区，否则逐字打印会攒成一大段才显示；
                    # echo=False（服务场景）跳过逐字打印，内容照常收集
                    if echo:
                        print(delta.content, end="", flush=True)
                    content_parts.append(delta.content)
                # 工具调用分片到达：首个分片带 id 和完整 name，arguments 被切成
                # 多个碎片散在后续分片里，必须按 index 拼回完整 JSON
                for tc in delta.tool_calls or []:
                    slot = tool_calls_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
        except APITimeoutError:
            # 消息开头的 \n 顺便截断流式中已打印一半的行
            _emit(f"\n[LLM 响应超时（{API_TIMEOUT} 秒），本轮终止，请重新提问]", echo, warning=True)
            break
        except APIConnectionError:
            _emit("\n[连不上 API 服务器：请检查网络和 BASE_URL 配置]", echo, warning=True)
            break
        except APIStatusError as e:
            # 401 key 无效 / 429 限流 / 500 服务端出错等都走这里
            _emit(f"\n[API 返回错误 {e.status_code}: {e.message}]", echo, warning=True)
            break
        if content_parts and echo:
            print()  # 流式打印不带换行，收尾补一个，避免下一条输出接在同一行

        final_content = "".join(content_parts)

        # G3 成本记账：本轮 LLM 调用已完成，此刻 messages 还没 append 本轮产物，
        # 正好就是这次请求发送出去的输入；输出 = 回复文本 + 工具调用参数。
        # API 出错走上面的 break，到不了这里——没产出不计费
        round_chars = context_stats_size(messages) + len(final_content)
        if tool_calls_acc:
            round_chars += sum(len(slot["arguments"]) for slot in tool_calls_acc.values())
        budget.record(round_chars)

        # 没调工具 = 最终回答，结束本轮。流式没有现成的 message 对象，
        # assistant 消息要手工组装塞回 messages，会话落盘才完整
        if not tool_calls_acc:
            messages.append({"role": "assistant", "content": final_content})
            break

        # 有工具调用：按 OpenAI 协议拼出 id/type/function 三件套再塞回 messages，
        # 下一轮 LLM 才能看到自己调过什么工具、结果是什么
        assistant_tool_calls = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for slot in tool_calls_acc.values()
        ]
        messages.append({
            "role": "assistant",
            "content": final_content or None,
            "tool_calls": assistant_tool_calls,
        })

        # 并行执行多个工具调用：独立请求并发跑，节约串行等待时间。
        # run_one_tool 在模块级定义，保证协议合法（messages 按 LLM 调用顺序追加）
        with ThreadPoolExecutor(max_workers=len(assistant_tool_calls)) as pool:
            # map 保证按入参顺序返回，但内部并发执行
            results = list(pool.map(run_one_tool, assistant_tool_calls))

        for call, name, args, result in results:
            # args 是 None 说明 JSON 解析失败，打印原始字符串
            args_repr = args if args is not None else f"(JSON 解析失败: {call['function']['arguments']!r})"
            _emit(f"  -> [调用工具] {name}({args_repr})", echo,
                  extra={"event": "tool_call", "tool": name})
            # 即使失败也必须回一条工具消息：否则有 tool_call 却没有对应结果，
            # 消息序列不符合协议，下一次 API 调用会被服务端直接拒绝
            # G4：所有工具结果统一视为不可信文本（含错误串——信任不靠枚举来源），
            # 回传前包进带随机 nonce 的数据边界，防资料里藏的注入指令被模型执行
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": wrap_untrusted(name, result),
            })

        # 工具结果回传后，回到 while 内循环顶部
        # LLM 看到工具结果，决定下一步：继续调工具 or 直接给出文字答案

    # 所有路径都从循环内 break 出来：正常回答/超时/断连/熔断统一在这里交出内容
    return final_content
