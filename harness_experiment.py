"""Harness 侧对比实验：星辰商城客服工单（对照组是 agent.py 手写单体 loop）。

同一个任务两边各跑一遍，对比四个维度：
agent loop / tool registry / permission gate / session store。

工具刻意各用一种注册方式，实测「实现与 schema 同源」的成本：
- search_knowledge → Tool 子类（my_knowledge_tool.py，已有）
- calculate        → Tool 子类（本文件）
- remember         → @tool_action 装饰器自动展开（hello-agents 0.2.9 内置机制）

用法（在项目根目录）：
  uv run python harness_experiment.py                  # 跑工单题面（默认）
  uv run python harness_experiment.py --recall         # 新进程=新会话，验证跨会话召回
  uv run python harness_experiment.py --question "..." # 换题（陷阱追问用）
  uv run python harness_experiment.py --dry            # 只打印注册表与 schema，不调 LLM
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()  # 必须在 import agent 之前：agent.py 模块级创建 OpenAI client

import agent  # 复用裸 loop 的工具实现——两边工具行为一致，只对比框架层
from hello_agents import FunctionCallAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import Tool, ToolParameter
from hello_agents.tools.base import tool_action

from my_knowledge_tool import KnowledgeSearchTool

TICKET_QUESTION = (
    "顾客买了 45 元的台灯和 39 元的小风扇，用微信支付。请回答："
    "① 这单收多少运费？"
    "② 台灯无理由退货的话，寄回运费谁承担、退款几天到账？"
    "③ 把给顾客的完整答复记到备忘里，下次值班直接用。"
)
RECALL_QUESTION = "上次台灯顾客的咨询，我们最后是怎么答复的？请复述要点。"


class CalculateTool(Tool):
    """calculate 的 Tool 子类封装：实现复用 agent.calculate，schema 单独手写。

    注意这里 schema 是第二份事实来源——agent.py 的 TOOLS 里已经写过一次
    同名工具的 schema，这就是「双处维护、容易漂移」的活例子。
    """

    def __init__(self):
        super().__init__(
            name="calculate",
            description="计算数学表达式，返回结果。用于金额、差额等算术。",
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        return agent.calculate(parameters["expression"])

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Python 语法的算术表达式，如 45+39、134-99",
                required=True,
            )
        ]


class MemoryNoteTool(Tool):
    """remember 的 @tool_action 封装：方法签名 + docstring 自动生成 schema。

    expandable=True + @tool_action：注册时 ToolRegistry 把它展开成独立的
    remember 工具，参数定义只存在于方法签名这一处——对比 CalculateTool
    的手写 schema，体感两种方案的维护成本差。
    """

    def __init__(self):
        super().__init__(name="memory", description="长期记忆存取", expandable=True)

    @tool_action("remember", "把一条结论性信息追加到长期记忆，下次会话可召回")
    def _remember(self, content: str) -> str:
        """记录结论到长期记忆。

        Args:
            content: 要记住的结论性内容，一两句话，含关键数字
        """
        return agent.remember(content)

    # 抽象方法兜底实现：展开后父工具自身不再被调用，但 ABC 要求可实例化
    def run(self, parameters: Dict[str, Any]) -> str:
        return self._remember(**parameters)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="content",
                type="string",
                description="要记住的结论性内容",
                required=True,
            )
        ]


def build_system_prompt() -> str:
    """客服角色 + 长期记忆注入。

    与 agent.py 的 build_system_prompt 同构：memory.md 全文进 system prompt。
    对比点：agent.py 在主循环文件里内联做这件事；harness 侧没有自动机制，
    必须显式接线——session store 在两个世界里都是自己动手，只是位置不同。
    """
    parts = [
        "# 角色\n你是星辰商城的客服助手。所有政策条款（运费/退货/退款/会员/积分）"
        "必须以 search_knowledge 返回的资料为准，禁止编造。",
        "# 工作方式\n"
        "- 业务问题先 search_knowledge，涉及金额用 calculate，"
        "最终给顾客的答复用 remember 存一条。\n"
        "- 按顾客的问题逐条回答，给出完整可发送给顾客的答复文本。",
    ]
    memory_path = Path(agent.MEMORY_PATH)
    if memory_path.exists():
        memory = memory_path.read_text(encoding="utf-8").strip()
        if memory:
            parts.append(f"# 历史值班记录（长期记忆）\n{memory}")
    return "\n\n".join(parts)


class MeteredFunctionCallAgent(FunctionCallAgent):
    """带计量的 FunctionCallAgent：LLM 轮数、工具调用次数与耗时。

    用继承插入计量而不是改 harness 源码——这本身就是对比素材：
    agent.py 的同类计量只能内联在 823 行的主循环里，harness 一个子类就挂上了。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_rounds = 0
        self.tool_calls = 0
        self.tool_seconds = 0.0
        self.call_log = []

    def _invoke_with_tools(self, messages, tools, tool_choice, **kwargs):
        self.llm_rounds += 1
        return super()._invoke_with_tools(messages, tools, tool_choice, **kwargs)

    def _execute_tool_call(self, tool_name, arguments):
        self.tool_calls += 1
        start = time.perf_counter()
        result = super()._execute_tool_call(tool_name, arguments)
        elapsed = time.perf_counter() - start
        self.tool_seconds += elapsed
        self.call_log.append((tool_name, arguments, elapsed))
        print(f"  🔧 [{self.tool_calls}] {tool_name} {arguments} -> {elapsed:.2f}s")
        return result


def build_agent() -> MeteredFunctionCallAgent:
    registry = ToolRegistry()
    registry.register_tool(KnowledgeSearchTool())
    registry.register_tool(CalculateTool())
    registry.register_tool(MemoryNoteTool())  # 自动展开为独立的 remember
    return MeteredFunctionCallAgent(
        name="星辰客服-harness",
        llm=HelloAgentsLLM(),
        system_prompt=build_system_prompt(),
        tool_registry=registry,
        max_tool_iterations=5,  # 默认 3：本任务约 4-5 次工具调用会撞线
    )


def print_metrics(agent_obj: MeteredFunctionCallAgent, total_seconds: float):
    print("\n=== 实验数据（harness 侧）===")
    print(f"LLM 轮数: {agent_obj.llm_rounds}（上限 max_tool_iterations=5）")
    print(f"工具调用: {agent_obj.tool_calls} 次，累计 {agent_obj.tool_seconds:.2f}s")
    print(f"总耗时: {total_seconds:.2f}s（总耗时 - 工具耗时 ≈ LLM 纯推理时间）")


def main():
    parser = argparse.ArgumentParser(description="harness 侧对比实验：星辰商城客服工单")
    parser.add_argument("--recall", action="store_true", help="模拟新会话，只问召回问题")
    parser.add_argument("--question", default=None, help="自定义问题（陷阱追问用）")
    parser.add_argument("--dry", action="store_true", help="只构建注册表打印 schema，不调 LLM")
    args = parser.parse_args()

    agent_obj = build_agent()

    if args.dry:
        registry = agent_obj.tool_registry
        print("注册表中的工具:", registry.list_tools())
        for schema in agent_obj._build_tool_schemas():
            fn = schema["function"]
            print(f"\n[{fn['name']}] {fn['description']}")
            print("  参数:", fn["parameters"])
        return

    question = RECALL_QUESTION if args.recall else (args.question or TICKET_QUESTION)
    print(f"\n=== 问题 ===\n{question}\n")

    start = time.perf_counter()
    answer = agent_obj.run(question)
    total = time.perf_counter() - start

    print(f"\n=== 最终回答 ===\n{answer}")
    print_metrics(agent_obj, total)


if __name__ == "__main__":
    main()
