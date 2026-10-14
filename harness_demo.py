from dotenv import load_dotenv
from hello_agents import ToolRegistry, HelloAgentsLLM, ReActAgent
# 方式一：register_function 快速接入
from agent import search_knowledge
# 方式二：Tool 子类
from my_knowledge_tool import KnowledgeSearchTool

load_dotenv()

# 方式一
# registry = ToolRegistry()
# registry.register_function(
# 	name="search_knowledge",
# 	description="星辰商城知识库检索：返回与问题最相关的 top-3 资料（带来源文件名和相似度）",
# 	func=search_knowledge
# )
# print(registry.get_tools_description())

# 方式二
registry = ToolRegistry()
registry.register_tool(KnowledgeSearchTool())
print(registry.execute_tool("search_knowledge", "星辰商城的退货政策是什么"))
print(registry.get_tools_description())
print(registry.get_all_tools()[0].to_dict())

CUSTOM_REACT_PROMPT = """你是一个能调用外部工具的智能助手。

可用工具如下：
{tools}

回应格式，必须严格遵守：
- 第一行以 "Thought: " 开头（英文冒号加一个空格），写你的思考过程
- 第二行以 "Action: " 开头，只能是以下两种之一：
- 工具名[参数]，例如：search_knowledge[退货政策]
- Finish[最终答案]
- 不要 Markdown 加粗（**），不要中文冒号，不要 JSON
- Thought 和 Action 各占一行，Action 必须单行
- Thought 中不要出现"Action:"这个词

示例：
Thought: 我需要先查询知识库中关于退货政策的资料。
Action: search_knowledge[退货政策]

Question: {question}
History: {history}
"""
llm = HelloAgentsLLM()
agent = ReActAgent(name="商城客服", llm=llm, tool_registry=registry, custom_prompt=CUSTOM_REACT_PROMPT)
answer = agent.run("星辰商城的退货政策是什么？")
print(f"\n最终答案：{answer}")