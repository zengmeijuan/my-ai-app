from typing import Dict, Any, List
from hello_agents.tools import Tool, ToolParameter
from my_agent.agent import search_knowledge

class KnowledgeSearchTool(Tool):
	def __init__(self):
		super().__init__(
			name="search_knowledge",
			description="星辰商城知识库检索：返回与问题最相关的 top-3 资料（带来源文件名和相似度）"
		)

	def run(self, parameters: Dict[str, Any]) -> str:
		query = parameters.get("input") or parameters.get("query") or ""
		return search_knowledge(query)

	def get_parameters(self) -> List[ToolParameter]:
		return [
			ToolParameter(
				name="input",
				type="string",
				description="要检索的问题或关键词",
				required=True
			)
		]