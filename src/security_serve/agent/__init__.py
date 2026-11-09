"""security_serve.agent —— G0 重构版 agent 包（五层边界）。

层次与依赖单向：cli（入口组装）→ loop（agent 循环）→ {tools（注册表+9 工具）,
safety（G 防线，G1-G3 填肉）, memory（会话/长期记忆/压缩）} → config（全部调参）。
每个边界就是改动隔离点——改任何一层的理由都只落在一个文件。

旧版 src/my_agent/agent.py 冻结为 A 阶段历史教学物（B2/E/evals 的引用不受影响），
本包是 G 阶段的可部署产品。
"""
