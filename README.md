# my-ai-app 学习路线（求职版）

从零手写 agent 到转岗 agent 应用开发的实战项目。路线对标 [Agent-Learning-Hub](https://github.com/) 的 Learning Todo List，当前进度：里程碑 A 已完成（A-3 跳过，改在 B2 用 LangChain @tool 实现）。

## 已有资产

- agent.py：手写 agent loop，9 个工具（天气/计算/读文件/列目录/grep/知识库检索/联网搜索/网页抓取/长期记忆），带 MAX_STEPS、超时、API 异常兜底
- rag.py + rag_index.json：星辰商城知识库，chunk / embed / 余弦检索 top-3，带来源标注
- session.json：会话记忆落盘，启动自动恢复（上下文膨胀问题已在 A-2 解决：超过阈值自动压缩旧消息）
- memory.md：长期记忆，启动时注入 system prompt

执行顺序：A → B → B2 → C → D → E → G（F 可选支线，按兴趣插入）。


## 里程碑 A：补强 agent.py —— 小型 harness 化

简历亮点：手写 agent loop 与上下文管理。

核心步骤：

- [x] 流式输出：stream=True，回答逐字打印 → 验证：CLI 体验质变
- [x] 上下文压缩：会话超过阈值时把旧消息总结成摘要，替代无脑全量保留 → 验证：连续对话 50 轮，上下文字符数有上界
- [ ] 工具注册表重构：装饰器模式，一个函数同时生成实现和 schema（现在 TOOL_FUNCTIONS 和 TOOLS 两处手工维护，容易漂移）→ 验证：新加一个工具只改一处（跳过：改在里程碑 B2 用 LangChain @tool）
- [x] 多工具并行执行：独立的 tool_call 并发跑（线程池或 asyncio）→ 验证：「北京和上海哪个温度高」耗时约等于单次调用

## 里程碑 B：精读一个现代 agent harness（Stage 3）

面试谈资：架构对比。选型二选一：hello-agents（中文 / Python / OpenAI 原生 API，最贴现有技术栈）或 learn-claude-code（想深挖 coding agent 形态）。

核心步骤：

- 精读第四章、第七章，加工具
- 重点看 9.3 的 ContextBuilder 和 9.6长程智能体实战，第八章浏览即可

- [ ] 跑通它的最小示例
- [ ] 给它加一个自己的工具
- [ ] 找到并读懂四个关键组件：agent loop、tool registry、permission gate、context compaction / session store
- [ ] 对比实验：同一个任务用「裸 agent loop」（agent.py 即是）和 harness 各实现一遍 → 验证：对比笔记写清楚裸 loop 差在哪

产出：对比笔记 + 可运行的 harness demo。
对比结论：裸 loop 不是跑起来弱，是改起来贵
- 裸loop把工具实现、schema、循环、观测、记忆五层焊在一个文件里，任何一层的改动半径都是整个文件；harness把每层拆成边界（子类、注册表、构造参数），每个边界就是改动隔离点

## 里程碑 B2：LangGraph 实战 —— 重写客服 agent

简历亮点：LangGraph 项目支撑。顺序逻辑：裸 loop（已完成）→ 精读 harness（B）→ 编排框架（B2），三层抽象各实现一遍。

核心步骤：

- [ ] 用 LangGraph 重写星辰商城客服 agent，工具直接复用 agent.py 现有 9 个函数 → 验证：功能对齐原版
- [ ] 攻关四个原版没有的能力：checkpoint（替代 session.json 手工落盘）、interrupt（危险工具人工确认）、subgraph（给 search_knowledge 套检索子图）、streaming → 验证：四个能力各有可演示场景
- [ ] 顺带掌握 LangChain 最小核心抽象：message / model / runnable 是什么、能读懂它的代码即可，不深入 chain 写法
- [ ] 精读 langchain-ai/open_deep_research 源码（LangGraph 官方 deep research 范例，面试聊架构直接可用）

## 里程碑 C：迷你评测（Stage 7 提前）

简历亮点：评测意识，大多数候选人没有。动机：路线自己的 Learning Principles 就写着「Add evals before you add more agents」。

核心步骤：

- [ ] 建 20~30 条固定测试集：问题 + 期望答案要点 + 应调用的工具
- [ ] eval.py 可重复运行，输出汇总：正确率、工具调用次数、token 成本、延迟、失败分类
- [ ] 失败分类四档：检索没命中 / 工具选错 / 编造引用 / 其他
- [ ] 接入 LangSmith（或 LangFuse）看 trace：打开一次完整调用的 trace，定位一次真实失败 → 验证：能说清失败发生在 prompt / 工具 / 检索 / 模型哪一层
- [ ] 根据评测结果回头调 rag.py 的 chunking 和检索策略 → 验证：改动前后正确率对比可见

产出：eval.md 或 eval.py，可重复跑的评测表。

## 里程碑 D：Skills 与 MCP（Stage 5）

选它的原因：MCP 学完马上能用在自己每天用的工具上，正反馈最强。

核心步骤：

- [ ] 理清三者区别：tool 是可调用接口，skill 是可复用流程知识，prompt 是一次性指令
- [ ] 用 FastMCP 把 search_knowledge（或 grep）包装成 MCP server
- [ ] 在 Claude Code 里接入自建 MCP server 并实际调用 → 验证：Claude Code 能查到星辰商城知识库
- [ ] 写一个最小 SKILL.md：name、何时使用、步骤、验收标准
- [ ] 了解 A2A / ACP 协议定位（agent 互连 / 宿主接入），概念层面能答即可，不实现

## 里程碑 E：多 agent 压缩版（Stage 4，用 LangGraph 做）

原则：做小不做大，一个三角色系统足以理解协调问题，不碰 CrewAI 类模板化框架。

核心步骤：

- [ ] 用 LangGraph 的 supervisor 模式搭 research → write → review → revise 小系统
- [ ] 复用 web_search / web_fetch 工具
- [ ] 明确每个 agent 的职责边界、输入输出 schema、停止条件
- [ ] 实际处理至少一种问题：循环、任务漂移、上下文膨胀，记录处理过程
- [ ] 得出自己的结论：这个任务什么时候单 agent 反而更好

产出：一个小型多 agent 系统。

## 里程碑 F（可选支线）：浏览器 agent（Stage 6）

按兴趣和目标岗位决定做不做。做的话用 browser-use，只操作公开网页，记录截图和动作日志方便复盘。

## 里程碑 G：安全加固与服务化（Stage 7 剩余 + Stage 8）

简历亮点：项目闭环——别人能 clone 下来跑。两个现成的安全素材就在本项目里。

核心步骤：

- [ ] read_file 加路径白名单（现在可读任意路径，是越权风险）
- [ ] calculate 的 eval 换成基于 ast 的白名单解析（现在的空 builtins 沙箱可用 __class__ 链绕过）
- [ ] 危险工具加人工确认机制，加成本 / 步数上限
- [ ] prompt injection 防线：往知识库塞一条「忽略以上指令…」类注入用例，验证 agent 不执行 → 验证：注入测试不生效
- [ ] FastAPI 服务化或 CLI 打包，确定一种部署形态
- [ ] README 收尾：运行方式、key 配置、如何扩展工具、已知限制

产出：可部署、有权限边界、有 README 的完整项目。

---

## 面试故事线

手写 agent loop（A）→ 架构对比（B）→ LangGraph 实战（B2）→ 评测体系（C）→ MCP 扩展（D）→ 多 agent 协调（E）→ 服务化上线（G）。每一步都能回答「为什么需要它」：裸 loop 撑不住才上 harness，手写 checkpoint 太脏才用 LangGraph，没有评测就是盲改，不能服务化就不算产品。

## 参考资料速查

- 路线总纲：Agent-Learning-Hub README（Stage 0-8 + Project Ladder）
- harness 选型：datawhalechina/hello-agents、shareAI-lab/learn-claude-code
- LangGraph：langchain-ai.github.io/langgraph、langchain-ai/open_deep_research
- 评测参照：OpenAI Evals、LangSmith、AgentBench / SWE-bench
- MCP：modelcontextprotocol.io、官方 MCP Servers 合集
- 浏览器 agent（F）：browser-use/browser-use
