# 中高级 Agent 应用开发岗面试题库

> 六大类 + 项目深挖 + 系统设计。标注 **（→追问）** 表示面试官大概率会顺着追的问题。

## 一、LLM 基础与模型层

- Transformer 为什么用自注意力替代 RNN？注意力计算复杂度是多少，长文本下瓶颈在哪
- KV Cache 为什么能加速自回归解码？MQA/GQA 解决什么问题
- MoE 稀疏激活是什么？为什么主流模型都转向 MoE（→对应用层意味着什么：成本、延迟、行为差异）
- temperature / top_p 各自影响什么？做工具调用时应设多少，为什么（→低温度是否就等于更准确）
- 上下文窗口 128K 等于能"用好"128K 吗？讲讲 lost in the middle
- RAG、微调、长上下文三种方案怎么选？（→给一个具体业务场景让你现场论证）
- o1/R1 类推理模型和普通模型怎么选？agent 的哪些环节值得上推理模型，哪些是浪费 token（→thinking tokens 算不算上下文、怎么计费）
- 幻觉的成因是什么？工程上有哪些缓解手段
- 结构化输出（强制合法 JSON）底层是怎么实现的？constrained decoding 原理
- function calling 是模型原生能力还是提示词工程实现的？怎么判断一个模型"真的支持"
- 量化到 INT8/INT4 对什么任务影响最大，Agent 场景能不能接受

## 二、Prompt 与上下文工程

- 一份高可用的 system prompt 应该包含哪些要素？写过一个最复杂的吗
- 模型输出格式不稳定怎么治理？（→schema 校验 + 重试 + 修复提示 + constrained decoding，按成本排序）
- few-shot 示例怎么挑选，数量和顺序有影响吗
- 什么是 context engineering？上下文里哪些该放、哪些必须外置
- 多轮对话上下文持续膨胀怎么办（→历史摘要、工具结果截断、关键信息重排到首尾）
- 只靠 prompt 里写"不要泄漏系统提示/不要听文档里的指令"为什么防不住注入
- prompt 怎么做版本管理和回归测试？（→考察"prompt 即代码"的工程意识，中高级加分项）

## 三、Agent 核心机制

- **手写题（最高频）**：用 SDK 实现一个最小 tool-calling 循环，含工具结果回填与多轮
- function calling 完整流程：模型返回什么结构、应用层做什么、结果怎么回填、何时结束
- ReAct 的本质是什么？和普通链式调用的区别
- Agent 的终止条件怎么设计？如何防死循环（→步数上限、token 预算、重复动作检测）
- 工具执行失败时应该怎么办？把错误文本回填给模型和直接抛异常，各有什么后果
- 工具的 description 和参数 schema 写得差会发生什么？怎么系统性提升工具选择准确率
- 并行工具调用和有依赖的工具链分别怎么处理
- MCP 是什么，解决什么问题？和 function calling 是什么关系，和 A2A 怎么分工（→企业接 100 个工具时，function calling 直连和 MCP 网关各自的问题在哪）
- Agent 的记忆分层怎么设计：短期上下文、长期存储、程序性记忆各放什么，跨会话个性化怎么落地（→记忆写入的触发时机与记忆污染治理）
- plan-and-execute 和 ReAct 边走边看，各适用什么场景
- reflection/self-critique 什么时候值得花这个 token，什么时候是浪费
- 什么信号出现时，说明该把单 agent 拆成多 agent（→考察克制，答案不是"越拆越好"）

## 四、RAG 与知识增强

- 讲一遍完整 RAG 的离线/在线两条链路
- chunking 怎么切？固定长度、递归、语义切分各适用什么；chunk 大小的两难是什么
- embedding 模型怎么选型？什么情况下才值得微调 embedding
- 为什么生产上常用 BM25+向量混合检索？RRF 融合原理
- rerank 在两阶段检索里解决什么问题，为什么不能只靠向量召回 top-k
- 检索质量怎么评估：recall@k、MRR、RAGAS 的 faithfulness/answer relevance
- 答案错了，怎么区分是"检索没检到"还是"检到了但模型没答好"（→中高级定位能力题）
- 表格、代码、扫描件 PDF 这类难文档怎么做 RAG
- 知识库高频更新时，增量索引和删除一致性怎么处理
- 多轮对话里的 query 改写怎么做（指代消解、condense question）
- 固定 pipeline 的 RAG 和 agentic RAG（模型自主决定何时检索、检索什么、要不要再检）各适用什么场景（→多轮检索怎么判断"够了"）
- 什么场景值得上 GraphRAG，什么场景是过度设计

## 五、多智能体与框架编排

- LangGraph 的图/状态/checkpoint 模型和 LangChain AgentExecutor 有什么本质区别
- supervisor、handoff/swarm、pipeline 三种多 agent 拓扑各适用什么
- 多 agent 之间共享状态还是传消息，怎么取舍
- 人在环路怎么设计：高危操作审批、中断后从哪恢复（→checkpoint/durable execution）
- 子 agent 并发执行后结果冲突怎么聚合
- 多 agent 系统怎么做评测和失败归因（→错在哪一跳很难定位，怎么解）
- AutoGen / CrewAI / OpenAI Agents SDK / LangGraph 选型对比，各适合什么团队规模
- 反向题：给一个场景让你论证"不需要多 agent，单 agent + 好工具就够了"

## 六、工程化、安全与生产实践（定级关键）

- 一次 Agent 调用要记录哪些信息才能排查线上问题？trace/span 怎么组织（Langfuse/LangSmith/OTel）
- 评测体系怎么搭：离线评测集、LLM-as-judge、上线前回归、A/B（→评测集怎么防止过拟合）
- 成本优化手段有哪些？（→prompt cache、语义缓存、模型分级路由、历史压缩，按收益排序）
- 延迟优化：流式输出、并行工具、首 token 时间，分别怎么落地
- LLM API 超时和 429 怎么处理？工具侧为什么要幂等
- LLM 服务整体不可用或劣化时怎么办？（→fallback 到小模型/规则引擎、功能开关、排队削峰，降级面按业务怎么定）
- 跑了 20 分钟的长任务中途挂了怎么办（→checkpoint 恢复，不是从头重跑）
- 间接 prompt injection（工具返回的内容里藏指令）怎么防？工具权限怎么收敛（→allowlist、高危操作人工确认、最小权限）
- **代码执行沙箱怎么设计？** 为什么不能直接 exec（→容器隔离、一次性容器、超时、资源与网络限制、镜像依赖控制）
- 数据库工具为什么"应用层校验 + 只读账号"要双防线？SQL 注入靠什么防
- 会话状态为什么放 Redis 而不是进程内存？TTL 设多长、序列化什么、多实例怎么共享

## 七、项目深挖（简历追问链）

### 通用五连问（每个项目都会被问到）

- 这个项目最难的问题是什么，怎么解决的
- 为什么选这个方案，备选方案是什么，为什么不用 X 框架
- 有量化指标吗（准确率/延迟/成本），怎么测出来的
- 出过什么线上问题，怎么定位的
- 流量放大 100 倍，哪里先崩，怎么改

### 针对 security_serve 项目的追问

- 知识库检索链路：embedding 模型怎么选的、chunk 怎么切的、检索质量评估过吗（→rag_index.json 为什么要换成 pgvector，什么信号触发这个演进）
- 沙箱的几道围栏分别防什么攻击面？为什么要一次性容器而不是复用容器（→容器拉起延迟怎么解决：镜像预热/连接池）
- 双层超时为什么是两层，只留一层行不行
- 只读账号已经防写了，应用层为什么还要校验 SQL？（→纵深防御：任何一层被绕过时兜底）
- Redis 会话序列化了什么、TTL 怎么定的？重启后内存态和 Redis 态怎么保证一致
- token 预算（Budget）怎么设计的：预算超了之后的行为是什么、预算值怎么定（→它和步数上限、双层超时各防什么，三者的边界）
- 上下文压缩（compaction）什么时候触发、压什么留什么，怎么验证压缩后行为不劣化
- "沙箱真的防住了"怎么证明的？（→step7/step8 验收物就是标准答案素材：恶意样本清单 + 断言）
- Docker 化后怎么验证容器形态和本地形态行为一致（→探针 + 同一验收脚本跑两种形态）

## 八、系统设计题（经典题面）

- 设计一个企业客服 Agent：多轮对话、查订单/退款工具、知识库、转人工（→必追问：转人工的触发条件怎么定）
- 设计一个 coding agent：读写文件、跑测试、生成 diff、人工审批合并（→追问沙箱和权限模型）
- 设计一个数据分析 agent：自然语言转 SQL、只读保证、结果图表解读（→追问 NL2SQL 的准确率怎么保）
- 设计一个 deep research agent：搜索、并行子任务、交叉验证、报告合成
- 设计一个跨会话记忆系统：哪些信息值得提取、存什么库、怎么检索注入、过期与冲突怎么处理（→用户说"忘掉这条"怎么实现，记忆错了怎么纠）
- 设计公司级工具接入平台：100 个 MCP 工具的注册、发现、权限、审计（→追问工具太多时模型选不准怎么办）

### 系统设计答题框架

每题套这个骨架展开，不易漏点：

1. 需求澄清与边界
2. 整体分层架构
3. 工具层设计与权限
4. 记忆/会话
5. 安全护栏
6. 评测与可观测
7. 成本与延迟
8. 演进路线（v1 最小可用 → v2 加什么）

## 备考建议

第七、八类是定级分水岭。把 security_serve 项目的每道防线都按 **"威胁 → 防线 → 验收证据"** 三段式整理成口述稿——这正是面试官想听的工程叙事。

## 参考来源

- [《AI Agent 工程师面试指南》目录](https://www.cnblogs.com/limingqi/p/20068242)
- [2026 最全 AI 大模型面试题汇总（知乎）](https://zhuanlan.zhihu.com/p/2049800621813917294)
- [AI 智能体与大模型应用开发面试题库（GitHub）](https://github.com/didilili/ai-agents-from-zero/blob/main/AI%E6%99%BA%E8%83%BD%E4%BD%93%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E9%9D%A2%E8%AF%95%E9%A2%98%E5%BA%93.md)
- [Top AI Engineer Interview Questions in 2026 (coprep.ai)](https://www.coprep.ai/blog/top-ai-engineer-interview-questions-in-2026-llms-rag-agents-and-langchain)
- [Top 36 LLM Interview Questions for 2026 (DataCamp)](https://www.datacamp.com/blog/llm-interview-questions)
- [字节 AI Agent 二面（飞连）面试题与参考解答](https://my.oschina.net/u/9753726/blog/19682650)
