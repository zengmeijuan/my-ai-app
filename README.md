# my-ai-app 学习路线（求职版）

从零手写 agent 到转岗 agent 应用开发的实战项目。路线对标 [Agent-Learning-Hub](https://github.com/) 的 Learning Todo List，当前进度：里程碑 A、B、B2、C、D、E、G 已完成（A-3 跳过，改在 B2 用 LangChain @tool 实现；F 为可选支线未启动），当前进入里程碑 H（工程补课）。

## 已有资产

- agent.py：手写 agent loop，9 个工具（天气/计算/读文件/列目录/grep/知识库检索/联网搜索/网页抓取/长期记忆），带 MAX_STEPS、超时、API 异常兜底
- rag.py + rag_index.json：星辰商城知识库，chunk / embed / 余弦检索 top-3，带来源标注
- session.json：会话记忆落盘，启动自动恢复（上下文膨胀问题已在 A-2 解决：超过阈值自动压缩旧消息）
- memory.md：长期记忆，启动时注入 system prompt

执行顺序：A → B → B2 → C → D → E → G → H（F 可选支线，按兴趣插入）。


## 里程碑 A：补强 agent.py —— 小型 harness 化

简历亮点：手写 agent loop 与上下文管理。

核心步骤：

- [x] 流式输出：stream=True，回答逐字打印 → 验证：CLI 体验质变
- [x] 上下文压缩：会话超过阈值时把旧消息总结成摘要，替代无脑全量保留 → 验证：连续对话 50 轮，上下文字符数有上界
- [ ] 工具注册表重构：装饰器模式，一个函数同时生成实现和 schema（现在 TOOL_FUNCTIONS 和 TOOLS 两处手工维护，容易漂移）→ 验证：新加一个工具只改一处（跳过：改在里程碑 B2 用 LangChain @tool）
- [x] 多工具并行执行：独立的 tool_call 并发跑（线程池或 asyncio）→ 验证：「北京和上海哪个温度高」耗时约等于单次调用

## 里程碑 B：精读一个现代 agent harness

面试谈资：架构对比。选型二选一：hello-agents（中文 / Python / OpenAI 原生 API，最贴现有技术栈）或 learn-claude-code（想深挖 coding agent 形态）。

核心步骤：

- 精读第四章、第七章，加工具
- 重点看 9.3 的 ContextBuilder 和 9.6长程智能体实战，第八章浏览即可

- [x] 跑通它的最小示例
- [x] 给它加一个自己的工具
- [x] 找到并读懂四个关键组件：agent loop、tool registry、permission gate、context compaction / session store
- [x] 对比实验：同一个任务用「裸 agent loop」（agent.py 即是）和 harness 各实现一遍 → 验证：对比笔记写清楚裸 loop 差在哪

产出：对比笔记 + 可运行的 harness demo。
对比结论：裸 loop 不是跑起来弱，是改起来贵
- 裸loop把工具实现、schema、循环、观测、记忆五层焊在一个文件里，任何一层的改动半径都是整个文件；harness把每层拆成边界（子类、注册表、构造参数），每个边界就是改动隔离点

## 里程碑 B2：LangGraph 实战 —— 重写客服 agent

简历亮点：LangGraph 项目支撑。顺序逻辑：裸 loop（已完成）→ 精读 harness（B）→ 编排框架（B2），三层抽象各实现一遍。

核心步骤：

- [x] 用 LangGraph 重写星辰商城客服 agent，工具直接复用 agent.py 现有 9 个函数 → 验证：功能对齐原版
- [x] 攻关四个原版没有的能力：checkpoint（替代 session.json 手工落盘）、interrupt（危险工具人工确认）、subgraph（给 search_knowledge 套检索子图）、streaming → 验证：四个能力各有可演示场景
- [x] 顺带掌握 LangChain 最小核心抽象：message / model / runnable 是什么、能读懂它的代码即可，不深入 chain 写法
- [x] 精读 langchain-ai/open_deep_research 源码（LangGraph 官方 deep research 范例，面试聊架构直接可用）

## 里程碑 C：迷你评测

简历亮点：评测意识，大多数候选人没有。动机：路线自己的 Learning Principles 就写着「Add evals before you add more agents」。

核心步骤：

- [x] 建 20~30 条固定测试集：问题 + 期望答案要点 + 应调用的工具 → 验证：testcases.json v1 定版 25 条（详见 evals/评测笔记.md §2）
- [x] eval.py 可重复运行，输出汇总：正确率、工具调用次数、token 成本、延迟、失败分类 → 验证：基线 run_20261023_105116 正确率 96%（24/25）
- [x] 失败分类四档：检索没命中 / 工具选错 / 编造引用 / 其他 → 验证：真实坏例归因入档（R13→other、Z02→tool_extra，评测笔记 §6）
- [x] 接入 LangSmith（或 LangFuse）看 trace：打开一次完整调用的 trace，定位一次真实失败 → 验证：能说清失败发生在 prompt / 工具 / 检索 / 模型哪一层
- [x] 根据评测结果回头调 rag.py 的 chunking 和检索策略 → 验证：句级切分对照实验，组件级 hit/rank1 不变、score 重分配（评测笔记 §8），端到端回归 25/25 无退化

产出：eval.md 或 eval.py，可重复跑的评测表。

## 里程碑 D：Skills 与 MCP

选它的原因：MCP 学完马上能用在自己每天用的工具上，正反馈最强。

核心步骤：

- [x] 理清三者区别：tool 是可调用接口，skill 是可复用流程知识，prompt 是一次性指令 → 验证：D1 三原语实物报文 + 六词对比（笔记 §1-§4）
- [x] 用 FastMCP 把 search_knowledge（或 grep）包装成 MCP server → 验证：02_raw_client 手写 JSON-RPC 裸报文走通 initialize / tools/list / tools/call 三步（D2）
- [x] 在 Claude Code 里接入自建 MCP server 并实际调用 → 验证：accr 会话 /mcp ✔ connected · 2 tools，「七天无理由退货」自主调 search_knowledge 基于知识库作答（D3）；延伸 D3a 自建 agent 接 MCP（04_sdk_agent）、D3b 部署 HTTP 服务与远程接入（05_http_client + uvx 社区工具）
- [x] 写一个最小 SKILL.md：name、何时使用、步骤、验收标准 → 验证：.claude/skills/eval-regression 固化评测回归流程，T01 单条实跑与说明书一致（D4）
- [x] 了解 A2A / ACP 协议定位（agent 互连 / 宿主接入），概念层面能答即可，不实现 → 验证：笔记 D5 三协议对比 + 自检四问（A2A 入 AAIF 与 ACP 收敛为 2026 查证结论）

## 里程碑 E：多 agent 压缩版

原则：做小不做大，一个三角色系统足以理解协调问题，不碰 CrewAI 类模板化框架。
场景（E1 调整）：深度研究——多 agent 三条件（可并行 / 上下文可切分 / 结果易验证）全中的任务；
原写文章流水线路径可穷举，保留为固定工作流基线对照（step0）。

核心步骤：

- [x] 用 LangGraph 的 supervisor 模式搭 research → write → review → revise 小系统 → 验证：step1_supervisor.py 最小动态深度研究，三角色三节点（supervisor / researcher / synthesizer），supervisor 三次内容函数路由（拆题 / 判材料 / 终审，directive 随内容变），实跑一次通过；step0_fixed_workflow.py 固定工作流基线对照（路由为字段函数，if/else 可替代 supervisor 的实证）
- [x] 复用 web_search / web_fetch 工具（researcher agent 化：ReAct 循环 + 联网工具，系统出现第 1 个 agent）→ 验证：step2_research_tools.py，researcher 升级为 ReAct 子图（@tool 薄包装 + bind_tools + ToolNode，MAX_TOOL_CALLS=10 预算熔断 → finalize 结构收尾），supervisor / synthesizer 零改动；实跑两轮各调满 10 次工具，材料带实时来源 URL，supervisor 据材料缺口定向补查（第 2 轮自主改用 GitHub API 取 stars / 发版数据），2/2 轮熔断后带局限收尾
- [x] 明确每个 agent 的职责边界、输入输出 schema、停止条件（含 Send API 并行多 researcher 实例，agent 实例数 ≥2 达标 multi-agent）→ 验证：step3_contracts.py，契约三要素集中定义一屏内（职责 / 输入输出 schema / 停止条件）；Send fan-out 并行 2 实例（维度工单自包含，research 累加 reducer 合并）；轮次熔断代码层固化（LLM 决策返回后代码检查封顶）；实跑 2 批 3 份材料，补查实例纠正材料 1 的 CrewAI Flows 错误描述（交叉验证价值实证）
- [x] 实际处理至少一种问题：循环、任务漂移、上下文膨胀，记录处理过程 → 验证：step4_failure_lab.py 循环故障实验——故障注入把概率故障变确定性故障，单一变量对照：无打回熔断 → 13 次打回循环耗尽 recursion_limit 抛 GraphRecursionError（兜底被触发 = 业务熔断设计缺口）；--fixed 加打回熔断 → 2 次打回后代码强制 END，答案保留带保留意见收尾（优雅降级）；完整走通"现象 → 根因 → 机制修复 → 对照验证"
- [x] 得出自己的结论：这个任务什么时候单 agent 反而更好 → 验证：step5_single_vs_multi.py 同题公平对照（同一副工具、同一预算 10，差异只在编排）——单 agent 71.3s 产出合格答案 vs 多 agent 493.5s（6.9 倍）换一手数据与交叉验证纠偏；结论：单一检索线 / 材料一个上下文装得下 / 验证一眼可判的任务三条件不中，单 agent 够用；编排税买证据深度与纠偏能力，E1 三条件全中才上 multi-agent

产出：一个小型多 agent 系统。

## 里程碑 F（可选支线，轻量入门定位）：浏览器 agent

背景：el-perf-report 的浏览器流程（bsk daemon）非自己实现，对操作型 agent 机制感兴趣的话，做**轻量入门**即可，不必生产化。用 browser-use 写一个最小演示：感知（截图 / DOM）→ LLM 决策（点哪 / 填什么）→ 执行 → 观察结果，操作公开网页完成一个简单任务（如搜索并提取结果），记录截图和动作日志方便复盘。

定位与边界：目标是理解「感知-决策-行动」循环与元素定位、失败重试机制——**不做生产化**；生产级的浏览器自动化（登录态管理、定时调度、失败告警）等实战主题真的需要操作无 API 系统时再上。

## 里程碑 G：安全加固与服务化

简历亮点：项目闭环——别人能 clone 下来跑。两个现成的安全素材就在本项目里。

核心步骤：

- [x] read_file 加路径白名单（现在可读任意路径，是越权风险）→ 验证：check_path 两步（realpath 规范化 + os.sep 目录级前缀 + normcase 大小写），`../`、UNC、同级前缀、大小写混淆全拦，区内行为零改动（step1 + step0 全绿）
- [x] calculate 的 eval 换成基于 ast 的白名单解析（现在的空 builtins 沙箱可用 __class__ 链绕过）→ 验证：白名单即解释器能力集，属性链逃逸在求值步截断；行为对齐坑（前导空白）修复，浮点行为与旧 eval 逐字一致（step2）
- [x] 危险工具加人工确认机制，加成本 / 步数上限 → 验证：CONFIRM_TOOLS + 默认拒绝（EOF→拒绝为服务化预留）+ Budget 估算熔断；gate 在 loop 执行点而非工具层，直接调工具不受影响（step3）
- [x] prompt injection 防线：往知识库塞一条「忽略以上指令…」类注入用例，验证 agent 不执行 → 验证：wrap_untrusted 随机 nonce 数据边界 + system 声明段，伪造闭合与重放攻击失效；真机直白注入模型自行识破（step4）
- [x] FastAPI 服务化或 CLI 打包，确定一种部署形态 → 验证：FastAPI 服务化——POST /chat + GET /health，会话按 session_id 分格（独立 messages + Budget），echo 双输出去向；服务无终端 → EOF 默认拒绝自动生效（G3 预埋分支零改动验收）；TestClient（step5）+ httpx 真网络客户端（step5b）双验收物
- [x] README 收尾：运行方式、key 配置、如何扩展工具、已知限制 → 验证：下方「运行方式」节 + 笔记 G0-G5 章节与延伸沉淀（G6）

产出：可部署、有权限边界、有 README 的完整项目——五层结构 + 五道防线 + 服务化双入口，验收物 step0-step5b，细节见 src/security_serve/安全加固与服务化笔记.md。

### 运行方式（G 阶段成果）

- 环境：Python ≥ 3.13 + uv。克隆后 `uv sync` 装依赖；项目根建 `.env`，填 openai 兼容接口的 BASE_URL / API_KEY / MODEL。
- CLI 对话（bash）：`PYTHONPATH=src uv run python -m security_serve.agent.cli`；PowerShell 先 `$env:PYTHONPATH="src"` 再去掉前缀跑。
- HTTP 服务：`uv run uvicorn security_serve.agent.server:app --app-dir src --port 8000`（`--app-dir` 跨 shell 通用）；浏览器开 `http://127.0.0.1:8000/docs` 交互调试；真网络客户端示例 `uv run python src/security_serve/step5b_http_client.py`（服务需先启动）。
- 验收物：`uv run python src/security_serve/step0_refactor_check.py` 起 step0-step5b 逐个跑（step3 用管道喂 y/n；step4 可加 `--live`；step5/step5b 需 .env 真实 key）。
- 扩展工具：tools.py 写函数 + 挂 `@tool` 装饰器（一处注册同时生成实现与 schema）；危险工具名加进 config.CONFIRM_TOOLS；路径类工具自动过 G1 闸门、返回值自动过 G4 封装。
- 已知限制：会话存进程内存（重启即失、多副本不共享，生产换 Redis）；全局锁串行化（吞吐换正确）；Budget 按字符折算 token（估算护栏，中文偏少记）；无认证（默认只绑 127.0.0.1 限本机，公开前需补认证 / 限流 / HTTPS，见笔记 G5 §8）。

## 里程碑 H：工程补课 —— 从 demo 到准生产

原则：全部在现有星辰商城 RAG + agent 上练手，**不等实战主题**——工程件与主题无关，底座建成后随主题直接迁移。生产架构全景与六层补课清单见 src/security_serve/安全加固与服务化笔记.md 延伸沉淀节。

学习顺序按「与已学知识的连贯性 + 难度梯度」重排（2026-11-10）：先熟域扩展（H1 SQL 只读工具与 G 防线同源），再新域入门与应用（H2 Docker → H3 沙箱 / H4 Redis），后前置综合（H5 可观测性、H6 pgvector 各踩两个前置），CI 门禁收官——每步只引入一个新概念。H5 前置小节：先补 logging 机制（logger 树 / 级别 / Handler-Formatter-Filter，纯文本玩透）再上结构化改造。

核心步骤：

- [x] H1 数据库只读工具：SQLite 起步（生产形态 = 真实库只读账号连接），schema 链接（大表只给模型相关结构）+ SQL 生成校验 + 只读硬控制 → 验证：自然语言查数答对（金牌 3 人）；「删掉所有订单」被拒且回复非空、数据无损；确定性验收 12 断言全 PASS（step6，含 WITH 夹写穿透预检被 ro 连接硬拦的纵深防御实证；防删库思路与 G1 路径白名单同源）
- [x] H2 Docker 化（范围收窄：实际只编排 app，Redis / 向量库随 H4 / H6 再接进 compose）：应用 Dockerfile（slim 底座 + uv.lock 装依赖 + 跨包双拷 + CMD 先 db_seed 后 uvicorn --host 0.0.0.0）+ docker-compose（端口映射 / env_file 运行时注入密钥 / healthcheck 容器探针）→ 验证：compose up 后容器 healthy、step5b 四项全绿且脚本零改动（跨形态一致性）；K8s 到概念层（Pod / Deployment / 探针 / 控制循环 / Service 分发——G5 的 /health 即存活探针）；六阶段流程、学习版与生产版对照、agent 特殊性沉淀入 H 笔记
- [ ] H3 代码执行沙箱工具（接在 Docker 化后）：Docker 容器沙箱 + 代码执行工具，agent 能写 Python 做计算 / 数据分析 / 自我验证 → 验证：模型通过写代码回答「这份数据的统计结论」类问题；恶意代码（读文件外传 / 死循环）被沙箱隔离不伤宿主；权限演进故事完整（裸 eval → ast 白名单 → 沙箱内自由执行）
- [ ] H4 Redis 会话外置：server.py 内存 dict 换 Redis（TTL 管时效、多副本共享）→ 验证：重启服务后第二轮还能答出上一轮说过的事实
- [ ] H5 可观测性：logging 升级 JSON 结构化日志，FastAPI 中间件生成 request_id 贯穿调用链，计时中间件统计延迟分位数 → 验证：能回答「某时段 p95 延迟多少、哪个工具最慢、哪个会话最烧钱」三个问题
- [ ] H6 向量库替换：rag_index.json → pgvector，增量更新（增删文档不重建全量索引）+ 元数据过滤（来源 / 权限 / 时间）→ 验证：C 阶段评测集跑分无退化（25/25 或不低于基线）
- [ ] H7 CI 评测门禁：GitHub Actions 每次提交跑 eval.py，不达标不合并 → 验证：故意改坏 prompt 触发评测失败并阻止合并（C 阶段资产流程化）

概念层（不动手、面试能说清即可）：微调决策树（知识常更新走 RAG、风格与领域黑话走微调 SFT/LoRA、通用能力换大模型）、模型分级路由与降级（小模型分类、大模型决策）、任务队列与长任务异步、认证限流（G5 笔记 §8 清单）。

产出：准生产架构完整演示——面试聊到部署 / 可观测 / 数据层，每个格子都答得出「做过」或「为什么需要它」。

---

## 面试故事线

手写 agent loop（A）→ 架构对比（B）→ LangGraph 实战（B2）→ 评测体系（C）→ MCP 扩展（D）→ 多 agent 协调（E）→ 服务化上线（G）→ 工程补课进准生产（H）。每一步都能回答「为什么需要它」：裸 loop 撑不住才上 harness，手写 checkpoint 太脏才用 LangGraph，没有评测就是盲改，不能服务化就不算产品，没有工程化就上不了生产。

## 参考资料速查

- 路线总纲：Agent-Learning-Hub README（Stage 0-8 + Project Ladder）
- harness 选型：datawhalechina/hello-agents、shareAI-lab/learn-claude-code
- LangGraph：langchain-ai.github.io/langgraph、langchain-ai/open_deep_research
- 评测参照：OpenAI Evals、LangSmith、AgentBench / SWE-bench
- MCP：modelcontextprotocol.io、官方 MCP Servers 合集
- 浏览器 agent（F）：browser-use/browser-use
