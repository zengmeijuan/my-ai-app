# mcp_and_skill 笔记（里程碑 D：Skills 与 MCP）

> 里程碑 D 学习笔记，对应 README「里程碑 D：Skills 与 MCP」。
> 目标：理清 agent / skill / MCP 三个主角与六词对比（D1）、FastMCP 把商城工具包成
> server 并接入 Claude Code（D2/D3）、把评测回归流程固化成 SKILL.md（D4）、
> MCP/A2A/ACP 协议定位（D5）。每节带「生产对照」——学习产物要能对照真实生产环境印证。
> 组织原则：**先说清本体核心（是什么、为什么存在），再谈规范，最后才对比**。

## D1 概念地基：agent / skill / MCP 三个主角，与六词对比

### 1. 三个主角是什么、为什么产生

**agent——为什么产生？LLM 本体只会"文本进、文本出"。** 它没有手（不能执行任何操作）、
没有持续记忆（对话窗口外的都忘了）、不会多步（一次生成完就结束）。agent 就是给 LLM
装上三件套的工程封装：**工具**（手：调 calculate、查知识库）、**循环**（多步：调完工具
结果回填，模型接着决策，直到任务完成）、**状态**（消息轨迹 + 长期记忆）。本项目锚定：
`agent.py` 的 while 循环 + 9 个工具 + function calling 是裸写版；`step1_graph.py` 用
LangGraph 是它的图化重写。跑「台灯 45 元小风扇 39 元一共多少钱」，模型自主决定调
calculate、拿到 84、组织答复——这就是 agent 与纯 LLM 对话的分水岭。

**MCP——为什么产生？工具从哪来，适配成本爆炸。** agent 有了工具接口（function calling），
但工具提供方五花八门：每个宿主（Claude Desktop、Cursor、自建 agent）都为每个工具
单独写适配——M 个宿主 × N 个工具 = M×N 份适配代码。MCP（Model Context Protocol，
Anthropic 2024-11 开源）把「宿主 ↔ 工具提供方」的接口标准化：**工具方实现一次
MCP server，任何支持 MCP 的宿主都能接**，M×N 退化为 M+N。类比 USB：设备不为每台
电脑单独设计接口。已被 OpenAI、Google 等采纳，事实标准。

**skill——为什么产生？教 agent 流程知识的两个老办法都有痛点。** 想让 agent 学会
"一类事怎么做"，原始手段是写进 system prompt——但 prompt 全文常驻，技能多了
爆上下文、稀释注意力，且散落在代码里难复用难审计。skill（Agent Skills）把流程写成
**独立的 markdown 包**（何时使用/步骤/验收标准），靠**渐进披露**：只有几十 token 的
name + description 常驻上下文，触发才读全文——几百个 skill 也不爆上下文；独立文件
随 git 入库，可 review、可跨项目跨宿主复用。

**三者关系一句话**：agent 是干活的**主体**（宿主），MCP 是它连接外部工具的**统一插口**
（协议），skill 是注入它的**岗位 SOP**（流程知识）。本里程碑实战正是两件事：把商城
工具插上 MCP 口（D2/D3），把评测流程写成 skill（D4）。

### 2. MCP 协议层规范概要与三原语

「协议」= 宿主与 server 之间共同遵守的**规范约定**。它由两层规范叠成（详版地图见 D2 §3，这里先列要点）：

- **JSON-RPC 2.0 层**（借用的已有标准，规定消息形态）：请求（method + params + id，
  **必须响应**，回同一个 id）；通知（无 id，**禁止响应**）；响应（同 id + result/error 二选一）。
- **MCP 层**（规定语义内容）：生命周期定死——`initialize` 握手（互报版本与能力）→
  `notifications/initialized` → 才允许工作请求；传输定死——stdio（每行一条 UTF-8
  JSON-RPC）或 HTTP。

**三原语**：协议规定 server 能提供的东西只有三个容器，区分维度是**决策者是谁**：

| 原语 | 是什么 | 决策者 | 类比一家店铺 |
|---|---|---|---|
| **tools** | 模型可调用的动作（执行、有副作用） | 模型（model-controlled） | 能帮你**办事**的窗口 |
| **resources** | 可读取的数据（只读，GET 语义） | 宿主应用（application-controlled） | 能给你**翻阅**的资料架 |
| **prompts** | 用户可选的提示模板 | 用户（user-controlled） | 帮你**起稿**的填空表单 |

记忆钩子：**模型管动作、应用管数据、用户管模板**。"原语"的意义：协议不规定 server
能提供多少种服务，只规定服务只能装进这三个容器——client 只要懂三个容器就能对接任何
server，这是协议的杠杆。

### 3. 三原语实物：报文长什么样

三种原语是**同一个套路**：server 端用装饰器声明一段能力，客户端用两个 method 跟它
打交道（`xxx/list` 问你有什么 + `xxx/call|read|get` 用某一个）。看实物：

**tools（本项目真实报文）**——`@mcp.tool()` 读函数名/docstring/类型注解生成 schema
存进工具表；`tools/list` 回 schema（宿主将其转换成模型可调工具，**description 是
模型选不选的唯一依据**）；`tools/call` 传 `arguments`（动态，模型决定传什么），回
`content` 内容块列表：

```
--> {"jsonrpc":"2.0","id":3,"method":"tools/call",
     "params":{"name":"calculate","arguments":{"expression":"45+39"}}}
<-- {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"84"}],"isError":false}}
```

工具内部报错也走正常响应，`isError:true` 标记——错误语义在 result 里，不在 JSON-RPC
error 层。

**resources（对照例子）**——声明用 URI 定位（`@mcp.resource("mall://policies/return")`，
装饰器参数是资源地址不是函数名）；`resources/read` 只传 `uri`（固定地址）不传业务
参数，读的是现成数据；读回来给不给模型看由宿主应用决定——"应用管数据"。

**prompts（对照例子）**——`@mcp.prompt()` 声明模板函数（返回 messages 列表）；
`prompts/get` 传 name + arguments（填模板），返回填好的消息（role=user）插进对话；
用户在输入框敲斜杠命令的那一刻就是 prompts/get 被触发——"用户管模板"。

收拢对照：

| | server 声明 | 问有什么 | 用某一个 | 传的参数 | 谁触发 | 返回 |
|---|---|---|---|---|---|---|
| tools | `@mcp.tool()` | tools/list | tools/call | arguments（动态） | 模型 | content 内容块 |
| resources | `@mcp.resource(uri)` | resources/list | resources/read | uri（固定地址） | 宿主应用 | contents |
| prompts | `@mcp.prompt()` | prompts/list | prompts/get | arguments（填模板） | 用户 | messages 模板 |

本项目只实现 tools 不是偷懒：政策全文"喂给模型"由自家 RAG 检索器干了（application
controlled 不假外求），话术常驻 system prompt（不需要模板机制）——商城场景里模型缺
的只有**动作**。真要用另两个原语：装饰器一换、method 一换，套路完全一样。

### 4. 六词对比（收拢）

前三节立完本体与规范，现在把使用层三词和协议层三原语放到一张表里：

| 词 | 层 | 一句话 | 谁决定用 | 上下文占用 | 本项目实例 |
|---|---|---|---|---|---|
| tool | 使用层 | 可调用接口：函数 + JSON Schema | 模型自主选择调用 | 全部工具 schema 常驻每轮 | `calculate`、`search_knowledge`（step1_graph） |
| prompt | 使用层 | 一次性指令文本：人设/规范 | 开发者/用户写死 | 全文常驻 | `build_system_prompt()` 的客服人设 |
| skill | 使用层 | 可复用流程知识包 | 模型按 description 判断 / 用户触发 | 仅 name+description 常驻 | D4 将写的 `eval-regression` |
| tools | 协议层 | server 暴露的动作 | 模型 | 宿主注入为模型可调工具 | D2 server 暴露的双工具 |
| resources | 协议层 | server 暴露的数据 | 宿主应用 | 宿主按需读取 | 知识库原文（暂不实现） |
| prompts | 协议层 | server 暴露的模板 | 用户主动选择 | 触发时才注入 | 如「客服工单回复」模板 |

使用层三词精讲：

- **tool 是 agent 的手**：本体是函数 + schema（名字、参数、描述），schema 每轮进模型
  上下文，模型决定何时调。B2 的 `@tool` 装饰器就是为了消除"实现和 schema 两处手工
  维护"的漂移。关键性质：结构化、有 schema、模型自主调用。
- **prompt 是上岗培训**：客服人设、"金额必须走 calculate 不得心算"全在 system prompt。
  关键性质：写死的文本、每个 token 常驻——写多了贵且稀释注意力。
- **skill 是岗位 SOP**：何时使用→什么步骤→什么标准，靠渐进披露解决 prompt 的两大
  痛点（上下文经济、独立复用）。

三词分工一句话：**tool 是能力（做事），prompt 是约束（人设与边界），skill 是方法
（把一类事做好的流程）**。

最容易翻车的点：**协议层 ≠ 使用层**。MCP 协议回答"server 能提供什么"（三原语），
宿主回答"拿到之后怎么用"（tools 注入为模型可调工具、prompts 映射成 slash command、
resources 按需进上下文）。**skill 在 MCP 里没有对应原语**——它是宿主自有的知识包装
机制。对应关系：

```
使用层（宿主概念）          协议层（MCP 原语）           本项目落点
tool  ────────────────→  tools（model 控制）      ← D2 server 暴露 search_knowledge/calculate
prompt ───────────────→  prompts（user 控制）     ← 宿主 system prompt 常驻
skill ─────×──────────   （无对应原语，宿主自有机制）  ← .claude/skills/eval-regression/
（数据注入）──────────→  resources（application 控制）← 本项目暂不实现，概念能答即可
```

面试钩子：被问"tool 和 skill 区别"，先声明层次——"你说的是使用层三词还是协议层
三原语？"——一句话把概念分层立住，比背定义高级。

## D2 FastMCP 实战：把商城工具包成 MCP server

产物：`src/mcp_and_skill/01_mcp_server.py`（server）+ `02_raw_client.py`（裸报文客户端）。

### 1. FastMCP 概念与 server 写法

**FastMCP / MCPServer 是什么**：官方 SDK 提供的类，"**一台 MCP server**"的抽象。
实例化它 = 创建一个按 MCP 协议通信的 server 对象。三个核心动作：

- `MCPServer("xingchen-mall")`——建一台 server，取名字（握手时 serverInfo 里回的就是它）；
- `@mcp.tool()`——**登记**：import 执行到这行时，读函数名、docstring、类型注解，
  生成 JSON Schema，存进 server 实例内存里的工具表；
- `mcp.run()`——**启动并阻塞**：进入事件循环监听 stdin，每读到一行 JSON-RPC 就处理
  （握手/查工具/调工具），结果写回 stdout，直到 stdin 关闭才退出。返回 None——
  调用它的意义在于让当前进程转变为持续运行的服务，而非取得返回值。

变量名叫 `mcp` 只是惯用缩写，与 `mcp` 包撞名纯属巧合，它就是普通实例变量。

**server 写法：协议适配层，零业务复制**。两个工具都只是薄包装，实现直接 import
my_agent.agent 的 `calculate` / `search_knowledge`——与被测图共用同一份业务代码，
MCP 只是一层协议适配。包装层的真正价值在 docstring：**schema 的 description 是
模型选工具的唯一依据**，要写成面向模型的操作指令（"任何金额计算必须调用本工具，
禁止心算"），不是面向人的 API 文档。参数 schema 由函数类型注解自动生成（pydantic）
——注解即 schema，签名即契约。

**版本注意**：官方 SDK 2.x 已把 `FastMCP` 改名 `MCPServer`（`mcp.server.mcpserver`），
API 几乎同构；2.x 新增 `outputSchema`/`structuredContent` 结构化输出、`auth`（OAuth
入口）、`resource_security`（默认拒路径穿越）——生产化配置的现成入口。

### 2. mcp.server 概念及核心

**mcp 包 = 官方 SDK**（PyPI 上的 `mcp`，`uv add mcp` 装的），是 MCP 协议的代码实现
——协议是规范文档，SDK 是按规范实现协议交互的库（类比：HTTP 是协议，requests 是库）。

包结构按**角色**分目录：

```
mcp/
├── server/   ← 当服务端（提供工具的一方）的所有代码，MCPServer 类住这里
├── client/   ← 当客户端（宿主/调用方）的所有代码
└── shared/   ← 两端共用的类型定义
```

一个包装两端，因为 MCP 连接天生有双方：宿主当 client、工具提供方当 server。
我们写的是提供方，所以 import 落在 `mcp.server`；D3 里 Claude Code 内部扮演的正是
`mcp.client` 角色。

**stdio 接线的核心事实**：进程创建时自带三个标准流——stdin（标准输入）、stdout（标准输出）、
stderr（标准错误）。宿主拉起 server 子进程时接管 stdin/stdout 作为协议通道，stderr
留给日志（不参与协议，排障生命线）。协议帧格式 = 每行一条 UTF-8 JSON-RPC。

**导入机制**：server 脚本不需要 `sys.path.insert` 就能 `import my_agent`——uv 把
本项目以 editable 模式装进 .venv，落地物 `site-packages/my_ai_app.pth` 写入 src
绝对路径，python 启动时自动追加进 sys.path（与脚本位置、cwd 无关）。判定规则：
**uv 管包定义内的（my_agent）；包定义外的（langGraph_agent）才需要手动 insert**。

### 3. 02_raw_client.py 核心

**整体在干嘛**：不用 SDK 客户端，用 30 行裸代码扮演**最小宿主**——subprocess 拉起
server、往 stdin 写报文、从 stdout 读报文，使协议的完整交互过程对学习者可见。SDK 把报文的构造
与解析封装成方法调用，raw_client 直接读写原始报文，协议细节全程可见。

**接线**：`subprocess.Popen(SERVER, stdin=PIPE, stdout=PIPE, text=True,
encoding="utf-8")`——PIPE 表示子进程的 stdin/stdout 不接键盘屏幕，两条流重定向给
父进程（proc.stdin/proc.stdout）。proc 是 Popen 实例，子进程的句柄：
句柄不是进程本体——句柄销毁进程照常运行，操作进程必须经由句柄。收尾两行是一套
优雅关闭：`proc.stdin.close()` 关闭输入流（server 读到 EOF 退出循环），
`proc.wait(timeout=10)` 阻塞等待子进程退出并回收资源，带超时保护（实测 close 后
server 还要 ~2s 才真正退出——关闭输入流不等于进程立即终止，server 还需自行完成收尾）。

**四帧时序**（协议规范的最小执行）：

1. `initialize` 握手：客户端自报协议版本/能力 → server 回 capabilities。**响应里
   server 宣告 `prompts`/`resources`/`tools` 三原语能力**——`"capabilities":{...}`
   是能力声明（原语键名出现 = 支持该原语；`listChanged:false` 等开关 = 无变更推送
   附加特性），外加 `protocolVersion`（协商确定的协议版本）与 `serverInfo`（服务端标识信息）。
2. `notifications/initialized`：无 id、无响应的通知帧——send 后**故意不 recv**。
3. `tools/list`：server 遍历内存工具表，回每个工具的 `inputSchema`——宿主就拿这份
   schema 注入成模型可调工具。
4. `tools/call`：`{"name":"calculate","arguments":{"expression":"45+39"}}` →
   `{"content":[{"text":"84","type":"text"}],"isError":false}`——动作的决策权在
   模型（宿主端），server 只负责执行。

**MCP 协议规范地图（详版，两层规定，报文每个字段都有出处）**：

「协议」= 两层规范叠成——读到任何报文先问"这字段是哪层规定的"：

第一层 **JSON-RPC 2.0**（规定消息形态）：

| 形态 | 长相 | 规矩 |
|---|---|---|
| 请求 | method + params + id | **必须回**（result 或 error，带同一个 id） |
| 响应 | 同 id + result/error 二选一 | 对请求的回话 |
| 通知 | 无 id | **禁止回**任何东西 |

标准错误码（负数是 JSON-RPC 定的码表）：-32700 解析错误 / -32600 无效请求 /
-32601 方法不存在 / **-32602 无效参数**（未握手就 tools/list 实测撞到的）/ -32603 内部错误。

第二层 **MCP**（规定语义内容）：

- 生命周期定死：initialize（params: protocolVersion/capabilities/clientInfo）→
  notifications/initialized → 才允许工作请求。握手 = 版本与能力协商；没握手直接
  tools/list → error -32602（实测）。
- method 名定死（三原语各一组）：

  | 原语 | method | params | result |
  |---|---|---|---|
  | tools | tools/list | 无 | {tools: [{name, description, inputSchema}]} |
  | tools | tools/call | {name, arguments} | {content: [{type:"text", text}], isError} |
  | resources | resources/list、resources/read | {uri} | {contents: [...]} |
  | prompts | prompts/list、prompts/get | {name, arguments} | {messages: [...]} |

  响应的 `content` 是内容块**列表**（允许一次返回多块：文本/图片/资源），文本块的
  正文在 `text` 字段——读响应 = 按 spec 结构一层层拆。
- 传输定死：stdio（每行一条 UTF-8 JSON-RPC，stderr 留给日志不参与协议）或
  streamable-http。
- 权威出处：**modelcontextprotocol.io/specification**——字段争议查 spec，不猜。

raw_client 对照自查（30 行代码 = spec 的最小执行）：每行一条 ✓ / jsonrpc 字段 ✓ /
握手顺序 ✓ / 有 id 必等回 ✓ / 通知不等 ✓ / method 名照 spec ✓。

方法论：碰到任何陌生协议，第一动作找它的 spec，先抓三样——**消息结构（字段表）、
生命周期（顺序规矩）、方法清单（可调用哪些方法）**，三样到手任何报文都能自己读。

交互式看报文/试调用（浏览器 UI，值得自己玩一次）：
`npx @modelcontextprotocol/inspector uv run python src/mcp_and_skill/01_mcp_server.py`

### 4. `__name__` 与子进程的两种进入方式

01_mcp_server.py 被 Popen 拉起时会执行 `if __name__ == "__main__": mcp.run()` 吗？
**会**。Popen 启动一个全新解释器进程、把脚本当命令行参数，
与终端手敲 `python 01_mcp_server.py` 完全等价，该文件作为 `__main__` 模块运行。

`__name__` = Python 为每个模块自动设置的**模块名属性**，值取决于文件怎么被拉起：
直接跑 → `"__main__"`（if 成立，run() 执行，进程阻塞成 server）；被 import →
模块名 `"mcp_server"`（if 不成立）。实测：`import mcp_server` 后打印
`__name__ = mcp_server`，进程立刻结束、没有阻塞。

分界表——if 只守住 if 块，**顶层代码两种方式都执行**：

| 代码 | Popen 直接跑 | 被 import |
|---|---|---|
| `load_dotenv` / `mcp = MCPServer()` / `@mcp.tool()` 登记 | ✅ | ✅（import 能成功正靠它先供上 API_KEY） |
| `mcp.run()` | ✅ 阻塞成 server | ❌ 跑完顶层自然退出 |

if 守卫的意义：同一文件两种身份互不干扰——**当脚本**被宿主 Popen 拉起（生产形态），
**当库**被 import 做测试/检查工具注册，绝不误启动一个阻塞的 server。没有 if 守卫，
任何人 import 都会把当前进程阻塞在 stdin 上。

## D3 接入 Claude Code 实际调用（实战闭环）

产物：五个编号教学物（`01_mcp_server.py` server｜`02_raw_client.py` 裸报文客户端｜
`03_sdk_client.py` SDK 客户端｜`04_sdk_agent.py` 自建 agent 接 MCP｜`05_http_client.py`
HTTP 传输客户端）+ 两条命令把 server 接进真实宿主，走通闭环
「写配置 → 宿主拉起 → 模型自主调用 → 知识库作答」。

### 1. 接入动作：两条命令

```bash
# 注册：写配置，仅此而已，这一步没有任何连接发生
claude mcp add xingchen-mall -- uv run python src/mcp_and_skill/01_mcp_server.py
# 验证：真实拉起子进程做健康检查（Popen → initialize 握手 → tools/list，通了才 ✔）
claude mcp list   # → xingchen-mall: uv run python src/... ✔ Connected
```

| 片段 | 含义 |
|---|---|
| `claude mcp` | 官方 CLI 的 MCP 配置管理子命令 |
| `add xingchen-mall` | 注册 + 起名（宿主列表里标识这台 server） |
| `--` | 硬分隔符：后面整串原封不动成为"启动命令"，防止被误解析 |
| `uv run python src/...` | 启动命令本体——宿主将来 Popen 的就是这串命令行 |

同家族：`claude mcp get xingchen-mall`（单条详情）、`claude mcp remove`（删除）；
`add` 加 `--scope` 选层级，不写默认 local。

**scope 三级**：

| scope | 存哪 | 谁能用 | 场景 |
|---|---|---|---|
| `local`（默认） | `~/.claude.json` 的本项目段 | 你 + 这个项目 | 个人调试（本里程碑用的） |
| `project` | 项目根 `.mcp.json` | 随 git 入库，clone 即得，全团队 | 团队共享 server |
| `user` | `~/.claude.json` 顶层 | 你，跨所有项目 | 个人常用工具箱 |

### 2. 注册 JSON 与 Popen 的血缘

local 注册的实际落盘内容：

```json
"xingchen-mall": {
  "type": "stdio",
  "command": "uv",
  "args": ["run", "python", "src/mcp_and_skill/01_mcp_server.py"],
  "env": {}
}
```

`command` + `args` 拼起来正是 raw_client 里 `SERVER = [程序, 参数...]` 的**同构物**
——注册表存的就是「如何 Popen 启动该 server」的完整配置：`type: "stdio"`（管道接管 stdin/stdout）、
`env: {}`（额外注入的环境变量，空 = 继承宿主）。注册里写**相对路径**能用，因为宿主
以项目根为 cwd 拉起子进程，`uv run` 找得到 pyproject 与 .venv；而脚本内部的 .env
用 `__file__` 绝对路径加载（D2 的 cwd 坑）——双保险。

### 3. 宿主生态：accr、Claude Code 与 .claude.json

**accr 不是 Claude Code 的客户端，是同一引擎构建的两个宿主**（地位并列，不是上下级）：
官方 **Claude Agent SDK**（`@anthropic-ai/claude-agent-sdk`）是构建 agent 应用的库
（对话循环/工具系统/MCP 客户端能力），Claude Code 是官方基于它构建的产品，accr 是公司
（内部 npm 源，`@annto/accr`，自述"代码审查工具"）基于**同一套 SDK** 构建的产品。证据：
accr 会话读同一份 `~/.claude.json`、/mcp 面板一致、`claude mcp add` 注册的 server
在 accr 里照常 Connected——**MCP 客户端能力是 SDK 公共件**，配置写一次，两边都能用。

**MCP 接入到了谁**：配置态接入的是共享配置层；运行态接入的是**当前跑着的宿主进程**
（本例是 accr）——它 Popen server、它的 MCP Client 按 JSON-RPC 通信、它把 schema 注入
模型。换官方 `claude` 会话则由 Claude Code 进程做同样的事。

**.claude.json = 用户级全局总账本**（配置与运行状态混装），四类字段：

1. **身份与统计**：userID / machineID / numStartups（启动次数）/ firstStartTime
2. **UI 引导记号**：hasCompletedOnboarding / tipsHistory——全是"别再烦用户"的标记
3. **projects（核心）**：按项目绝对路径索引的条目表，每条含 `mcpServers`（local
   注册住这）、`allowedTools`（"不再询问"的预授权规则）、`enabled/disabledMcpjsonServers`
   （对 project scope server 的开关）、`hasTrustDialogAccepted`（目录信任弹窗结果）
4. **使用统计与缓存**：skillUsage / clientDataCache 等

**配置分层**（容易混，一张表钉死）：

| 文件 | 层级 | 管什么 |
|---|---|---|
| `~/.claude.json` | 用户级总账本 | 全局状态 + 各项目状态（含 local MCP 注册） |
| `~/.claude/settings.json` | 用户级设置 | 权限规则、模型偏好 |
| 项目 `.mcp.json` | 项目级、入 git | 团队共享的 MCP 注册（project scope） |
| 项目 `.claude/settings.json` | 项目级、入 git | 团队共享的项目设置 |

### 4. 运行时序：接入怎么发生、为什么自动调用

「接入」分两层说：**配置态**——注册躺在共享配置层（.claude.json）；**运行态**——
**谁启动会话，谁拉起 server**，各拉各的子进程副本，互不干扰。

**启动时序**（会话启动那一刻，宿主自动做三件事）：

```
① 读 ~/.claude.json 本项目的 mcpServers 段，找到启动配置
② 按配置 Popen 拉起子进程 → initialize 握手 → tools/list 收走 2 个工具 schema
③ 把 schema 转换成模型 tool 定义，注入本轮会话——从此这些工具每轮对模型可见
```

**对话时序**（提问「七天无理由退货的条件是什么？」之后）：

```
① 问题连同工具清单一起送模型
② 模型读到 description（"政策类问题必须先调用本工具"）→ 输出 tool_calls
   ——模型只产出"调用意图 + 参数"，它自己执行不了任何东西
③ 宿主按 name 路由到 xingchen-mall 子进程，往 stdin 写 tools/call 报文
④ server 执行 _search_knowledge，结果从 stdout 回来
⑤ 宿主把结果作为 ToolMessage 回填消息轨迹 → 模型基于资料生成最终回答
```

「自动调用」没有魔法 = **模型的决策 + 宿主的执行循环**——就是 B 阶段亲手写过的那个
`while` 带 tool_calls 就执行、回填、再喂模型的循环；唯一区别：B 阶段工具是同进程
Python 函数（bind 直传），现在是跨进程 JSON-RPC。**模型视角毫无变化**——看到的都是
schema、做的都是产出 tool_calls。MCP 的全部意义：让跨进程工具在模型视角下与本地函数调用形式一致。
宿主执行前的批准弹窗是宿主的安全设计（工具有真实副作用），协议本身不管这个。

### 5. D3a：SDK 客户端与自建 agent 接 MCP

01 脚本是 server 侧，02-05 全是客户端侧，教学递进：**02 手写报文**（不经 SDK，直接构造与解析协议帧）→
**03 SDK 客户端**（协议交互封装为方法调用）→ **04 真 agent**（LLM 决策 + MCP 执行）→
**05 换 HTTP 传输**（协议层不变，传输层替换）。

**手写报文 vs SDK 封装**（03 与 02 同一套协议三步，SDK 全部封装成方法调用）：

| 协议步骤 | 02 手写报文 | 03 SDK 客户端 |
|---|---|---|
| 握手 | `initialize` 报文 + `notifications/initialized` | `await session.initialize()`（通知自动补） |
| 查表 | `tools/list` + 手动等 id 对齐 | `await session.list_tools()` |
| 调用 | `tools/call` + 手动解析 content | `await session.call_tool(name, args)` |

SDK 内部做的事：后台任务把字节流切帧、按 id 把响应配对到请求、JSON 解析成对象、
连接生命周期管理。报文键名是 camelCase（`inputSchema`），SDK 对象属性是
snake_case（`t.input_schema`）——读报文看 spec 键名，操作对象看 Python 属性名。

**04 的双客户端视角**：sdk_agent 进程同时是两个客户端的宿主——对上是
**AsyncOpenAI**（模型 API 客户端，把 openai 库的同步客户端换成异步版，每个调用加
`await`），对下是 **ClientSession**（MCP server 客户端）。角色是相对的：对模型
它是工具的执行方，对 server 它是调用的发起方。

**决策层与执行层的分工**（与 B 阶段 agent.py 逐行同构，唯一差异是执行走跨进程）：

```
工具表：MCP 的 inputSchema 本身就是 JSON Schema，包装成 OpenAI function 定义格式即成 tools 参数
   openai_tools = [{"type": "function", "function": {"name": t.name,
                   "description": t.description, "parameters": t.input_schema}} ...]
循环：模型产 tool_calls（决策）→ args = json.loads(...) → await session.call_tool（执行）
      → 按 tool_call_id 回填 role:"tool" 消息 → 循环到模型给出文字回答
护栏：for _ in range(10) 防失控连环调用（B 阶段同款）
```

schema 转换为何只需一行：**三家模型 API 的消息格式不同，但工具参数的描述格式都是
JSON Schema**——转换只是调整外层消息结构，参数定义原样复用。LangChain 的
`bind_tools`、Google 的 function calling，各自都有适配层完成这一转换。

**混合工具源架构**（实际应用中本地函数 + 多个 MCP server 混用的做法）：模型只见
**一张统一工具表**；宿主侧维护注册表 `name → 执行器闭包`（本地函数直接调、MCP 工具
转发 `session.call_tool`），按名字分发。Claude Code 的命名空间前缀
`mcp__<server名>__<工具名>`（如 `mcp__xingchen-mall__search_knowledge`）就是为
多 server 混用不撞名。

**async 四件套**（读 03-05 的门槛）：

| 概念 | 是什么 |
|---|---|
| 协程对象 | 调用 async 函数得到的对象，封装了待执行的函数体，被 await 或事件循环调度才执行 |
| 事件循环 | `asyncio.run()` 创建的调度机构，负责推进所有已注册的协程 |
| `await` | 挂起当前协程：等待 IO 期间把控制权交还事件循环，由其执行其他协程 |
| `asyncio.run(main())` | 创建事件循环 → 运行 main 协程至结束 → 关闭事件循环 |

`main()` 直接调用只会得到 coroutine object + RuntimeWarning("never awaited")——
协程从未被调度执行。同步阻塞 = 当前进程停下来等待一个操作完成（02 单次教学场景够用）；
异步 = 等待期间事件循环可执行其他协程（SDK 面向并发场景：同时处理多工具调用/多会话）。

**Popen vs stdio_client**：上下层关系，非竞品。`subprocess.Popen` 是底层原语——
创建子进程、返回原始字节管道，流的重定向策略由调用方选择（PIPE/DEVNULL/继承终端）；
`stdio_client` 是封装层——内部同样调用 Popen，但固定接管 stdin/stdout、放行
stderr（留给日志），并在其上启动后台任务做帧切分、JSON-RPC 解析、按 id 配对请求
与响应，交给你的 `read/write` 已是消息级通道而非原始字节流。**stderr 不要被占用**：
server 里 print 默认写 stdout，输出会混入协议报文流导致解析失败——调试信息必须写 stderr。

**参数对象（parameter object）**：`StdioServerParameters(command, args)` =
封装"如何启动 server"的配置对象；`XxxParameters / XxxConfig / XxxOptions` 后缀
是通用命名惯例——参数多到不宜散着传时，打包成一个对象。

### 6. D3b：server 部署成 HTTP 服务与远程接入

**加一个启动参数即部署为 HTTP 服务**：`01_mcp_server.py` 加 http 分支——`mcp.run(transport="streamable-http")`，
FastMCP 内置 uvicorn 起常驻服务，监听 `127.0.0.1:8000`，endpoint 路径 `/mcp`。
同一份工具表、同一套协议三步，两种传输的差异只在字节流的承载方式。

**stdio vs HTTP 本质对照**：

| 维度 | stdio | streamable-http |
|---|---|---|
| server 形态 | 客户端 Popen 创建的子进程 | 独立常驻的网络服务 |
| 寻址 | 启动命令（command+args） | url（`http://127.0.0.1:8000/mcp`） |
| 生命周期 | 随客户端起止（客户端退出，server 进程随之退出） | 各自独立，服务先起、客户端后连 |
| 关系 | 一宿主一副本（谁启动谁拉起） | 一份服务，多客户端共享 |
| 谁能访问 | 只有拉起它的进程 | 知道 url 且网络可达的任何人 |

**HTTP 传输下各协议动作的日志形态**（server 侧日志完整展示了协议动作如何映射为 HTTP 请求）：
`POST /mcp 200`（请求-响应：initialize/tools/list/tools/call）｜
`POST /mcp 202 Accepted`（通知 HTTP 化：notifications/initialized 发出即接受，
无需等待响应）｜`GET /mcp`（SSE 推送流：服务端主动推送的通道）｜
`DELETE /mcp`（会话结束）｜每个客户端一个 session ID（服务端区分多客户端会话）。

05 与 03 唯一差异在入口：`stdio_client(params)` 换成
`streamable_http_client(URL)`——initialize / list_tools / call_tool 一字不变。
**传输层替换，协议层不变。**（注意：mcp SDK 2.x 里函数名从 1.x 的
`streamablehttp_client` 改成了 `streamable_http_client`，且 yield 从三元组变
二元组——按报错信息调整即可。）

**进程控制 vs 协议平等**：stdio 下宿主对子进程拥有完整的生命周期控制权（父子进程
关系），但**协议地位完全平等**——MCP 协议约束下双方按报文交互，不存在命令与服从关系。

**社区工具两形态**（接入别人部署的 MCP）：

```bash
claude mcp add web-fetch -- uvx mcp-server-fetch        # Python 系：uvx 临时拉环境跑
claude mcp add some-server -- npx -y @some/mcp-server   # Node 系：npx 拉包跑
# 已部署的 HTTP 服务则直接指 url：
claude mcp add xingchen-mall-remote --transport http http://127.0.0.1:8000/mcp
```

uvx/npx 的本质：**启动命令的扩展形态**——无需预装，运行时下载包/构建临时环境再执行，
执行完即丢弃环境。stdio 的"拉起子进程"机制不变，只是启动命令从"运行本地脚本"
变为"运行社区发布的包"。

**撞名案例**（web-fetch）：注册的 MCP server 与环境预置的 skill 同名，模型按
description 匹配抓网页时选了 skill——判定方法：Skill 工具调用形如
`Skill(web-fetch)`，MCP 工具形如 `mcp__web-fetch__fetch`。教训：**名字是公共
资源**，注册名要避开常用词（或接受模型按语义自选）。

### 7. 实操验证记录与收束

**接入宿主闭环**：新会话验证（accr 宿主）：`/mcp` → `xingchen-mall · ✔ connected ·
2 tools`；问「七天无理由退货的条件是什么？」→ 弹出 `search_knowledge` 批准框 →
基于知识库 top-3 资料作答；计算题 → `calculate`。**闭环走通：自建 server 第一次被
生产级宿主里的模型自主调用。**

**D3a 机器验证**：03 三步全通（握手 → 2 工具 → calculate=84）；04 真 agent 两条
问题全通——计算题自主调 `calculate` 得 84，政策题自主调 `search_knowledge`
（top-1 相似度 0.622）组织回答。

**D3b 三步全通**：① 部署——HTTP 常驻服务起在 127.0.0.1:8000；② 自接自——
Claude Code 以 http 注册 `xingchen-mall-remote` ✔，05 客户端三步全通
（calculate=84）；③ 社区工具——`uvx mcp-server-fetch` 接入 ✔。

为什么 MCP 能接入（收束）：**开放标准**——规范公开（modelcontextprotocol.io），
宿主只实现客户端侧、server 只实现服务端侧，双方代码互不依赖，遵循同一份规范即
天然兼容；宿主不挑实现语言（command 就是普通命令行）、不挑传输（stdio 换
streamable-http，协议层一字不动）。M×N 退化为 M+N，在接入成功的这一刻兑现——
这个 server 拿去接任何 MCP 宿主，一字不改。

## D4 SKILL.md 实战：评测回归流程固化成 skill

产物：`.claude/skills/eval-regression/SKILL.md`（单文件，零新代码——评测本体是
`evals/eval.py`，skill 只是说明书）。

### 1. skill 的定位与机制

**skill = 注入给模型的程序性知识**（何时做、按什么步骤做、怎么判断结果）——扩展
模型的**知识**，不扩展**能力**。读完评测回归的 SKILL.md，模型不会凭空多出任何执行
本领：跑命令靠 Bash 工具、读文件靠 Read、找历史结果靠 Glob——skill 只负责让模型
"知道该用哪些已有工具、按什么顺序、怎么解读输出"。

**SKILL.md 格式**：frontmatter（`---` 包裹的 YAML）+ Markdown 正文。

```markdown
---
name: eval-regression          ← 与目录名一致；小写字母/数字/连字符
description: 何时触发的描述     ← 唯一触发依据：写用户可能的原话 + 自动触发场景
---
正文：给模型的操作指令
```

**渐进披露两阶段**：索引阶段——宿主启动扫描，只读 name + description（几十 token
常驻，description 用于任务匹配）；执行阶段——匹配成功才把正文注入模型。所以：

- description 写"什么时候用我"（触发词），不是功能介绍；
- frontmatter 是给宿主消费的元数据：会话里看不到、Markdown 预览模式可能隐藏——
  源码模式可见（D4 实测踩过的疑问）；
- name 与目录名不一致会导致注册失败。

**存放层级**：项目 `.claude/skills/`（随 git 入库，clone 即得，团队共享——对应
D3 的 project scope 思路）；用户 `~/.claude/skills/`（个人，D3 任务3 里公司分发
的 60+ 个都在这）。

### 2. eval-regression 的写作要点

- 正文五节：**本体与分工**（eval.py 是本体，不要重写）→ **命令清单**（全量 /
  `--graph step5` 对比 / `--category` / `--id` / `--reclassify`，每条带注释）→
  **执行步骤**（前置检查 .env → 按意图选命令 → 等完成 → 对比历史 → 聚合汇报，
  明确"FAIL 多时按归因类别聚合，不逐条复述"）→ **结果解读**（五个失败归因类别
  的含义与修复方向、SKIP 不计分母）→ **注意事项**（真实消耗 token、
  `retrieved_context` 是归因证据、不改 testcases.json 来让测试通过）。
- 正文只写操作指令，不写背景科普——背景在 evals/评测笔记.md，skill 不重复
  （skill 与本体文档的分工）。

### 3. 实操验证

T01 单条实跑 PASS（关键词判定 kw、tools=2、4.66s、结果落盘
`run_20261031_180432.json`）——说明书里的命令与输出格式与实际一致。
用户侧验证：新开会话说"跑一下评测回归"，观察宿主触发 `Skill(eval-regression)`
并按流程执行（skill 在宿主启动时扫描，当前会话不可靠）。

### 4. 上下文注入机制全景对比

提示词、tool、MCP、skill 都是"给模型注入上下文以改变行为"的机制，差别只在
**注入什么、何时注入、谁决定触发**：

| 机制 | 注入什么 | 加载时机 | 谁决定触发 | 本项目实例 |
|---|---|---|---|---|
| 提示词（prompt） | 一次性任务指令 | 随请求发送，用完即弃 | 人（每次手动） | B 阶段 SYSTEM_PROMPT |
| tool（宿主内置） | 执行能力 | 会话启动注入工具表 | 模型决策调用 | Bash / Read / Glob |
| MCP | 外部系统能力 | 会话启动注入工具表 | 模型决策调用 | calculate / search_knowledge |
| SKILL | 流程/规范知识 | 任务匹配时加载正文 | 模型按 description 匹配 | eval-regression |
| CLAUDE.md | 常驻全局规范 | 每次会话无条件注入 | 无（总是注入） | 全局行为指南 |
| Hooks | 宿主侧自动动作 | 固定生命周期点 | 宿主（绕过模型决策） | —（未用） |
| 子代理（subagent） | 独立上下文的执行架构 | 显式委派时 | 模型或人 | —（未用） |
| 自定义斜杠命令 | 显式触发的 prompt 模板 | 人输入 `/` 时 | 人显式调用 | /commit 等 |

逐对辨析（易混的边界）：

- **tool vs MCP**：同是工具，差在进程边界——内置工具在宿主进程内；MCP 跨进程、
  标准协议、第三方可分发（D2/D3 的全部内容）。
- **skill vs 提示词**：复用性与载体——一次性要求直接说；反复使用的流程落成文件。
- **skill vs CLAUDE.md**：加载策略——常驻（每次都注入，放全局规范）vs 按需
  （触发才加载，放专项流程）。常驻内容放 skill 浪费每轮上下文，专项流程塞
  CLAUDE.md 撑爆日常指令。
- **Hooks vs skill**：触发者——hook 由宿主在固定事件点自动执行，绕过模型决策；
  skill 需要模型匹配后主动加载。强制格式化/拦截敏感文件用 hook，流程知识用 skill。
- **斜杠命令 vs skill**：最接近的一对，差在触发者——斜杠命令由人显式调用，
  skill 由模型按语义自动匹配（两者正在融合，skill 也可被 `/` 调用）。
- **子代理**：不注入知识，改变**执行架构**——独立上下文窗口、可并行、保护主上下文
  （大范围搜索、多路调研）。
- **RAG**（本项目 search_knowledge）：管"依据什么材料"——动态检索的资料；
  skill 管"怎么做"——预先写好的静态流程知识。

**选型决策线**：做不了 → 找工具（要接外部系统 → MCP）；会做但不知道团队流程 →
skill；只此一次 → 直接说（提示词）；每次都要 → CLAUDE.md。

**两轴收束**：这条轴的两端是改权重（微调）↔ 改上下文（提示词），中间全部机制都是
上下文注入的工程化变体，彼此的差别只有两个变量——**注入时机**（常驻 / 按需 /
事件驱动）与**触发者**（人 / 模型 / 宿主）。选型先问这两个问题。

### 5. skill 的定位精确化与联合工作

skill = 注入给模型的**程序性知识**——何时做、按什么步骤做、怎么判断结果。它扩展
模型的**知识**，不扩展**能力**：读完评测回归的 SKILL.md，模型跑命令仍靠 Bash、读
文件仍靠 Read——skill 只负责让模型知道"该用哪些已有工具、按什么顺序、怎么解读输出"。

**联合工作的对象**（skill 不单独干活）：

| 对象 | 关系 |
|---|---|
| 工具（tools） | 最核心：skill 是说明书，工具是执行者——skill 指挥的每一步都落在工具调用上 |
| 宿主运行时 | skill 机制本身的实现者：启动扫描注册、description 匹配、触发时注入正文 |
| MCP server | 企业场景标配组合：skill 写流程（操作规范），MCP 提供内部系统的工具能力 |
| CLAUDE.md | 分工而非联合：常驻规范 vs 按需流程（见 §4 逐对辨析） |
| 附带脚本/文件 | 进阶形态：skill 目录可含脚本、模板、参考文档，正文按需引用——从说明书升级为资源包 |

### 6. 斜杠命令系统：`/` 面板的四来源与注册表

**输入 `/` 看到的列表，四个来源**（指令存在哪 × 触发者）：

| 来源 | 指令存在哪 | 触发者 | 本环境实况 |
|---|---|---|---|
| 内置命令 | Claude Code 程序代码内 | 人 | /help /clear /model /mcp /init 等 |
| skill（可调用形态） | `skills/*/SKILL.md` | 人或模型 | 列表大头：60+ 装入 skill 中的可调用者 |
| 自定义 commands | `commands/*.md` | 人 | 两个目录都不存在，此来源为空 |
| MCP prompts 原语 | server 进程内 | 人 | xingchen-mall 只实现 tools，此来源为空 |

**commands 机制**：文件即命令——`.claude/commands/deploy.md` → `/deploy`（项目级，
随 git 入库）、`~/.claude/commands/`（用户级）、子目录 = 命名空间。frontmatter 支持
`description`（列表说明）/ `allowed-tools`（预授权）/ `argument-hint` / `model`。
正文的动态特性：`$ARGUMENTS`（参数填充）、`` !`command` ``（宿主在组装 prompt 时
执行 shell 并填入结果）、`@path`（附文件内容）。

**commands 正文 vs SKILL.md 正文的粒度差异**：command 是 **prompt 模板**（任务
委托——用户显式触发，意图明确，写成一封指令邮件）；SKILL.md 是**流程手册**
（模型自动匹配，需要自足的完整上下文）。两者可组合：command 做薄入口（一句
"按 eval-regression skill 的流程跑 $ARGUMENTS"），skill 存厚流程。独立建 command
的时机：希望人主动发起，而非等模型匹配。

**五个机制确认点**：

1. skill 默认支持 `/` 显式调用——与模型自动匹配构成双触发，零配置。
2. MCP server 出现在 `/` 列表**靠实现而非配置**——server 写了 `@mcp.prompt()`
   （prompts 原语），宿主接入后自动列为 `/mcp__服务名__模板名`；没实现，配什么都不出现。
3. commands/skill 目录在**会话启动时**扫描——新建文件当前会话不可见，新会话才生效。
4. 列表只载入命令名 + description（轻量披露，与 skill 渐进披露同思路）——完整
   正文在选中那一刻才展开。
5. 选中 commands = 展开模板（替换参数）后**作为消息发给模型**，不是运行脚本；
   唯一的真执行是 `` !`command` ``，且由宿主执行，模型只看到结果。

**内置命令的两层执行路径**（`/` 列表里混排着本质不同的两类）：

| | 本地命令（/clear /model /mcp） | prompt 型（commands / skill / MCP prompts） |
|---|---|---|
| 执行者 | 宿主自己 | 模型 |
| 经过模型？ | 否（UI 层逻辑：拦截 → 查表 → 直接执行内部逻辑） | 是（模板展开后注入对话） |
| 与 B 阶段手写循环的关系 | input() 之后、调 LLM 之前多一道前置拦截 | 同一个循环，消息内容被替换为模板 |

**命令注册表：内存运行时结构，不在磁盘**。宿主进程把各来源装配成一张运行时表
（每条：name / description / 来源类型 / 处理方式），`/` 面板即此表的过滤视图；
会话结束表随进程消失。落盘的是**原料**：内置命令在程序代码里（本机
`/d/Software/nodejs/` 下 npm 全局包）、commands/skill 在各自目录、MCP prompts
不落盘（握手后经 `prompts/list` 从 server 拉取，server 掉线其条目即消失）。
复用 D3 的两分法：**配置态落盘（~/.claude.json 存 MCP server 启动配方、目录存
指令文件），运行态在内存（注册表是装配产物）**——与"注册只是写配置，连接发生在
运行时"同一条原理。

## D5 协议定位：MCP vs A2A vs ACP（概念层，不实现）

产物：无代码——概念定位 + 自检辨析。查证基线：MCP 与 A2A 现由同一中立基金会治理
（Agentic AI Foundation，2025-12 成立，初始贡献 MCP/Goose/AGENTS.md；A2A 于
2026-08 捐入），互补定位为基金会与社区的一致口径。

### 1. 总体地图与 MCP 的边界

```
MCP  纵向连接：agent ↔ 工具/数据      （D2-D4 已实战）
A2A  横向连接：agent ↔ agent          （Google 提出，与 MCP 同属 AAIF）
ACP  横向连接：agent ↔ agent 的另一条路线（IBM 提出，已向 A2A 对齐）
```

MCP 的边界（为什么需要横向协议）：本项目的 `search_knowledge` 是**无状态的被动
工具**——没有自己的 LLM、不做决策，"查不查、查什么"全由宿主模型决定；交互单元
`tools/call` 是一次函数调用。若对端是**完整的 agent**（自带模型/记忆/决策循环），
需要的是"委托一个任务"（可能长时、可能反问、可能多轮）而非"调用一个函数"；跨框架
agent 协作还需要 agent 间的公共语言。这两件事 MCP 都没规定——A2A 的生态位。

### 2. A2A 核心概念（Google 2025-04 提出 → 2026-08 入 AAIF）

| A2A 概念 | 是什么 | MCP 对照 |
|---|---|---|
| Agent Card | agent 的自我描述文档（`/.well-known/` 约定地址）：技能列表、url、认证方式、流式/推送能力开关——**agent 级**对外承诺 | `tools/list` 是**工具级**声明 |
| Task（任务） | 一等公民，带生命周期：submitted → working → input-required（反问中）→ completed/failed，支持长时运行与状态推送 | `tools/call` 一问一答，无任务态 |
| 不透明协作 | 刻意设计：不暴露对方内部思考、记忆、工具明细——只共享任务语义 | MCP 的 inputSchema 把工具参数结构全暴露 |
| 基座 | HTTP + JSON-RPC + SSE（Web 标准件） | streamable-http 同款选型思路 |

"不透明"是 A2A 与 MCP 最深层的差异：MCP 透明（宿主模型要精确填参数，所以摊开
schema）；A2A 黑盒（协作以解耦为代价，对端内部实现是私事）。

### 3. ACP：另一条横向路线，已收敛；重名警报

**IBM 的 ACP（Agent Communication Protocol）**：IBM Research 2025 年初提出，
选型不同——异步消息驱动（MIME 多模态消息，非任务对象）、强调人类在环（人与
agent 在同一条消息流交替参与）。现状：属 Linux 基金会 BeeAI 项目，2025 年中起
与 A2A 对齐（BeeAI agent 可作为 A2A server 运行），独立协议活跃度下降。生态位
竞争后的收敛是协议格局常态（对照容器编排 Docker Swarm → Kubernetes）。

**重名警报**：ACP 还指 **Zed 的 Agent Client Protocol**——编辑器 ↔ code agent
（Zed 接入 Gemini CLI、Claude Code 这类），与 IBM 的 ACP 零关系。web-fetch 撞名
案例的协议版：名字是公共资源，协议命名也一样，搜资料先分语境。

### 4. 三协议对比总表

| 维度 | MCP | A2A | ACP（IBM） |
|---|---|---|---|
| 连接方向 | 纵向：agent ↔ 工具/数据 | 横向：agent ↔ agent | 横向：agent ↔ agent |
| 对端是什么 | 无状态的工具 | 完整的黑盒 agent | 完整的 agent（消息化） |
| 交互单元 | `tools/call`（函数调用） | task（带生命周期的任务） | message（多模态消息流） |
| 发现机制 | `tools/list`（工具级） | Agent Card（agent 级） | agent 注册/清单 |
| 透明度 | 透明（schema 全暴露） | 刻意不透明 | 消息级 |
| 治理 | AAIF（2025-12 捐入） | AAIF（2026-08 捐入） | LF BeeAI，向 A2A 对齐 |

### 5. 本项目映射与判断口诀

xingchen-mall 的两种化身：

- **MCP 版（已交付）**：`search_knowledge` 是工具，决策权在宿主模型（"政策类问题
  必须先检索"写在宿主 system prompt）——暴露原子能力。
- **A2A 版（假想）**：把 B 阶段带 LLM 循环的完整客服图部署为 A2A server，Agent
  Card 声明"能回答商城售前售后问题"；总系统经 Agent Card 发现、委托任务、收最终
  答案——决策权在星辰商城 agent 自己，内部图/prompt/工具不暴露。

判断口诀：**暴露原子能力、决策留给调用方 → MCP；暴露完整任务、决策留给服务方 →
A2A。**

### 6. 自检四问与易混点

1. MCP server 为何不能直接当 A2A agent？——三条：无自身决策、无任务生命周期、
   无 agent 级自我描述。不是技术不能，是语义不合格——承诺不了 A2A 合同的内容。
2. **最易混点：把 tools/list 的透明思维带入 Agent Card**——Agent Card 不声明
   内部工具（技能 + 访问信息而已），内部实现全部藏起。MCP 的透明是已内化的心智
   模型，学 A2A 要主动切换到黑盒视角。
3. 决策权归属：原子能力+调用方决策 → MCP 工具；完整任务+服务方决策 → A2A agent。
4. 两个 ACP：IBM 版连 agent ↔ agent（消息流+人在环）；Zed 版连编辑器 ↔ code
   agent。同缩写零关系。

### 7. 附属辨析：多 client / workflow / 多 agent / A2A 的边界

**client ≠ agent**：client 是传输层概念（API 客户端对象，只管请求收发），agent
是架构层概念（决策循环）。一个 agent 可用多个 client，一个 client 可伺候多个
agent。本项目反例：step5 的 judge_llm、eval.py 的归因分析员——独立 LLM 角色，
但单次结构化调用、无循环，连 agent 都不算。

**多 agent 的三条判定**（同时满足）：① 成员有独立决策循环（自己的 prompt/工具，
自己决定下一步）；② 职责分工明确；③ 成员间显式交接（消息/任务移交）。分界线一条：
**控制流是 LLM 决定的还是代码决定的**（C 阶段 workflow vs agent 的那条界）。

| 形态 | 定性 |
|---|---|
| 多个 client / 多次 LLM 调用 | 传输层复用，什么都不是 |
| 一张图多节点、控制流代码写死（即使节点内调 LLM） | workflow（工作流） |
| 多成员各带循环，supervisor（LLM 路由）或消息传递协作 | 进程内多 agent 系统 |
| 独立部署、互不信任的黑盒 agent 跨系统协作 | A2A 领域（网络协议） |

**进程内多 agent ≠ A2A**：前者是**架构模式**（同程序、共享内存状态、开发者全控、
不需要网络协议）；后者是**网络协议**（独立部署、黑盒、跨系统）。同构对照：同进程
函数互调不需要 HTTP，跨进程才需要 MCP；同系统多 agent 协作不需要 A2A，跨系统才
需要。**协议只解决边界外的问题，边界内是架构设计。**

**本路线的实战落点**：workflow 已在 B2 实战（LangGraph 图、step5 检索子图）；
进程内多 agent 在里程碑 E 实战（LangGraph supervisor 模式，research → write →
review → revise 三角色，明确职责边界/输入输出 schema/停止条件）；A2A / ACP 按路线
设定概念层能答即可，不实现。
