# Python 导入与运行：一个模块的两种命运

> 定位：知识体系本身，不是错题本。主线只有一条——
> **每个 .py 文件（模块）只有两种命运：被运行、被导入。
> 运行决定环境（sys.path 怎么初始化、入口以什么身份执行），
> 导入决定依赖（名字怎么定、文件去哪找、代码何时执行），
> 两条线在"身份"上交汇分野，最后落到环境挂载与工程决策。**
>
> 章节即旅程：运行侧（§1）→ 导入侧（§2）→ 定名（§3）→ 身份（§4）
> → 环境（§5）→ 组织（§6）→ 工程（§7）→ 分诊（§8）→ 全景（§9）。

---

## 0. 总纲：两种命运

一个 .py 文件就是一个**模块**（Python 组织代码的最小单元）。
它的内容只会以两种方式进入执行：

```python
python cli.py                        # 命运 A：被运行（作为程序入口）
import cli                           # 命运 B：被导入（作为别人的依赖）
```

两种命运贯穿一切差异，先给对照总表，后文逐格展开：

| 维度 | 被运行 | 被导入 |
|---|---|---|
| 触发方式 | `python file.py` / `-m` / 交互式 | import 语句 |
| `__name__` | `'__main__'` | 完整模块名（如 `pkg.mod`） |
| `__package__` | `''`（直跑时无包身份） | 父包名（如 `pkg`） |
| 相对导入 | ✗ 无处落地 | ✓ 可用 |
| 对 sys.path 的贡献 | **决定** sys.path[0] | 不改变（命中缓存直接复用） |
| 代码执行时机 | 程序的起点 | 第一次 import 时（之后走缓存） |
| 典型角色 | 入口、脚本 | 库成员、依赖 |

> 全文最重要的一句话：**同一个文件，两种命运，身份不同、环境不同、
> 能用的写法也不同——所有"导入玄学"都是这两种命运的错位。**

---

## 1. 运行侧：敲下 `python cli.py` 的瞬间发生了什么

先认识本节主角——**sys.path：一个目录字符串的列表，是后续一切
import 的搜索范围**。import 找一个名字，就是拿它从头到尾逐目录找；
列表里没有装着目标模块的目录，就 ModuleNotFoundError。这个列表是
怎么被填出来的？import 还一个都没发生时，解释器先做两件独立的
准备（挂载、注入），做完后 sys.path 成形，入口才开始执行：

### 1.1 先扫清三个目录术语（后文反复用）

三个"目录"极易混淆，先同框实测——在项目根直跑一个躺在 Temp 的脚本：

```
cwd         = E:\workspace\studying\ai\my-ai-app   ← 我敲命令的地方
__file__    = D:\...\Temp\cwd_probe\show.py        ← 脚本文件自己躺在哪
sys.path[0] = D:\...\Temp\cwd_probe                ← 脚本所在目录（注入项）
```

**cwd（Current Working Directory，当前工作目录）**：进程"站在哪个
目录"，是每个进程的属性。**继承自启动它的 shell，与脚本躺在哪无关**。
它管**文件读写**：进程内一切相对路径（`open`、`Path`）都相对 cwd
解析，**不相对脚本文件**；它不管 import（import 只看 sys.path）。
经典坑：读"脚本旁边的文件"别靠 cwd——换个目录启动就找不到，正解是
`Path(__file__).parent / 'x.json'` 定位脚本自身。进程内
`os.getcwd()` 查看、`os.chdir()` 修改（只改本进程，不影响父 shell）。
node 世界的 `process.cwd()` 是同一个概念。

**脚本位置（`__file__` / sys.path[0]）**：脚本文件在磁盘上的位置；
sys.path[0] 是它被注入进搜索路径的形态（注入细节 §1.3）。

**解释器标准位置**：Python 自带代码的家——解释器安装目录下的几个
标准子目录（实测）：

```
...\cpython-3.13\python313.zip  ← 标准库的 zip 形式（加速加载）
...\cpython-3.13\DLLs           ← 解释器自带二进制扩展
...\cpython-3.13\Lib            ← 标准库（json、os、还有 site.py 都在这）
```

它与 site-packages **分家**，是理解环境隔离的钥匙：

| | 解释器标准位置 | site-packages |
|---|---|---|
| 装什么 | Python 自带（json、os、site） | 第三方（openai、mcp） |
| 谁写入 | 装 Python 时就有 | uv/pip 装包时 |
| 跟着谁走 | **解释器**（uv 管理的 cpython） | **虚拟环境**（.venv） |

"跟着谁走"即虚拟环境的隔离原理：**同一解释器可配多个 venv——
标准位置共享，site-packages 各自隔离**。

### 1.2 准备一：site 模块挂载（把"别人的家"接进来）

**site 模块是 Python 标准库自带的模块**（site.py，本体就在标准库
目录里，`site.__file__` 可验证），**不是第三方依赖**——名字里的
site 指"本机安装站点"（这台机器/这个环境的安装布局），它是官方
派来做本机初始化的"装配工"，解释器每次启动自动执行它。它做两件事：

- 把 **site-packages**（第三方包安装目录）加进 sys.path；
- 扫描 site-packages 下所有 **.pth 文件**，把其中每行路径也追加进
  sys.path（这是 editable install 的挂载通道，§5 展开）。

做完这一步，标准库和所有已安装第三方包就"天生可导"了。

> 辨析三个形近角色：**site 模块 = 装配工（自带），
> site-packages = 被挂载的第三方仓库（目录），.pth = 扩展清单（文件）**。
> 装的是第三方包，干活的模块是自带的——方向别搞反。

### 1.3 准备二：确定入口，注入 sys.path[0]

sys.path[0]——列表第一项、搜索优先级最高的位置——由启动方式决定，
解释器启动时自动计算并塞入：

| 运行方式 | sys.path[0] | 直觉理解 |
|---|---|---|
| `python src/xx/cli.py` 直跑 | 脚本所在目录 | 只带上了 cli.py 的"邻居们" |
| `python -m pkg.mod` | 当前工作目录 | 先把 cwd 放进去，再按模块名找 |
| `python -c "..."` / 交互式 | `''`（当前目录） | 同上 |

直觉：**你从哪儿启动，就先从哪儿找**——直跑 cli.py 时必须让它的
邻居目录可搜，否则它连同目录的兄弟模块都 import 不到。

### 1.4 汇总：sys.path 的初始内容 = 三个来源

两件准备 + 环境变量，拼出 §1 开头那个 sys.path 的初始内容公式：

```
sys.path = [ 启动方式注入的第一项 ]            ← §1.3，搜索优先级最高
         + [ PYTHONPATH 环境变量里的目录 ]      ← 手工配置的唯一进料口
         + [ 解释器标准位置 ]                   ← §1.2：标准库、site-packages、.pth 展开
```

**PYTHONPATH 补充**：操作系统环境变量，值是目录列表，解释器每次启动
读出并追加。插入位置在注入项之后、标准库之前——**优先级高于标准库**，
放个 json.py 进去会遮蔽真 json（坑）。定位：**不安装、不改代码，给
sys.path 手工加目录的官方通道**，但它是临时/兜底手段（脱离 editable
环境时救急），日常正路是 .pth（装一次永久生效）。设法三档：命令前缀
`PYTHONPATH=src uv run ...`（仅本次）、`set`（当前窗口）、
`setx`/系统面板（持久，之后所有新进程继承）。环境变量按进程继承——
"sys.path 里莫名多出目录"的常见来源是 **IDE 运行配置注入了它**。

本项目实测（`uv run python -c` 打印 sys.path，逐项注解）：

```
''                                    ← -c 启动注入的当前目录（§1.3）
D:\...\cpython-3.13...\python313.zip  ← 标准库压缩包
D:\...\cpython-3.13...\Lib            ← 标准库（json 在这）
E:\...\.venv\Lib\site-packages        ← 第三方包（openai 在这）
E:\...\my-ai-app\src                  ← ★ editable install 挂进来的（.pth，§1.2）
```

> 直接回答一个常见疑问："绝对导入的顶层名默认是 src 下面的吗？"
> **不是，任何项目都不是。** 顶层名能找到的唯一原因是：恰好有一个
> 目录在 sys.path 里且装着它。json 靠标准库位置，openai 靠
> site-packages，my_agent 靠 .pth 挂载——三个都是"环境事实"，
> 换台没跑过 `uv sync` 的机器，最后一个立刻 ModuleNotFoundError。

### 1.5 sys.path 的生命周期：进程私有，每次从零重建

两个常见疑问，一起答：

**每个文件一份 sys.path 吗？——不是，是每个进程一份。** sys 是
进程内唯一的模块，sys.path 是它身上的列表：同一进程内，入口与导入
链上任何深度的模块看到的是**同一个列表对象**，谁改它全进程生效
（`sys.path.insert` hack 的"传染性"即此）；不同进程完全独立——
实测：进程 A 里 `sys.path.append('I_WAS_HERE')`，进程 B 查它，
False。文件从不"拥有"sys.path，它被哪个进程加载，就用哪个进程的。

**site-packages 没变，每次启动还要重新挂吗？——要，零缓存。**
sys.path 活在**进程内存**里：进程结束内存释放，下次启动是全新进程 +
空白列表，必须重新执行一遍 site 装配（挂 site-packages、扫 .pth、
读 PYTHONPATH）。"没变化"是磁盘视角，新进程的内存不知道任何历史。
实测三连：往 site-packages 写一个临时 .pth → 下次启动立刻出现在
sys.path；删掉 → 再下次立刻消失。没有任何"记住上次结果"的机制。

不缓存的原因：sys.path 头部（注入项、PYTHONPATH）每次都随运行
上下文变，本就要现算；.pth 扫描只是读几个小文本文件（毫秒级），
单独做持久缓存收益极小。别混淆：**无缓存的是 sys.path 的构建；
有磁盘缓存的是模块字节码（__pycache__/*.pyc）**——import 的性能
优化发生在后者。

实验顺带发现一条 .pth 隐藏规则：**路径必须真实存在，否则被静默
跳过**——指向不存在的目录，.pth 等于白写，且无任何报错。

类比收束：像每次开机重新读配置——配置文件没动过，但内存是空的，
必须重读一遍。

### 1.6 入口执行：以 `__main__` 的身份

入口文件以 `__name__ = '__main__'`、`__package__ = ''` 的身份开始
从头到尾执行。**注意时序**：sys.path 在此之前就已定死——
运行方式决定寻址环境，之后的一切 import 都在这个环境里进行。

---

## 2. 导入侧：import 语句的四步流水线

入口开始执行代码，遇到 import 时，导入侧启动。一条 import 的完整
翻译过程是四步：

```
① 定名     把语句翻译成一个"完整名字"（dotted name）。
② 查缓存   拿完整名查 sys.modules：已有 → 直接复用，到此结束。
③ 寻址     未命中才拿完整名字按 sys.path 逐目录搜索，定位到文件。
④ 执行     文件从头到尾执行一遍，产生模块对象，塞回 sys.modules。
```

**① 定名。** 绝对导入与相对导入的**唯一分叉点**，§3 专门展开。
绝对=字面照抄；相对=由 `__package__` 换算。查缓存与寻址都以这个
**完整名字符串**为准，所以定名必须最先发生。

**② 查缓存。** 所有已导入模块都登记在 `sys.modules` 里——一个
**dict：key 是完整模块名（'json'、'security_serve.agent.loop'），
value 是模块对象本身**。拿①的完整名查它：命中即复用，寻址与执行
全部跳过。这就是 import 幂等的机制——同一模块 import 十遍，代码
只执行一遍，顶层副作用只在第一次发生。key 用全名，也正是相对导入
必须先换算成完整名的原因（不换算就没法查字典）。

实测佐证：解释器刚启动、一行 import 都没写时，表里已有 55 个模块
（sys、builtins……解释器启动自己依赖的）；`import
security_serve.agent.loop` 一句过后总数跳到 1300+——一条 import
把它整条依赖链全部执行并登记；重复 import 返回的就是表里那个对象
（`模块 is sys.modules['全名']` 为 True）。

**③ 寻址。** 拿定好的完整名字，先取**顶层名**（第一段）从 sys.path
逐目录找，找到后沿包的层级一路深入。任何一段找不到 →
`ModuleNotFoundError`。两条定名路线在这里汇合，之后完全同路。

**④ 执行。** 找到的文件以"被导入"身份执行（命运 B）：`__name__` =
完整模块名，`__package__` = 父包名。导入的本质是**执行别人的代码**——
这解释了为什么 `__init__.py` 能放初始化逻辑，也解释了循环导入为什么
会炸（对方执行到一半，名字还不存在）。

**递归成树。** 被导入的模块自己内部又有 import……整个程序的执行
结构就是：**一棵以"被运行的入口"为根、以"导入链"为枝的模块树**。
运行只有一次（根），导入层层展开（枝），每片叶子都是一次"执行 +
缓存"。

---

## 3. 定名：绝对与相对两种写法

**判断标准只有一条：语句开头有没有 `.`。** 有 `.` 是相对导入，
没有是绝对导入。

```python
# 绝对导入：从顶层名写完整路径（名字字面写死，与"我是谁"无关）
import json
from openai import OpenAI
from security_serve.agent.config import MAX_STEPS

# 相对导入：从"我所在的包"写起（名字由 __package__ 换算而来）
from .config import MAX_STEPS     # .  = 当前包
from ..other import x             # .. = 上一级包
from ...a import x                # ...= 上两级，点数无语法上限
```

### 3.1 绝对导入：字面定名

完整名字照抄语句，然后汇入 §2 流水线（查缓存 → 未命中才寻址）。
它**只依赖 sys.path、
不依赖自身身份**——单文件直跑、被 import、-m 启动，怎么都能解析。

### 3.2 相对导入：身份换算定名

完整名字 = 自己的 `__package__` + 点数换算（`.` 换掉 `__package__`，
`..` 换掉它的父级，以此类推）。所以相对导入**依赖"被导入"命运**——
这是 §4 的主题。

层级规则与硬边界：点数没有语法上限，但**不能越过顶层包**。
实测 `pkg/sub/` 内部：

| 写法 | 换算目标 | 结果 |
|---|---|---|
| `from .a import x` | pkg.sub.a | ✓ |
| `from ..a import x` | pkg.a | ✓ |
| `from ...a import x` | 越过 pkg 顶 | ✗ `attempted relative import beyond top-level package` |

`pkg` 之上没有包了，点无处落地。另有一个语法限制：相对导入只能用于
`from ... import ...`，`import .config` 直接是语法错误。

### 3.3 依赖差异（§7 选型的伏笔）

| | 绝对导入 | 相对导入 |
|---|---|---|
| 完整名怎么来 | 字面写死 | `__package__` 换算 |
| 依赖什么 | src 在 sys.path | 模块有包身份（被导入） |
| 直跑单文件 | ✓ 照样能解析 | ✗ 身份缺失就报错 |
| 重命名包/移动目录 | 每处都要改 | 内部引用不用改 |

---

## 4. 身份：两种命运的交汇与分野

§1 说运行给身份（`__main__` + 空包），§2 说导入给身份（全名 + 父包）。
现在把两边合到一张表——同一文件，两种命运：

| 打开方式 | `__name__` | `__package__` |
|---|---|---|
| `python cli.py` 直跑（被运行） | `'__main__'` | `''` |
| 被 import（被导入） | 全名，如 `security_serve.agent.cli` | 父包名，如 `security_serve.agent` |

实测佐证：`import security_serve.agent.loop` 后
`__name__ = 'security_serve.agent.loop'`、`__package__ = 'security_serve.agent'`。

### 4.1 if __name__ == '__main__'：兼容两种命运的通行写法

入口文件常常既想直跑又想被 import 不产生副作用，标准写法：

```python
def main():
    ...

if __name__ == '__main__':   # 只在被运行（命运 A）时为真
    main()
```

本质就是 §0 总表的第一个行为差异：直跑给 `__main__`，导入给全名，
用身份区分"我是不是程序入口"。

### 4.2 相对导入为什么直跑会挂

相对导入定名需要 `__package__`（§3.2），而直跑命运给的是 `''`——
定名无处落地，报
`attempted relative import with no known parent package`。
这就是"import 它一切正常、直跑它当场报错"同一段代码的根因：
**不是文件变了，是打开方式决定了命运。**

### 4.3 -m 如何改变命运

`python -m security_serve.agent.cli` 不按文件路径执行，而是
**按完整模块名加载**：模块以正常的包成员身份出生
（`__package__ = 'security_serve.agent'`），同时 cwd 进入 sys.path[0]
（§1.3）。一次同时修好身份与路径——这是包内模块的标准运行方式。

### 4.4 身份与路径是两条独立的轴

身份决定"相对导入能不能**定名**"；路径决定"定好的名字**找不找得到**"。
两条轴互相独立，后面 §8 的报错分诊全靠它们。

---

## 5. 环境：寻址原料的生产线

§2 的寻址只认 sys.path。sys.path 里的第三方目录是哪来的？
答案是**安装**——安装的本质就是把目录送进 sys.path，送法有两种。

### 5.1 普通安装：复制进 site-packages

**site-packages** 是 pip/uv 存放第三方包的标准目录（本项目虚拟环境下
是 `.venv\Lib\site-packages`），site 模块启动时自动挂入 sys.path
（§1.2）。`import openai` 零配置，因为 openai 的代码被**复制**躺在那里。

对自己项目这样做的问题：装完再改源码，环境里跑的还是旧副本——开发期不可用。

### 5.2 editable install：不复制，只留指针

**editable install（可编辑安装）是"安装你自己项目"的方式**：
不复制代码，只在 site-packages 写一个 `.pth` 文件，内容是指向源码
目录的一行路径。每次解释器启动，site 模块读它把该路径挂进 sys.path
（§1.2 的通道）——源码原地不动、改动立即生效，"editable"即此意。

实测本项目（.venv/Lib/site-packages/）：

```
my_ai_app.pth   文件内容就一行：
                E:\workspace\studying\ai\my-ai-app\src
```

### 5.3 什么时候执行

`uv sync`、`uv add`、`uv pip install -e .` 这类安装动作时执行。
uv 管理项目时**默认把项目本身以 editable 方式装进环境**——
`uv sync` 之后 `import my_agent` 能用，不是自动的，是 uv 替你做了
editable install。

### 5.4 本项目的连带事实

- editable 挂载的是**整个 src/ 目录**，不是只挂 pyproject 声明的包。
  security_serve 没在 pyproject 声明，`import security_serve` 照样解析
  ——依赖这个副作用时心里要有数；
- `uv run python -m security_serve.agent.cli` 不需要 PYTHONPATH；
- `PYTHONPATH=src` 只在不依赖 editable 挂载时才需要（如系统 python
  直接跑），uv 环境里是冗余保险。

---

## 6. 组织形态：单文件与包

命运 B（被导入）的模块住在哪里、怎么分家，就是组织形态问题。

### 6.1 模块与包：叶子与容器

- **模块（单文件）**：一个 .py，叶节点，不能含子模块；
- **包**：目录形态的"容器模块"，`__path__` 是目录，能继续容纳
  子模块/子包，形成 `pkg.sub.mod` 层级命名空间（不同包下同名模块
  互不冲突）。

| | 模块（单文件） | 包（目录） |
|---|---|---|
| `__file__` | 指向 .py 文件 | 指向 `__init__.py` |
| `__path__` | 无（叶子） | 有（目录） |
| 向下容纳 | 不能 | `from pkg import sub` 一路向下 |
| 使用常态 | 直跑/被导入皆可 | 以"被导入"为常态 |

### 6.2 regular 与 namespace：包的两种形态

- **regular package**：目录里有 `__init__.py`。它是包的**凭证与
  执行点**：① 标记 regular，包内模块的 `__package__` 由此建立；
  ② 导入包时最先执行的代码（可放包级 API 提升、`__all__`、初始化，
  最小形态只写 docstring）；③ 搜索语义确定，无跨目录合并歧义；
- **namespace package**（PEP 420）：无 `__init__.py` 的纯容器，
  `__file__ = None`，设计目的是跨多个 sys.path 条目合并同名目录——
  site-packages 里若混入同名目录会静默拼入，正式代码建议补
  `__init__.py` 消除歧义。

判别特征两行输出：

```
security_serve (namespace): __file__ = None
agent (regular):            __file__ = ...\security_serve\agent\__init__.py
```

本项目实况：`agent/` 是 regular；`security_serve/` 这层是 namespace
（纯目录分层），现状够用。

---

## 7. 工程决策：写法即契约

### 7.1 没有"万能写法"

§3.3 的依赖差异决定：**没有一种导入写法天然同时满足"直跑"和
"被导入"两种使用方式**。选定写法 = 选定优先支持哪种命运。
所以先问使用方式，再定写法：

- **一次性入口**（step 教学文件）：只直跑、从不被 import →
  平铺/任意绝对导入，怎么顺手怎么来；
- **库成员**（agent 包的 tools/memory/loop）：注定被多种方式使用
  （cli 启动、脚本导入、FastAPI 导入）→ 写法必须在所有方式下成立：
  **完整名绝对导入**（或相对导入，二选一并全包一致）。

本项目落地：

```
src/security_serve/
  step*.py            # 入口：平铺 import / import agent.tools（包方式）
  agent/              # regular 包：包内统一完整名绝对导入
    config.py / tools.py / safety.py / memory.py / loop.py / cli.py
```

入口两种跑法等价可用（editable 保证 src 在 sys.path；绝对导入不查身份）：

```
uv run python src/security_serve/agent/cli.py   # 直跑
uv run python -m security_serve.agent.cli       # 模块方式
```

---

## 8. 报错分诊：两条轴定位一切导入问题

所有导入报错先分诊到两条轴——**身份轴（§4）还是路径轴（§1/§5）**：

| 报错关键字 | 哪条轴 | 根因 | 解法 |
|---|---|---|---|
| `attempted relative import with no known parent package` | 身份 | 直跑命运 `__package__` 为空，相对导入无法定名 | -m 启动 / 改绝对导入；**与 sys.path 无关，改路径救不了** |
| `attempted relative import beyond top-level package` | 身份 | 点数越过顶层包（§3.2） | 减点数，或改绝对导入 |
| `ModuleNotFoundError: No module named 'X'` | 路径 | sys.path 里没有装着 X 的目录 | 检查挂载（editable/.pth/PYTHONPATH）与启动方式 |

口诀：报错有 **relative import** 字样 → 查身份（`__package__`）；
报错是 **No module named** → 查路径（打印 sys.path 逐项看）。

---

## 9. 全景收束：运行与导入的完整旅程

```
$ python -m security_serve.agent.cli
   │
   ├─【运行侧准备】§1
   │    ├ 建 sys.path：启动注入[0] + PYTHONPATH + 标准位置
   │    ├ site 模块：挂 site-packages + 展开 .pth（src/ 由此进场，§5）
   │    └ 入口以 __main__ 身份开始执行（命运 A）
   │
   └─【导入侧展开】§2-§4
        └ 遇 import → ① 定名：绝对=字面 / 相对=__package__ 换算（§3）
                     ② 查缓存 sys.modules[全名]：命中即复用
                     ③ 寻址：sys.path 逐目录搜索
                     ④ 执行：以全名身份运行（命运 B）→ 登记进 sys.modules
                        └ 模块内部又有 import……层层展开成模块树
```

记忆钩子：**运行给环境（sys.path 怎么定、入口是谁），导入给依赖
（名字怎么定、文件去哪找）；绝对导入问路（sys.path），相对导入认人
（__package__）；身份由命运（运行/导入）决定，src 进不进 path 由
环境（editable/.pth）决定——命运与环境，是两条独立的轴。**
