"""step9a_logging_basics.py —— H5 前置小节：logging 机制纯文本玩透（零 API 成本）

先于结构化改造（H5-1）把 logging 五件套玩透：logger 树、级别两道闸、
Handler-Formatter、Filter、传播开关。全部用项目真实的 logger 名字演示
（security_serve.agent.loop / security_serve.server，与 loop.py / server.py
里 getLogger 的名字同源），实验现象直接回答一个现状问题：

  为什么服务里那些 log.info 看不见、log.warning 看得见？
  （server.py docstring 只记了现象，本实验拆开机制。）

 LogRecord 产生的条件：调用级别的数字 ≥ logger 的有效级别的数字

跑法：uv run python src/security_serve/step9a_logging_basics.py
只动标准库 logging，不碰业务代码、不调 API；logging 默认输出到 stderr，
print 输出到 stdout——stderr 无缓冲、stdout 重定向到管道时块缓冲，两流合并
查看会出现"日志跑到讲解前面"的错序，main 开头给 stdout 开行缓冲对齐时序。
"""
import logging
import sys
import tempfile
from pathlib import Path

# 与 loop.py / server.py 同名的两个 logger——实验即复现服务现场
LOOP = "security_serve.agent.loop"
SERVER = "security_serve.server"


def hr(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def reset() -> None:
    """清空全部配置回到出厂状态，保证每个小节独立复现。

    logging 是进程级全局状态：root 的 handlers / level、每个具名 logger 的
    自有 handler / level / propagate，上一小节改过的都会带到下一小节——
    演示脚本必须显式重置，这也是生产代码里常被忽视的坑（import 副作用）。
    """
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.WARNING)  # 出厂值：basicConfig 未被调用时的默认级别
    for name in (LOOP, SERVER, "security_serve.agent", "security_serve"):
        lg = logging.getLogger(name)
        for h in lg.handlers[:]:
            lg.removeHandler(h)
        lg.setLevel(logging.NOTSET)  # 0 = 未设置，有效级别沿父链向上找
        lg.propagate = True
    # handlers 清空后 root 回到"零 handler"状态，后续小节的 basicConfig 恢复可生效
    # （它的生效条件就是 root 零 handler；force 是它的调用参数，与此处无关）


# [1] logger 树：点分名字即父子关系 ------------------------------------------------
def demo_1_tree() -> None:
    reset()
    hr("[1] logger 树：点分名字就是父子关系")
    loop_log = logging.getLogger(LOOP)
    server_log = logging.getLogger(SERVER)
    print(f"getLogger({LOOP!r}).parent        = {loop_log.parent.name}")
    print(f"getLogger({SERVER!r}).parent     = {server_log.parent.name}")
    print(f"root logger 的名字               = {logging.getLogger().name!r}（root 没有名字）")
    twice = logging.getLogger(LOOP)
    print(f"同名两次 getLogger 是同一对象    = {twice is loop_log}（注册表单例，名字即钥匙）")
    print("结论：命名即树。security_serve.agent.loop 的父链是 agent → security_serve → root。")


# [2] 级别第一道闸：现状复现 ------------------------------------------------------
# LogRecord 产生的条件：调用级别的数字 ≥ logger 的有效级别的数字，root级别默认30
def demo_2_level_gate() -> None:
    reset()
    hr("[2] 级别第一道闸：什么都没配时，info 为什么看不见")
    loop_log = logging.getLogger(LOOP)
    root = logging.getLogger()
    print(f"loop_log.level        = {loop_log.level}（NOTSET：未设置，有效级别沿父链找）")
    eff = loop_log.getEffectiveLevel()
    print(f"loop_log 有效级别     = {eff} = {logging.getLevelName(eff)}（继承自 root 出厂值 WARNING）")
    print(f"root.handlers         = {root.handlers}（空的：没有任何 handler）")
    print("\n-- 实验 A：loop_log.info(...) —— 20 < 30，第一道闸直接拦下，管线根本不启动")
    loop_log.info("这条 info 不会出现")
    print("   （上面无输出 = 被拦。没有 LogRecord 产生，连 handler 都没见到）")
    print("\n-- 实验 B：loop_log.warning(...) —— 30 >= 30 过闸，但树上没有 handler……")
    loop_log.warning("这条 warning 出现了，且只有光秃秃一行")
    print("   （输出来自 logging.lastResort：全树零 handler 时的兜底出口，")
    print("     WARNING 级别、直写 stderr、无时间戳无名字——这就是服务里 warning")
    print("     '神秘可见'的机制，也是它长得那么简陋的原因）")


# [3] basicConfig：root 的级别与 handler 从这来 ----------------------------------
# logging.basicConfig 这个handler函数将root级别线从 30 降到 20,并且配置Formatter
def demo_3_basicconfig() -> None:
    reset()
    hr("[3] basicConfig(level=INFO)：让 info 现形")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    root = logging.getLogger()
    print(f"basicConfig 做了三件事：root.level={logging.getLevelName(root.level)}，")
    print(f"挂 root.handlers={root.handlers}（StreamHandler→stderr），配了 Formatter。")
    print("（注：basicConfig 只在 root 零 handler 时生效，重复调用被忽略，3.8+ 需 force=True）")
    print("\n-- 同一条 info，重新发一次：")
    logging.getLogger(LOOP).info("现在现形了：有名字、有级别、有时间戳")
    print("   （输出来自 root 的 StreamHandler：消息沿父链传播上来，被 root 的 handler 发射）")


# [4] Handler 与 Formatter：一条消息，两个去处，两种格式 --------------------------
# 第一道闸：调用级别 vs logger的有效级别，如果小于则不产生logRecord
# 第二道闸：消息级别 vs handler自己的lever,如果小于则这个出口不出，别的出口照常
# 第一道闸拦下 = 全局消失；第二道闸拦下 = 只拦自己这个出口
def demo_4_handlers(tmp: Path) -> None:
    reset()
    hr("[4] Handler 与 Formatter：多去向 + 第二道闸")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log_file = tmp / "playground.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
    loop_log = logging.getLogger(LOOP)
    loop_log.addHandler(fh)  # 子 logger 挂自己的 handler
    print("已给 loop_log 挂 FileHandler（竖线分隔格式），root 仍有终端 StreamHandler。")
    print("\n-- 发一条 info：消息先过 logger 闸（INFO 放行），再沿父链把 handler 全问一遍")
    loop_log.info("一条消息，两个去处")
    print(f"   终端有（root handler）+ 文件有（自有 handler）→ {log_file} 内容：")
    print("   " + log_file.read_text(encoding="utf-8").strip().replace("\n", "\n   "))
    print("\n-- 第二道闸：handler 有自己的级别。fh.setLevel(ERROR) 后——")
    fh.setLevel(logging.ERROR)
    loop_log.info("这条只去终端，不去文件")
    print(f"   文件仍是 1 行 → 同一条消息在不同出口的去留，由 handler 级别独立裁决。")
    fh.setLevel(logging.INFO)  # 还原，供后续小节使用


# [5] Filter：按模块分流 ----------------------------------------------------------
def demo_5_filter(tmp: Path) -> None:
    reset()
    hr("[5] Filter：按模块分流（谁的消息能过这个口）")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root_handler = logging.getLogger().handlers[0]
    # filter 收到 LogRecord，返回 True 放行——只放行 server 模块的消息
    root_handler.addFilter(lambda rec: rec.name == SERVER)
    server_log = logging.getLogger(SERVER)
    loop_log = logging.getLogger(LOOP)
    print("已给 root 的终端 handler 挂 filter：只放行 security_serve.server 的消息。")
    print("\n-- server_log.info(...) 与 loop_log.info(...) 各发一条：")
    server_log.info("server 的消息：过了 filter，终端可见")
    loop_log.info("loop 的消息：被 filter 拦下，终端不可见")
    print("   （只看到 server 那条 = filter 在 handler 层按 logger 名分流，")
    print("     生产里'只收业务日志、滤掉三方库噪音'就是这一招）")


# [6] propagate：传播开关 ---------------------------------------------------------
# 开关配置的是 子logger发的消息，是否被root的handler处理
def demo_6_propagate(tmp: Path) -> None:
    reset()
    hr("[6] propagate：切断传播，各归各")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log_file = tmp / "propagate.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s|%(message)s"))
    loop_log = logging.getLogger(LOOP)
    loop_log.addHandler(fh)
    loop_log.propagate = True
    print("loop_log 有文件 handler，root 有终端 handler，propagate=True。")
    loop_log.info("传播开着：文件、终端各收一份")
    print(f"   文件 1 行、终端 1 行。现在 loop_log.propagate = False——")
    loop_log.propagate = False
    loop_log.info("传播关了：只进文件，终端绝收不到")
    content = log_file.read_text(encoding="utf-8").strip().splitlines()
    print(f"   文件变 2 行、终端没新增 → propagate=False 切断沿父链的 handler 查找，")
    print("   消息只走自有 handler。生产里'本模块日志只落文件、不刷终端'即此开关。")


def main() -> None:
    # stderr（logging）无缓冲、stdout（print）重定向到管道时块缓冲——合并查看
    # 时日志会"穿越"到讲解前面；行缓冲让 print 立即出，时序与代码顺序一致
    sys.stdout.reconfigure(line_buffering=True)
    print("H5-0 前置小节：logging 机制纯文本玩透（实验现象在 stderr，讲解在 stdout）")
    tmp = Path(tempfile.mkdtemp(prefix="step9a_"))
    demo_1_tree()
    demo_2_level_gate()
    demo_3_basicconfig()
    demo_4_handlers(tmp)
    demo_5_filter(tmp)
    demo_6_propagate(tmp)
    print(f"\n{'=' * 62}")
    print("五件套一句话：名字定树、logger 级别定'配不配进管线'、handler 级别定")
    print("'这个口出不出'、Formatter 定长相、Filter 定谁过、propagate 定上不上溯。")
    print(f"实验日志文件（可自行查看后删除）：{tmp}")
    print("=" * 62)


if __name__ == "__main__":
    main()
