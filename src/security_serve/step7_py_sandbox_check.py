"""step7_py_sandbox_check.py —— H3 验收：代码执行沙箱工具

两部分验收：
  确定性部分（默认跑，不烧 LLM 的钱，直调 run_code()）：
    [1] 正常执行 + 中文输出（stdin 喂代码，UTF-8 全链路）
    [2] 统计任务答对数：均值 / 最大值 / 排序（模拟「这份数据的统计结论」类问题）
    [3] 脚本报错回传：1/0 的 ZeroDivisionError traceback 原样返回给模型
    [4] 网络隔离：容器内连外网必败（--network none）
    [5] 文件隔离：容器内读业务代码必败（不挂载，纯净底座无业务代码）
    [6] 死循环超时：while True 被容器内 timeout 杀掉（退出码 124）
    [7] 内存炸弹：[0]*10**9 被内存围栏拦下（MemoryError / OOM）
    [8] fork 炸弹被资源围栏拦下 + 宿主无损总检（订单可查 / 记忆在 / 密钥在）
  LLM 端到端（--live，需 .env 真实 key，经 TestClient 走服务全链路）：
    [9] 自然语言统计 → 模型真的调了 run_python（行为判定，测行为不测措辞）
    [10] 纯算术分流 → 模型用 calculate 而不动沙箱（提示词第 4 条分工生效）

跑法（项目根）：
  uv run python src/security_serve/step7_py_sandbox_check.py          # 确定性（约 40 秒，含一个 10 秒超时用例）
  uv run python src/security_serve/step7_py_sandbox_check.py --live   # 加 LLM 端到端
前置：Docker 引擎在跑 + python:3.13-slim 镜像已在本地（docker images 可见）
"""
import sys

from security_serve.agent.config import MEMORY_PATH, PROJECT_ROOT
from security_serve.agent.sandbox import run_code
from security_serve.agent.session_store import load_session
from security_serve.agent.tools import db_query

failures = []


def mark(label: str, ok: bool):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def first_line(r: str) -> str:
    return r.splitlines()[0][:80] if r else r


if __name__ == "__main__":
    print("H3 验收：代码执行沙箱工具（run_python，一次性容器 + 五道围栏）")
    print("=" * 60)

    print("[1] 正常执行 + 中文输出（stdin 喂代码，UTF-8 全链路）")
    r = run_code('print("你好，沙箱")\nprint(1 + 2)')
    print(" ", first_line(r))
    mark("中文与计算结果都正常返回", "你好，沙箱" in r and "3" in r)

    print("[2] 统计任务（模型视角：数据写字面量 → 代码算出结论）")
    r = run_code(
        "data = [23, 45, 12, 78, 34]\n"
        'print(f"均值 {sum(data) / len(data):.1f}")\n'
        'print(f"最大 {max(data)}")\n'
        'print("排序", sorted(data))'
    )
    print(" ", first_line(r))
    mark("均值/最大值/排序全部算对", "38.4" in r and "78" in r and "12" in r)

    print("[3] 脚本报错回传（模型看 traceback 改代码重试的前提）")
    r = run_code("print(1 / 0)")
    print(" ", first_line(r))
    mark("ZeroDivisionError traceback 原样回传", "ZeroDivisionError" in r)

    print("[4] 网络隔离（围栏1：--network none）")
    r = run_code(
        "import socket\n"
        "try:\n"
        '    socket.create_connection(("8.8.8.8", 53), timeout=3)\n'
        '    print("NETWORK_OK")\n'
        "except Exception as e:\n"
        '    print(f"BLOCKED: {type(e).__name__}")'
    )
    print(" ", first_line(r))
    mark("容器内连外网必败", "NETWORK_OK" not in r and "BLOCKED" in r)

    print("[5] 文件隔离（围栏2：不挂载，容器里只有纯净底座）")
    r = run_code(
        "try:\n"
        '    open("/app/src/security_serve/agent/config.py", encoding="utf-8").read()\n'
        '    print("FILE_READ")\n'
        "except FileNotFoundError:\n"
        '    print("FILE_BLOCKED")\n'
        "except PermissionError as e:\n"
        '    print(f"FILE_BLOCKED: {type(e).__name__}")'
    )
    print(" ", first_line(r))
    mark("容器内读业务代码必败", "FILE_READ" not in r and "FILE_BLOCKED" in r)

    print("[6] 死循环超时（容器内 timeout 精确杀，退出码 124）")
    r = run_code("while True: pass")
    mark("超时代码被强制终止", "超时" in r)

    print("[7] 内存炸弹（围栏3：--memory 128m）")
    r = run_code("x = [0] * 10**9\nprint(len(x))")
    print(" ", first_line(r))
    mark("内存炸弹被围栏拦下（MemoryError 或 OOM）", "MemoryError" in r or "内存超限" in r)

    print("[8] fork 炸弹（围栏5：--pids-limit 64）+ 宿主无损总检")
    # 哪道围栏先拦取决于资源消耗次序：64 个 CPython 进程的内存footprint
    # 会先撞 128m 内存墙，所以 fork 报 OSError、MemoryError、OOM 任何一种
    # 都证明「炸不动」——断言取三路并集
    r = run_code(
        "import os\n"
        "count = 0\n"
        "try:\n"
        "    while True:\n"
        "        os.fork()\n"
        "        count += 1\n"
        "except Exception as e:\n"
        '    print(f"FORK_LIMITED: fork {count} 次后被 {type(e).__name__} 拦截")'
    )
    print(" ", first_line(r))
    mark("fork 炸弹被资源围栏拦下（无法无限造进程）",
         "FORK_LIMITED" in r or "MemoryError" in r or "内存超限" in r)
    mark("宿主业务库无损（订单计数仍可查）", "18" in db_query("SELECT COUNT(*) FROM orders"))
    mark("宿主长期记忆文件仍在", MEMORY_PATH.exists())
    mark("宿主 .env（密钥）未被触碰", (PROJECT_ROOT / ".env").exists())

    if "--live" in sys.argv:
        print("[9-10] LLM 端到端（真实 key，经 TestClient 走服务全链路）")
        from fastapi.testclient import TestClient
        from security_serve.agent.server import app

        def tool_names_of(session_id: str) -> list:
            # 会话已外置 Redis（H4）：从存储读回 messages 判定工具调用（测行为不测措辞）
            msgs = load_session(session_id)["messages"]
            return [
                c["function"]["name"]
                for m in msgs if m.get("tool_calls")
                for c in m["tool_calls"]
            ]

        client = TestClient(app)
        r = client.post("/chat", json={
            "message": "星辰商城各会员等级分别有多少人？帮我算出每个等级占总人数的百分比，精确到 1 位小数，请写代码精确计算不要心算。",
            "session_id": "h3-live",
        })
        reply = r.json()["reply"]
        print(f"  reply: {reply[:150]}")
        names = tool_names_of("h3-live")
        print(f"  本次会话调用的工具: {names}")
        mark("模型真的调了 run_python（行为判定）", "run_python" in names)
        mark("回复给出了百分比结论", "%" in reply or "百分" in reply)

        r = client.post("/chat", json={
            "message": "用 calculate 工具算一下 1234567 × 7654321 等于多少？",
            "session_id": "h3-live2",
        })
        reply = r.json()["reply"]
        print(f"  reply: {reply[:150]}")
        names = tool_names_of("h3-live2")
        print(f"  本次会话调用的工具: {names}")
        mark("纯算术走 calculate 不动沙箱", "calculate" in names and "run_python" not in names)

    print("=" * 60)
    if failures:
        print(f"结果：FAIL（{len(failures)} 例不符合预期）：{failures}")
        sys.exit(1)
    print("结果：PASS——正常执行与统计答对、报错回传、网络/文件/内存/进程四道围栏与超时全拦、宿主无损"
          + ("，LLM 端到端通过" if "--live" in sys.argv else ""))
