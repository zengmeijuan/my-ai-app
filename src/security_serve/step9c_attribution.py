"""step9c —— H5-4 验收：工具耗时 + LLM 成本归因。

零 API / 零 Redis：patch 掉 LLM client（手造流式 chunk，含 usage 收尾块）与
session 读写，run_agent_turn 本体是待测主角不 patch——中间件/日志/工具全部真跑。

断言五件事：
[1] 每轮 LLM 调用落一行 event=llm_usage（model/prompt/completion/total/duration_ms 齐）
[2] Budget 记账切到真实 tokens：响应 used_tokens == 两轮 total_tokens 之和（精确值）
[3] tool_call 行带 duration_ms>0（真 calculate 执行，round 3 位保证非零）
[4] 降级：usage 收尾块没送达时回落字符折算（无 llm_usage 行、used_tokens 仍 >0）
[5] 请求侧确实带了 stream_options={"include_usage": True}（没有它收尾块根本不会来）

聚合演示：按 event 分组求和 duration_ms——"LLM 占了多少、工具占了多少"的归因账。

跑法（项目根）：
  默认：   PYTHONUTF8=1 uv run python src/security_serve/step9c_attribution.py
  --live： uv run --with fakeredis python src/security_serve/step9c_attribution.py --live
           （fakeredis 内存 redis + 真 uvicorn + 真 LLM，验证 stream_options 在真实网关的兼容性）
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

tmp = Path(tempfile.mkdtemp(prefix="step9c_"))
import security_serve.agent.config as config
config.LOG_PATH = tmp / "server.jsonl"  # 先 patch 再导入（模块级 setup 读它）

import security_serve.agent.loop as loop_mod
import security_serve.agent.server as srv

# 将load_session、save_session改为假函数
srv.load_session = lambda sid: None
srv.save_session = lambda sid, sess: None

# ---------- 假 LLM：手造流式 chunk（SimpleNamespace 足够模拟协议属性访问） ----------
create_kwargs = []  # 记录每次 create 的入参,供 [5] 断言

# NS 是 SimpleNamespace 的别名，标准库提供的空容器对象，模拟被测对象的访问方式
def make_chunks(text=None, tool_call=False, usage=None):
    """按 OpenAI 流式协议形状造 chunk 列表;收尾块 choices=[] 只带 usage。"""
    chunks = []
    if tool_call:
        chunks.append(NS(choices=[NS(delta=NS(content=None, tool_calls=[
            NS(index=0, id="call_x1",
               function=NS(name="calculate", arguments='{"expression": "1+1"}')),
        ]))], usage=None))
    if text:
        chunks.append(NS(choices=[NS(delta=NS(content=text, tool_calls=None))], usage=None))
    chunks.append(NS(choices=[], usage=usage))
    return iter(chunks)

# 模拟的是 LLM API 调用本身
def fake_create(**kwargs):
    create_kwargs.append(kwargs)
    n = len(create_kwargs)
    if n == 1:   # 会话 A 第 1 轮:发工具调用,收尾块带真实 usage
        return make_chunks(tool_call=True, usage=NS(prompt_tokens=100, completion_tokens=20, total_tokens=120))
    if n == 2:   # 会话 A 第 2 轮:最终回答,收尾块带真实 usage
        return make_chunks(text="42", usage=NS(prompt_tokens=110, completion_tokens=10, total_tokens=120))
    return make_chunks(text="ok")  # 会话 B(降级样本):收尾块不带 usage


loop_mod.client = NS(chat=NS(completions=NS(create=fake_create)))

from fastapi.testclient import TestClient

failures = []

# ---------- --live 模式:真 uvicorn + 真 LLM,验 stream_options 网关兼容 + 真实归因 ----------
if "--live" in sys.argv:
    import os
    import subprocess
    import threading
    import time

    from fakeredis import TcpFakeServer

    # 构造redis服务，监听6393端口
    server = TcpFakeServer(("127.0.0.1", 6393))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    env = {**os.environ, "REDIS_URL": "redis://127.0.0.1:6393/0", "PYTHONUTF8": "1"}
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "security_serve.agent.server:app",
         "--app-dir", str(root / "src"), "--port", "8004"],
        env=env, stdin=subprocess.DEVNULL, cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    import httpx

    live_log = root / "src/security_serve/agent/data/server.jsonl"
    before = len(live_log.read_text(encoding="utf-8").strip().splitlines()) if live_log.exists() else 0
    for _ in range(60):
        try:
            if httpx.get("http://127.0.0.1:8004/health", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        sys.exit("FAIL: --live 服务 30 秒未就绪")

    r = httpx.post("http://127.0.0.1:8004/chat",
                   json={"message": "(30+12)*2 等于多少", "session_id": "attr-live"},
                   headers={"X-Request-ID": "attr-live-1"}, timeout=120)
    if r.status_code != 200:
        failures.append(f"[live] 请求失败: {r.status_code}")
    proc.terminate()

    rows = [json.loads(line) for line in
            live_log.read_text(encoding="utf-8").strip().splitlines()][before:]
    live_usage = [row for row in rows if row.get("event") == "llm_usage"]
    live_tool = [row for row in rows if row.get("event") == "tool_call"]
    if not live_usage:
        failures.append("[live] 无 llm_usage 行——usage 收尾块没来(stream_options 未生效或网关不支持)")
    for row in live_usage:
        if row.get("total_tokens", 0) <= 0:
            failures.append(f"[live] llm_usage total_tokens 异常: {row}")
    if live_tool and not all(row.get("duration_ms", 0) > 0 for row in live_tool):
        failures.append(f"[live] tool_call 缺 duration_ms: {live_tool}")
    if live_usage:
        total_cost = sum(row.get("total_tokens", 0) for row in live_usage)
        print(f"[live] 真实 LLM 调用 {len(live_usage)} 轮,合计 {total_cost} tokens;"
              f"工具 {len(live_tool)} 次")
        # 端到端只数 /chat 行：轮询 /health 也各落一行 http_request，
        # 不过滤 path 会把健康检查的几毫秒混进端到端账里
        e2e = [row for row in rows
               if row.get("event") == "http_request" and row.get("path") == "/chat"]
        print(f"[live] 归因: LLM {sum(row.get('duration_ms', 0) for row in live_usage):.0f}ms"
              f" vs 工具 {sum(row.get('duration_ms', 0) for row in live_tool):.1f}ms"
              f" vs 请求端到端 {sum(row.get('duration_ms', 0) for row in e2e):.0f}ms")
    if failures:
        print("FAIL：")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nPASS(--live)：stream_options 网关兼容、llm_usage 真实 tokens、tool_call 耗时全落日志")
    sys.exit(0)

client = TestClient(srv.app)

# ---------- 会话 A:一次请求内跑两轮 LLM(工具轮 + 回答轮),真 calculate 执行 ----------
r_a = client.post("/chat", json={"message": "1+1 等于几", "session_id": "attr-a"},
                  headers={"X-Request-ID": "attr-rid-1"})
if r_a.status_code != 200:
    failures.append(f"[A] 请求失败: {r_a.status_code}")

# ---------- 会话 B:usage 收尾块缺席,走字符折算降级 ----------
r_b = client.post("/chat", json={"message": "随便说点什么", "session_id": "attr-b"},
                  headers={"X-Request-ID": "attr-rid-2"})
if r_b.status_code != 200:
    failures.append(f"[B] 请求失败: {r_b.status_code}")


def records() -> list:
    lines = config.LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


# [1] llm_usage 行:字段齐、带 request_id、每轮一行
usage_rows = [r for r in records() if r.get("event") == "llm_usage"]
if len(usage_rows) != 2:
    failures.append(f"[1] llm_usage 行数 {len(usage_rows)} != 2")
for row in usage_rows:
    missing = [k for k in ("model", "prompt_tokens", "completion_tokens", "total_tokens", "duration_ms") if k not in row]
    if missing:
        failures.append(f"[1] llm_usage 缺字段 {missing}: {row}")
    if row.get("request_id") != "attr-rid-1":
        failures.append(f"[1] llm_usage 的 request_id 异常: {row.get('request_id')!r}")

# [2] Budget 真实 tokens 入账:120 + 120 = 240(响应体的 used_tokens 就是预算账本现值)
if r_a.json().get("used_tokens") != 240:
    failures.append(f"[2] 会话 A used_tokens {r_a.json().get('used_tokens')} != 240(真实 tokens 未入账)")

# [3] tool_call 行带 duration_ms(真 calculate,微秒级靠 round 3 位保住非零)
tool_rows = [r for r in records() if r.get("event") == "tool_call"]
if len(tool_rows) != 1:
    failures.append(f"[3] tool_call 行数 {len(tool_rows)} != 1")
for row in tool_rows:
    if not row.get("duration_ms", 0) > 0:
        failures.append(f"[3] tool_call 缺 duration_ms 或为零: {row}")

# [4] 降级:会话 B 无 llm_usage 行;used_tokens>0 说明字符折算还在工作
b_usage = [r for r in usage_rows if r.get("request_id") == "attr-rid-2"]
if b_usage:
    failures.append(f"[4] 无 usage 的会话不应落 llm_usage 行: {b_usage}")
if not r_b.json().get("used_tokens", 0) > 0:
    failures.append(f"[4] 降级口径未记账: used_tokens={r_b.json().get('used_tokens')}")

# [5] stream_options 已随请求发出
for kw in create_kwargs:
    if kw.get("stream_options") != {"include_usage": True}:
        failures.append(f"[5] create 未带 stream_options(收尾块不会来): {kw}")

# ---------- 归因聚合演示:按 event 分组求和 ----------
durs = {}
for row in records():
    if "duration_ms" in row:
        durs.setdefault(row["event"], []).append(row["duration_ms"])
print("[归因] 各事件耗时合计(ms):")
for ev, vals in sorted(durs.items()):
    print(f"    {ev:14s} n={len(vals)}  sum={round(sum(vals), 1)}ms  max={round(max(vals), 1)}ms")
llm_sum = sum(durs.get("llm_usage", []))
tool_sum = sum(durs.get("tool_call", []))
if llm_sum and tool_sum:
    share = tool_sum / (llm_sum + tool_sum) * 100
    print(f"    → 工具耗时占比 {share:.1f}%（LLM {round(llm_sum, 1)}ms vs 工具 {round(tool_sum, 1)}ms）")

if failures:
    print(f"\nFAIL（{len(failures)} 项）：")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nPASS：llm_usage 真实 tokens 入账/工具耗时落日志/降级口径兜底/stream_options 显式请求/归因聚合全过")
