"""step9b —— H5-3 验收：计时中间件 + 延迟分位数聚合。

默认零 API / 零 Redis（patch 三个依赖），TestClient 进程内发请求（中间件栈
全程真跑，计的是真实流逝时长），断言四件事：
[1] 正常请求每个都有一行 event=http_request：duration_ms>0、字段齐、带 request_id
[2] 业务抛异常也有计时行且 status=500——故障请求不留统计盲区
[3] 400（空消息）与 404（未注册路由）也计时——中间件在最外圈，所有 HTTP 响应都有行
[4] percentile 聚合与手算一致；真实日志聚合出 p50/p95/max

--live：连真 Redis / 真 LLM 跑端到端（需容器 redis 与 API 可用；fakeredis 起
本地内存 redis 则零容器依赖），发 3 条真请求后聚合。

为什么看 p95 不看平均值：长尾被平均掩盖——99 个 10ms + 1 个 10s，平均
110ms 显得健康，但 1% 的用户在等 10 秒；agent 服务 LLM 调用天然长尾。

跑法（项目根）：
  默认：   uv run python src/security_serve/step9b_latency.py
  --live： uv run --with fakeredis python src/security_serve/step9b_latency.py --live
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="step9b_"))
import security_serve.agent.config as config
config.LOG_PATH = tmp / "server.jsonl"  # 先 patch 再导入 server（模块级 setup 读它）

import security_serve.agent.server as srv


def percentile(sorted_vals: list, p: float) -> float:
    """p 分位数（最近秩法）：升序序列里 p 比例的值不慢于返回值。

    idx = ceil(p*n)-1 再 clamp 进 [0, n-1]——无插值、无第三方依赖，与
    Prometheus histogram bucket 的语义近似（bucket 上界即"不慢于"）；
    numpy 的 linear 插值法结果略有差异但同量级，学习版不做插值。
    """
    if not sorted_vals:
        raise ValueError("空序列算不了分位数")
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def load_records() -> list:
    """读（patch 后的）日志文件，逐行 json.loads——聚合的数据源。"""
    lines = config.LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


failures = []

# ---------- 默认模式：TestClient + patch ----------
if "--live" not in sys.argv:
    def fake_turn(messages, budget, echo=True):
        """按消息长度 sleep 出 5 档延迟（5-25ms），聚合才有形状。"""
        n = len(messages[-1]["content"]) % 5
        time.sleep(0.005 * (n + 1))
        return "mocked"

    srv.load_session = lambda sid: None
    srv.save_session = lambda sid, sess: None
    srv.run_agent_turn = fake_turn

    from fastapi.testclient import TestClient
    # raise_server_exceptions=False：业务异常转成 500 响应（贴近真实 uvicorn 行为），
    # 否则异常直接 re-raise 给测试进程，[2] 的断言没法写
    client = TestClient(srv.app, raise_server_exceptions=False)

    # [1] 正常请求 ×20
    for i in range(20):
        r = client.post("/chat", json={"message": "x" * (i + 1), "session_id": f"s{i}"})
        if r.status_code != 200:
            failures.append(f"[1] 请求 {i} 失败: {r.status_code}")

    http_rows = [r for r in load_records() if r.get("event") == "http_request"]
    if len(http_rows) != 20:
        failures.append(f"[1] http_request 行数 {len(http_rows)} != 20")
    bad = [r for r in http_rows if not (
        r.get("duration_ms", 0) > 0 and r.get("path") == "/chat"
        and r.get("method") == "POST" and r.get("status") == 200)]
    if bad:
        failures.append(f"[1] 计时行字段缺失/错误: {bad[:2]}")
    no_rid = [r for r in http_rows if "request_id" not in r]
    if no_rid:
        # H5-2/H5-3 的交点：洋葱顺序——计时中间件必须注册在 request_id 之后（内层）
        failures.append(f"[1] {len(no_rid)} 行缺 request_id（洋葱顺序错了）")

    # [2] 异常路径：业务炸了也有计时行
    def boom(messages, budget, echo=True):
        raise RuntimeError("业务爆炸")

    srv.run_agent_turn = boom
    r = client.post("/chat", json={"message": "boom", "session_id": "sx"})
    if r.status_code != 500:
        failures.append(f"[2] 业务异常未转 500: {r.status_code}")
    err_rows = [r for r in load_records()
                if r.get("event") == "http_request" and r.get("status") == 500]
    if len(err_rows) != 1 or err_rows[0].get("duration_ms", 0) <= 0:
        failures.append(f"[2] 异常路径缺计时行或 duration 异常: {err_rows}")
    srv.run_agent_turn = fake_turn

    # [3] 400 / 404 也计时
    r400 = client.post("/chat", json={"message": "   ", "session_id": "s"})
    r404 = client.get("/no-such-path")
    other = {(r.get("path"), r.get("status")) for r in load_records()
             if r.get("event") == "http_request"}
    if (("/chat", 400) not in other or ("/no-such-path", 404) not in other):
        failures.append(f"[3] 400/404 缺计时行: {other}")

    # [4] 聚合：手算对照 + 真实日志
    vals = list(range(1, 101))  # 已知序列 1..100
    checks = [(0.95, 95), (0.50, 50), (0.0, 1), (1.0, 100)]
    for p, expect in checks:
        got = percentile(sorted(vals), p)
        if got != expect:
            failures.append(f"[4] percentile(p={p}) 手算对照失败: expect {expect} got {got}")
    durations = sorted(r["duration_ms"] for r in http_rows if r["status"] == 200)
    if len(durations) < 5:
        failures.append(f"[4] 可聚合样本不足: {len(durations)}")
    print(f"[聚合] n={len(durations)}  p50={percentile(durations, 0.5)}ms  "
          f"p95={percentile(durations, 0.95)}ms  max={durations[-1]}ms")

else:
    # ---------- --live 模式：真 Redis / 真 LLM 端到端 ----------
    from fakeredis import TcpFakeServer

    server = TcpFakeServer(("127.0.0.1", 6392))
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()

    env = {**os.environ, "REDIS_URL": "redis://127.0.0.1:6392/0", "PYTHONUTF8": "1"}
    root = Path(__file__).resolve().parents[2]  # src/security_serve/step9b → 项目根
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "security_serve.agent.server:app",
         "--app-dir", str(root / "src"), "--port", "8003"],
        env=env, stdin=subprocess.DEVNULL, cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    import httpx
    live_log = root / "src/security_serve/agent/data/server.jsonl"
    for _ in range(60):
        try:
            if httpx.get("http://127.0.0.1:8003/health", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        sys.exit("FAIL: --live 服务 30 秒未就绪")

    for i, rid in enumerate(["live-p01", "live-p02", "live-p03"]):
        r = httpx.post("http://127.0.0.1:8003/chat",
                       json={"message": f"{i + 1}+{i + 2} 等于几", "session_id": "live-p"},
                       headers={"X-Request-ID": rid}, timeout=120)
        if r.status_code != 200:
            failures.append(f"[live] 请求 {rid} 失败: {r.status_code}")

    proc.terminate()
    rows = [json.loads(l) for l in
            live_log.read_text(encoding="utf-8").strip().splitlines()]
    durs = sorted(r["duration_ms"] for r in rows if r.get("event") == "http_request"
                  and r.get("path") == "/chat")
    if len(durs) < 3:
        failures.append(f"[live] http_request 行不足 3: {len(durs)}")
    no_rid = [r for r in rows if r.get("event") == "http_request" and "request_id" not in r]
    if no_rid:
        failures.append(f"[live] {len(no_rid)} 行缺 request_id")
    if durs:
        print(f"[live 聚合] n={len(durs)}  p50={percentile(durs, 0.5)}ms  "
              f"p95={percentile(durs, 0.95)}ms  max={durs[-1]}ms（真 LLM 延迟）")

# ---------- 汇总 ----------
if failures:
    print(f"\nFAIL（{len(failures)} 项）：")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nPASS：计时行全覆盖（正常/异常/400/404）、字段完整带 request_id、聚合与手算一致")
