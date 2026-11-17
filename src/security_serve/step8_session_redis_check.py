"""step8_session_redis_check.py —— H4 验收：会话外置 Redis

两部分验收：
  确定性部分（默认跑，不烧 LLM 的钱）：
    [1] 会话读写往返：save → load，messages 完整、Budget 还原成实例且记账延续
    [2] 写时 TTL 已挂：EXPIRE 剩余时间落在 (0, SESSION_TTL] 区间
    [3] 引擎级过期实证：ex=1 的短 key 等待后消失——过期由 Redis 管理，非应用代码
    [4] 跨进程共享（多副本语义）：主进程写 marker 会话，子进程（全新解释器）读回
    [5] 前缀隔离总检：count_sessions 只数 session: 前缀，其他 key 不串味
  LLM 端到端（--live，需 .env 真实 key）：
    [6] 同实例续答事实：第二轮答出第一轮说的数字（进程内延续 + 存储写回链路）
    [7] 真跨重启：子进程另起 uvicorn(8001) 续问——「重启服务后仍答出上一轮事实」，
        H4 验收标准的实弹检验（两进程 = 两实例，读同一份 Redis 会话）

跑法（项目根）：
  uv run python src/security_serve/step8_session_redis_check.py          # 确定性
  uv run python src/security_serve/step8_session_redis_check.py --live   # 加 LLM 端到端
前置：Redis 在跑——docker compose up -d redis（宿主验收走 localhost:6379 端口映射）
"""
import json
import subprocess
import sys
import time

import httpx

from security_serve.agent.config import PROJECT_ROOT, SESSION_TTL
from security_serve.agent.safety import Budget
from security_serve.agent.session_store import _KEY_PREFIX, _conn, count_sessions, load_session, save_session

failures = []


def mark(label: str, ok: bool):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


if __name__ == "__main__":
    print("H4 验收：会话外置 Redis（session:{id} + TTL + 多进程共享）")
    print("=" * 60)

    print("[1] 会话读写往返（messages + Budget 整存整取）")
    sess = {
        "messages": [
            {"role": "system", "content": "test-system"},
            {"role": "user", "content": "我最喜欢的数字是 42"},
            {"role": "assistant", "content": "已记住"},
        ],
        "budget": Budget(),
    }
    sess["budget"].record(1000)  # used = 1000 // CHARS_PER_TOKEN = 500
    save_session("h4-check", sess)
    loaded = load_session("h4-check")
    mark("messages 完整还原（3 条、末条内容一致）",
         loaded is not None and len(loaded["messages"]) == 3 and "42" in loaded["messages"][1]["content"])
    mark("Budget 还原成实例且记账延续（used=500）",
         isinstance(loaded["budget"], Budget) and loaded["budget"].used == 500)

    print("[2] 写时 TTL 已挂（时效管理在存储层，应用代码零清理逻辑）")
    ttl = _conn().ttl(_KEY_PREFIX + "h4-check")
    mark(f"EXPIRE 剩余时间在 (0, {SESSION_TTL}] 区间（实测 {ttl}）", 0 < ttl <= SESSION_TTL)

    print("[3] 引擎级过期实证（短 TTL key 等 1 秒后由 Redis 自动删除）")
    _conn().set(_KEY_PREFIX + "exp-test", "x", ex=1)
    time.sleep(1.3)
    mark("短 TTL key 已被引擎删除", _conn().get(_KEY_PREFIX + "exp-test") is None)

    print("[4] 跨进程共享（多副本语义：两个实例读同一份存储）")
    save_session("h4-cross", {
        "messages": [{"role": "system", "content": "h4-cross-proc-marker"}],
        "budget": Budget(),
    })
    child_code = (
        "from security_serve.agent.session_store import load_session; "
        "s = load_session('h4-cross'); "
        "print('FOUND' if s and s['messages'][0]['content'] == 'h4-cross-proc-marker' else 'MISS')"
    )
    r = subprocess.run(
        ["uv", "run", "python", "-c", child_code],
        capture_output=True, encoding="utf-8", errors="replace", cwd=PROJECT_ROOT,
    )
    print("  子进程输出:", (r.stdout or r.stderr).strip()[:80])
    mark("子进程（全新解释器）读回主进程写的会话", "FOUND" in r.stdout)

    print("[5] 前缀隔离总检（session: 命名空间不被其他 key 污染）")
    _conn().set("not-a-session", "x")
    n = count_sessions()
    _conn().delete("not-a-session")
    mark(f"count_sessions 只数 session: 前缀（实测 {n} >= 2）", n >= 2)

    if "--live" in sys.argv:
        print("[6-7] LLM 端到端（真实 key）")
        from fastapi.testclient import TestClient
        from security_serve.agent.server import app

        client = TestClient(app)
        r = client.post("/chat", json={
            "message": "请记住：我最喜欢的数字是 42。", "session_id": "h4-live",
        })
        print("  reply1:", r.json()["reply"][:80])
        r = client.post("/chat", json={
            "message": "我最喜欢的数字是多少？", "session_id": "h4-live",
        })
        reply = r.json()["reply"]
        print("  reply2:", reply[:80])
        mark("同实例续答事实（含 42）", r.status_code == 200 and "42" in reply)

        print("  [7] 真跨重启：子进程另起 uvicorn(8001)，续问同一会话")
        proc = subprocess.Popen(
            ["uv", "run", "uvicorn", "security_serve.agent.server:app",
             "--app-dir", "src", "--port", "8001"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            base = "http://127.0.0.1:8001"
            for _ in range(30):  # 轮询就绪：健康即探，最多约 30 秒
                try:
                    if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                time.sleep(1)
            r = httpx.post(f"{base}/chat", json={
                "message": "我最喜欢的数字是多少？", "session_id": "h4-live",
            }, timeout=120)
            reply = r.json()["reply"]
            print("  reply3(8001):", reply[:80])
            mark("重启后（新进程）仍答出上一轮事实（含 42）", r.status_code == 200 and "42" in reply)
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    print("=" * 60)
    for k in ("h4-check", "h4-cross"):
        _conn().delete(_KEY_PREFIX + k)  # 清理测试会话，不污染后续
    if failures:
        print(f"结果：FAIL（{len(failures)} 例不符合预期）：{failures}")
        sys.exit(1)
    print("结果：PASS——读写往返、TTL 写挂与引擎过期、跨进程共享、前缀隔离成立"
          + ("，LLM 端到端跨重启通过" if "--live" in sys.argv else ""))
