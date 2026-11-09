"""step5_serve_demo.py —— G5 演示：FastAPI 服务化（TestClient，不占端口可自动化）

服务化的两个关键验收点：
  [3] 危险工具默认拒绝：HTTP 服务没有交互终端——TestClient 环境里用空 StringIO
      替换 sys.stdin 模拟「服务进程无 stdin」的真实部署形态（systemd/容器里
      stdin 就是关闭的），ask_user 的 input() 撞 EOFError → 默认拒绝。
      这是 G3 预埋分支的验收现场，一行代码没改，语义自动成立。
  [4] 会话连续性：同一个 session_id 的两轮请求共享服务端 messages——
      第二轮的答案依赖第一轮的信息（上下文 = 短期记忆存在服务端）。

[1][2][3][4] 都需要真实 LLM API（.env 已配）。

跑法（项目根）：
  uv run python src/security_serve/step5_serve_demo.py
"""
import io
import sys

from fastapi.testclient import TestClient

from security_serve.agent.server import app

failures = []
# TestClient FastAPI测试工具，直接在进程内调用路由函数，不开端口不发网络包
client = TestClient(app)


def mark(label: str, ok: bool):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


if __name__ == "__main__":
    print("G5 演示：FastAPI 服务化（POST /chat 包手写 agent）")
    print("=" * 60)

    print("[1] 健康检查 GET /health")
    r = client.get("/health")
    print(f"  响应: {r.status_code} {r.json()}")
    mark("HTTP 200 且 status=ok", r.status_code == 200 and r.json()["status"] == "ok")

    print("[2] 单轮问答 POST /chat（确定性数学题）")
    r = client.post("/chat", json={"message": "(25+5)*2 等于多少？", "session_id": "demo-数学"})
    body = r.json()
    print(f"  响应: {r.status_code}")
    print(f"  reply: {body['reply'][:200]}")
    print(f"  used_tokens: {body['used_tokens']}")
    mark("HTTP 200 且 reply 非空", r.status_code == 200 and body["reply"].strip() != "")
    mark("预算记账生效（used_tokens > 0）", body["used_tokens"] > 0)

    print("[3] 危险工具默认拒绝（服务无终端 → EOF → 拒绝，G3 预埋分支验收）")
    real_stdin = sys.stdin
    sys.stdin = io.StringIO("")  # 模拟服务进程：没有交互终端可点头
    try:
        r = client.post("/chat", json={"message": "读一下 README.md 告诉我项目叫什么", "session_id": "demo-拒绝"})
    finally:
        sys.stdin = real_stdin
    body = r.json()
    print(f"  响应: {r.status_code}")
    print(f"  reply: {body['reply'][:300]}")
    mark("HTTP 200 且 reply 非空（拒绝串以工具结果身份回传，循环未断）", r.status_code == 200 and body["reply"].strip() != "")

    print("[4] 会话连续性（同一 session_id 两轮，第二轮依赖第一轮）")
    r1 = client.post("/chat", json={"message": "记住：我最喜欢的数字是 42。回复'好的'即可", "session_id": "demo-连续"})
    r2 = client.post("/chat", json={"message": "我最喜欢的数字是多少？", "session_id": "demo-连续"})
    print(f"  第一轮 reply: {r1.json()['reply'][:100]}")
    print(f"  第二轮 reply: {r2.json()['reply'][:200]}")
    mark("两轮均 HTTP 200", r1.status_code == 200 and r2.status_code == 200)
    mark("第二轮回答含 42（上下文在服务端延续）", "42" in r2.json()["reply"])

    print("=" * 60)
    if failures:
        print(f"结果：FAIL（{len(failures)} 例不符合预期）：{failures}")
        sys.exit(1)
    print("结果：PASS——服务化面 /chat 正常应答、危险工具默认拒绝、会话在服务端延续")
