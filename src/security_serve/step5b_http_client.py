"""step5b_http_client.py —— G5 补充演示：httpx 真网络客户端（与 step5 的 TestClient 对照）

step5 用 TestClient 在进程内直呼 app（不开端口、不发网络包）；本脚本走**真实 HTTP**：
先用 uvicorn 把服务起起来（软件服务器监听端口），再用 httpx 作为客户端从网络打请求进来。
服务端代码（server.py）和客户端代码（本文件）是两个独立进程、两份独立源码，
只靠 HTTP 协议交流——这就是"如何调用服务"的代码形态。
httpx 是客户端库：一行 post() 背后是构造报文 → 建 TCP 连接 → 发送 → 阻塞等响应 → 解析回 r 对象。

跑法（两个终端，端口须一致）：
  终端1 起服务（断开 stdin 模拟真实部署；交互启动则 stdin 活着，遇危险工具会在服务终端阻塞等人）：
    PowerShell:  $null | uv run uvicorn security_serve.agent.server:app --app-dir src --port 8000
    bash:        uv run uvicorn security_serve.agent.server:app --app-dir src --port 8000 < /dev/null
  终端2 跑本脚本：
    uv run python src/security_serve/step5b_http_client.py

[1] 前置探测 GET /health——连不上说明服务没起，打印启动命令后退出（等价浏览器地址栏敲 URL）
[2] 单轮问答 POST /chat——json= 参数自动序列化 body 并设 Content-Type 头；timeout 必须给足，
    httpx 默认 5 秒，LLM 调用撑不到
[3] 会话连续性——同一 session_id 的两个**独立 HTTP 请求**，第二轮答案依赖第一轮：
    上下文存在服务端会话表里，不在请求里（真网络形态下看得最清楚）
[4] 错误路径——空消息直接 400，客户端读 r.status_code 和 r.json()["detail"]，不烧 LLM 的钱

注意：若服务以交互终端启动且模型触发危险工具，[需确认] 提示会打到**服务终端**——
真网络形态下 ask_user 发生在服务进程，客户端这边只看到请求迟迟不回。
"""
import sys

import httpx

BASE = "http://127.0.0.1:8000"

failures = []


def mark(label: str, ok: bool):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


if __name__ == "__main__":
    print("G5 补充演示：httpx 真网络客户端（服务必须已在另一终端启动）")
    print("=" * 60)

    print("[1] 前置探测 GET /health")
    try:
        r = httpx.get(f"{BASE}/health", timeout=5)
    except httpx.ConnectError:
        print("  连接失败：服务未启动。先在另一个终端跑（PowerShell）：")
        print("    $null | uv run uvicorn security_serve.agent.server:app --app-dir src --port 8000")
        sys.exit(2) # 退出码，0=成功、非0=失败、1=断言失败（服务行为不符预期）、2=环境错误（服务没起）
    print(f"  响应: {r.status_code} {r.json()}")
    mark("HTTP 200 且 status=ok", r.status_code == 200 and r.json()["status"] == "ok")

    print("[2] 单轮问答 POST /chat（确定性数学题）")
    r = httpx.post(f"{BASE}/chat", json={"message": "(3+4)*5 等于多少？", "session_id": "step5b"}, timeout=120)
    # 把 dict 序列化成 JSON 文本塞进请求体 + 自动设 Content-Type: application/json 头
    body = r.json()
    print(f"  响应: {r.status_code}")
    print(f"  reply: {body['reply'][:120]}")
    print(f"  used_tokens: {body['used_tokens']}")
    mark("HTTP 200 且 reply 非空", r.status_code == 200 and body["reply"].strip() != "")
    mark("预算记账贯通（used_tokens > 0）", body["used_tokens"] > 0)

    print("[3] 会话连续性（同一 session_id 两轮独立 HTTP 请求，第二轮依赖第一轮）")
    r1 = httpx.post(f"{BASE}/chat", json={"message": "记住：我最喜欢的数字是 42。回复'好的'即可", "session_id": "step5b-连续"}, timeout=120)
    r2 = httpx.post(f"{BASE}/chat", json={"message": "我最喜欢的数字是多少？", "session_id": "step5b-连续"}, timeout=120)
    print(f"  第一轮 reply: {r1.json()['reply'][:100]}")
    print(f"  第二轮 reply: {r2.json()['reply'][:150]}")
    mark("两轮均 HTTP 200", r1.status_code == 200 and r2.status_code == 200)
    mark("第二轮回答含 42（上下文在服务端延续）", "42" in r2.json()["reply"])

    print("[4] 错误路径：空消息 POST /chat → 400（框架直接拒绝，不烧 LLM 的钱）")
    r = httpx.post(f"{BASE}/chat", json={"message": "   "}, timeout=5)
    print(f"  响应: {r.status_code} {r.json()}")
    mark("HTTP 400 且 detail 提示", r.status_code == 400 and "message" in r.json()["detail"])

    print("=" * 60)
    if failures:
        print(f"结果：FAIL（{len(failures)} 例不符合预期）：{failures}")
        sys.exit(1)
    print("结果：PASS——真网络面 health 探活、/chat 应答、会话延续、400 错误路径全部符合预期")
