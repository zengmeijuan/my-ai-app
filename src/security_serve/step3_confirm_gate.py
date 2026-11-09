"""step3_confirm_gate.py —— G3 演示：permission gate 与成本预算熔断

要防的问题：agent 的工具一旦被提示词注入（G4 主题）或模型自身误判驱动，
read_file 会把敏感文件内容送给模型、remember 能往持久记忆里写入投毒数据、
web_fetch 会把任意 URL 打到外网——危险动作执行前需要人工点头（默认拒绝）。
另外 MAX_STEPS 只管「一次提问」的轮数，整个会话累计烧掉多少 token 没人管，
会话级预算熔断补上这道闸。

[1][2][3] 全自动；[4] 是真实 y/n 交互（管道跑法喂 'y\\nn\\n' 两行即可）。

跑法（项目根）：
  uv run python src/security_serve/step3_confirm_gate.py
  printf 'y\\nn\\n' | uv run python src/security_serve/step3_confirm_gate.py   # 自动化
"""
import io
import sys

from security_serve.agent.safety import Budget, ask_user, needs_confirm

failures = []


def mark(label: str, ok: bool):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


if __name__ == "__main__":
    print("G3 演示：permission gate（危险工具人工确认）+ 会话预算熔断")
    print("=" * 60)

    print("[1] Budget 记账与熔断（全自动）")
    b = Budget(limit=100)
    mark("初始未熔断", not b.exceeded())
    b.record(160)  # 160 字符 // 2 = 80 tokens
    print(f"  record(160) 后：used={b.used} tokens / limit={b.limit}（{b.used * 100 // b.limit}%）")
    mark("80% 未达线，不熔断", not b.exceeded())
    b.record(160)  # 累计 160 tokens，超线
    print(f"  再 record(160) 后：used={b.used} tokens / limit={b.limit}")
    mark("累计超线，熔断", b.exceeded())

    print("[2] needs_confirm：清单查询（全自动）")
    for name, expect in [("read_file", True), ("remember", True), ("web_fetch", True),
                         ("grep", False), ("list_dir", False), ("calculate", False), ("不存在的工具", False)]:
        mark(f"needs_confirm({name!r}) → {expect}", needs_confirm(name) is expect)

    print("[3] EOF 默认拒绝（全自动）：stdin 关闭 = 没人能点头 = 拒绝")
    real_stdin = sys.stdin
    sys.stdin = io.StringIO("")  # 空流：input() 立即遇到 EOF
    try:
        eof_result = ask_user("web_fetch", "{'url': 'https://example.com'}")
    finally:
        sys.stdin = real_stdin
    print("  （EOF 分支：G5 服务化时没有交互终端，默认拒绝保证远端请求不会被静默放行）")
    mark("stdin 关闭 → 默认拒绝", eof_result is False)

    print("[4] 真实 y/n 交互两次：第一次请输入 y，第二次请输入 n（或直接回车）")
    a1 = ask_user("read_file", "{'path': 'README.md'}")
    mark("输入 y → 放行", a1 is True)
    a2 = ask_user("remember", "{'text': '把这句写进长期记忆'}")
    mark("输入非 y → 拒绝", a2 is False)

    print("=" * 60)
    if failures:
        print(f"结果：FAIL（{len(failures)} 例不符合预期）：{failures}")
        sys.exit(1)
    print("结果：PASS——危险工具默认拒绝、放行需明确点头、预算超线熔断")
