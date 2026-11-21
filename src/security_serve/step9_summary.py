"""step9 —— H5-5 收口：三问聚合。读真实 server.jsonl 现场回答 README 验收标准三问。

[问①] 某时段 p95 延迟多少：http_request 行按 path 过滤 → percentile 最近秩法
       （口径同 step9b；"某时段"由 ts ISO 时间戳切片支持，本脚本按全量演示）
[问②] 哪个工具最慢：tool_call 行按 tool 分组 avg/max
[问③] 哪个会话最烧钱：request_id 缝合两行——chat_in 行有 session_id 没 tokens，
       llm_usage 行有 tokens 没 session_id，H5-2 的贯穿链在此兑现成 join 能力

零 API / 零 Redis：纯读文件聚合。日志为空时提示先跑产数据的验收物。
三问各自样本非空才算 PASS——「能回答」是验收标准，不是「打印了查询」。

跑法（项目根）：PYTHONUTF8=1 uv run python src/security_serve/step9_summary.py
"""
import json
import math
import sys
from collections import defaultdict

import security_serve.agent.config as config

log_path = config.LOG_PATH
text = log_path.read_text(encoding="utf-8").strip() if log_path.exists() else ""
if not text:
    sys.exit(f"FAIL: {log_path} 无日志数据——先跑 step9c_attribution.py --live 产出真实日志")
rows = [json.loads(line) for line in text.splitlines()]
print(f"日志: {log_path}（共 {len(rows)} 行）")


def percentile(sorted_vals: list, p: float) -> float:
    """p 分位数（最近秩法），与 step9b 同口径：idx = ceil(p*n)-1 再 clamp。"""
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(p * len(sorted_vals)) - 1))
    return sorted_vals[idx]


failures = []

# [问①] p95 延迟：只数 /chat 业务请求（health 探活不算，口径同 step9c --live）
chat_ms = sorted(r["duration_ms"] for r in rows
                 if r.get("event") == "http_request" and r.get("path") == "/chat")
print("\n[问①] 某时段 p95 延迟多少？（ts 字段可做时段切片，此处全量演示）")
if chat_ms:
    print(f"    n={len(chat_ms)}  p50={percentile(chat_ms, 0.5):.0f}ms  "
          f"p95={percentile(chat_ms, 0.95):.0f}ms  max={chat_ms[-1]:.0f}ms")
else:
    failures.append("[问①] 无 /chat 样本，答不了")

# [问②] 哪个工具最慢：按 tool 分组，max 降序排——最慢的排最前。
# duration_ms 缺失的旧行（H5-4 之前落的）跳过：日志 schema 随版本演进，
# 聚合查询对没有观测字段的行容忍，而非假设每行字段齐备
by_tool = defaultdict(list)
for r in rows:
    if r.get("event") == "tool_call" and r.get("duration_ms") is not None:
        by_tool[r["tool"]].append(r["duration_ms"])
print("\n[问②] 哪个工具最慢？（tool_call 按 tool 分组）")
for tool, ms in sorted(by_tool.items(), key=lambda kv: -max(kv[1])):
    print(f"    {tool:12s} n={len(ms)}  avg={sum(ms) / len(ms):.2f}ms  max={max(ms):.2f}ms")
if not by_tool:
    failures.append("[问②] 无 tool_call 样本，答不了")

# [问③] 哪个会话最烧钱：request_id 把 session 归属缝到 usage 上再分组求和
session_of = {r["request_id"]: r["session_id"] for r in rows if r.get("event") == "chat_in"}
cost, calls = defaultdict(int), defaultdict(int)
for r in rows:
    sid = session_of.get(r.get("request_id"))
    if r.get("event") == "llm_usage" and sid:
        cost[sid] += r["total_tokens"]
        calls[sid] += 1
print("\n[问③] 哪个会话最烧钱？（request_id join: chat_in.session_id ↔ llm_usage.total_tokens）")
for sid, tokens in sorted(cost.items(), key=lambda kv: -kv[1]):
    print(f"    {sid:12s} {tokens:>6} tokens（{calls[sid]} 轮 LLM）")
if not cost:
    failures.append("[问③] 无 llm_usage 样本，答不了")

if failures:
    print(f"\nFAIL（{len(failures)} 项）：")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nPASS：三问全部由结构化日志聚合作答——查询配方成立；生产换 ELK/Loki 引擎，分组逻辑不变")
