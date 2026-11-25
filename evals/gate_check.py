"""gate_check.py - H7 CI 门禁判定器：读最新评测结果 JSON，不达标退出码 1 拦截合并

CI 判定不靠 eval.py 的退出码（正常跑完恒 0，只有无用例/全 SKIP 才非 0），靠
结果 JSON 里的数字——评测数字是门禁的唯一事实源。两种评测、两种前缀、两种阈值：
  --metric pass 对端到端（eval.py → results/run_*.json）：25 条真 LLM 有随机方差，
    阈值 24（容忍 1 条抖动）——硬卡 25/25 会随机误杀健康提交；
  --metric hit 对组件级（retrieval_eval.py → results/retrieval_*.json）：检索是
    纯函数、确定性可复现，阈值 17 硬卡满分——这里不达标必然是代码坏了。

跑法：
  uv run python evals/gate_check.py --metric pass --min 24   # 端到端门禁
  uv run python evals/gate_check.py --metric hit  --min 17   # 组件级门禁
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
PREFIX = {"pass": "run_", "hit": "retrieval_"}  # 指标 → 结果文件前缀（两种评测器各归各）


def main():
    parser = argparse.ArgumentParser(description="CI 门禁判定：最新评测结果 ≥ 阈值才放行")
    parser.add_argument("--metric", choices=["pass", "hit"], required=True)
    parser.add_argument("--min", dest="min_required", type=int, required=True)
    args = parser.parse_args()

    files = sorted(RESULTS_DIR.glob(f"{PREFIX[args.metric]}*.json"))
    if not files:
        sys.exit("门禁失败：找不到评测结果文件（评测器没跑？）")
    latest = max(files, key=lambda p: p.stat().st_mtime)  # mtime 最新 = 最近一次跑
    data = json.loads(latest.read_text(encoding="utf-8"))
    score, total = data[args.metric], data["total"]
    verdict = "放行" if score >= args.min_required else "拦截"
    print(f"门禁判定: {args.metric} = {score}/{total} | 阈值 ≥{args.min_required} | {verdict} | 来源 {latest.name}")
    if score < args.min_required:
        sys.exit(1)


if __name__ == "__main__":
    main()
