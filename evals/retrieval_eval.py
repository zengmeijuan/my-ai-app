"""retrieval_eval.py - C5 检索组件级评测执行器

与 eval.py（端到端）构成两级证据链的组件层：直接评 rag.retrieve 本身，
不经 LLM、不经 agent 图。检索是纯函数（同输入同输出，无模型方差），
组件级基线可精确复现——端到端挂了先看这里，分得清「检索没命中」和
「资料在但模型没用好」。

判定口径：expected_sources 中每个来源都需在 top-k 中至少出现一个 chunk
才记 HIT；同时记录 best_rank（期望来源的最好名次）与 best_score（最高
相似度），作为 C5 调优的前后对比基线。

运行：
  uv run python evals/retrieval_eval.py            # 全量（retrieval_cases）
  uv run python evals/retrieval_eval.py --id R02   # 单条调试
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from my_agent import rag  # uv editable install，包路径免配置


def run_case(case: dict, index: dict | None) -> dict:
    """跑单条：retrieve → 判定来源覆盖 → 采集名次/分数/延迟"""
    t0 = time.perf_counter()
    results = rag.retrieve(case["query"], index)
    latency = round(time.perf_counter() - t0, 2)
    """
        results = [
            {"source": "退货政策.md", "score": 0.5691, ...}, # 排第 1
            {"source": "配送说明.md", "score": 0.2808, ...}, # 排第 2
            {"source": "会员权益.md", "score": 0.2332, ...}, # 排第 3
        ]
        got_rank = {"退货政策.md": 1, "配送说明.md": 2, "会员权益.md": 3}
    """
    got_rank: dict[str, int] = {}
    for i, r in enumerate(results):
        got_rank.setdefault(r["source"], i + 1)  # 来源 → 首次出现名次（setdefault 保先，同源多个 chunk 时取最靠前的）
    missing = [s for s in case["expected_sources"] if s not in got_rank] # 期望来源在不在 top-3
    hit = not missing
    best_rank = min(got_rank[s] for s in case["expected_sources"]) if hit else None # 期望来源排第几
    best_score = max(
        (r["score"] for r in results if r["source"] in case["expected_sources"]), # 期望来源最高相关度
        default=None,
    )
    return {
        "id": case["id"],
        "query": case["query"],
        "expected_sources": case["expected_sources"],
        "top_k": [{"source": r["source"], "score": round(r["score"], 4)} for r in results],
        "hit": hit,
        "missing_sources": missing,
        "best_rank": best_rank,
        "best_score": round(best_score, 4) if best_score is not None else None,
        "latency_s": latency,
    }


def main():
    parser = argparse.ArgumentParser(description="检索组件级评测（C5 前置）")
    parser.add_argument("--cases", default=str(Path(__file__).parent / "testcases.json"))
    parser.add_argument("--id", dest="case_id", help="只跑指定用例（调试）")
    args = parser.parse_args()

    data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = data["retrieval_cases"]
    if args.case_id:
        cases = [c for c in cases if c["id"] == args.case_id]
    if not cases:
        sys.exit("没有匹配的用例")

    # 后端分流：pgvector 数据在 PG（sync-pg 灌库），无需文件版索引；
    # json 才读 rag_index.json 并校验构建模型一致
    backend = rag.RAG_BACKEND
    index = None
    if backend == "json":
        if not rag.INDEX_PATH.exists():
            sys.exit(f"索引不存在，先运行: uv run python -m my_agent.rag build")
        index = json.loads(rag.INDEX_PATH.read_text(encoding="utf-8"))
        if index["model"] != rag.EMBED_MODEL:
            sys.exit(f"索引由 {index['model']} 构建，与当前 {rag.EMBED_MODEL} 不一致，请重新 build")

    print(f"被测: rag.retrieve | backend: {backend} | embed: {rag.EMBED_MODEL} | top_k: {rag.TOP_K} | 用例: {len(cases)} 条")
    records = [run_case(c, index) for c in cases]

    hits = [r for r in records if r["hit"]]
    rank_dist = {rank: sum(1 for r in hits if r["best_rank"] == rank) for rank in (1, 2, 3)}
    avg_rank = round(sum(r["best_rank"] for r in hits) / len(hits), 2) if hits else None
    summary = {
        "total": len(records),
        "hit": len(hits),
        "hit_rate": round(len(hits) / len(records) * 100, 1),
        "rank_dist": rank_dist,
        "avg_best_rank": avg_rank,
        "latency_avg_s": round(sum(r["latency_s"] for r in records) / len(records), 2),
        "backend": backend,
        "testcases_version": data["version"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "records": records,
    }

    print(f"\n{'=' * 62}")
    print(f"检索组件级评测汇总  testcases {data['version']}")
    print(f"命中率(hit@{rag.TOP_K}): {summary['hit']}/{summary['total']} = {summary['hit_rate']}%"
          f" | 命中名次: rank1={rank_dist[1]} rank2={rank_dist[2]} rank3={rank_dist[3]}"
          f" | 平均名次: {avg_rank} | 延迟 avg {summary['latency_avg_s']}s")
    for r in records:
        if r["hit"]:
            print(f"[{r['id']}] HIT  | rank={r['best_rank']} | score={r['best_score']}")
        else:
            top_desc = " ".join(f"{i['source']}({i['score']})" for i in r["top_k"])
            print(f"[{r['id']}] MISS | 缺 {r['missing_sources']} | top{rag.TOP_K} 实际: {top_desc}")
    print("=" * 62)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入: {out_file}")


if __name__ == "__main__":
    main()
