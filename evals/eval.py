"""eval.py —— 里程碑 C2：客服 agent 可重复评测执行器

被测对象：B2 的 LangGraph 图（--graph step1|step5，默认 step1）
评测集：evals/testcases.json（固定基准，版本化）
判定：关键词全命中且无 judge_required 标注 → 直接通过；未全命中/要点为空/带标注
      → LLM judge 兜底（judge_required 用例的 notes 作为 rubric 随 prompt 送达判定者）
      行为判定：expected_tools 为 must-call 语义（该调的必须调；空数组 = 禁调任何工具）
失败分类（C3，只对 FAIL 归因）：检索没命中 / 工具选错（漏调|误调）/ 编造引用 / 其他
      规则先行：error 与行为层失败纯规则零成本；答案层失败交 LLM 三选一
      证据：record.retrieved_context（每次 search_knowledge 返回内容截断存档）——采集决定归因上限
配置适配（C3 preflight）：expected_tools 含被测图未注册的工具 → 用例 SKIP 不计分母
      （配置性失败 ≠ agent 失败；step5 只注册 search_knowledge，tool 类用例会被 SKIP）
隔离：每条用例独立 invoke；remember 的写入被重定向到临时文件且每条前重置（防记忆污染）；
      温度固定为 0（step5 的 judge_llm 也用 patch 后的 llm 重建）

跑法（项目根）：
  uv run python evals/eval.py                          # 全量 25 条，step1 图
  uv run python evals/eval.py --graph step5            # 检索子图版（对比实验）
  uv run python evals/eval.py --category retrieval     # 只跑检索类
  uv run python evals/eval.py --id T01                 # 单条调试
  uv run python evals/eval.py --reclassify evals/results/run_xxx.json
                                                       # 历史 run 离线重分类（无检索证据时归因降级）
结果：evals/results/run_<时间戳>.json（重复运行天然可对比；方差观察 = 跑多次对比）
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
# sys.path 的作用是：这个目录下的直接子项（文件/子目录）可以当作顶层名字被导入
# sys.path默认来源：1.脚本所在目录 2.环境变量里的目录（这个项目uv打包module-name = "my_agent"，所以src目录天然在sys.path中）3.标准库目录（json、pathlib这些）4.site-packages第三方包
sys.path.insert(0, str(ROOT / "src")) #
sys.path.insert(0, str(ROOT / "src" / "langGraph_agent"))

load_dotenv(ROOT / ".env")

import my_agent.agent as agent_mod  # noqa: E402  需在 patch 前确保可导入

# ============== 可配置区 ==============

# 每百万 token 单价（美元，输入, 输出）。网关模型按名称前缀匹配，查不到则只报 token 数。
PRICE_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}
EVAL_MEMORY = Path(__file__).parent / ".eval_memory.md"  # 评测专用记忆文件（原 memory.md 零接触）


# ============== 环境装配：导入被测图并打隔离补丁 ==============

def load_graph(name: str):
    """惰性导入被测图；导入后统一打补丁（温度/记忆路径），返回 (可 invoke 的图, 注册工具名集合, 判定用 llm)"""
    if name == "step1":
        import step1_graph as m
    elif name == "step5":
        import step5_subgraph as m  # 其内部 from step1_graph import llm，同一对象
    else:
        raise ValueError(f"未知被测图: {name}")

    # 补丁 1：温度固定（llm 为 step1/step5 共享的同一对象，bind_tools/bound 链共享属性）
    m.llm.temperature = 0.0
    # 补丁 2：记忆重定向——写入路径（my_agent.agent 内 remember 用）+ 读取路径（step1 的 build_system_prompt 用）
    # 为什么既要修改agent和step1_graph下的MEMORY_PATH都要修改？因为from X import Y 是值拷贝，不是引用
    agent_mod.MEMORY_PATH = EVAL_MEMORY
    import step1_graph as s1
    s1.MEMORY_PATH = EVAL_MEMORY
    # 补丁 3：step5 的 judge_llm 在导入时已用默认温度构造，用 patch 后的 llm 重建
    if name == "step5":
        m.judge_llm = m.llm.with_structured_output(m.Verdict)
    return m.graph, {t.name for t in m.tools}, m.llm  # llm 同时是 judge/归因的判定 LLM，随被测图走，避免硬编码 step1


# ============== 判定 ==============

class JudgeVerdict(BaseModel):
    is_pass: bool = Field(description="回答是否合格")
    reason: str = Field(description="一句话判定理由")


def normalize(text: str) -> str:
    """字面命中的归一化：去空格 + 小写（中文『1-3 个工作日』vs『1-3个工作日』）"""
    return text.replace(" ", "").lower()


def judge_answer(llm, question: str, points: list[str], answer: str, rubric: str = "") -> tuple[bool, str, int, int]:
    """LLM 质检员：要点未命中或为空时的兜底判定。宽松面：允许措辞不同、信息更丰富；
    严格面：关键事实不得缺失、不得答反、不得编造。
    rubric：用例级特别判定口径（judge_required 用例传入 notes）——话术缺陷/元回复这类
    形态问题通用标准看不见，口径必须显式送达判定者（写在 notes 里但不传进 prompt，
    judge 只能按通用标准自觉发挥，R04 的推诿话术依旧漏网）。
    返回多带本次调用的 token 用量（评测器自身开销，单列入账）。
    token 直取 AIMessage.usage_metadata 而非 callback handler：graph.invoke 的
    config 残留在 contextvar，之后的 invoke 做 config 合并时 handler 被深拷贝，
    回调累加丢失（实测 run_case 路径恒为 0、单独调用却正常）；usage 附在返回值上
    不依赖 handler 传播，与 extract_tokens 同一路径，零框架黑盒假设。"""
    prompt = f"""你是客服回答的质检员。判断【客服回答】是否合格。
合格标准：正确回答了【用户问题】，且满足全部【期望要点】——允许措辞不同、内容更丰富，
但关键事实不得缺失、不得答反、不得编造知识库中不存在的政策。
用户问题：{question}
期望要点：{json.dumps(points, ensure_ascii=False) if points else "（无字面要点，按常识判断回答是否如实、不编造、切题）"}
{f"特别判定口径（必须严格执行，违反即不合格）：{rubric}" if rubric else ""}
客服回答：{answer}"""
    bound = llm.bind_tools([JudgeVerdict], tool_choice="JudgeVerdict")
    resp = bound.invoke(prompt)
    usage = getattr(resp, "usage_metadata", None) or {}
    verdict = JudgeVerdict(**resp.tool_calls[0]["args"])
    return verdict.is_pass, verdict.reason, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def grade(llm, case: dict, answer: str, called_tools: list[str]) -> dict:
    """综合判定：答案维度（关键词→judge 兜底）+ 行为维度（must-call）
    judge_required 标注（notes 内）强制走 judge：要点全命中也可能有话术缺陷
    （R04 教训——「如需归档请您那边补录」类推诿话术，关键词判定看不见）"""
    points = case.get("expected_points", [])
    norm_answer = normalize(answer)
    hit = [p for p in points if normalize(p) in norm_answer]
    missed = [p for p in points if p not in hit]

    notes = case.get("notes", "")
    judge_used, judge_reason = False, ""
    j_tin = j_tout = 0
    if points and not missed and "judge_required" not in notes:
        answer_ok = True                      # 关键词全命中且无强制标注，不花 judge 的钱
    else:
        rubric = notes if "judge_required" in notes else ""  # 口径随标注送达判定者
        answer_ok, judge_reason, j_tin, j_tout = judge_answer(llm, case["question"], points, answer, rubric)
        judge_used = True

    must = case.get("expected_tools", [])
    called = set(called_tools)
    behavior_ok = all(t in called for t in must) and (
        not (must == [] and called)           # 空数组 = 禁调任何工具
    )
    return {
        "answer_ok": answer_ok,
        "behavior_ok": behavior_ok,
        "points_hit": hit,
        "points_missed": missed,
        "judge_used": judge_used,
        "judge_reason": judge_reason,
        "judge_tokens_in": j_tin,
        "judge_tokens_out": j_tout,
    }


# ============== C3 失败分类四档 ==============

# 分类码 → 控制台 ASCII 标签（Windows GBK 控制台安全）；JSON 里存分类码本身
CLASS_LABELS = {
    "retrieval_miss": "RETRIEVAL",
    "tool_miss": "TOOL-MISS",
    "tool_extra": "TOOL-EXTRA",
    "tool_wrong": "TOOL-WRONG",   # tool_miss/tool_extra 在汇总里合并的桶名
    "fabrication": "FABRICATION",
    "other": "OTHER",
}


class FailureClass(BaseModel):
    category: Literal["retrieval_miss", "fabrication", "other"] = Field(
        description="失败归因：retrieval_miss=检索没命中（资料撑不起答案）；"
                    "fabrication=编造引用（答案与资料矛盾或无中生有）；other=其他（资料够但没用好/元回复泄漏等）")
    reason: str = Field(description="一句话归因理由，须指认具体证据")


def extract_retrieved(messages) -> str:
    """拼接 search_knowledge 的 ToolMessage 内容（每段截断），作为失败归因的检索证据"""
    chunks = []
    for msg in messages:
        if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "search_knowledge":
            chunks.append(str(msg.content)[:400])
    return "\n---\n".join(chunks)


def classify_failure(record: dict, llm) -> dict:
    """只对 FAIL 归因。规则先行：error→其他、行为层→工具选错（漏调/误调）零成本可判；
    答案层失败交 LLM 三选一（检索没命中/编造引用/其他），证据是检索返回原文"""
    out = {"failure_class": "", "failure_reason": ""}   # judge_tokens 不在此初始化——PASS 短路会
    # 用 0 覆盖掉 grade 刚写入的 judge token（run_case 里 record.update 本函数的返回值）
    if record["pass"]:
        return out
    if record["error"]:
        out.update(failure_class="other", failure_reason=f"执行异常: {record['error'][:80]}")
        return out
    if not record["behavior_ok"]:
        missing = [t for t in record.get("expected_tools", []) if t not in record["called_tools"]]
        if missing:
            out.update(failure_class="tool_miss", failure_reason=f"漏调工具 {missing}")
        else:
            out.update(failure_class="tool_extra", failure_reason=f"禁调却调了 {record['called_tools']}")
        return out
    prompt = f"""你是客服 agent 的失败归因分析员。下面这条回答被判不合格，请把失败归入一类：
- retrieval_miss：调用了知识库检索，但返回资料撑不起正确答案（没查到/查偏了）
- fabrication：答案与检索资料矛盾，或编造了资料中不存在的政策/数字
- other：资料其实足够但模型没用好、把过程汇报当答复、答非所问等
用户问题：{record["question"]}
缺失要点：{json.dumps(record.get("points_missed", []), ensure_ascii=False)}
质检员意见：{record.get("judge_reason") or "（未经 judge，关键词未命中）"}
检索返回资料：{record.get("retrieved_context") or "（无检索证据）"}
agent 回答：{record["answer"]}"""
    bound = llm.bind_tools([FailureClass], tool_choice="FailureClass")
    resp = bound.invoke(prompt)
    usage = getattr(resp, "usage_metadata", None) or {}
    args = resp.tool_calls[0]["args"]
    out.update(failure_class=args["category"], failure_reason=args["reason"],
               judge_tokens_in=usage.get("input_tokens", 0), judge_tokens_out=usage.get("output_tokens", 0))
    return out


def reclassify(path: str, llm):
    """离线重分类历史 run：规则层完整可用；缺 retrieved_context 时 LLM 归因降级（采集决定归因上限）。
    历史 record 没有 expected_tools 字段（C3 才加），从当前测试集按 id 回填——golden dataset 是判定语义的权威"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data["records"]
    tc_map = {c["id"]: c.get("expected_tools", [])
              for c in json.loads((Path(__file__).parent / "testcases.json").read_text(encoding="utf-8"))["cases"]}
    n_fail = sum(1 for r in records if not r["pass"])
    print(f"重分类: {path} | 被测图: {data.get('graph')} | FAIL {n_fail} 条")
    for r in records:
        if r["pass"]:
            continue
        r.setdefault("expected_tools", tc_map.get(r["id"], []))
        r.setdefault("retrieved_context", "")
        r.update(classify_failure(r, llm))
        print(f"[{r['id']}] {CLASS_LABELS.get(r['failure_class'], '?')} | {r['failure_reason'][:70]}")
    out = Path(path).with_name(Path(path).stem + "_reclassed.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {out}")


# ============== 单条执行 ==============

def extract_tool_calls(messages) -> list[str]:
    """从消息轨迹提取工具名序列（按调用顺序）"""
    names = []
    for msg in messages:
        # 对象属性的安全写法，消息列表里 HumanMessage/ToolMessage 根本没有 tool_calls 属性，直接访问会 AttributeError
        if getattr(msg, "tool_calls", None):
            names.extend(tc["name"] for tc in msg.tool_calls)
    return names


def extract_tokens(messages) -> tuple[int, int]:
    """累加轨迹中所有 AIMessage 的 usage_metadata；网关不返回时给 0"""
    tin = tout = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if usage:
            tin += usage.get("input_tokens", 0)
            tout += usage.get("output_tokens", 0)
    return tin, tout


def run_case(graph, case: dict, llm, graph_name: str = "") -> dict:
    """跑单条：隔离重置记忆 → invoke → 采集（答案/轨迹/token/延迟/检索证据）→ 判定 → 失败归因（C3）"""
    # .write_text(s) 等价于 open(path, "w", encoding="utf-8").write(s)
    EVAL_MEMORY.write_text("", encoding="utf-8")  # 每条从零记忆开始，保证可重复
    t0 = time.perf_counter()
    try:
        result = graph.invoke(
            {"messages": [("user", case["question"])]},
            config={
                "recursion_limit": 15,
                # LangSmith trace 标注：run_name 让 trace 列表一眼认出用例，metadata 供网页端按图/用例/类别筛选
                "run_name": f"eval:{graph_name}:{case['id']}",
                "metadata": {"graph": graph_name, "case_id": case["id"], "category": case["category"]},
            },
        )
        latency = time.perf_counter() - t0
        messages = result["messages"]
        answer = str(messages[-1].content)
        called_tools = extract_tool_calls(messages)
        tin, tout = extract_tokens(messages)
        record = {
            "id": case["id"], "category": case["category"],
            "question": case["question"], "answer": answer,
            "called_tools": called_tools, "expected_tools": case.get("expected_tools", []),
            "n_tool_calls": len(called_tools),
            "latency_s": round(latency, 2),
            "tokens_in": tin, "tokens_out": tout,
            "retrieved_context": extract_retrieved(messages), # 检索文本
            "error": "",
        }
        record.update(grade(llm, case, answer, called_tools))
    except Exception as e:
        record = {
            "id": case["id"], "category": case["category"],
            "question": case["question"], "answer": "",
            "called_tools": [], "expected_tools": case.get("expected_tools", []),
            "n_tool_calls": 0,
            "latency_s": round(time.perf_counter() - t0, 2),
            "tokens_in": 0, "tokens_out": 0,
            "retrieved_context": "",
            "error": f"{type(e).__name__}: {e}",
            "answer_ok": False, "behavior_ok": False,
            "points_hit": [], "points_missed": case.get("expected_points", []),
            "judge_used": False, "judge_reason": f"执行异常: {e}",
            "judge_tokens_in": 0, "judge_tokens_out": 0,
        }
    record["pass"] = record["answer_ok"] and record["behavior_ok"]
    record.update(classify_failure(record, llm))  # C3：只对 FAIL 归因，PASS 零成本
    return record


# ============== 汇总与报告 ==============

def percentile(sorted_vals: list[float], p: float) -> float:
    """无插值的经验分位数（迷你样本够用）"""
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, round(p / 100 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def cost_of(model: str, tin: int, tout: int) -> float | None:
    price = PRICE_PER_MTOK.get(model)
    return (tin * price[0] + tout * price[1]) / 1_000_000 if price else None


def aggregate(records: list[dict]) -> dict:
    """汇总表"""
    latencies = sorted(r["latency_s"] for r in records) # 所有延迟排序
    tin, tout = sum(r["tokens_in"] for r in records), sum(r["tokens_out"] for r in records) # 全量 token 求和
    by_cat = Counter(r["category"] for r in records) # 每类多少条
    pass_by_cat = Counter(r["category"] for r in records if r["pass"]) # 每类通过多少条
    cost = cost_of(os.environ.get("MODEL", ""), tin, tout) # 折算成本
    # 失败分类统计（C3 四档：tool_miss/tool_extra 合并进「工具选错」桶）
    fail_by_class = Counter(
        "tool_wrong" if r["failure_class"] in ("tool_miss", "tool_extra") else r["failure_class"]
        for r in records if r.get("failure_class"))
    return {
        "total": len(records),
        "pass": sum(1 for r in records if r["pass"]),
        "accuracy": round(sum(1 for r in records if r["pass"]) / len(records) * 100, 1),
        "by_category": {c: f"{pass_by_cat[c]}/{by_cat[c]}" for c in by_cat},
        "latency_avg_s": round(mean(latencies), 2),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "tokens_in": tin, "tokens_out": tout,
        "cost_usd": round(cost, 4) if cost is not None else None,
        "judge_calls": sum(1 for r in records if r["judge_used"]),
        "judge_tokens_in": sum(r.get("judge_tokens_in", 0) for r in records),
        "judge_tokens_out": sum(r.get("judge_tokens_out", 0) for r in records),
        "errors": sum(1 for r in records if r["error"]),
        "failures_by_class": dict(fail_by_class),
    }


def print_report(records: list[dict], summary: dict, graph_name: str):
    print(f"\n{'=' * 62}\n评测汇总  被测图: {graph_name}")
    print(f"用例: {summary['pass']}/{summary['total']}  正确率: {summary['accuracy']}%")
    print("按类别: " + "  ".join(f"{c} {v}" for c, v in summary["by_category"].items()))
    print(f"延迟: 平均 {summary['latency_avg_s']}s | p50 {summary['latency_p50_s']}s | p95 {summary['latency_p95_s']}s")
    cost = f" ≈${summary['cost_usd']}" if summary["cost_usd"] is not None else ""
    print(f"Token: in {summary['tokens_in']} / out {summary['tokens_out']}{cost} | judge 调用: {summary['judge_calls']} 次"
          + (f" | 执行异常: {summary['errors']}" if summary["errors"] else ""))
    if summary.get("judge_tokens_in") or summary.get("judge_tokens_out"):
        print(f"judge 侧 token: in {summary['judge_tokens_in']} / out {summary['judge_tokens_out']}"
              f"（评测器自身开销，单列不入图内账）")
    if summary["failures_by_class"]:
        print("失败分类: " + "  ".join(f"{CLASS_LABELS[c]}={n}" for c, n in summary["failures_by_class"].items()))
    if summary.get("skipped"):
        print("SKIP(被测图未注册所需工具，不计分母): " + "  ".join(f"{s['id']}缺{s['missing']}" for s in summary["skipped"]))
    print("=" * 62)
    for r in records:
        mark = "PASS" if r["pass"] else "FAIL"
        src = "kw" if (r["answer_ok"] and not r["judge_used"]) else "judge"
        line = f"[{r['id']}] {mark} | {src:<5} | tools={r['n_tool_calls']} | {r['latency_s']}s"
        if not r["pass"]:
            line += f" | {CLASS_LABELS.get(r['failure_class'], '?'):<11} | {r['failure_reason'][:55]}"
        print(line)


# ============== 主流程 ==============

def main():
    parser = argparse.ArgumentParser(description="客服 agent 评测执行器")
    parser.add_argument("--graph", choices=["step1", "step5"], default="step1")
    parser.add_argument("--cases", default=str(Path(__file__).parent / "testcases.json"))
    parser.add_argument("--category", help="只跑指定类别（retrieval/tool/combo/refusal/chitchat）")
    parser.add_argument("--id", dest="case_id", help="只跑指定用例（调试）")
    parser.add_argument("--reclassify", help="对历史 run 的结果 JSON 离线重分类（C3）")
    args = parser.parse_args()

    data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.case_id:
        cases = [c for c in cases if c["id"] == args.case_id]
    if not cases:
        sys.exit("没有匹配的用例")

    graph, registered, llm = load_graph(args.graph) # registered 注册工具名集合，llm 兼任 judge/归因的判定 LLM

    if args.reclassify:
        reclassify(args.reclassify, llm)
        return

    # preflight 配置适配检查：expected_tools 含被测图未注册的工具 → SKIP，不进分母（配置性失败 ≠ agent 失败）
    runnable, skipped = [], []
    for case in cases:
        missing = [t for t in case.get("expected_tools", []) if t not in registered]
        if missing:
            skipped.append({"id": case["id"], "missing": missing})
            print(f"[{case['id']}] SKIP | 被测图未注册工具 {missing}")
        else:
            runnable.append(case)
    if not runnable:
        sys.exit("所有匹配用例都因配置缺失被 SKIP")

    print(f"被测图: {args.graph} | 用例: {len(runnable)} 条 (SKIP {len(skipped)}) | 注册工具: {sorted(registered)} | 温度: {llm.temperature}")
    records = []
    for i, case in enumerate(runnable, 1):
        print(f"\n--- ({i}/{len(cases)}) {case['id']} {case['question'][:40]}")
        record = run_case(graph, case, llm, args.graph)
        records.append(record)
        mark = "PASS" if record["pass"] else "FAIL"
        print(f"[{record['id']}] {mark} | tools={record['n_tool_calls']} | {record['latency_s']}s"
              + (f" | {record['error'][:60]}" if record["error"] else ""))

    summary = aggregate(records)
    summary.update({"graph": args.graph, "testcases_version": data["version"],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "skipped": skipped, "records": records})
    print_report(records, summary, args.graph)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入: {out_file}")

    EVAL_MEMORY.unlink(missing_ok=True)  # 清理评测临时记忆


if __name__ == "__main__":
    main()
