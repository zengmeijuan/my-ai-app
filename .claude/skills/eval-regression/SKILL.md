---
name: eval-regression
description: 客服 agent 的评测回归流程。当用户要求"跑评测 / 回归测试 / 评测对比 / 看正确率 / 分析失败用例"，或改动了 src/my_agent 与 src/langGraph_agent 下的 agent 逻辑需要验证没有回归时使用。执行 evals/eval.py、对比历史结果、解读失败归因。
---

# 评测回归流程

## 本体与分工

评测执行器是 `evals/eval.py`（已有代码，不要重写）。本 skill 只负责：何时跑、跑哪个命令、结果怎么解读、怎么对比历史。

- 被测对象：LangGraph 客服图（`--graph step1` 默认全功能图；`step5` 检索子图版，用于对比实验）
- 评测集：`evals/testcases.json`（固定基准，版本化，25 条）
- 结果落盘：`evals/results/run_<时间戳>.json`（每次运行一个新文件，历史天然可对比）

## 命令

```bash
# 在项目根执行
uv run python evals/eval.py                          # 全量回归（默认 step1 图）
uv run python evals/eval.py --graph step5            # 检索子图版（对比实验）
uv run python evals/eval.py --category retrieval     # 只跑一类（retrieval/tool/combo/refusal/chitchat）
uv run python evals/eval.py --id T01                 # 单条调试
uv run python evals/eval.py --reclassify evals/results/run_xxx.json   # 历史 run 离线重分类
```

## 执行步骤

1. 前置检查：确认项目根 `.env` 存在且含 `API_KEY`（缺则停止并告知用户，不要猜路径）。
2. 按用户意图选命令：改了检索逻辑 → 全量；只调某条用例 → `--id`；对比两个图 → 各跑一次。
3. 运行并等待完成（全量 25 条约几分钟，输出逐条 PASS/FAIL）。
4. 解读汇总（见下），再与 `evals/results/` 里最近一次 `run_*.json` 对比：
   正确率变化、新增/消失的 FAIL、失败分类迁移、延迟与 token 波动。
5. 向用户汇报：正确率、与基线的差异、FAIL 清单及归因；FAIL 多时先按归因类别聚合，
   给出最可能的根因，不要逐条复述。

## 结果解读

汇总行字段：正确率（pass/total）、按类别、延迟（avg/p50/p95）、token 与成本、
judge 调用次数、失败分类统计。

失败归因类别（只对 FAIL 标注）：

| 标签 | 含义 | 修复方向 |
|---|---|---|
| RETRIEVAL | 检索没命中，资料撑不起答案 | 调检索（分块/嵌入/阈值） |
| TOOL-MISS | 该调的工具没调 | 查工具 description 与 system prompt |
| TOOL-EXTRA | 禁调却调了 | 同上 |
| FABRICATION | 答案与检索资料矛盾或无中生有 | 收紧 prompt 对编造的限制 |
| OTHER | 资料够但没用好 / 答非所问 / 元回复泄漏 | 逐条看 record 定位 |

SKIP 不计入分母：用例 `expected_tools` 含被测图未注册的工具时跳过（配置性失败 ≠ agent 失败）。

## 判定机制（理解输出用）

- 答案维度：关键词全命中且无 `judge_required` 标注 → 直接 PASS；否则 LLM judge 兜底
  （宽松：允许措辞不同；严格：关键事实不得缺失、不得答反、不得编造）。
- 行为维度：`expected_tools` 是 must-call 语义；空数组 = 禁调任何工具。
- 温度固定 0，每条用例前重置记忆（写入重定向到 `.eval_memory.md`），保证可重复。
- judge 与归因是评测器自身的 LLM 开销，单列入账，不算进被测图。

## 注意事项

- 评测会真实调用 LLM，消耗 token；全量一次约几美分（汇总里有 `cost_usd`）。
- 结果 JSON 里 `retrieved_context` 是检索证据存档，分析 FABRICATION / RETRIEVAL 时先看它。
- Windows 控制台中文若乱码，用 `PYTHONIOENCODING=utf-8` 前缀重跑。
- 不要修改 `evals/testcases.json` 来让测试通过；发现用例本身有误，向用户指出并确认。
