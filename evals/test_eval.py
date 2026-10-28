"""test_eval.py - 评测器自身的单测（C 收官自评产物）

评测器是所有评测数字的裁判，裁判说谎没人知道——got_rank 覆盖 bug 与
judge 口径断裂（rubric 写在 notes 里但从未进 prompt）都是靠人眼质疑发现
的，不是自测发现的。本文件把两处伤疤及关键判定语义锁成断言：任何改动
踩到这些语义，pytest 当场报警。全部 FakeLLM / monkeypatch，零网络成本。

pytest：发现所有test_开头的函数，执行里面的assert，断言为假就记一条失败
assert：python本身的语法关键字，右侧代码为true，则什么都不发生，为false，抛AssertionError,程序中断
断言：在此处判定某件事必然发生，若不成立，程序拒绝继续

跑法（项目根）：
  uv run pytest evals/ -q
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval as ev  # noqa: E402
import retrieval_eval  # noqa: E402
from my_agent import rag as rag_mod  # noqa: E402


# ============== FakeLLM：捕获 prompt 与调用次数，判定结果可配置 ==============

class FakeLLM:
    """模拟 llm.bind_tools([...], tool_choice=...).invoke(prompt) 接口，
    返回 AIMessage 形状对象（tool_calls + usage_metadata），与 eval.py
    judge_answer / classify_failure 的真实调用路径一致

    呼应：
    bound = llm.bind_tools([JudgeVerdict], tool_choice="JudgeVerdict")
    resp = bound.invoke(prompt)
    usage = getattr(resp, "usage_metadata", None) or {}
    verdict = JudgeVerdict(**resp.tool_calls[0]["args"])
    """

    def __init__(self, is_pass=True, category="other"):
        self.is_pass = is_pass
        self.category = category
        self.prompts: list[str] = []

    def bind_tools(self, schemas, tool_choice=None):
        outer = self
        schema = schemas[0]

        class _C:
            # 通过FakeLLM调用invoke时，外部的prompt会作为参数，传进了 FakeLLM 的 invoke
            def invoke(self, prompt, config=None):
                outer.prompts.append(prompt)
                # schema拿到的是pydantic 类本身，然后通过schema.__name__，区分是judge还是归因，以此返回不同的格式
                args = ({"is_pass": outer.is_pass, "reason": "fake-reason"}
                        if schema.__name__ == "JudgeVerdict"
                        else {"category": outer.category, "reason": "fake-reason"})
                return SimpleNamespace(
                    tool_calls=[{"name": schema.__name__, "args": args}],
                    usage_metadata={"input_tokens": 0, "output_tokens": 0},
                )

        return _C()


# ============== normalize：字面命中的归一化 ==============

def test_normalize_strips_spaces_and_lowers():
    assert ev.normalize("1-3 个工作日") in ev.normalize("退款 1-3个工作日 内到账")
    assert ev.normalize("PASS") == "pass"


# ============== grade：判定矩阵（含 rubric 送达断言） ==============

def make_case(**kw):
    case = {"id": "X", "question": "q?", "category": "retrieval",
            "expected_points": [], "expected_tools": [], "notes": ""}
    case.update(kw)
    return case


POINTS_HIT = "积分一年有效，过期自动清零"  # 同含「一年」「清零」


def test_grade_all_hit_no_label_skips_judge():
    llm = FakeLLM()
    r = ev.grade(llm, make_case(expected_points=["一年", "清零"]), POINTS_HIT, ["search_knowledge"])
    assert r["answer_ok"] is True and r["judge_used"] is False
    assert llm.prompts == []  # 免费通道：judge 零调用
    assert r["judge_tokens_in"] == 0 and r["judge_tokens_out"] == 0  # 免检也无 token 开销


def test_grade_judge_required_forces_judge_and_sends_rubric():
    llm = FakeLLM()
    case = make_case(expected_points=["一年", "清零"],
                     notes="judge_required：不得出现推诿话术，出现即 FAIL")
    r = ev.grade(llm, case, POINTS_HIT, [])
    assert r["judge_used"] is True
    assert len(llm.prompts) == 1                      # 要点全命中也强制走 judge
    assert "特别判定口径" in llm.prompts[0]           # rubric 必须送达判定者（口径断裂回归锚）
    assert "推诿话术" in llm.prompts[0]
    assert "judge_tokens_in" in r and "judge_tokens_out" in r  # token 字段常在（真 LLM 才有值）


def test_grade_missed_points_goes_to_judge():
    llm = FakeLLM(is_pass=False)
    r = ev.grade(llm, make_case(expected_points=["一年", "清零", "三倍"]), POINTS_HIT, [])
    assert r["judge_used"] is True and r["answer_ok"] is False   # judge 拒绝能打回
    assert r["points_missed"] == ["三倍"]


def test_grade_empty_points_goes_to_judge():
    llm = FakeLLM()
    r = ev.grade(llm, make_case(), "你好，我是客服助手", [])
    assert r["judge_used"] is True and r["answer_ok"] is True


def test_grade_mustcall_satisfied_extra_calls_not_punished():
    llm = FakeLLM()
    r = ev.grade(llm, make_case(expected_tools=["calculate"]), "84", ["calculate", "search_knowledge"])
    assert r["behavior_ok"] is True          # 多调不罚


def test_grade_mustcall_missing_fails():
    llm = FakeLLM()
    r = ev.grade(llm, make_case(expected_tools=["calculate"]), "84", [])
    assert r["behavior_ok"] is False        # 漏掉要罚


def test_grade_forbidden_tool_call_fails():
    llm = FakeLLM()
    r = ev.grade(llm, make_case(expected_tools=[]), "你好", ["search_knowledge"])
    assert r["behavior_ok"] is False         # 空数组 = 禁调任何工具


def test_grade_forbidden_none_called_ok():
    llm = FakeLLM()
    r = ev.grade(llm, make_case(expected_tools=[]), "你好", [])
    assert r["behavior_ok"] is True


# ============== classify_failure：规则层四向 ==============

def base_record(**kw):
    rec = {"pass": False, "error": "", "behavior_ok": True, "question": "q?",
           "expected_tools": [], "called_tools": [], "judge_reason": "",
           "retrieved_context": "", "answer": "ans", "points_missed": []}
    rec.update(kw)
    return rec


def test_classify_error_is_other():
    r = ev.classify_failure(base_record(error="TimeoutError: 60s"), FakeLLM())
    assert r["failure_class"] == "other"


def test_classify_tool_miss():
    r = ev.classify_failure(base_record(behavior_ok=False,
                            expected_tools=["calculate"], called_tools=[]), FakeLLM())
    assert r["failure_class"] == "tool_miss"


def test_classify_tool_extra():
    r = ev.classify_failure(base_record(behavior_ok=False,
                            expected_tools=[], called_tools=["search_knowledge"]), FakeLLM())
    assert r["failure_class"] == "tool_extra"


def test_classify_pass_is_free():
    llm = FakeLLM()
    r = ev.classify_failure(base_record(**{"pass": True}), llm)
    assert r["failure_class"] == ""
    assert llm.prompts == []                 # PASS 零成本：LLM 不被触达


def test_classify_answer_layer_uses_llm_with_evidence():
    llm = FakeLLM(category="retrieval_miss")
    r = ev.classify_failure(base_record(judge_reason="缺关键事实",
                            retrieved_context="资料原文…", points_missed=["30 天"]), llm)
    assert r["failure_class"] == "retrieval_miss"
    assert "缺关键事实" in llm.prompts[0]        # judge 意见作为归因证据进 prompt
    assert "资料原文…" in llm.prompts[0]
    assert "judge_tokens_in" in r                # 归因 LLM 调用的 token 同样入账（FakeLLM 下为 0）


# ============== percentile：经验分位数边界 ==============

def test_percentile_empty_is_zero():
    assert ev.percentile([], 50) == 0.0


def test_percentile_single():
    assert ev.percentile([5.0], 50) == 5.0
    assert ev.percentile([5.0], 95) == 5.0


def test_percentile_general():
    vals = sorted([1.0, 2.0, 3.0])
    assert ev.percentile(vals, 50) == 2.0
    assert ev.percentile(vals, 95) == 3.0


# ============== extract_*：轨迹采集 ==============

def test_extract_tool_calls_in_order_across_messages():
    msgs = [
        SimpleNamespace(type="ai", tool_calls=[{"name": "search_knowledge"}]),
        SimpleNamespace(type="tool", tool_calls=None),           # ToolMessage 无此属性
        SimpleNamespace(type="ai", tool_calls=[{"name": "calculate"}, {"name": "calculate"}]),
    ]
    assert ev.extract_tool_calls(msgs) == ["search_knowledge", "calculate", "calculate"]


def test_extract_tokens_sums_and_defaults_zero():
    msgs = [
        SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5}),
        SimpleNamespace(usage_metadata=None),                    # 网关不返回时给 0
        SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1}),
    ]
    assert ev.extract_tokens(msgs) == (11, 6)


def test_extract_retrieved_only_search_knowledge_and_truncates():
    msgs = [
        SimpleNamespace(type="tool", name="search_knowledge", content="A" * 500),
        SimpleNamespace(type="tool", name="calculate", content="不该收"),
        SimpleNamespace(type="tool", name="search_knowledge", content="B"),
    ]
    out = ev.extract_retrieved(msgs)
    assert len(out) == 400 + 5 + 1              # 400 截断 + "\n---\n"(5字符) + "B"
    assert "不该收" not in out and out.endswith("B")


# ============== retrieval_eval.run_case：got_rank 保先语义（覆盖 bug 回归锚） ==============

FAKE_RESULTS = [
    {"source": "A.md", "text": "t1", "score": 0.90},
    {"source": "B.md", "text": "t2", "score": 0.50},
    {"source": "A.md", "text": "t3", "score": 0.30},   # 同源第二次出现——旧 bug 在此覆盖 rank
]

# 将函数登记进fixture 注册表，本质是执行patched_retrieve = pytest.fixture(patched_retrieve)
# pytest调用测试函数时，看到参数名，只会去注册表查找，而不是去python作用域找同名函数
# 标注后，patched_retrieve作为入参时，pytest在调用测试函数前会执行patched_retrieve()，所以实参是patched_retrieve的执行结果
@pytest.fixture
def patched_retrieve(monkeypatch):
    # my_agent.rag 模块里名叫 retrieve 的函数，临时换成一个“无论问什么、一律返回 FAKE_RESULTS”的假函数;测试结束后自动换回原货
    # lambda q, index: FAKE_RESULTS 替代品。lambda是单行匿名函数，参数名为q,index。等价于def fake_retrieve(q, index):return FAKE_RESULTS
    monkeypatch.setattr(rag_mod, "retrieve", lambda q, index: FAKE_RESULTS)
    return {"chunks": []}


def test_run_case_same_source_keeps_first_rank(patched_retrieve):
    r = retrieval_eval.run_case({"id": "P", "query": "q", "expected_sources": ["A.md"]}, patched_retrieve)
    assert r["hit"] is True
    assert r["best_rank"] == 1                    # 取首次出现名次，不是覆盖成 3
    assert r["best_score"] == 0.90                # 期望来源最高分


def test_run_case_partial_missing(patched_retrieve):
    r = retrieval_eval.run_case({"id": "P", "query": "q", "expected_sources": ["A.md", "C.md"]}, patched_retrieve)
    assert r["hit"] is False and r["missing_sources"] == ["C.md"]
    assert r["best_rank"] is None


def test_run_case_all_missing(patched_retrieve):
    r = retrieval_eval.run_case({"id": "P", "query": "q", "expected_sources": ["Z.md"]}, patched_retrieve)
    assert r["hit"] is False and r["best_rank"] is None and r["best_score"] is None
