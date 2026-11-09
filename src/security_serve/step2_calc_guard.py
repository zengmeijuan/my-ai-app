"""step2_calc_guard.py —— G2 演示：calculate 从裸 eval 到 ast 白名单求值器

要防的问题：eval(expression, {"__builtins__": {}}) 并不安全——空 builtins 只挡
名字查找，挡不住属性访问。经典逃逸链 ().__class__.__bases__[0].__subclasses__()
不需要任何内置名就能拿到全部已加载类，是继续通向 os 等危险模块的起点；
且出错时报 NameError（"名字没定义"），误导排查方向。

新方案（safety.safe_eval）：ast.parse 解析成语法树后递归求值，全程无 eval——
解释器只实现 Constant（仅 int/float）/ BinOp（七种算术）/ UnaryOp（正负号）
三类节点，白名单即解释器能力集，其余节点一律拒绝。附带幂指数守卫防数值炸弹。

跑法（项目根）：
  uv run python src/security_serve/step2_calc_guard.py
"""
import sys

from my_agent.agent import calculate as old_calculate
from security_serve.agent.tools import calculate as new_calculate

failures = []


def mark(label: str, ok: bool):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


if __name__ == "__main__":
    print("G2 calculate 演示：裸 eval（旧）vs ast 白名单求值器（新）")
    print("=" * 60)

    print("[1] 合法算术：新旧逐字一致")
    for expr in ["(25+5)*2", "2**10", "7//2", "7%3", "-3+5", "3.5*2", " 25 + 5 "]:
        o, n = old_calculate(expr), new_calculate(expr)
        mark(f"{expr!r} → {n}" + ("" if o == n else f"（旧版 {o!r} 不一致！）"), o == n)

    print("[2] 算术错误：两边都清晰返回（措辞有意变更）")
    print(f"  旧版 1/0 → {old_calculate('1/0')}")
    print(f"  新版 1/0 → {new_calculate('1/0')}")

    print("[3] 注入对照：旧 eval 的洞 vs 新白名单")
    probes = [
        "__import__('os').getcwd()",
        "().__class__.__bases__[0].__subclasses__()",
        "x + 1",
        "'hello'",
    ]
    for p in probes:
        o = old_calculate(p)
        n = new_calculate(p)
        print(f"  探针 {p!r}")
        # 旧版属性链真的执行了：截断展示它返回的东西
        print(f"    旧 eval → {o[:100] + ('...' if len(o) > 100 else '')}")
        print(f"    新 ast  → {n}")
        mark(f"新版拒绝 {p!r}", n.startswith("错误：包含不允许的语法"))

    print("[4] 数值炸弹：新版秒拒（旧版会算数亿位大数，这里不跑它）")
    bomb = new_calculate("9**9**9")
    print(f"  新版 9**9**9 → {bomb}")
    mark("炸弹被守卫拦下", bomb.startswith("错误：幂指数过大"))

    print("=" * 60)
    if failures:
        print(f"结果：FAIL（{len(failures)} 例不符合预期）：{failures}")
        sys.exit(1)
    print("结果：PASS——白名单求值器合法放行、注入与炸弹全拒")
