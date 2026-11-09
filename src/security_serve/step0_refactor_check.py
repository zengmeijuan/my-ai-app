"""step0_refactor_check.py —— G0 验收：重构前后行为不变的对照检查

G0 只搬结构不添功能（硬约束）。本文件逐项对照旧版（src/my_agent/agent.py，冻结）
与新版（src/security_serve/agent/）的确定性输出：工具实现逐个调用比对、
schema 逐字比对、注册表完整性比对。任何一项不一致即 FAIL——结构变了，行为一个
字符都不能变。

G2 起的例外：calculate 的注入类拒绝语义有意变更（裸 eval → ast 白名单求值器），
该项不再跨版对照（见文件内标注），新版权面行为由 step2_calc_guard.py 覆盖。

联网类工具（web_search / web_fetch）依赖外部网络，不在自动对照内（CLI 冒烟覆盖）。

跑法（项目根）：
  uv run python src/security_serve/step0_refactor_check.py
"""
import json
import sys

from my_agent.agent import (
    calculate as old_calculate,
    get_weather as old_get_weather,
    grep as old_grep,
    list_dir as old_list_dir,
    read_file as old_read_file,
    search_knowledge as old_search_knowledge,
    TOOL_FUNCTIONS as old_tool_functions,
    TOOLS as old_tools,
)
from agent.tools import (
    calculate as new_calculate,
    get_weather as new_get_weather,
    grep as new_grep,
    list_dir as new_list_dir,
    read_file as new_read_file,
    search_knowledge as new_search_knowledge,
    TOOL_FUNCTIONS as new_tool_functions,
    TOOLS as new_tools,
)

failures = []


def check(label: str, old, new):
    """单项对照：一致 PASS，不一致记录并打印 diff 摘要。"""
    if old == new:
        print(f"  [PASS] {label}")
    else:
        failures.append(label)
        print(f"  [FAIL] {label}")
        print(f"    旧版: {old!r:.200}")
        print(f"    新版: {new!r:.200}")


def schema_view(tools: list) -> dict:
    """schema 列表按名排序转 dict，消除顺序差异后逐字比对内容。"""
    return {t["function"]["name"]: json.dumps(t, ensure_ascii=False, sort_keys=True) for t in tools}


if __name__ == "__main__":
    print("G0 重构对照检查：旧版（冻结）vs 新版（security_serve/agent）")
    print("=" * 60)

    print("[1] 工具名集合一致")
    check("工具名集合", sorted(old_tool_functions), sorted(new_tool_functions))

    print("[2] schema 逐字一致（按名排序比对）")
    old_schemas = schema_view(old_tools)
    new_schemas = schema_view(new_tools)
    check("schema 名称集合", sorted(old_schemas), sorted(new_schemas))
    for name in sorted(old_schemas):
        if name in new_schemas:
            check(f"schema[{name}]", old_schemas[name], new_schemas[name])

    print("[3] 确定性工具输出逐个对照")
    check("get_weather('北京')", old_get_weather("北京"), new_get_weather("北京"))
    check("get_weather(' Atlantis')", old_get_weather(" Atlantis"), new_get_weather(" Atlantis"))
    check("calculate('(25+5)*2')", old_calculate("(25+5)*2"), new_calculate("(25+5)*2"))
    check("calculate('2**10')", old_calculate("2**10"), new_calculate("2**10"))
    # G2 有意变更：注入类输入的拒绝语义改变，不再跨版逐字对照。
    #   旧版 eval: 计算错误: name '__import__' is not defined（误导性的 NameError，
    #   且属性链 ().__class__.__bases__[0].__subclasses__() 仍可执行，空 builtins 挡不住）
    #   新版 ast:  错误：包含不允许的语法（Call）——见 step2_calc_guard.py
    print("  [跳过] calculate('__import__') 跨版对照（G2 有意变更，step2 覆盖）")
    check("read_file('README.md') 前 500 字符", old_read_file("README.md")[:500], new_read_file("README.md")[:500])
    check("read_file('不存在.txt')", old_read_file("不存在.txt"), new_read_file("不存在.txt"))
    check("list_dir('src/my_agent')", old_list_dir("src/my_agent"), new_list_dir("src/my_agent"))
    check("list_dir('不存在目录')", old_list_dir("不存在目录"), new_list_dir("不存在目录"))
    check("grep('uv_build', 'pyproject.toml')", old_grep("uv_build", "pyproject.toml"), new_grep("uv_build", "pyproject.toml"))
    check("grep('无效正则[')", old_grep("[invalid"), new_grep("[invalid"))
    check("search_knowledge('退货')", old_search_knowledge("退货"), new_search_knowledge("退货"))

    print("[4] 联网工具（web_search / web_fetch）跳过自动对照（外部网络依赖，CLI 冒烟覆盖）")

    print("=" * 60)
    if failures:
        print(f"结果：FAIL（{len(failures)} 项不一致）：{failures}")
        sys.exit(1)
    print("结果：PASS——重构后行为与旧版完全一致")
