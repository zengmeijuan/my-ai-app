"""step1_path_guard.py —— G1 演示：文件系统工具的路径白名单闸门

要防的问题：read_file / list_dir / grep 的路径参数完全由 LLM（或注入文本）可控，
没有闸门时可以读盘上任意文件。G1 给这三个文件系统出口加同一道白名单闸：
工作区外（config.ALLOWED_ROOTS 之外）一律拒绝。

闸门两步核心（下方用例逐条对应）：
  1. realpath 规范化：.. . 符号链接全部消解成真实路径——不是见到 .. 就拒，
     消化后仍在区内就放行（用例 2）；逃到区外才拒（用例 3）
  2. 目录级前缀判断：target == base 或 target.startswith(base + os.sep)——
     补 os.sep 防同级目录前缀骗过（用例 5 + 裸 startswith 对照）

跑法（项目根）：
  uv run python src/security_serve/step1_path_guard.py
"""
import sys

from security_serve.agent.config import PROJECT_ROOT
from security_serve.agent.safety import check_path
from security_serve.agent.tools import read_file

failures = []


def case(label: str, path: str, expect_ok: bool):
    """跑一个用例：check_path 的放行结果与预期比对，同时打印规范化产物。"""
    ok, info = check_path(path)
    mark = "PASS" if ok == expect_ok else "FAIL"
    if ok != expect_ok:
        failures.append(label)
    verdict = "放行" if ok else "拒绝"
    print(f"  [{mark}] {label}")
    print(f"        输入: {path}")
    print(f"        → {verdict}: {info}")


if __name__ == "__main__":
    print(f"G1 路径白名单演示（允许根: {PROJECT_ROOT}）")
    print("=" * 60)

    print("[1] check_path 五用例")
    case("工作区内相对路径，放行", "README.md", True)
    case(".. 消化后仍在区内，放行（不是见到 .. 就拒）", "src/../README.md", True)
    case(".. 逃逸到工作区外，拒绝", "../pyproject.toml", False)
    case("绝对路径越界（系统文件），拒绝", r"C:\Windows\win.ini", False)
    case("同级目录前缀陷阱，拒绝", f"{PROJECT_ROOT}-evil\\secret.txt", False)

    print("[2] 对照：裸 startswith 为什么会被用例 5 骗过")
    evil = f"{PROJECT_ROOT}-evil\\secret.txt"
    naive = evil.startswith(str(PROJECT_ROOT))
    print(f"  裸 startswith 判定 '{evil}' → {naive}")
    print("  同级目录确实以项目根为前缀，所以闸门里必须补 os.sep 再比")

    print("[3] 工具层闸门生效（read_file 实调，LLM 拿到的就是这段错误串）")
    print(f"  read_file(r'C:\\Windows\\win.ini') →\n  {read_file(r'C:\Windows\win.ini')}")

    print("=" * 60)
    if failures:
        print(f"结果：FAIL（{len(failures)} 例不符合预期）：{failures}")
        sys.exit(1)
    print("结果：PASS——白名单闸门全部用例符合预期")
