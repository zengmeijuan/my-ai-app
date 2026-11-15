"""sandbox.py —— H3 代码执行沙箱：把模型写的 Python 代码放进一次性容器里跑。

权限演进三幕：裸 eval（G1 之前）→ ast 白名单 safe_eval（只算算术，图灵不完备）→
沙箱内自由执行（本模块）。执行类工具的行为由模型现场定义（填空题变作文题），
防线强度必须匹配能力开放度——白名单管不住任意代码，改靠隔离环境兜底。

五道围栏（全 docker 原生参数，零自研）+ 双层超时：
    --network none（断网）/ 不挂载不传 -e（纯净文件系统）/ --memory / --cpus /
    --pids-limit（防 fork 炸弹）；
    容器内 GNU timeout 精确杀超时代码——外层杀 docker CLI 进程时容器还在跑会
    泄漏，杀容器内主进程则随容器一起死；外层 subprocess timeout 只做兜底。

代码经 stdin 喂入（python -），不落盘——容器里连写代码的文件都没有，跑完即焚。

已知限制：主进程在 Windows 上，靠 subprocess 调 wsl 里的 docker。容器形态下
（H2 镜像）主进程自己就在容器里，调不到宿主 docker——生产解法是 socket 挂载
或远程 daemon API。
"""
import subprocess

from security_serve.agent.config import (
    SANDBOX_CPUS,
    SANDBOX_IMAGE,
    SANDBOX_MAX_CODE_CHARS,
    SANDBOX_MAX_OUTPUT_CHARS,
    SANDBOX_MEMORY,
    SANDBOX_PIDS_LIMIT,
    SANDBOX_TIMEOUT,
)


def _truncate(text: str) -> str:
    if len(text) <= SANDBOX_MAX_OUTPUT_CHARS:
        return text
    return text[:SANDBOX_MAX_OUTPUT_CHARS] + f"\n... [输出过长，已截断到 {SANDBOX_MAX_OUTPUT_CHARS} 字符]"


def run_code(code: str) -> str:
    """在一次性容器里执行模型写的代码，返回 stdout 或错误说明串。

    边界：代码超长直接拒绝；所有失败（超时/OOM/脚本异常/docker 故障）都以
    错误串返回不抛异常——与 db_query 同思路：模型拿得到错误信息，agent 循环
    不至于中断，还能看着报错改代码重试。

    退出码语义（容器主进程是 timeout）：
        0   正常结束
        124 timeout 杀掉超时代码
        137 被 SIGKILL（内存围栏触发 OOM）
        1   脚本自身抛异常（traceback 在 stderr）
        125 docker 自身失败（daemon 不在 / 镜像不在），容器根本没起来
    """
    if len(code) > SANDBOX_MAX_CODE_CHARS:
        return f"错误：代码长度 {len(code)} 字符超过上限 {SANDBOX_MAX_CODE_CHARS}，请精简后重试"

    # cmd命令按照列表顺序逐元素拼
    # wsl -u root   第1段：跨系统跳板（Windows 进 Linux，以 root 身份执行后面整条）
    # docker run    第2段：告诉引擎"创建并启动一个新容器"
    # --rm --network none --memory=...  第3段：docker 的选项区（容器的"出生条件"）
    # -i python:3.13-slim   第4段：-i 是最后一个 docker 选项；镜像名结束选项区
    # timeout 10 python -   第5段：镜像名之后的东西 docker 不再解析,整体作为【容器内的主进程命令行】

    # 镜像名左边的参数是docker CLI在Linux侧解析的配置；右边的timeout 10 python - 是进容器内部去执行的命令
    cmd = [
        "wsl", "-u", "root", "docker", "run",
        "--rm",                        # 用完即焚：主进程退出容器自动删除
        "--network", "none",           # 围栏1：断网，容器内碰不到外网和宿主内网
        # 围栏2：不挂载任何目录、不传 -e——容器里只有纯净底座，无业务代码无密钥
        f"--memory={SANDBOX_MEMORY}",          # 围栏3：内存上限
        f"--cpus={SANDBOX_CPUS}",              # 围栏4：CPU 半核
        f"--pids-limit={SANDBOX_PIDS_LIMIT}",  # 围栏5：进程数上限，fork 炸弹造不出新进程
        "-i",                          # 保持 stdin 打开：代码经 stdin 喂入
        SANDBOX_IMAGE,
        "timeout", str(SANDBOX_TIMEOUT), "python", "-",
    ]
    try:
        # subprocess.run 和 subprocess.Popen 都是创建子进程，run是Popen的便携封装，它内部创建Popen然后一直阻塞直到子进程退出，收输出打包进 proc.stdout/stderr，主进程才继续
        # Popen是并行，子进程启动后立即返回，在后台活着，父进程不被强制等待
        proc = subprocess.run(
            cmd,                    # 列表形式：第0个是程序，其余是参数
            input=code,             # 要写进子进程stdin的内容；写完自动关闭stdin
            capture_output=True,    # 等价 stdout=PIPE + stderr=PIPE：两个口子接管道，跑完从 proc.stdout/stderr 取回
            encoding="utf-8",       # 按 UTF-8 解码输出
            errors="replace",       # 遇到解不了的坏字节替换成 �，不抛 UnicodeDecodeError
            timeout=SANDBOX_TIMEOUT + 15,  # 子进程 25 秒不退就杀掉，抛 TimeoutExpired（外层兜底）：正常时容器内 timeout 先触发
        )
    except subprocess.TimeoutExpired:
        return f"错误：沙箱执行外层超时（>{SANDBOX_TIMEOUT + 15} 秒），请重试"
    except FileNotFoundError:
        return "错误：找不到 wsl 命令——本工具依赖 WSL2 里的 Docker 引擎"

    # returncode是退出码，是穿透四层传递上来的：python（脚本退出码）→ timeout 命令（转发或改写）→ 容器退出码 → docker run CLI → wsl.exe → proc.returncode
    # stdout 输出内容
    rc = proc.returncode
    out = _truncate(proc.stdout.strip())
    err = _truncate(proc.stderr.strip())

    if rc == 0:
        return out if out else "（执行成功，无输出。结果需要用 print() 打印才能拿到）"
    if rc == 124:
        head = f"错误：代码执行超时（超过 {SANDBOX_TIMEOUT} 秒被强制终止）"
        return head + (f"\n超时前输出:\n{out}" if out else "")
    if rc == 137:
        return "错误：内存超限，代码被 OOM 杀掉（沙箱内存上限 128m）"
    if rc == 125:
        return f"错误：Docker 引擎执行失败（容器未启动）: {err}"
    # rc == 1 等脚本异常退出：traceback 在 stderr，原样回传给模型改代码
    body = err if err else out if out else "（无输出）"
    return f"执行出错（退出码 {rc}），错误信息:\n{body}"
