"""02_raw_client.py —— 手写 JSON-RPC 裸客户端：逐帧打印 MCP stdio 报文（D2 教学物）

MCP stdio 传输的本质：子进程的 stdin/stdout 上，每行一条 JSON-RPC 2.0 消息。
本脚本不用任何 SDK 客户端，直接 subprocess 拉起 server、往 stdin 写报文、
从 stdout 读报文——initialize 握手、tools/list、tools/call 全程协议可见，
配合 D1 的三原语概念看「模型管动作」长什么样。不追求健壮（一次教学投喂）。

跑法（项目根）：uv run python src/mcp_and_skill/02_raw_client.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台显示 UTF-8 报文（不影响协议本身）

# SERVER 普通 Python 变量,用于后续程序的启动。sys.executable 代表程序，path为参数
# sys.executable——当前正跑着的解释器本体
# str(Path(__file__).resolve().parent / "01_mcp_server.py") 从当前文件位置，推出server脚本（01_mcp_server.py）的位置
SERVER = [sys.executable, str(Path(__file__).resolve().parent / "01_mcp_server.py")]


def send(proc, obj):
    """把一帧请求发出去（Python 对象 → JSON 文本 → 管道）"""
    line = json.dumps(obj, ensure_ascii=False)  # ① dict 编码成 JSON 字符串，ensure_ascii=False 让中文原样输出
    print(f"--> {line}")                        # ② 抄送给屏幕看（教学用，可删）
    proc.stdin.write(line + "\n")               # ③ 写进 server 的 stdin，\n 是帧分隔符。此时数据在本地缓冲区
    proc.stdin.flush()                          # ④ 强制立刻送出


def recv(proc):
    """收一帧响应（管道 → JSON 文本 → Python 对象）"""
    line = proc.stdout.readline()               # ① 阻塞读一行：server 不回，就停在这干等
    print(f"<-- {line.strip()}")                # ② 抄送
    return json.loads(line)                     # ③ JSON 字符串解码回 dict，交还给调用方


def main():
    # subprocess.Popen 从当前进程中启动一个子进程，并且新进程一直在后台活着，Popen 马上返回
    # subProcess.run 启动子进程->等他跑完->给结果，适合跑完就丢的命令
    # SERVER 要启动的命令，Popen 把列表第 0 个元素当作要启动的程序
    # stdin 标准输入，默认接键盘，谁在用：input()、sys.stdin.readline()、server读报文
    # stdout 标准输出，默认接屏幕，谁在用：print()、server回报文
    # stderr 标准错误，默认接屏幕，谁在用：报错 traceback、日志、server 的启动信息
    # stdin=subprocess.PIPE 子进程的stdin接一条管道，不设的话，子进程的stdin会继承当前进程的终端键盘
    # stdout=subprocess.PIPE 子进程的 stdout接一条管道，不设的话，它 print 直接到屏幕，会拿不到报文
    # 进程创建时自带这三个口子，接到哪里，由启动者决定。如果终端敲python mcp_server.py，stdin←键盘，stdout/stderr→屏幕。如果raw_client里Popen(SERVER,stdin=PIPE,stdout=PIPE),stdin接proc.stdin，stdout接proc.stdout
    # text=True 管道按文本模式工作
    # encoding = "utf-8" 文本模式的解码编码,只写 text=True 不写 encoding 时，Python 用系统默认编码——Windows 是 GBK，而 server 发的是 UTF-8，就会解码出错或乱码
    # proc代表并指向子进程，但只是”子进程的遥控器“，不是子进程本身
    proc = subprocess.Popen(SERVER, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, encoding="utf-8")

    # 1. initialize 握手：客户端自报协议版本与能力，server 回它的能力与边界
    # protocolVersion: "2024-11-05"——我讲这个版本的协议；server 认就继续，不认就报错断开；
    # capabilities: {}——我（客户端）没什么特殊能力，空着声明；
    # clientInfo: {"name": "raw-client", ...}——自报家门，纯身份信息。
    send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "raw-client", "version": "0.1"}}})
    # recv 回来的响应里有 serverInfo（"xingchen-mall"）和 server 的 capabilities
    recv(proc)
    # 2. initialized 通知：notification 无 id、无响应，纯粹告知「握手完成」
    send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    # 3. tools/list：server 暴露什么工具、schema 长什么样，全在这一帧的响应里
    send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    resp = recv(proc)
    tools = resp["result"]["tools"]
    print(f"\n===== server 暴露 {len(tools)} 个工具: {[t['name'] for t in tools]} =====\n")

    # 4. tools/call：动作语义的活教材——客户端（宿主）替模型执行，结果回填消息轨迹
    send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "calculate", "arguments": {"expression": "45+39"}}})
    r = recv(proc)
    print(f"===== calculate 结果: {r['result']['content'][0]['text']} =====\n")

    # search_knowledge 首次调用会懒加载 embedding 模型，等几秒属正常
    send(proc, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "search_knowledge",
                           "arguments": {"query": "七天无理由退货的条件是什么？"}}})
    r = recv(proc)
    print(f"===== search_knowledge 结果（前 120 字）: {r['result']['content'][0]['text'][:120]} =====")

    proc.stdin.close()  # 发结束信号
    proc.wait(timeout=10) # 确认退场+回收资源+超时保护


if __name__ == "__main__":
    main()
