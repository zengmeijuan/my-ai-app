"""server.py —— G5 服务化：FastAPI 包手写 agent，POST /chat 对外提供对话能力。

CLI 是「一个终端 + 一个用户 + 一次对话循环」；服务化的本质是把会话状态从
进程级单例变成请求级资源：每个 session_id 一份独立 messages（上下文）+
独立 Budget（成本账本）。H4 起会话外置 Redis（session_store）：服务重启
不失忆、多实例共享同一会话、TTL 管时效——每次请求从存储 load、结束写回，
进程内存里不再养会话表。

G3 预埋的 ask_user EOFError→默认拒绝在此验收：HTTP 服务没有交互 stdin，
危险工具一律拒绝——没人能点头 = 没人同意。echo=False 的中间过程走日志：
熔断/超时（warning 级）在服务终端可见；工具调用等 info 级默认被 root logger
的 WARNING 级别拦下，要看到它们需 logging.basicConfig(level=logging.INFO)。

并发取舍：全局锁把整个「取会话 → 跑内循环 → 写回」包成临界区，同一进程内
同一时刻只有一个请求在跑（吞吐换正确，学习版够用）；多副本下两实例并发写
同一会话仍有「后写覆盖先写」竞态，生产的形态是 per-session 锁 + 会话读写
分离。

跑法（项目根；--app-dir 由 uvicorn 把 src 加进 sys.path，等效 PYTHONPATH 但跨 shell 通用——
PYTHONPATH=src 是 bash 专属前缀，PowerShell/cmd 会把它当程序名报错）：
  PowerShell:  $null | uv run uvicorn security_serve.agent.server:app --app-dir src --port 8000
  bash:        uv run uvicorn security_serve.agent.server:app --app-dir src --port 8000 < /dev/null
（管道/重定向断开 stdin 模拟真实部署；交互终端直接跑 stdin 还活着，ask_user 会阻塞等人敲 y/N）
测试（httpx 最稳——curl 在 Windows GBK 控制台传中文 body 会坏编码，PowerShell 引号转义另有一套）：
  httpx.post("http://127.0.0.1:8000/chat", json={"message": "(25+5)*2 等于多少"}).json()

  uvicorn是Web服务器（应用服务器），作用是绑定端口 → 收到 HTTP 报文 → 按协议解析 → 调用我们模块里的 app 对象 → 拿到响应 → 写回网络
  /docs文档地址：http://127.0.0.1:8000/docs
"""
import logging
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from security_serve.agent.config import LOG_PATH
from security_serve.agent.loop import run_agent_turn
from security_serve.agent.logsetup import request_id_var, setup
from security_serve.agent.memory import build_system_prompt
from security_serve.agent.safety import Budget
from security_serve.agent.session_store import count_sessions, load_session, save_session

# 创建FastAPI应用实例，是整个web应用的骨架：路由表、请求校验、响应序列化都长在它身上
app = FastAPI(title="my-ai-app agent server", version="G5")
# H5-1 结构化日志：显式配置 root（文件口 JSON + 控制台口人读），替代默认的
# "info 被拦、warning 走 lastResort" 的零配置状态；日志文件是三问验收的数据源
setup(LOG_PATH)
# 获取 logger（记录器）实例，logging内部维护logger注册表，同名logger全进程只有一份，已存在直接返回，不存在才新建
log = logging.getLogger("security_serve.server")

# 中间件注册时倒序插队，新注册的插到列表头部。构建时逆序包裹，后注册的包在外壳上，先被请求碰到
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """H5-3 请求计时：每请求一行 event=http_request 日志，聚合出延迟分位数。

    注册顺序有讲究：FastAPI 的 add_middleware 是 insert(0)，后注册的包在
    更外层、先执行（与直觉相反）——所以计时必须注册在 request_id **之前**
    （本文件里定义在其上方），成为它的内层、后执行，此刻 rid 已 set 进
    ContextVar；且每层 BaseHTTPMiddleware 跑在独立 task（context 是各自
    创建时刻的副本），外层读不到内层 set 的值——顺序反了 request_id 字段
    就是 None，step9b [1] 断言验收的正是这一点。perf_counter 是单调时钟
    ——只测流逝时长，不受系统对时/NTP 回拨干扰，计时用它而非 time.time()。
    call_next 抛异常（依赖挂了/业务炸了）也要计时落日志：故障请求从统计
    里消失是最危险的盲区，finally 兜住记 500，异常本身继续向上抛给框架。
    """
    start = time.perf_counter()
    status = 500
    try:
        # call_next内部做的事：把“剩余整链”(下一层中间件 → 路由 → 端点)扔进新 task 启动，如果中间件忘记调用它，请求死在这一层
        # 这层函数体在此暂停，控制权交给下游链路，等它办完事（响应头抵达内存流）才从这一行恢复
        response = await call_next(request)
        status = response.status_code
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        log.info(f"{request.method} {request.url.path} -> {status}",
                 extra={"event": "http_request", "method": request.method,
                        "path": request.url.path, "status": status,
                        "duration_ms": round(duration_ms, 1)})
    return response


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """H5-2 请求链路标识：每个请求一个 request_id，贯穿它的全部日志行。

    上游/网关传来的 X-Request-ID 优先（跨服务链路贯通惯例），没有则本服务
    生成。set 进 ContextVar 后本请求的所有日志自动带上它（Formatter 读），
    响应头回写同一 id 供客户端对账。头部是外部输入，截断防超长撑日志。
    """
    rid = (request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16])[:64]
    # token 里存的是旧值，request_id_var被设置为新值
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)  # 恢复现场：恢复成set(rid) 之前的值。中间件常驻，请求之间不残留
    response.headers["X-Request-ID"] = rid
    return response

# 本进程串行化锁：同进程并发请求对同一会话的「load → 改 → 写回」不能交错
# （Redis 本身线程安全，锁护的是读改写序列的原子性）；多副本竞态见 docstring 说明
_sessions_lock = threading.Lock()

# 会进行类型检查，如果校验失败，则返回422
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    used_tokens: int


def _get_or_create_session(session_id: str) -> dict:
    """按 session_id 取会话：先从 Redis load（不存在/已过期返回 None），不存在则建。

    system prompt 打底 + 独立预算账本。CLI 形态的 load_session 从 session.json
    恢复单会话快照，这里从 Redis 按 id 恢复任意会话——跨重启恢复由此补全。"""
    sess = load_session(session_id)
    if sess is None:
        # 业务数据走 extra 字段（聚合可查），消息只留人读描述——H5-1 的核心改法
        log.info("创建新会话", extra={"event": "session_new", "session_id": session_id})
        sess = {
            "messages": [{"role": "system", "content": build_system_prompt()}],
            "budget": Budget(),
        }
    return sess

# @app.post 语法糖，等价于：chat = app.post("/chat")(chat)
# app.post("/chat") 先执行，返回一个注册器函数；再把 chat 传进去，效果是把 {方法: POST, 路径: /chat} → chat 这条映射登记进app的路由表
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # 系统边界校验：空消息/纯空白直接拒绝，不烧 LLM 的钱
    if not req.message.strip():
        # raise 是 Python 主动抛出异常的语句，chat函数会立即中断，之后的代码一行不执行，FastAPI框架会将异常包装成HTTP错误提供给客户端
        raise HTTPException(status_code=400, detail="message 不能为空")

    # 全局锁包住「load 会话 → 跑内循环 → 写回」全程：并发请求串行化，
    # 同一会话的 messages 不会被两个请求同时改写
    with _sessions_lock:
        sess = _get_or_create_session(req.session_id)
        sess["messages"].append({"role": "user", "content": req.message})
        log.info("收到用户消息", extra={
            "event": "chat_in",
            "session_id": req.session_id,
            "msg_index": len(sess["messages"]) - 1,  # 去掉 system 打底那条
        })

        # echo=False：中间过程走日志，回答从返回值拿
        reply = run_agent_turn(sess["messages"], sess["budget"], echo=False)

        # 写回 Redis：每请求结束落一次盘（H4 外置的核心动作），TTL 顺手续期
        save_session(req.session_id, sess)

        return ChatResponse(
            session_id=req.session_id,
            reply=reply,
            used_tokens=sess["budget"].used,
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sessions": count_sessions()}
