"""logsetup.py —— H5-1 JSON 结构化日志：显式配置 + 自建 Formatter。

H5-0 实验的结论：默认状态下具名 logger 继承 root 的出厂 WARNING，info 全被
第一道闸拦下；零 handler 时 warning 靠 lastResort 光秃秃现形。结构化改造做的
正是"配置"这件事——显式挂两个 handler，正是 H5-0 [4] 的直接应用：

  文件口  FileHandler + JsonFormatter  → 一行 JSON，给机器聚合（三问验收的数据源）
  控制台口 StreamHandler + 人读格式     → 开发终端保持可读

Formatter 的核心动作：LogRecord → dict → json.dumps 一行。业务字段不再藏进
消息文本，而是经 log.info("...", extra={...}) 挂到 LogRecord 上，由 Formatter
平铺进 JSON——"消息里藏数据"升级为"字段里放数据"，日志从给人读升级为可查询。

设计取舍：
- extra 用排除法收集（LogRecord 标准属性不进 JSON，其余全收）：调用方加新
  观测字段零改动 Formatter；但 key 撞标准属性名会被静默丢弃，字段名避开即可
- ensure_ascii=False 保中文可读（文件 utf-8）；default=str 兜底非序列化对象
- 进程内调用一次 setup()（uvicorn 导入 server.py 时）；stdout/stderr 两流的
  缓冲差异与 utf-8 输出问题见 step9a 实验脚本
"""
import json
import logging
from datetime import datetime
from logging import FileHandler, Formatter, StreamHandler
from pathlib import Path

# LogRecord 的全部标准属性：这些由 logging 机制自己填，extra 的 key 撞上会被
# 覆盖/丢弃，收集时排除——Formatter 只收业务字段
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "sinfo", "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(Formatter):
    """一条 LogRecord 序列化成一行 JSON：固定四字段 + extra 业务字段平铺。"""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            # 非_RESERVED中标准属性的，属于额外的字段，需要保留
            if key not in _RESERVED:
                data[key] = value
        # exc_info：异常信息在record 里是三元组（异常类型/值/traceback 对象），json.dumps 序列化不了，必须先 formatException 转成字符串
        if record.exc_info:
            data["traceback"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def setup(log_file: Path, level: int = logging.INFO, console: bool = True) -> None:
    """挂 root：文件口落 JSON（聚合查询数据源），控制台口落人读格式。

    都挂 root——收全树消息（security_serve.* 与 uvicorn 的日志一起进文件，
    集中收集视角下 uvicorn.access 也是观测数据）。WARNING 之后不再走
    lastResort（root 有 handler 了），终端告警由控制台口接管，恢复带时间戳。
    """
    root = logging.getLogger()  # 不带名字，获取root（logger树的根）
    root.setLevel(level)        # 将第一道闸的线从出厂 WARNING(30) 拨到 INFO(20)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = FileHandler(log_file, encoding="utf-8")    # 造一个文件出口，往磁盘文件写
    fh.setFormatter(JsonFormatter())                # 这个出口的排版 = JSON
    root.addHandler(fh)

    if console:
        # 第二个出口，目的地默认 stderr
        ch = StreamHandler()
        ch.setFormatter(Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(ch)
