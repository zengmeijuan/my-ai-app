# Dockerfile —— H2 镜像配方：python:3.13-slim 底座 → uv.lock 装依赖 → 拷代码 → 起服务
# 教学映射：uv.lock 锁依赖 ↔ Dockerfile 锁环境；G5 的 uvicorn 服务进程 ↔ 容器里的主进程
FROM python:3.13-slim

# 装 uv（PyPI 单文件二进制）：镜像内沿用项目同一套依赖管理，严格按 uv.lock 复现环境
# ENV：给镜像里所有后续层和运行时设置环境变量。国内网络：pip / uv 走镜像源拉包（生产换成公司内部源，改这两处 URL 即可）
# RUN：build时执行的命令，装uv本身
ENV UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设定后续指令的工作目录，也是容器启动时的默认目录
WORKDIR /app

# 先只拷依赖清单再 sync：依赖没变只改代码时，这层命中缓存，重建秒级（层缓存思想）
COPY pyproject.toml uv.lock ./
# uv sync --frozen，严格按照uv.lock装包，--no-dev跳过pytest等开发依赖，--no-install-project只装依赖不装项目本身，我们的服务靠PYTHONPATH导入，不走包安装
RUN uv sync --frozen --no-dev --no-install-project

# 拷运行涉及的代码：security_serve（主服务）+ my_agent（tools.py 的 search_knowledge
# 复用其 rag 模块——跨包依赖，漏拷即 ModuleNotFoundError，容器实跑抓出）
COPY src/security_serve ./src/security_serve
COPY src/my_agent ./src/my_agent
ENV PYTHONPATH=/app/src
# venv 的 python / uvicorn 直接可用（uv sync 默认落在 /app/.venv）
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# CMD：容器启动时执行的命令--镜像的主进程
# 启动前置建库（db_seed 幂等重建：容器无状态，每次启动回到确定种子）→ 再起服务
# --host 0.0.0.0 是必须的：容器内监听 127.0.0.1 只有容器自己能访问（G5 回环边界知识的容器版）；0.0.0.0监听所有网卡
CMD ["sh", "-c", "python -m security_serve.agent.db_seed && uvicorn security_serve.agent.server:app --host 0.0.0.0 --port 8000"]
