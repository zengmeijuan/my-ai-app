#!/usr/bin/env bash
# H4 效果演示：会话外置 Redis —— 重启服务不失忆
# 用法（WSL 内）: bash /mnt/e/workspace/studying/ai/my-ai-app/h4_demo.sh
set -e
cd /mnt/e/workspace/studying/ai/my-ai-app
SID="h4-demo2"

echo "== [0] 确保 app 容器在跑 =="
docker compose up -d app 2>&1 | tail -1
for i in $(seq 1 25); do curl -s -m 2 http://localhost:8000/health >/dev/null 2>&1 && break; sleep 1; done

echo ""
echo "== [1] 第一轮对话：告诉 agent 一个事实 =="
curl -s -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d "{\"message\":\"A fun fact about me: my favorite number is 42. Just reply 'got it' — do NOT call any tools.\",\"session_id\":\"$SID\"}" --max-time 120
echo ""

echo ""
echo "== [2] 重启前：Redis 里已有该会话（key + 剩余TTL）=="
docker exec my-ai-app-redis-1 redis-cli --scan --pattern 'session:*'
echo "ttl(session:$SID) = $(docker exec my-ai-app-redis-1 redis-cli ttl session:$SID)"

echo ""
echo "== [3] 重启 app 容器（redis 原地不动）=="
docker restart my-ai-app-app-1 >/dev/null
for i in $(seq 1 25); do curl -s -m 2 http://localhost:8000/health >/dev/null 2>&1 && break; sleep 1; done
echo "app 已重启并就绪"

echo ""
echo "== [4] 重启后：会话 key 仍在（数据在 redis 进程，不随 app 生死）=="
docker exec my-ai-app-redis-1 redis-cli --scan --pattern 'session:*'

echo ""
echo "== [5] 第二轮对话：同 session_id 续问 —— H4 验收时刻 =="
curl -s -X POST http://localhost:8000/chat -H 'Content-Type: application/json' \
  -d "{\"message\":\"What is my favorite number? Answer with just the number.\",\"session_id\":\"$SID\"}" --max-time 120
echo ""
echo ""
echo "== 演示结束：若上方回复含 42，即「重启服务后仍答出上一轮事实」达成 =="
