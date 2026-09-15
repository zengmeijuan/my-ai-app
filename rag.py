"""rag.py - 朴素 RAG（Retrieval-Augmented Generation，检索增强生成）学习实现

RAG = 先「查资料」再「作答」，把大模型的开放式问答变成有据可依的问答。
核心解决两个问题：
1. 知识边界：模型不知道私有/最新数据（本项目即 docs/ 下的客服知识库）
2. 幻觉：让模型只依据检索到的资料作答并标注引用，而不是自由发挥

两条管道：
- 离线管道（build 子命令）：docs/*.md → 按段落切块 → 本地 embedding 模型向量化
  → 存入 rag_index.json（学习用「文件版向量库」；生产用 FAISS/Milvus/Qdrant 等）
- 在线管道（ask 子命令）：问题 → 向量化 → 与全部 chunk 算余弦相似度 → 取 top-3
  → 拼进提示词 → 网关 chat 模型生成带引用的回答

与 agent.py 的关系：agent.py 是「会用工具的通用助手」，rag.py 是「专注查知识库的
问答器」。下一课可把本文件的检索做成 agent.py 的一个新工具（search_knowledge），
即 Agentic RAG——RAG 退化为 agent 工具箱里的一件工具。

embedding 模型：jinaai/jina-embeddings-v2-base-zh（本地 ONNX 推理，免费离线，
161M 参数，768 维，中英双语）——网关不支持 embedding 接口，故走本地路线。

运行：
  uv run python rag.py build                      # 构建索引（docs/ 变更后重跑）
  uv run python rag.py ask 退货的钱多久能退回来    # 提问
"""
import os

# ============== 0. 环境变量（必须在 import fastembed 之前！）==============

# huggingface_hub 在 import 时读取进程环境变量，之后再改无效，所以放在文件最顶部。
# HF_ENDPOINT：国内镜像，不走镜像直连 huggingface.co 大概率超时
# HF_HUB_DISABLE_XET：新版默认启用 Xet 存储协议，它绕过镜像直连官方 CAS 服务器
#   导致 401；关掉后走镜像支持的普通 HTTP 下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import sys

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI

load_dotenv()  # 读 .env 拿 API_KEY / BASE_URL / MODEL（与 agent.py 同款配置）


# ============== 1. 配置 ==============

DOCS_DIR = "docs"      # 语料目录：里面的 .md 文件是知识库
INDEX_PATH = "rag_index.json"  # 索引文件：chunk 文本 + 向量，即本项目的「向量库」本体
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-zh"  # 本地 embedding 模型
CHAT_MODEL = os.environ.get("MODEL", "gpt-4o-mini")  # 生成侧用网关 chat 模型
TOP_K = 3              # 每次检索取最相似的 3 个 chunk
API_TIMEOUT = 60       # chat API 超时秒数（与 agent.py 一致）

# 懒加载单例模式：embedding 模型很重（161M 参数），全进程只加载一次，之后处处复用
_model = None

def get_embed_model() -> TextEmbedding: # 注解：返回一个 TextEmbedding 实例
    global _model  # 声明：我要操作的是模块级全局变量 _model
    if _model is None:  # 还没加载过？
        _model = TextEmbedding(EMBED_MODEL) # 才真正加载，并记在全局变量里
    return _model   # 返回缓存的那个实例


# ============== 2. 离线管道：docs → 切块 → 向量化 → 存索引 ==============

def load_chunks() -> list[dict]:
    """读取 docs/ 下全部 .md，按空行切段落，一段 = 一个 chunk。

    切块（chunking）策略：段落级切分。每段自带小节标题（markdown 里标题
    与正文之间不留空行），标题参与向量化，chunk 主题更明确、检索更准。
    生产系统会用滑动窗口/递归分割等策略并考虑相邻 chunk 重叠，防答案被切断。
    """
    chunks = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
            text = f.read()
        for para in text.split("\n\n"):
            para = para.strip()
            if para:  # 跳过空段
                chunks.append({"source": fname, "text": para})
    return chunks


def build_index():
    """离线管道主函数：语料 → 向量索引文件。docs/ 变更后重跑。"""
    chunks = load_chunks()
    if not chunks:
        print(f"未在 {DOCS_DIR}/ 找到任何 .md 文件，先创建语料")
        return

    model = get_embed_model()
    # embed() 接受字符串列表，返回生成器（惰性产出 numpy 数组），逐个转 list 存 JSON
    # 最内层：[c["text"] for c in chunks] 将chunks列表中的text字段抽出来，得到一个纯字符串列表，比如 ["退货政策…", "发票问题…", ...]
    # 第二层：model.embed(字符串列表) 把整批文本交给模型向量化。它的返回值不是列表，而是生成器——惰性的，你不索取它就不计算。每次 next（或被 for 循环消费）才产出一个 numpy数组
    # 第三层（外层）：[v.tolist() for v in ...]  for 循环消费那个生成器，每拿到一个向量 v（numpy 数组），就调 .tolist() 把它转成普通 Python 列表
    vectors = [v.tolist() for v in model.embed([c["text"] for c in chunks])]

    index = {
        "model": EMBED_MODEL,  # 记录构建模型：换模型必须重建索引（维度和语义空间都不同）
        "chunks": chunks,      # chunk 原文：检索命中后要拿给模型看
        "vectors": vectors,    # 与 chunks 一一对应的向量（JSON 只能存普通 list）
    }
    # "w" 是写模式（write）：文件不存在就创建；已存在就先清空再写
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        # dump 落盘，dumps 出串。将index对象序列化成 JSON 格式直接写进文件
        json.dump(index, f, ensure_ascii=False)

    n_files = len({c["source"] for c in chunks})
    print(f"索引构建完成：{n_files} 个文件 → {len(chunks)} 个 chunk → {INDEX_PATH}")


# ============== 3. 在线管道：问题 → 检索 → 生成 ==============

def retrieve(query: str, index: dict) -> list[dict]:
    """在索引里找与 query 最相似的 top_k 个 chunk。

    相似度用余弦（cosine）：两向量夹角的余弦值，只看方向不看长度，
    取值 -1~1，越大越相似。文本没法直接算距离，都先转成向量再比。
    学习版用 numpy 暴力算全量相似度（20 个 chunk 毫无压力）；生产向量库
    （FAISS/Milvus 等）的核心就是把 top-k 做到亿级向量也毫秒级。
    """
    model = get_embed_model()
    # np.array 把普通 Python 列表转换成 NumPy 的核心数据结构 ndarray（N 维数组）
    # next(...)——从生成器里拽出第一个结果
    query_vec = np.array(next(model.embed([query])))

    vectors = np.array(index["vectors"])  # 形状 (chunk 数, 768)
    # 余弦公式：点积 / (模长 × 模长)。一次矩阵乘算完所有 chunk 的点积
    # vectors @ query_vec: @ 是矩阵乘法。vectors 形状 (20, 768)，query_vec 形状 (768,)，相乘时 NumPy 把后者依次和每一行做点积（768 对数字两两相乘再求和）,得到形状(20,) 的一维数组——第 i 个数就是「第 i 个 chunk 与问题的点积」
    # np.linalg.norm(vectors, axis=1)按行算每个 chunk 向量的模长
    # scores 最后得到scores (20,)，20个相似度，纯 numpy 浮点数组
    scores = (vectors @ query_vec) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    )
    # argsort返回的是取数顺序，即下标，而不是排好序的值。默认升序，[::-1] 翻转成降序，取前 top_k 个下标
    top_idx = np.argsort(scores)[::-1][:TOP_K]

    return [
        {
            "source": index["chunks"][i]["source"],
            "text": index["chunks"][i]["text"],
            "score": float(scores[i]),   # numpy 标量转 Python float（JSON 友好）
        }
        for i in top_idx
    ]


def ask(query: str):
    """在线管道主函数：检索 top-3 → 拼提示词 → chat 模型生成带引用的回答。"""
    if not os.path.exists(INDEX_PATH):
        print(f"索引不存在，请先运行: uv run python rag.py build")
        return
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)
    if index["model"] != EMBED_MODEL:
        print(f"索引由 {index['model']} 构建，与当前配置 {EMBED_MODEL} 不一致，请重新 build")
        return

    # 1. 检索（Retrieval）
    results = retrieve(query, index)
    print("[检索结果]")
    for i, r in enumerate(results, 1):
        first_line = r["text"].splitlines()[0]  # chunk 第一行是文件/小节标题
        print(f"  [{i}] {r['source']} · {first_line}（相似度 {r['score']:.4f}）") # :.4f 四舍五入到 4 位
    print()

    # 2. 拼增强提示词（Augmented）：把检索到的资料塞进上下文，让模型有据可依
    context = "\n\n".join(
        f"资料[{i}]（来源：{r['source']}）\n{r['text']}"
        for i, r in enumerate(results, 1)
    )
    prompt = (
        "你是星辰商城的客服助手。仅依据下方编号资料回答用户问题，"
        "不要使用资料之外的知识，回答末尾单独一行标注引用的资料编号（如：引用：[1][2]）。\n"
        "资料中没有相关信息时，如实告知「知识库中没有相关信息」，禁止编造。\n\n"
        f"{context}\n\n"
        f"用户问题：{query}"
    )

    # 3. 生成（Generation）：复用 agent.py 同款网关 chat 模型
    client = OpenAI(
        api_key=os.environ["API_KEY"],
        base_url=os.environ.get("BASE_URL", "https://api.openai.com/v1"),
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        timeout=API_TIMEOUT,
    )
    print(f"AI: {resp.choices[0].message.content}")


# ============== 4. 命令行入口 ==============

# 直接运行本文件时才执行，被 import 时不执行（与 agent.py 同款判断）
# sys.argv 脚本名及之后的词，python及前面的不算。uv run python rag.py，sys.argv = ['rag.py']，长度只有 1
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：")
        print("  uv run python rag.py build                  # 构建索引（docs/ 变更后重跑）")
        print("  uv run python rag.py ask <问题>             # 提问，如：")
        print("      uv run python rag.py ask 退货的钱多久能退回来")
    elif sys.argv[1] == "build":
        build_index()
    elif sys.argv[1] == "ask":
        if len(sys.argv) < 3:
            print("请提供问题，如：uv run python rag.py ask 退货的钱多久能退回来")
        else:
            # 问题含空格时 shell 会拆成多个 argv，拼回完整句子
            ask(" ".join(sys.argv[2:]))
    else:
        print(f"未知命令：{sys.argv[1]}（可用：build / ask）")
