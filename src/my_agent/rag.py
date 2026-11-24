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
  uv run python -m my_agent.rag build                      # 构建文件版索引（docs/ 变更后重跑）
  uv run python -m my_agent.rag sync-pg                    # 增量同步 pgvector（content_hash diff，H6-2）
  uv run python -m my_agent.rag sync-pg --full             # 全量重建（TRUNCATE 重灌）
  uv run python -m my_agent.rag ask 退货的钱多久能退回来    # 提问（检索后端由 RAG_BACKEND 分流）
"""
import os
from pathlib import Path

# ============== 0. 环境变量（必须在 import fastembed 之前！）==============

# huggingface_hub 在 import 时读取进程环境变量，之后再改无效，所以放在文件最顶部。
# HF_ENDPOINT：国内镜像，不走镜像直连 huggingface.co 大概率超时
# HF_HUB_DISABLE_XET：新版默认启用 Xet 存储协议，它绕过镜像直连官方 CAS 服务器
#   导致 401；关掉后走镜像支持的普通 HTTP 下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import hashlib
import json
import re
import sys

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI

from my_agent import vector_store

load_dotenv()  # 读 .env 拿 API_KEY / BASE_URL / MODEL（与 agent.py 同款配置）


# ============== 1. 配置 ==============

# 数据目录：语料与索引都在包内 data/ 下，用文件自身定位（cwd 无关）
_DATA_DIR = Path(__file__).resolve().parent / "data"
DOCS_DIR = _DATA_DIR / "docs"      # 语料目录：里面的 .md 文件是知识库
INDEX_PATH = _DATA_DIR / "rag_index.json"  # 索引文件：chunk 文本 + 向量，即本项目的「向量库」本体
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-zh"  # 本地 embedding 模型
CHAT_MODEL = os.environ.get("MODEL", "gpt-4o-mini")  # 生成侧用网关 chat 模型
TOP_K = 3              # 每次检索取最相似的 3 个 chunk
API_TIMEOUT = 60       # chat API 超时秒数（与 agent.py 一致）

# 检索后端分流：json = 文件版向量库（rag_index.json + numpy 余弦，默认）；
# pgvector = 数据库版（PG doc_chunks 表 + <=> 余弦距离，H6-1）。
# 换库不换模型：两条后端共用同一 fastembed 编码与同一 TOP_K，近邻只在来源上不同——
# 这是「双后端评测无退化」成立的前提。外置开关 = 同一份代码两种部署形态（与 DSN 同思想）
RAG_BACKEND = os.environ.get("RAG_BACKEND", "json")

# 文档级元数据映射（H6-3）：source → 业务类别。元数据是「文档之外的知识」——
# 正文里读不出来，必须显式登记；生产由文档系统随文档下发，学习版手工映射表。
# 用途：入库写 doc_chunks.category 列；检索可 WHERE category 过滤（先过滤后排序）；
# 增量 diff 对 (content_hash, category) 二元组——元数据变更也触发重灌。
# 唯一事实源：tools.py 的 search_knowledge 类别 enum 从这里派生。
DOC_META = {
    "会员权益.md": "会员积分",
    "支付方式.md": "支付发票",
    "退货政策.md": "售后退货",
    "账号安全.md": "账号安全",
    "配送说明.md": "配送运费",
}

# 懒加载单例模式：embedding 模型很重（161M 参数），全进程只加载一次，之后处处复用
_model = None

def get_embed_model() -> TextEmbedding: # 注解：返回一个 TextEmbedding 实例
    global _model  # 声明：我要操作的是模块级全局变量 _model
    if _model is None:  # 还没加载过？
        _model = TextEmbedding(EMBED_MODEL) # 才真正加载，并记在全局变量里
    return _model   # 返回缓存的那个实例


# ============== 2. 离线管道：docs → 切块 → 向量化 → 存索引 ==============

def load_chunks() -> list[dict]:
    """读取 docs/ 下全部 .md，句级切分（C5 对照实验版 v2）。

    切块（chunking）策略 v2：先按空行分段（段首行是标题），再把段落正文
    按中文句号切成句子，每句重复携带所属标题作为前缀——标题语义继续参与
    向量化，且一个 chunk 只装一个知识点。
    实验动机（v1 段落级的已证短板）：关键句埋在长段尾部被整段向量稀释，
    组件级基线 R12（忘记密码）score 仅 0.38、R06 0.42——找回方式只是
    「密码设置建议」段落的尾句。v1 基线存档：data/rag_index_paragraph_v1.json，
    评测基线 evals/results/retrieval_20261027_154446.json。
    """
    chunks = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
            text = f.read()
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            head, _, body = para.partition("\n")  # 标题行 | 正文（本文档每段都是「标题+正文」结构）
            sentences = [s.strip() for s in re.split(r"(?<=[。！？])", body) if s.strip()]
            for s in sentences:
                chunks.append({"source": fname, "text": f"{head}\n{s}"})
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


def sync_pg(full: bool = False):
    """离线管道 pgvector 版：语料 → 指纹 diff → 按需向量化 → 灌库 doc_chunks。

    H6-2 增量更新：每个文档算 content_hash（sha256，算在切块前的原始全文上，
    同一文档所有 chunk 共用一个指纹，文档一个字没动指纹就不变）。与库内对照面
    diff 出四类处置：
      新增（库无盘有）→ 向量化插入；
      变更 → DELETE 旧行 + 插新行——chunk 集合是整体替换语义
        （新切块结果行数都可能不同），DELETE+INSERT 比 逐行 UPDATE 干净；
      删除（库有盘无）→ 清行；
      不变 → 跳过，零计算（连 embedding 模型都不加载）。
    H6-3 变更判定升级为 (content_hash, category) 二元组，并按「产出依赖哪些
    输入」分层处置（变更类型决定重算范围）：
      新增（库无盘有）→ 向量化插入；
      内容变更（hash 不同）→ DELETE 旧行 + 重切重嵌——向量依赖正文，必须重算；
      类别变更（hash 同、category 不同）→ 一条 UPDATE 补列，不碰模型——向量
        与类别无关，重嵌就是给没变的产出付全价；
      删除（库有盘无）→ 清行；
      不变 → 跳过，零计算（连 embedding 模型都不加载）。
    --full 退回全量重建（TRUNCATE 重灌），首次升级 / 怀疑库脏时的兜底。
    前置：pgvector 容器在跑（docker compose up -d pgvector）。
    """
    docs_text = {
        fname: (DOCS_DIR / fname).read_text(encoding="utf-8")
        for fname in sorted(os.listdir(DOCS_DIR))
        if fname.endswith(".md")
    }
    if not docs_text:
        print(f"未在 {DOCS_DIR}/ 找到任何 .md 文件，先创建语料")
        return
    missing_meta = [s for s in docs_text if s not in DOC_META]
    if missing_meta:  # 新文档没登记类别：静默兜底会错标，显式失败逼登记
        sys.exit(f"错误：以下文档未在 DOC_META 登记类别，请先补映射：{missing_meta}")
    new_hashes = {
        src: (hashlib.sha256(text.encode("utf-8")).hexdigest(), DOC_META[src])
        for src, text in docs_text.items()
    }

    # 全量重建
    if full:
        vector_store.sync_all(_embed_rows(sorted(docs_text), new_hashes))
        return

    # ---- 增量 diff：磁盘新状态 (hash, category) vs 库内旧状态，三路分层 ----
    # 全量路径的建表/加列在 sync_all 里；增量路径先自己确保表结构升级（幂等）
    vector_store.ensure_table()
    old_hashes = vector_store.fetch_hashes()
    added           = [s for s in new_hashes if s not in old_hashes]                            # 库无盘有
    content_changed = [s for s in new_hashes if s in old_hashes and old_hashes[s][0] != new_hashes[s][0]]  # 正文变
    meta_changed    = [s for s in new_hashes if s in old_hashes and old_hashes[s][0] == new_hashes[s][0]
                       and old_hashes[s][1] != new_hashes[s][1]]                                # 仅类别变
    removed         = [s for s in old_hashes if s not in new_hashes]                            # 库有盘无

    for src in removed:
        n = vector_store.delete_by_source(src)
        print(f"  - 已删除 {src}（清掉 {n} 个旧 chunk）")

    for src in meta_changed:  # 轻量路径：纯列更新，模型不加载
        n = vector_store.update_category(src, new_hashes[src][1])
        print(f"  - 类别变更 {src}（UPDATE {n} 行，未重新向量化）")

    targets = added + content_changed
    if targets:
        for src in content_changed:  # 正文变走重型路径：先删旧行，再随新增一起插（整体替换语义）
            vector_store.delete_by_source(src)
        vector_store.insert_chunks(_embed_rows(targets, new_hashes))
    else:
        print("  无需重新向量化（embedding 模型未加载）")

    unchanged = len(new_hashes) - len(added) - len(content_changed) - len(meta_changed)
    print(f"增量同步完成：新增 {len(added)} / 内容变更 {len(content_changed)} / 类别变更 {len(meta_changed)} / 删除 {len(removed)} / 不变跳过 {unchanged}")


def _embed_rows(sources: list[str], hashes: dict[str, tuple[str, str]]) -> list[tuple[str, str, list[float], str, str]]:
    """把 diff 出的目标文档切块并向量化 → 待插行 [(source, text, vector, hash, category)]。

    load_chunks 切全量再按 source 过滤——51 个 chunk 的切块是毫秒级开销，
    不值得为省它另写一套按文档切块。模型懒加载在此处触发：纯删除或全不变
    的同步走不到这里，161M 参数模型根本不进内存。
    """
    targets = set(sources)
    chunks = [c for c in load_chunks() if c["source"] in targets]
    model = get_embed_model()
    return [
        (c["source"], c["text"], v.tolist(), *hashes[c["source"]])  # * 解包把 (hash, category) 元组摊平进五元组
        for c, v in zip(chunks, model.embed([c["text"] for c in chunks]))
    ]


# ============== 3. 在线管道：问题 → 检索 → 生成 ==============

def retrieve(query: str, index: dict | None, category: str | None = None) -> list[dict]:
    """在索引里找与 query 最相似的 top_k 个 chunk（按 RAG_BACKEND 分流）。

    category 非 None 时按元数据先过滤后排序（H6-3）——只有 pgvector 分支
    支持（WHERE 下推 SQL）；json 分支显式报错而非静默忽略：调用方以为过滤了
    其实没滤，是最危险的静默 bug，失败要败在调用点看得见的地方。

    相似度用余弦（cosine）：两向量夹角的余弦值，只看方向不看长度，
    取值 -1~1，越大越相似。文本没法直接算距离，都先转成向量再比。
    两条后端共享「问题向量化」前半程，只在「近邻来源」上分岔：
    - pgvector：查询向量交给 vector_store.search，排序下推进 SQL（<=>）
    - json：numpy 暴力算全量相似度（20 个 chunk 毫无压力）；
      生产向量库（FAISS/Milvus 等）的核心就是把 top-k 做到亿级也毫秒级
    """
    model = get_embed_model()
    # np.array 把普通 Python 列表转换成 NumPy 的核心数据结构 ndarray（N 维数组）
    # next(...)——从生成器里拽出第一个结果
    query_vec = np.array(next(model.embed([query])))

    if RAG_BACKEND == "pgvector":
        return vector_store.search(query_vec, TOP_K, category=category)  # 排序（和过滤）在 PG 里做

    if category is not None:
        raise ValueError("RAG_BACKEND=json 不支持 category 元数据过滤（无过滤语义，H6-0 缺陷③）；传 category=None 或切换 pgvector 后端")

    if index is None:  # json 后端必须有文件版索引
        raise ValueError("RAG_BACKEND=json 需要传入 rag_index.json 加载的 index")
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
    # json 后端才需要文件版索引；pgvector 后端数据在 PG 里，跳过读文件与模型校验
    index = None
    if RAG_BACKEND == "json":
        if not os.path.exists(INDEX_PATH):
            print(f"索引不存在，请先运行: uv run python -m my_agent.rag build")
            return
        with open(INDEX_PATH, encoding="utf-8") as f:
            index = json.load(f)
        if index["model"] != EMBED_MODEL:
            print(f"索引由 {index['model']} 构建，与当前配置 {EMBED_MODEL} 不一致，请重新 build")
            return

    # 1. 检索（Retrieval）——retrieve 内部按 RAG_BACKEND 分流，这里无感知
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
        print("  uv run python -m my_agent.rag build                  # 构建文件版索引（docs/ 变更后重跑）")
        print("  uv run python -m my_agent.rag sync-pg                # 增量同步 docs/ → pgvector（按 content_hash diff，前置 pgvector 容器）")
        print("  uv run python -m my_agent.rag sync-pg --full         # 全量重建（TRUNCATE 重灌，增量出问题时的兜底）")
        print("  uv run python -m my_agent.rag ask <问题>             # 提问，如：")
        print("      uv run python -m my_agent.rag ask 退货的钱多久能退回来")
    elif sys.argv[1] == "build":
        build_index()
    elif sys.argv[1] == "sync-pg":
        sync_pg(full="--full" in sys.argv[2:])
    elif sys.argv[1] == "ask":
        if len(sys.argv) < 3:
            print("请提供问题，如：uv run python -m my_agent.rag ask 退货的钱多久能退回来")
        else:
            # 问题含空格时 shell 会拆成多个 argv，拼回完整句子
            ask(" ".join(sys.argv[2:]))
    else:
        print(f"未知命令：{sys.argv[1]}（可用：build / ask）")
