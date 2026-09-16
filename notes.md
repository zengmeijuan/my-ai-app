# 学习笔记

> 个人学习备忘，忘了就来翻。按主题分节，新主题往后追加。

## JSON 四件套（json 模块）

**一句话**：带 `s` 的在字符串里进出，不带 `s` 的在文件里进出；dump 是出去（Python → JSON），load 是回来（JSON → Python）。

| 函数 | 方向 | 进/出什么 | 项目里的用例 |
|---|---|---|---|
| `json.dump(obj, f)` | Python → JSON | 写进**文件** | agent.py 退出时把对话存进 session.json；rag.py 构建完索引落盘 rag_index.json |
| `json.dumps(obj)` | Python → JSON | 返回**字符串** | agent.py 把工具执行结果转成字符串塞回对话给模型 |
| `json.load(f)` | JSON → Python | 从**文件**读 | agent.py 启动时从 session.json 恢复会话；rag.py ask 时读索引 |
| `json.loads(s)` | JSON → Python | 解析**字符串** | agent.py 解析模型返回的工具调用参数 |

**两个高频参数**（加在 dump/dumps 上）：

- `ensure_ascii=False`：中文原样写入；默认 `True` 会变成 `\u9000\u8d27` 这种转义，机器能读回、人没法看
- `indent=2`：缩进美化；调试打印结构用 `json.dumps(data, ensure_ascii=False, indent=2)`

**必背错误**：`json.JSONDecodeError` —— 解析非法 JSON 时抛出。外部来源的解析必须套 try/except（agent.py 启动恢复会话处是范例：文件损坏就当无事发生，重新开始）。

**易踩的 JSON 硬规则**：

1. 键和字符串必须**双引号**，`{'a': 1}` 直接报错（Python 字典单双皆可，JSON 不行）
2. **不允许注释**
3. 最后一个成员后**不能有逗号**
4. 布尔和空值是 `true / false / null`（Python 里是 `True / False / None`）

**记法口诀**：dump 落盘、dumps 出串、load 从盘、loads 从串。dump/dumps 是打包发货，load/loads 是签收拆包。
