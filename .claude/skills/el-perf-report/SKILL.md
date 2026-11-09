---
name: el-perf-report
description: EL 系统每周性能指标自动填报。从美的 RUM 监控页（monitor.midea.com）查询 EL 指定时间段的四个指标（首屏加载/页面完全加载/慢渲染/慢加载），换算后填入「TMS前端 0指标」在线表格（mapnew5.midea.com）的 EL 区四列。当用户要求"填报 EL 性能数据"、"查 EL 周/某时间段指标"、"填报本周性能"、"填报 10.24~10.30"等时使用。
---

# EL 性能周报填报

数据流：用户给出时间段 → RUM 监控页查询 → 计算四个指标 → 在线表格 EL 区填报。

## 前置条件

1. bsk daemon 与浏览器（Chrome 已登录两个系统）。若 `bsk status` 报 daemon 未启动，
   在 Windows 后台任务里执行（不要反复重试 detached 启动）：
   ```bash
   BSK_AUTO_START=0 bsk daemon start --foreground   # run_in_background 启动
   ```
   然后 `bsk session start` 获取 session id（下文 `$S` 代指）。
2. 若页面出现登录页/未登录，停下来请用户手动登录后继续，不要猜测凭据。

## 第一步：确定查询时间范围

用户的表达 → 起止日期：
- "近一周" / "最近一周"：起始 = 七天前，结束 = 当天
- "10.24~10.30" 这类区间：起始 = 10-24，结束 = 10-30（连字符 "10.31-11.6" 同理解析）
- "本周"：起始 = 本周一，结束 = 今天（当年当月缺位补 0，月份按自然月解析）

该区间后续有两处使用：查询时间戳（下文）、表格 A 列文本（格式化为 `M.D~M.D`，不补零，
见 4a 第 3 点）。

时间戳（北京时间 UTC+8，毫秒）：
```bash
START_MS=$(date -d "${START}T00:00:00+08:00" +%s%3N)   # 起始日 00:00:00
END_MS=$(date -d "${END}T23:59:59+08:00" +%s%3N)       # 结束日 23:59:59
```

## 第二步：监控页查询

导航（URL 直达设置时间范围，`timeRange=c_<起>_<止>`，不要操作时间选择器弹层——
手输值会被组件状态回滚）：

```
https://monitor.midea.com/monitoring/rum/web/view/frontView?systemId=2975443be4234955a2967443c5b4b1f9&nodeId=a352e216e8744097948c1b00af33af1b&monitorItemId=6d219429e16346a4bd6f69719afd2158&viewType=%25E4%25B8%25BB%25E8%25A7%2586%25E5%259B%25BE&timeRange=c_${START_MS}_${END_MS}
```

导航后确认右上角显示 `起始日 00:00:00 ~ 结束日 23:59:59`，不符则停下排查。
**数据异步加载**：导航后立即读会拿到空值，轮询等待文本 `首屏加载[\d.]+` 出现再读
（每 2~3 秒一次 evaluate，最多等 30 秒）。

### 2a. 读指标卡（首屏加载 / 页面完全加载）

页面主体在内层容器 `.md-scrollbar__wrap` 滚动，window.scrollTo 无效；指标卡需先滚到它：

```bash
bsk evaluate --session $S "(() => { const w=document.querySelector('.md-scrollbar__wrap'); w.scrollTop=560; return 'ok' })()"
bsk screenshot --session $S --out <临时目录>/cards.png
```

截图读「性能卡片」区：**首屏加载** 与 **页面完全加载** 两个数值（单位 ms）。
备用方式（文本直读）：
```bash
bsk evaluate --session $S "(() => { const t=document.body.innerText; const f=t.match(/首屏加载([\d.]+)/); const p=t.match(/页面完全加载([\d.]+)/); return f&&p ? f[1]+' | '+p[1] : 'NOT_FOUND' })()"
```

### 2b. 下载详情数据并统计慢渲染 / 慢加载

**必须先等查询完成再下载**（步骤 2a 的轮询等待即查询完成的信号），下载的就是当前查询条件下的全部结果。

1. 定位下载按钮：详情表格标题行右侧（"页面url"搜索框旁的下载图标）。evaluate 打标记再点击
   （按钮无 aria 标签，快照里找不到 ref；**evaluate 表达式用单引号包裹且避开 `!`**，见陷阱清单）：

```bash
bsk evaluate --session $S '(() => { const inp=[...document.querySelectorAll("input")].find(i=>i.placeholder==="页面url"); if(inp===undefined) return "INPUT_NOT_FOUND"; const btns=[...inp.parentElement.parentElement.querySelectorAll("button")]; const dl=btns[btns.length-1]; dl.setAttribute("data-bsk-target","dl"); return dl.className })()'
```

   返回 `INPUT_NOT_FOUND` 说明详情表格还没渲染完，等待后重试。

2. 点击下载并捕获文件（`--out` 无扩展名也行，内容是 xlsx）：

```bash
bsk download --session $S --selector "[data-bsk-target=dl]" --out <临时目录>/el-perf-export.xlsx --overwrite
```

3. 用 skill 自带脚本统计（标准库解析，无需 openpyxl）：

```bash
python <skill目录>/scripts/parse_export.py <临时目录>/el-perf-export.xlsx
# 输出: total=N slow_render=X slow_load=Y   （均为 >3000ms 严格大于统计）
```

`total` 就是页面总数（导出行数与页面分页总数天然一致）；slow_render/slow_load 直接用作分子。
导出的 xlsx 留档备查。备用方案（下载失败时）：切 100条/页后翻页读 DOM——第 2 个 table 的
tbody tr，每行 9 列，取第 2/3 列统计，总数从分页 `共 (\d+) 条` 读。

## 第三步：计算四个指标

| 表格列 | 计算 | 示例（探查时实测值） |
|---|---|---|
| 渲染耗时 (J) | 首屏加载 ms ÷ 1000，四舍五入 2 位小数 | 3245.26 → 3.25 |
| 慢渲染 (K) | `${slowRender}/${total}` | 146/228 |
| 加载耗时 (L) | 页面完全加载 ms ÷ 1000，四舍五入 2 位小数 | 1873.34 → 1.87 |
| 慢加载 (M) | `${slowLoad}/${total}` | 42/228 |

注意：total 取导出文件的 data rows 数（与页面分页总数一致）。数据实时更新，
指标卡与导出若间隔过久可能差 1~2 条，一次流程内尽快连续完成。

## 第四步：在线表格填报

表格页 URL（长，原样使用用户提供的链接或下面这条）：

```
https://mapnew5.midea.com//documentEdit.html?mc_widget_identifier=com.midea.tmtp.documentedit&fromUrl=MXPC&preview=0&fileName=TMS%25E5%2589%258D%25E7%25AB%25AF%25200%25E6%258C%2587%25E6%25A0%2587.xlsx&fileId=/chat/w/wuqm38/1785740112473/F7E0815806FAE3FF0DCE3B47550BF6D4/8B8C40BF695C6578C00976AE5E6DDBE3&fileSize=11278&locale=zh_CN&msgId=6ae932967e47b84b3172892c&sId=20240428155745900249&protocol=file5&bucketId=601901&expiredDay=1801292112
```

表格在跨域 iframe（webedit.midea.com）内，DOM evaluate 不可用；一切经**名称框 + 键盘**完成，
页面是 canvas 渲染，snapshot 看不到单元格内容。

### 4a. 定位目标行（A 列缺日期时按递增规律补写）

表头占 2 行，数据从第 3 行起；A 列日期格式 `M.D~M.D`——月/日**不补零**（`8.1~8.7`、
`8.29~9.4`），分隔符 `~`，无年份。探查基线：`10.24~10.30` 在第 16 行，新周顺延下一行（17、18…）。

1. `bsk snapshot` 找名称框（值形如 `X15` 的 textbox，记为 ref `$NB`）。
2. 逐行探测找最后非空行：fill `$NB` = `A16` → Enter → screenshot 看值 → 非空则试 `A17`、
   `A18`… 直到遇到空单元格，它的上一行即最后非空行，它本身即目标行。
3. **A 列没有目标日期时补写**：目标区间文本 = 用户给的日期格式化为 `M.D~M.D`
   （如用户说"10.31-11.6" → 写 `10.31~11.6`；月/日不补零，照表格惯例）。写入前校验递增规律：
   **上一行结束日 + 1 天 = 目标起始日**（10.30 + 1 = 10.31 ✓；跨月同理 9.30 + 1 = 10.1）。
   校验通过 → 在目标行 A 列逐键写入该文本（方式同 4b；注意文本含 `~`，press 时
   **必须用单引号**：`bsk press --session $S '~'`），再填 J~M 四格。
   校验不通过（中间缺了多周，如表格最后是 10.24~10.30 而要填 11.7~11.13）→ **停下来问用户**
   是否只补目标周，不要自行猜测。

### 4b. 写入四个单元格（J、K、L、M）

对每个单元格（顺序 J → K → L → M）：
```bash
bsk fill --session $S @$NB --value "J17"    # 跳转
bsk press --session $S Enter
bsk press --session $S "3"                   # 逐键输入数值/文本
bsk press --session $S "."
bsk press --session $S "2"
bsk press --session $S "5"
bsk press --session $S Enter                 # 确认，焦点自动移到下一行同列
```
- 慢渲染/慢加载文本含 `/`，逐键 press，但 `/` 必须加 `MSYS_NO_PATHCONV=1` 前缀（见陷阱清单）：
  `MSYS_NO_PATHCONV=1 bsk press --session $S '/'`。
- **表格自动保存**（几秒一次），写入即生效，不要试错式乱写；写错立即 Ctrl+Z 撤销重填。
- 写入四个格子后逐一读回验证：fill `$NB` = `J17` → Enter → snapshot 中名称框应显示 `J17`，
  且 snapshot 的 button 文本/ctx 里出现所写值（如 "3.25"）；不符则 Ctrl+Z 后重写。

### 4c. 汇报

向用户报告：时间范围、首屏加载/页面完全加载原始值、slowRender/total、slowLoad/total、
最终写入的四个值与单元格位置；若补写了 A 列日期，一并说明写入的文本与行号，并提醒核对表格。

## 陷阱清单

- 监控页时间范围**只能用 URL 参数设置**；时间选择器弹层手输会被组件状态覆盖。
- 监控页滚动容器是 `.md-scrollbar__wrap`，不是 window。
- 导航后数据**异步加载**：指标卡文本直读可能拿到 null，先轮询等待再读。
- **先查询完再下载**：等 2a 轮询通过后再点下载按钮，导出的才是目标时段的完整数据。
- evaluate 表达式**用单引号包裹**：双引号内的 `!`（如 `if(!inp)`）会被 bash 截断报语法错误。
  **`!` 在单引号内也会报同样的语法错误**（2026-11-06 实测），表达式一律避开 `!`，
  用 `x===undefined` 替代 `!x`。
- `/` 字符 press 时**加 `MSYS_NO_PATHCONV=1` 前缀**：Git Bash 会把以 `/` 开头的参数做 MSYS
  路径转换（`'/'` 变成 `D:/Software/Git/`，报 `unknown key`）。报错时单元格处于编辑态、
  已输入的字符悬在里面——补上 `/` 与剩余字符再 Enter 即可恢复，无脏数据。
- `~` 字符 press 时**用单引号**（`bsk press --session $S '~'`）：双引号内 `~` 会被 bash
  展开为 home 目录，报 `unknown key`，且前面已输入的字符会被后续 Enter 保存成残缺文本
  （需立即 Ctrl+Z）。
- 3000ms 边界是**严格大于**。
- 表格 iframe 跨域：名称框+键盘是唯一通道；canvas 渲染：读值靠 snapshot 的状态栏/编辑栏上下文。
- 表格自动保存：无"保存"按钮，写错必须当场 Ctrl+Z。
- 数值换算：监控页是 ms，表格填秒（÷1000 保留 2 位）。
