# 多 Agent 智能旅行规划系统

## 这是什么

可部署的 Web 应用（公网 URL，浏览器直接用）。用户用自然语言描述出行需求，系统完成：
**多轮补齐需求 → 工具取数生成行程 → 自我反思回炉 → 人在回路确认 → 输出方案 + 评估报告**。

技术栈：Python 3.11+ / LangGraph / Pydantic v2 / FastAPI + SSE / SQLite Checkpointer / 原生 HTML + TailwindCSS。
5 个 Agent 节点共享一个全局 `TravelState`，两条回流链路（机器自省环 + 用户反馈环）。

## 开工前必读

- **[docs/开发流程.md](docs/开发流程.md)** —— **实现顺序的唯一依据**。每个阶段写了「做什么 / 产出哪些文件 / 跑什么算通过 / 卡住了怎么办」。
- **[docs/方案设计.md](docs/方案设计.md)** —— 契约手册。查字段定义、阈值、公式、接口形状时翻它。

两份是配套的：流程管「现在做什么」，方案管「契约长什么样」。
**改了契约（字段、路由、阈值）必须同步改方案设计** —— 文档和代码不一致 = 文档作废。

## 当前进度

| 阶段 | 状态 |
|---|---|
| P0.1 建 venv + 钉版本 | ✅ `pyproject.toml` 已落盘（版本取自 `pip freeze`） |
| P0.2 验证 `interrupt()` 语义 | ✅ 结论已回填文档 |
| P0.3 验证 LLM 的 tool_calls 格式 | ✅ 5 场景全通过，**不需要归一化层** |
| P0.4 实测回填文档 | ✅ |
| **P1 契约层** | ✅ **完成** —— P1.1–P1.7 全部落盘，`ruff` 干净 / **98 passed** |

**P0.3 实测结论**：`deepseek-v4-flash` 返回标准 `tool_calls`，流式/并行/回灌三种路径都正常，未出现 DSML 文本泄漏 → `app/core/llm.py` 不做归一化层。**结论与模型名绑定，换模型必须重跑** `scratch/verify_tool_calls.py`。

**P1.2 实测结论**：LangGraph **确实支持** Pydantic `BaseModel` 作状态 schema + 其上的 `Annotated[list, add]` reducer（`tests/test_state.py::test_reducers_inside_real_graph` 在真实编译图上验证）。同时确认节点拿到的状态是**模型对象**（可属性访问 `state.review_comments[0].severity`），不是裸 dict。**升级 langgraph 后这个测试若变红，说明状态契约层行为变了。**

**下一步：进 P2 数据层**（顺序见 docs/开发流程.md 的 P2 段）。

**一件悬着的事 —— G5（内容收敛检测）怎么落地由你定。** 它现在实现不了：要比较「新旧 `draft_plan` 的相似度」，而 `draft_plan` 是**覆盖**语义，上一轮草稿已被覆盖，比无可比。已在 `route_after_review` 的 docstring 与 §5.2 标注。两条路：

- **加 `draft_history: Annotated[list[str], add]`** —— 跟 `review_history` 一个模式，G5 即可落地。代价是一个状态字段 + 每轮几 KB。
- **砍掉 G5** —— `MAX_REVIEW_RETRY=3` 已把上限压得很低，G5 最多再省 1–2 轮 LLM 调用。**为性能优化污染核心状态契约，是性价比最低的一类改动。**（推荐砍）

**测试全绿 ≠ 覆盖到位** —— 已实测三次，可复现：删掉 `exceptions.py` 的 `= None`、把路由的 `>=` 改成 `>`、把 `ToolResult` 的不变式判断反过来，每次都**只有新写的测试**才变红（前两次分别是「全绿」和「两条红」）。**新模块落盘必须有测试真的 import 它**，否则它的死活 pytest 不知道，输出上和「全对」长得一模一样。

P1 **完全不碰 LLM**，无需 API key。

**P1.4 定案（2026-09-28）**：接口分**两层**，吃的东西不一样 ——

| 层 | 谁调它 | `estimate_distance` 入参 |
|---|---|---|
| 工具门面 `TravelTools` | 节点（deterministic）/ LLM（agent） | `from_id, to_id`（LLM 只能给字符串） |
| 按域 Protocol | 门面 + registry | `a: POI, b: POI`（距离是经纬度纯函数） |

id → POI 的转换**只在门面一处**。三个按域 Protocol：`POIBackend`（`query_poi` + `get_opening_hours`）/ `DistanceBackend` / `HotelBackend`。**`IntercityBackend` 不预留** —— 契约未定义（§6.6），写了就是编。

顺带定下三处契约修正：**① Protocol 一律返回 `ToolResult` 而非裸值**（裸返回值表达不了失败，与硬红线 #4 冲突）；**② 门面与 Protocol 的方法名刻意不同**（`poi_query` vs `query_poi`），看调用点即可判断层级；**③ `OpeningHours` 加 `all_day`** —— 否则「全天开放」和「解析失败」撞在同一个 `open=None` 上。

**数据源决策（2026-09-28，已写入 §6.6）**：采纳 **高德 + AIGOHOTEL + 飞常准**；**不采纳 12306**（只有社区逆向实现，无公开 API/授权）与**小红书**（需人工扫码 + 个人账号 Cookie，公网部署有账号安全风险，且 UGC 无结构填不进 `POI`）。注意**飞常准要单列**：`TravelState` 当前没有任何字段装城际交通段，`plan_struct` 未定义它，`Transport` 只覆盖市内交通 —— 接它是**产品变更而非换数据源**，配置里先留 `INTERCITY_BACKEND` 开关，功能等契约定完再上。

**AIGOHOTEL / 飞常准都走路线 C（离线抓取）**，所以 MCP 客户端只是 `scripts/fetch_*.py` 的**开发期依赖**，不进 `pyproject.toml` 的 `dependencies`，运行时仍读本地 JSON、零外部调用 —— 路线 C 的可复现性不被破坏。

**P1.1 已完成**：目录骨架（`app/` 全部子包 + `tests/` + `data/` + `eval/` + `scripts/`）、`pyproject.toml`（ruff 显式钉规则集 + pytest 配置）、`.env.example`。
`ruff check .` 与 `pytest -q` 均为绿。**`scratch/` 被 ruff 整体排除**——那是一次性探针脚本，刻意写得啰嗦且宽异常兜底，与 lint 规则正面冲突。**`data/raw/` 与 `eval/reports/` 已补进 `.gitignore`**（此前缺，是硬红线 #6 的缺口）。

**P1.2 已完成**：`app/graph/state.py`（`TravelState` + `TravelRequirement` / `ReviewComment` / `EvalResult` / `ConfirmInput` + 字面量类型别名）、`tests/test_state.py`（6 个测试）。

**P1.2 顺手修补的两处文档契约缺口**（§3.1 是字段权威定义，但文档别处引用了它没有的字段）：

1. `unresolved_errors` —— 被 §4.4 载荷 / §5.2 的 G1 / §10 的 SSE 事件**引用了三处**，但 `TravelState` 里没有。**处理：做成 `@property` 派生视图，不加字段**。因为 `review_comments` 是覆盖语义、任何时刻恰好持有本轮全部意见，再存一份 error 子集就是第二份事实来源，迟早漂移。
2. `estimated_cost` —— 同样被 §4.4 / §10 引用，没有字段，且 §4.4 的输入清单里连唯一可能的来源 `plan_struct` 都没有。**处理：把 `plan_struct` 补进 §4.4 输入**，并注明它须与 `evaluate` 的 `budget_fit` 共用同一套预算估算，否则确认页的数字和评估报告的数字会对不上。

另：`TravelRequirementDelta`（§4.1 引用）**推迟到 P4.0** 定义 —— 其形状取决于 `with_structured_output` 的实测结果，而那项还没验。

## 环境

**必须用项目内的 `.venv`，不要用全局 Python。** Windows 下解释器路径：`.venv/Scripts/python.exe`

**跑 pytest 时加 `-X utf8`** —— 即 `.venv/Scripts/python.exe -X utf8 -m pytest`。
Windows 控制台默认 GBK，不加这个参数时**中文断言失败信息全是乱码**，没法读。
（`ruff check` 输出是 ASCII，不受影响。）

已装版本（不要随意升级）。**唯一权威来源是 `pyproject.toml`**（版本取自 `.venv` 的 `pip freeze`），下面是速查：

```
langgraph 1.2.7 · langgraph-checkpoint-sqlite 3.1.1 · langchain-core 1.6.4
langchain-openai 1.6.4 · pydantic 2.13.5 · fastapi 0.141.1 · uvicorn 0.53.0
ruff 0.16.8 · pytest 9.1.1 · pytest-asyncio 1.4.0
```

**pandas 故意没装** —— 推迟到 P8 数据接入阶段。

`scratch/verify_interrupt.py` 随时可重跑（不需 API key、不联网）。**升级 langgraph 版本后必须重跑** —— 这是全项目唯一一处依赖特定库版本行为的设计。

**写测试时永不把密钥写进断言。** pytest 的断言自省会**把实际值原文打印到终端** —— 失败路径就是一条泄露路径。已踩过：`assert get_settings().llm_api_key == ""` 在配有 `.env` 的机器上必然失败，于是真实 key 被打了出去。只断言与 `.env` 无关的性质（例如「不抛异常」）。

## 硬红线

完整 20 条见方案设计 §14。以下是最常踩的：

1. `interrupt()` **之前**的代码必须是纯读取、无副作用 —— 恢复时节点从头重跑，副作用会执行两次
2. 判定图是否暂停用 `snapshot.interrupts`，**不要用 `snapshot.next`**（二次 interrupt 时返回 `()`，与「图跑完」无法区分）
3. `self_review` 节点**不可被任何提速优化旁路**，有结构性单测守着
4. 工具**永不抛异常**，一律返回 `ToolResult(ok=False)`
5. Prompt 放 `app/graph/prompts/*.md`，**不塞进 Python 字符串**
6. 日志脱敏；`.env`、`data/raw/`、`data/app.db` 永不入库

## 明确不做

桌面 exe / 原生 App · 向量数据库 / RAG · Redis · LangServe · 前端框架（Vue/React） · 完整鉴权体系。

理由是这些对这个项目的**展示价值**为零或为负。取舍标准是「能不能体现 Agent 工程能力」，不是「能不能上线」。

## 参考决策

`docs/方案设计.md` §6.6 是数据源决策记录 —— 解释了为什么走「高德离线抓取 → 本地 JSON」而不是 MCP 实时调用（核心是可复现性，评估基线依赖它）。**改动数据层之前先读那一节。**
