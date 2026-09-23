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
| **P1 契约层** | 🟡 **进行中** —— P1.1 ✅ / P1.2 ✅ / P1.3~P1.7 ⬜ |

**P0.3 实测结论**：`deepseek-v4-flash` 返回标准 `tool_calls`，流式/并行/回灌三种路径都正常，未出现 DSML 文本泄漏 → `app/core/llm.py` 不做归一化层。**结论与模型名绑定，换模型必须重跑** `scratch/verify_tool_calls.py`。

**P1.2 实测结论**：LangGraph **确实支持** Pydantic `BaseModel` 作状态 schema + 其上的 `Annotated[list, add]` reducer（`tests/test_state.py::test_reducers_inside_real_graph` 在真实编译图上验证）。同时确认节点拿到的状态是**模型对象**（可属性访问 `state.review_comments[0].severity`），不是裸 dict。**升级 langgraph 后这个测试若变红，说明状态契约层行为变了。**

**下一步**：P1.3 `app/graph/edges.py`（三个路由纯函数，全分支单测），然后 P1.4 `tools/base.py`、P1.5/P1.6 异常与配置、P1.7 补齐 `test_edges.py` / `test_config.py`。P1 **完全不碰 LLM**，无需 API key。

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
