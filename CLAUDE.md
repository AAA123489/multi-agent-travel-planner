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
| **P1 契约层** | ✅ **完成** —— P1.1–P1.7 全部落盘 |
| **P2 数据与工具层** | ✅ **完成** —— 30 条 POI + 三个 mock backend + 四个工具 + registry 分发 |
| **P3 图流转打通** | ✅ **完成（★ 里程碑 M1）** —— 建图 + 5 个节点空壳 + 装饰器，四条路径全走通，`ruff` 干净 / **187 passed** |
| **P4.0 结构化输出实测** | ✅ **完成** —— ⚠️ **照字面写 `with_structured_output` 会 100% 400**，两处修正已回填 §4.1 / §4.5 / §12.2 |

**P0.3 实测结论**：`deepseek-v4-flash` 返回标准 `tool_calls`，流式/并行/回灌三种路径都正常，未出现 DSML 文本泄漏 → `app/core/llm.py` 不做归一化层。**结论与模型名绑定，换模型必须重跑** `scratch/verify_tool_calls.py`。

**P4.0 实测结论（2026-09-29）—— 三条都会静默或响亮地失败，务必先读**：

1. **`llm.with_structured_output(Schema)` 在 DeepSeek 上 100% 400。** 它的默认 method 是 `json_schema`（走 `response_format`），服务端不认。**必须显式 `method="function_calling"`**。「不传」与「显式传 `json_schema`」是同一个失败 —— 代码看着没写错，只是没写全。§4.1 / §8.2 的原文照抄会直接崩。
2. **`function_calling` 还必须先关 thinking**，否则 `Thinking mode does not support this tool_choice`（`deepseek-v4-flash` 默认开着 thinking，而它不接受被强制的 `tool_choice`）。
3. **关 thinking 的参数形状只有一个是对的**：`extra_body={"thinking": {"type": "disabled"}}`。`{"enable_thinking": False}` 与 `{"chat_template_kwargs": {"enable_thinking": False}}` **被服务端收下但不生效，且不报错** —— 直接对话两种都成功，看不出来。**唯一试金石是拿 `function_calling` 去调。** 与 P2 那个「Protocol 继承让漏实现静默返回 None」同一型：**不报错的失效比报错的失效难查一个量级。**

**P4.0 顺带验出的两条**：

- **`exclude_unset` 可以直接用，delta 模式前提成立。** 而且 prompt 里写不写「没提到的字段不要输出」**没有差别**（两个变体各 10 次，都 10/10 省略）。机制在 schema 上：全字段可空 → JSON Schema 的 `required` 为空 → 模型天然只填想填的。**推论：`TravelRequirementDelta` 必须全字段 `| None` —— 这是 delta 语义的载体，不是风格选择。**
- **`start_date` 没有归一化**：输入「10月1号」返回的就是 `'10月1号'`。§4.1 实现要点 4 那句「由 LLM 结合注入的 `today` 换算」在现有 prompt 下不成立，P4.1 要补正反例。
- 「禁止编造」**有效**：输入「想去成都玩」，5/5 只抽到 `destination`，没编 `days`/`travelers`/`origin`/`start_date`。

**P4.0 对 `app/core/llm.py`（P4.0-C，还没写）的硬要求**：「显式 `method="function_calling"`」和「关 thinking」必须**封在工厂一处**，节点代码不许自己建 `ChatOpenAI` —— 否则这两件事会在每个节点各踩一遍，而且第二件踩了**不报错**。

**尚未验**：主力档（`plan_generate`/`self_review`）关不关 thinking。留给 P4.2 —— thinking ON 时 `with_structured_output(fc)` 不可用、但 `bind_tools(tool_choice="auto")` 可用（已实测），所以保留推理能力的路是通的。**不能因为省 token 就默认把主力档的推理关掉。**

**P1.2 实测结论**：LangGraph **确实支持** Pydantic `BaseModel` 作状态 schema + 其上的 `Annotated[list, add]` reducer（`tests/test_state.py::test_reducers_inside_real_graph` 在真实编译图上验证）。同时确认节点拿到的状态是**模型对象**（可属性访问 `state.review_comments[0].severity`），不是裸 dict。**升级 langgraph 后这个测试若变红，说明状态契约层行为变了。**

**P3 已完成（2026-09-29）** —— ★ 里程碑 M1。

| 任务 | 产出 |
|---|---|
| P3.1 | `app/graph/nodes/{requirement_collect,plan_generate,self_review,user_confirm,evaluate}.py` —— 五个空壳（**空壳≠什么都不做**，见下） |
| P3.2 | `app/graph/nodes/decorators.py` —— `@traced_node`：计时、日志、`node_trace` 追加、异常兜底 |
| P3.3 | `app/graph/builder.py` —— 建图 + `build_graph(checkpointer, overrides=)` + `graph_config()` |
| P3.4 | `scratch/walk_graph.py` —— 打印拓扑 + 实际走完四条路径 |
| P3.5 | `tests/test_graph.py` —— **24 条**（拓扑 3 / 四条路径 4 / 不可旁路 2 / 装饰器 5 / 装配 5 / 校验 5） |
| — | `scratch/verify_traced_node.py`（新探针）+ `scratch/mutate_p3.py`（变异测试，可重跑） |

**四条路径全部符合预期**，其中路径③（用户拒绝一次）实测轨迹：

```
rc#1 → pg#1 → sr#1 → pg#2 → sr#2 → uc#1 → pg#3 → sr#3 → pg#4 → sr#4 → uc#2 → eval#1
```

用户修改后机器**重新拿到完整自省预算**，所以第二个循环又「先失败一次再通过」。`retry_count` 与 `user_revision_count` 全程互不污染。

**P3 实测踩出来的三个坑**（都写进文档了）：

1. **异常兜底会吞掉 `interrupt()`** —— `GraphInterrupt` 是 `Exception` 的子类，最自然的 `except Exception` 把它当成节点故障。后果不是「暂停失败」而是**暂停被换成死循环**：节点返回错误状态 → `user_confirmed` 仍是 False → 路由判成「用户要改」送回生成节点 → 转一圈再来 → 最后 `GraphRecursionError`。修法是 `except GraphBubbleUp: raise` 放在前面。**变异① 实测：删掉这一行，8 条测试同时变红。**
2. **`snapshot.values` 不是完整状态，只含被写过的通道。** 「一次通过」路径里没人写过 `retry_count`，`values["retry_count"]` 直接 `KeyError`，而同名默认值是 0。**P5 的 API 层读状态一律先过 `TravelState.model_validate(snapshot.values)`。**
3. **Pydantic 模型进 checkpoint 触发 msgpack 反序列化警告**，且那是一条**安全边界**（宽松模式可被篡改的 checkpoint 触发任意代码执行）。已实测出配方：`JsonPlusSerializer(allowed_msgpack_modules=[("app.graph.state", ...)])` 传给 `serde=`，警告归零。**P5 建 `services/checkpoint.py` 时照抄。**

**新增一处 P4 阻塞项：`plan_struct` 的 schema 全文未定义**（§4.2 已记录）。它是 `plan_generate` 的第二输出段、`self_review` 全部程序化校验、§8.2 的 `estimated_cost` 三处的前置，而 §3.1 只把它声明成裸 `dict`。**P3 刻意不编占位形状** —— 编出来要么被推翻、要么被人当成契约照抄。

**P3 定下的两处接口形状**：`build_graph(checkpointer, *, overrides={节点名: 替身})` 与 `graph_config(session_id)`。`overrides` **只换节点行为、不换图的连线**（`test_overrides_do_not_change_topology` 守着），所以结构断言不受它影响；`graph_config` 把 `thread_id` 与 `recursion_limit`（G4）绑在一处，避免漏写的那处静默退回默认值 25。

**默认空壳审核「永远不通过」是刻意的** —— 它让 G1 降级放行成为**默认可见**的行为，而不是要靠注入才看得到的分支。想看通过的那条路用 `overrides` 换行为（`tests/test_graph.py` 的 `review_failing_times()` 与 `scratch/walk_graph.py` 的 `review_passing_after()`）。

**下一步：P4.0-B / P4.0-C**。P4.0 的两个待验项里，**第一个（`with_structured_output` 实测）已验完**（见上）；**第二个 `plan_struct` 的 schema 仍待定**，而且它的字段可以从下游倒推（§4.3 七条校验 + §8.2 预算估算各自需要什么），不必凭空发明。之后才能写 `app/core/llm.py`（P4.0-C）。顺序见 docs/开发流程.md 的 P4 段。

**P2 已完成（2026-09-29）**：

| 任务 | 产出 |
|---|---|
| P2.1 | `data/poi_clean.json` —— 成都 15 + 杭州 15，真实景点名与经纬度 |
| P2.2 | `app/tools/backends/mock_backend.py` —— **三个类**（POI / 距离 / 酒店），按域各一个（§6.4） |
| P2.3 | `app/tools/{poi_query,opening_hours,distance,hotel_price}.py` —— 每个含**纯函数**（铁律②）+ 门面用的工具类 |
| P2.4 | `app/tools/registry.py` —— 按域选后端（经 `effective_backend`）→ 组装成 `TravelTools` 门面 |
| P2.5 | `tests/test_tools.py` 44 → **108 条** |

**新增了一个 §13 没列的文件：`app/tools/poi_store.py`** —— POI 内存索引 + 逐条容错加载。三域共用一份（不重复载入）。已在 §13 与 §6.5 补记。

**P2 顺手定下的四处契约**（已写入 §6.3 / §6.5 / §9.1）：

1. **`POIStore` 是 P2 版的「归一化层」**。§6.5 要求每个 backend 有 `_normalize_xxx` 纯函数对付「上游返回壳」—— 但 mock 读的是自家预处理产物，没有返回壳这回事。面对同一个敌人的等价物是**逐条容错**：坏行跳过 + warning，其余照常可用。**「空」和「坏」必须分开**（空 store = 这个城市没景点；全坏 = 事故），不能都表现成空列表。真实抓包 payload 那份工作留给 P8 接高德时做。
2. **实现方不要继承那四个 Protocol** —— **已实测**：显式继承会让漏实现的方法**静默返回 `None`**（`class Impl(POIBackend): pass` 能实例化，调用得 `None` 不报错）。继承反而削弱了检查。改用结构化实现 + `isinstance` 断言 + 「方法必须在自己 `__dict__` 里」的断言（`test_mock_backend_defines_protocol_methods_itself`）。
3. **`search_url` 做成派生属性，不加字段** —— 铁律③要求 URL 指向真实可点开的页面，但 §9.1 的 schema 没有 `url` 字段。它是 `name` + `city` 的函数，存一份就是第二份事实来源。与 `unresolved_errors` 同一处理方式。
4. **酒店兜底的「全局经验值」补上了具体数字**（§6.3 原本只说「用全局经验值兜底」却从没定义）：budget 250 / comfort 450 / luxury 900 元/晚，**写死的估计不是实测**，故必须同时置 `is_approx=True`。

**一处要留意的地方**：P2 验收的「武侯祠 → 宽窄巷子 3~6 km」**区间偏宽**——实测把 `ROAD_FACTOR` 从 1.3 改成 1.0（距离缩小 23%）时它**依然是绿的**（3.17 km 仍落在区间内）。真正守住这个系数的是 `test_estimate_applies_road_factor_and_is_always_estimated`（它直接对 Haversine × 1.3 断言）。**验收标准是给人看的粗筛，不是回归网**；两者都要有。

**G5（内容收敛检测）已砍（2026-09-28 定案）。** 它要比较「新旧 `draft_plan` 的相似度」，而 `draft_plan` 是**覆盖**语义 —— 上一轮草稿已被覆盖，比无可比；落地得先给 `TravelState` 加累加语义的 `draft_history`。而它能买到的收益，只是「在 `MAX_REVIEW_RETRY=3` 之上再省 1–2 轮 LLM 调用」。

**砍的理由是性价比，不是难度**：G1 已把这个环保在 3 圈内，G5 只是把短环再截短一点 —— **为一点性能优化去污染核心状态契约，是本项目里性价比最低的一类改动。**

于是闸门编号 **G1~G4 是跳号的**（没有 G5），**跳号是刻意的**，它是一处「这里删过一个闸门」的记录。看到编号不连续就想补一个回来 = 把已做的决策推翻一遍。理由写在 [方案设计.md](docs/方案设计.md) §5.2 与 `route_after_review` 的 docstring 两处。

**测试全绿 ≠ 覆盖到位** —— 已实测十六次，可复现。P1 的三次：删掉 `exceptions.py` 的 `= None`、把路由的 `>=` 改成 `>`、把 `ToolResult` 的不变式判断反过来，每次都**只有新写的测试**才变红（前两次分别是「全绿」和「两条红」）。P2 的四次（全部变红）：把 `MockPOIBackend.query_poi` 改名、酒店阈值 `<` 改成 `<=`、`ROAD_FACTOR` 1.3 改成 1.0、坏行容错改成整批失败。P3 的九次（**9/9 全被抓住**，脚本 `scratch/mutate_p3.py` 可重跑）：吞掉 interrupt、忘记清零 `retry_count`、把「生成→审核」改成「生成→确认」、空壳审核改成永远通过、`ask` 分支接错、漏掉 `recursion_limit`、`node_trace` 序号写死为 1、审核不自增 `retry_count`、`revise` 不校验空意见。

**新模块落盘必须有测试真的 import 它**，否则它的死活 pytest 不知道，输出上和「全对」长得一模一样。**变异测试是唯一能证明这件事的手段** —— 改一行、跑一遍、看红不红，比读覆盖率数字可靠。

P1 / P2 / P3 **完全不碰 LLM**，无需 API key（P3 的 187 条测试 0.6 秒跑完、不联网）。

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
