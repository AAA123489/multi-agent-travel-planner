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
| P0.1 建 venv + 钉版本 | ✅ |
| P0.2 验证 `interrupt()` 语义 | ✅ 结论已回填文档 |
| **P0.3 验证 LLM 的 tool_calls 格式** | ⬜ **下一步做这个** |
| P1 契约层 | ⬜ |

**下一步**：写 `scratch/verify_tool_calls.py`，确认 DeepSeek/GLM 返回的是标准 `tool_calls` 字段，而不是文本形式的 DSML（详见开发流程 P0.3）。这是唯一需要 API key 的 P0 任务。

## 环境

**必须用项目内的 `.venv`，不要用全局 Python。** Windows 下解释器路径：`.venv/Scripts/python.exe`

已装版本（不要随意升级）：

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
