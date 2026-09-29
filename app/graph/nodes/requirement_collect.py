"""`requirement_collect` 需求收集节点（P3.1 空壳）—— 对应 §4.1。

P3 阶段**不调用 LLM**：本文件只保留「图能不能走通」所依赖的那几行确定性逻辑
（必填项判定、追问轮数自增、`user_query` 消费后清空）。P4 会把 LLM 的结构化抽取
接进来，替换掉的只有 `_extract_delta` 这一处 —— 其余判定与 §4.1 的契约无关，不会被推翻。

> ⚠️ **P4 接 LLM 时走 `build_llm("cheap").structured(TravelRequirementDelta)`，
> 不要在本文件里写 `llm.with_structured_output(...)`。** 裸写法在 DeepSeek 上
> 100% 400（默认 method 是 `json_schema`），且要在每个节点各踩一遍 ——
> 两条纪律都封在 `app/core/llm.py`（§12.2），
> `tests/test_llm.py::test_nodes_never_build_their_own_chat_openai` 守着这件事。

## 「空壳」不等于「什么都不做」

开发流程 P3.1 写的是「每个 `return {}` + 一行埋点」。**照字面做，P3 的四条路径
一条都走不出来**：`missing_fields` 永远是空的，图会一路冲进生成节点。
真正该空的是**需要 LLM 的那部分**，而「必填项缺不缺」「问了几轮了」这类判断
本来就是程序化的（§4.1 把它们写在节点里而不是 prompt 里，正是这个原因）。

所以本文件的取舍是：**凡是「图怎么走」依赖的，留着；凡是「答得对不对」依赖的，
留空并标 TODO。** P4 填肉时不会被本文件判定的东西挡住。

## 尚未实现（P4）

- **`_extract_delta`**：LLM 结构化抽取（§4.1 要点 1）。P3 直接沿用已有需求。
- **追问话术**：§4.1 要点 3 要求「由 LLM 生成自然中文问句，不要模板化」。
  P3 用的是模板（`_QUESTION_TEMPLATE`），**这是刻意的欠债**，P4 必须换掉 ——
  模板问句是「请提供 days 字段」那类生硬表达的来源。
- **日期归一化**（要点 4）与**境外拦截**（要点 5）。
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import TravelRequirement, TravelState

logger = logging.getLogger(__name__)

# 必填三项（§4.1 要点 2）：缺了它们行程根本排不出来。
# 其余字段（origin / start_date / budget / preferences / pace / transport）
# 缺失**不阻塞** —— 由生成节点按默认值兜底并在行程里标注假设。
REQUIRED_FIELDS: tuple[str, ...] = ("destination", "days", "travelers")

# 字段名 → 中文名。追问话术里直接报英文字段名（「请提供 days」）是很典型的
# 机器腔，P4 换成 LLM 生成问句后这张表仍有用（渲染给 prompt 看）。
FIELD_LABELS: dict[str, str] = {
    "destination": "目的地城市",
    "days": "行程天数",
    "travelers": "出行人数",
    "origin": "出发城市",
    "start_date": "出发日期",
    "budget": "预算",
}

# ⚠ P4 必须替换成 LLM 生成的自然问句（§4.1 要点 3）。留在 P3 只是为了让
# 路径④「追问一轮」走得出来 —— 追问内容本身不是 P3 要验的东西。
_QUESTION_TEMPLATE = "为了帮你排出合适的行程，还想确认一下：{fields}？"


def missing_required(requirement: TravelRequirement) -> list[str]:
    """列出还没填的必填项，**顺序固定**（按 `REQUIRED_FIELDS`，不是 set 序）。

    顺序固定是为了可断言：话术里字段的先后每次都一样，测试才能直接比字符串。
    用 set 的话同样的输入会产出不同的问句，而那是纯随机噪声。
    """
    return [name for name in REQUIRED_FIELDS if getattr(requirement, name) is None]


def build_question(missing: list[str]) -> str:
    """把缺失字段拼成一句追问。**P4 换成 LLM，此处为占位。**"""
    labels = "、".join(FIELD_LABELS.get(name, name) for name in missing)
    return _QUESTION_TEMPLATE.format(fields=labels)


def _extract_delta(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """本轮从 `user_query` 里抽出的字段增量 —— **P3 返回空增量**。

    P4 在这里调用便宜档（`build_llm("cheap").structured(...)`，见模块 docstring
    的警告），返回**只含新增/修改字段**的 delta。用 delta 而不是全量，是为了避免
    LLM 把已经收集好的字段顺手抹掉（§4.1 要点 1）。

    ⚠️ 合并用 `model_dump(exclude_unset=True)`，**不要用 `exclude_none=True`**：
    delta 模型的字段全是 `| None`（P4.0 实测 Q2 的决定性结论），逐个判断
    「None 是没提到还是明确置空」会退化成猜。`exclude_unset` 能区分这两者 ——
    而它成立的前提正是「所有字段可空」，q.v. `TravelRequirementDelta`。

    返回空 dict 的含义是「这一轮什么都没抽到」—— 它与「抽到但值为 null」
    不同：后者应当**覆盖**掉旧值（用户说了「不设预算」，那就是没有预算）。
    """
    return {}


def requirement_collect(
    state: TravelState, config: RunnableConfig
) -> dict[str, Any]:
    """需求收集。出口见 §5.1 的 `route_after_collect`（G3 在路由里）。"""

    delta = _extract_delta(state, config)
    # 增量合并：本轮抽到的覆盖旧值，未抽到的保留（§3.2）。
    requirement = state.user_requirement.model_copy(update=delta)

    missing = missing_required(requirement)
    question = build_question(missing) if missing else None

    # `ask_round` 每进一次本节点就 +1（§4.1 输出清单）。
    # **不是「问了几次」，是「收集了几轮」** —— 最后一轮需求齐了也会 +1。
    # 两种计法都能工作，但 G3 的判据 `ask_round >= MAX_ASK_ROUNDS` 只在
    # 「每轮都 +1」下与「最多问 MAX 次」等价（路由看的是自增**之后**的值）。
    ask_round = state.ask_round + 1

    logger.info(
        "需求收集第 %d 轮：缺失 %s",
        ask_round, missing or "无",
        extra={"thread_id": config.get("configurable", {}).get("thread_id", "-")},
    )

    return {
        "user_requirement": requirement,
        "missing_fields": missing,
        "pending_question": question,
        "user_query": "",          # 消费掉本轮输入（§3.2：节点消费后清空）
        "ask_round": ask_round,
        # 还在追问 → collecting；问齐了 → 交给生成节点，状态进入 planning
        "stage": "collecting" if missing else "planning",
    }
