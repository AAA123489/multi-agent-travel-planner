"""`plan_generate` 行程生成节点（P3.1 空壳）—— 对应 §4.2。

P3 阶段**不调用 LLM、不调用工具**：本文件产出一段写明了是占位的 Markdown 草稿，
让下游节点有东西可读、让 `node_trace` 里能看到「生成被调了几轮」。

## P4 要在这里做的事（按 §4.2 的执行流程）

1. 取数：`poi_query` → `opening_hours` → `distance`（候选集内两两）→ `hotel_price`
2. 组装上下文 → 主力模型生成 Markdown 草稿
3. 同时输出机器可读的 `plan_struct`

**工具由节点代码确定性调用，不让 LLM 自主选工具**（§4.2 的核心工程决策）：
入参本来就是确定性的（城市名、POI 名），交给 LLM 选只会带来编造 POI 名、
重复调用、参数格式错三类故障。把不确定性关在「表达」环节，不关在「取数」环节。

## ⚠ P4 的阻塞项：`plan_struct` 的 schema 全文未定义

§3.1 把它声明成裸 `dict`（不校验），§4.2 只说了句「每天点位顺序」，
而 §8.2 的预算估算器要从它里面读出「门票单价 × 人数」「住宿晚数」这些量。
**具体字段（`days[].items[].poi_id` / `start` / `end` / `kind`？）一处也没写。**

所以本节点**刻意不写 `plan_struct`** —— 写一个占位形状出来，等 P4 定稿时
它要么被推翻、要么更糟：被人当成契约照抄。宁可这里空着，把缺口摆在明处。
（同 P1.2 对 `TravelRequirementDelta` 的处理。）

## 覆盖语义的两处清空

返回值里的 `review_comments: []` 与 `user_feedback: ""` **不是多余**：
它们是覆盖语义字段（§3.3），返回空列表/空串就是「清空」。这一步必须做 ——
否则第 2 轮的审核意见里会混着第 1 轮已修好的问题，LLM 回头去修已经没问题的
部分，改 A 忘 B、改 B 又坏 A，永远收敛不了（§3.3 的原话）。
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import TravelState

logger = logging.getLogger(__name__)

# ⚠ 占位草稿。**刻意把「这是占位」写在正文里**，而不是留一句像模像样的假行程：
# 假行程会让人以为 P3 已经能生成方案了，而这个字符串一路会流到前端确认页。
STUB_DRAFT_PLAN = (
    "> ⚠ 这是 P3 阶段的**占位草稿**，不是真实行程。\n"
    "> 行程生成节点（§4.2）要到 P4 才接入 LLM 与工具层。\n"
    "> 当前存在的意义：让 self_review / user_confirm / evaluate 有东西可读，\n"
    "> 并把图流转跑到通。\n"
)


def plan_generate(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """行程生成。出口是无条件边 → `self_review`（§4.2）。"""

    # 轮次不在这里记 —— 装饰器已经写了 `plan_generate#N`（§3.2）。
    # 节点里再数一遍是同一件事的第二份实现，两处迟早对不上。
    # 这里只记「本轮的输入是什么」：决定生成结果的正是这两个量。
    logger.info(
        "生成输入：retry=%d，用户反馈=%s",
        state.retry_count,
        "有" if state.user_feedback else "无",
        extra={"thread_id": config.get("configurable", {}).get("thread_id", "-")},
    )

    return {
        "draft_plan": STUB_DRAFT_PLAN,
        # plan_struct 留空，理由见模块 docstring 的「P4 的阻塞项」
        "review_comments": [],   # 覆盖语义：清空上一轮意见
        "user_feedback": "",     # 覆盖语义：反馈已被本轮消费
        "stage": "reviewing",
    }
