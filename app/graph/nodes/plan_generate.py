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

## `plan_struct` 的 schema 已定义（P4.0，2026-09-29）

原先的阻塞项已解除：`PlanStruct` / `PlanDay` / `PlanItem` 落在
`app/graph/state.py`，字段从下游消费者倒推而来（§4.2 有字段溯源表）。
`TravelState.plan_struct` 的类型相应从裸 `dict` 收紧为 `PlanStruct | None`。

## 生成流程里有一处「代码回填」，别写漏（§4.2）

LLM 只输出 `PlanItem.name`，**`poi_id` 由本节点的代码回填**：

```
模型给出 days[].items[].name
  → 拿候选清单回查（精确匹配 → 归一化匹配）
  → 命中：填上 poi_id
  → 未命中：poi_id 留 None（= 幻觉信号，交给 self_review 判）
```

**回填必须在这里做，不能推迟到 self_review。** 两个下游都要用它（§4.3 的
幻觉/折返校验、§8.2 的门票单价），在两处各查一次就是两份实现；而且
`plan_struct` 存了 `poi_id` 之后是**自洽**的 —— `evaluate` 不必再持有候选清单。

> ⚠️ **回查失败会触发回炉，可能形成环。** 模型把「宽窄巷子」写成「宽窄巷子景区」，
> 回查失败 → self_review 报幻觉 → 回炉 → 模型**大概率还会这么写**。所以 §4.2 的
> prompt 必须把候选名字**逐字列出**并要求照抄，且回查要做归一化（去「景区」「公园」
> 这类后缀）。真绕不出来时由 G1 兜底降级放行 —— 而不是在这里放宽判据
> （放宽就再也抓不到真幻觉了）。

## 覆盖语义的两处清空

返回值里的 `review_comments: []` 与 `user_feedback: ""` **不是多余**：
它们是覆盖语义字段（§3.3），返回空列表/空串就是「清空」。这一步必须做 ——
否则第 2 轮的审核意见里会混着第 1 轮已修好的问题，LLM 回头去修已经没问题的
部分，改 A 忘 B、改 B 又坏 A，永远收敛不了（§3.3 的原话）。
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import PlanDay, PlanItem, PlanStruct, TravelState

logger = logging.getLogger(__name__)

# ⚠ 占位草稿。**刻意把「这是占位」写在正文里**，而不是留一句像模像样的假行程：
# 假行程会让人以为 P3 已经能生成方案了，而这个字符串一路会流到前端确认页。
STUB_DRAFT_PLAN = (
    "> ⚠ 这是 P3 阶段的**占位草稿**，不是真实行程。\n"
    "> 行程生成节点（§4.2）要到 P4 才接入 LLM 与工具层。\n"
    "> 当前存在的意义：让 self_review / user_confirm / evaluate 有东西可读，\n"
    "> 并把图流转跑到通。\n"
)

# 占位 `plan_struct`。它**结构合法、内容不合格**，这是故意的 —— 两件事分开看：
#
#   - 结构合法，是为了让 `PlanStruct` 的约束（`HH:MM`、`kind` 字面量、frozen）
#     在 P3 的每条路径里都被真实构造一遍。全字段留空的话，这些约束直到 P4.2
#     才有第一行代码碰它们 —— 而那时出问题会被误当成「接入 LLM 引入的」。
#   - 内容不合格（`poi_id=None`、名字是占位串），是为了让 P4.3 的幻觉校验落地时
#     **正确地把它判为幻觉**。让空壳在真校验下表现为「有问题」，而不是伪装成
#     一份合格行程 —— 与 `evaluate` 空壳给 0 分而不是 85 分同一条姿态。
STUB_PLAN_STRUCT = PlanStruct(
    days=[
        PlanDay(
            day_index=0,
            items=[
                PlanItem(kind="attraction", name="（P4 占位景点）", start="09:00", end="11:00"),
                PlanItem(kind="meal", name="（P4 占位用餐）", start="12:00", end="13:00"),
            ],
        )
    ]
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
        "plan_struct": STUB_PLAN_STRUCT,   # 结构合法、内容不合格，见常量处说明
        "review_comments": [],   # 覆盖语义：清空上一轮意见
        "user_feedback": "",     # 覆盖语义：反馈已被本轮消费
        "stage": "reviewing",
    }
