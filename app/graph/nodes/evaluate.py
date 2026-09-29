"""`evaluate` 评估节点（P3.1 空壳）—— 对应 §4.5 / §8。出口 `END`。

## P3 空壳的取值：**全 0 分，不是「看起来还行」的假分**

`EvalResult` 是**给人看的结论**，它会一路流到前端的评估报告里。空壳阶段填一个
像模像样的 85 分，比填 0 危险得多 —— 0 分一眼就知道没实现，85 分会让人以为
评估模块已经能用了，甚至可能被当成基线数字引用。

所以这里全 0、`passed=False`，并把「尚未实现」写进 `judge_reason`：
**报告里必须能读出这个分数为什么是 0。**

## P4/P6 要在这里做的事（§8）

- `feasibility`（程序化）：从 100 分按七类问题扣分（§8.2 第 1 张表）
- `budget_fit`（程序化）：按 `estimated_cost / budget` 的比值分档（§8.2 第 2 张表）
  —— **必须与 `user_confirm` 的 `estimated_cost` 共用同一套预算估算**（§4.4）
- `requirement_match`（LLM-as-Judge，`temperature=0`，rubric 写死在 prompt 里）
- `total_score = 0.40*req + 0.35*feas + 0.25*budget`，`passed` 还带两条独立及格线
  （`feasibility >= 60`、`budget_fit >= 50`）—— 防「需求匹配满分但行程根本走不通」

前两项都读 `plan_struct`（schema 已于 P4.0 定义，见 `app/graph/state.py`）。
预算估算要用的量从它里面取：门票 = `PlanItem.poi_id` → `POI.price` × 人数；
住宿 = `PlanStruct.nights` × `hotel_price`；市内交通 = 相邻点位通勤时长之和。

> ⚠️ **通勤时长不存在 `plan_struct` 里，由 `DistanceTool` 现算。** 存一份就是
> 第二份事实来源，而且模型估的时长和工具算的必然打架。`plan_struct` 只存
> 「顺序与时刻」，距离类的东西全部现算 —— 与 `search_url` 做成派生属性同一条理。
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import EvalResult, TravelState

logger = logging.getLogger(__name__)

# 空壳的评分结果。四个分数都是 0，理由见模块 docstring。
STUB_EVAL = EvalResult(
    requirement_match=0.0,
    feasibility=0.0,
    budget_fit=0.0,
    total_score=0.0,
    passed=False,
    dimensions_detail={"stub": True},
    judge_reason="P3 空壳：评估模块（§8）尚未实现，故各项均为 0 分，不代表行程质量。",
)


def evaluate(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """评估最终方案，写入 `eval_result`。出口 `END`。"""

    logger.info(
        "评估完成（空壳）：%.1f 分，passed=%s，用了 %d 轮机器重试 / %d 轮用户修改",
        STUB_EVAL.total_score, STUB_EVAL.passed,
        state.retry_count, state.user_revision_count,
        extra={"thread_id": config.get("configurable", {}).get("thread_id", "-")},
    )

    # `model_copy` 而不是直接把常量塞进去：EvalResult 里还有 elapsed_ms 这类
    # 每次调用都不同的量，共用一个模块级实例会让「谁改了这条记录」变成
    # 跨会话串味。复制一份是廉价的隔离。
    # elapsed_ms 目前留 0：真实耗时应当由评估节点自己测（装饰器的耗时在日志里，
    # 但那包含了状态合并的开销，与「评估花了多久」不是同一个量）。
    return {"eval_result": STUB_EVAL.model_copy(), "stage": "done"}
