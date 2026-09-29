"""`user_confirm` 用户确认节点（P3.1）—— 对应 §4.4。**本节点不调用 LLM。**

与其他四个节点不同，本文件在 P3 就已经是**近乎完整**的实现：§4.4 的代码在
方案设计里是逐行给出的，而它依赖的全是确定性逻辑（读状态 → `interrupt()` →
按 resume 值分支）。P4 要补的只有 `estimated_cost` 一项（依赖 §8.2 的预算估算器）。

## 硬红线 #1：`interrupt()` **之前**的代码必须是纯读取

`interrupt()` 恢复时**节点函数从头重新执行一遍**，只有 `interrupt()` 的返回值
变成 resume 值。所以暂停点之前的任何副作用都会执行两次 —— 写日志计数、调 LLM、
改外部状态，全都不能放。

本文件的做法：`build_confirm_payload()` 是纯函数（只读状态、拼 dict），
**所有 `logger` 调用都放在 `interrupt()` 之后**。这与「日志是只读的」那种辩解
不同 —— 日志会重复打，而「确认页被渲染了两次」这种噪声在排查时会真的误导人。

## 异常输入不是 500，而是**再问一次**

resume 值校验失败，或 `action="revise"` 却没带 `feedback` → 不抛异常、不返回
错误状态，而是带上 `input_error` **再次 `interrupt()`**，让前端重新弹提示。

依据 `scratch/verify_interrupt.py` 场景 B 的实测：同一个节点内连续 `interrupt()`
可行，且**恢复时第 1 个 `interrupt()` 拿回的仍是上次那个值**，所以「校验失败 →
再问 → 新值」这条链是确定性的，不会串。

## 尚未实现

- **`estimated_cost`**（§4.4 载荷字段）：依赖 §8.2 的预算估算器，P6 落盘。
  它与 `evaluate` 的 `budget_fit` 必须**共用同一套计算**，否则确认页的数字和
  评估报告的数字会对不上（§4.4 的原话）。此处先置 `None`，不编。
- **`AUTO_APPROVE`**（§12：批量评估时自动通过 HITL）：P9 接批量评估时再定。
  它与 G2（用户修改上限）如何交互没有定义，现在写就是编。
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import ValidationError

from app.graph.state import ConfirmInput, TravelState

logger = logging.getLogger(__name__)


def build_confirm_payload(state: TravelState) -> dict[str, Any]:
    """拼出给前端的确认载荷（§4.4）。**纯函数：只读状态，可安全重跑。**

    载荷里的值都转成**可 JSON 序列化**的形态（`model_dump()`）——
    它会经由 SSE 送到浏览器，也会被写进 checkpoint。塞 Pydantic 模型进去，
    序列化时会在两个不同的地方各报一次错。
    """
    return {
        "type": "plan_confirm",
        "plan": state.draft_plan,
        # 含未解决的 warn —— 用户有权看到全部意见，不只是阻塞项
        "review_comments": [c.model_dump() for c in state.review_comments],
        # 达 G1 上限时的降级放行标记（§5.2：降级必须透明）
        "unresolved_errors": [c.model_dump() for c in state.unresolved_errors],
        "estimated_cost": None,        # TODO(P6)：§8.2 的预算估算器
        "budget": state.user_requirement.budget,
        "revision_count": state.user_revision_count,
    }


def invalid_reason(raw: Any) -> str | None:
    """校验 resume 值，返回**给用户看的中文原因**；合法则返回 `None`。

    拆成独立函数而不是内联在节点里：它是本节点唯一一段「可以出错但不该崩」的
    逻辑，独立出来才能单独测（节点本体必须跑在真实图上，因为 `interrupt()`
    只有在图里才有意义）。
    """
    try:
        decision = ConfirmInput.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = str(first["loc"][0]) if first["loc"] else "输入"
        return f"{field} 不合法：{first['msg']}"

    if decision.action == "revise" and not decision.feedback.strip():
        # 空反馈的 revise 等于让生成节点「凭感觉再写一版」——
        # 用户以为自己的意见被采纳了，而实际上一个字都没传下去。
        return "选择「提修改意见」时必须填写具体意见，否则行程没有可改的依据。"

    return None


def user_confirm(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """暂停等用户决定。出口见 §5.1 的 `route_after_confirm`（G2 不在此处）。"""

    payload = build_confirm_payload(state)     # 纯读取，重跑安全

    while True:
        raw = interrupt(payload)                # ← 暂停点
        reason = invalid_reason(raw)
        if reason is None:
            break
        # 带上一次的错误**再问一遍**。合法输入才会走到下面的状态写入。
        payload = {**payload, "input_error": reason}

    decision = ConfirmInput.model_validate(raw)

    # ↓ 以下只在 resume 之后执行，重复执行无妨（都在 interrupt() 之后）
    logger.info(
        "用户决定：%s（历史修改 %d 次）",
        decision.action,
        state.user_revision_count,
        extra={"thread_id": config.get("configurable", {}).get("thread_id", "-")},
    )

    if decision.action == "approve":
        return {
            "user_confirmed": True,
            "final_plan": state.draft_plan,
            "user_feedback": "",
            "stage": "evaluating",
        }

    return {
        "user_confirmed": False,
        "user_feedback": decision.feedback,
        "user_revision_count": state.user_revision_count + 1,
        # ⚠ **retry_count 清零**是 §2.3 的契约：用户介入后，机器重新获得完整的
        # 自省预算。不清零的话，用户改了三轮之后机器就再没有回炉机会了 ——
        # 两条链路的计数**互不污染**正是这么落地的。
        "retry_count": 0,
        # 用户接手后，机器那一轮的意见作废。（`plan_generate` 也会清一次；
        # 这里清是因为「作废」这件事在语义上发生在此刻，不是等到下次生成。）
        "review_comments": [],
        "stage": "planning",
    }
