"""条件边路由函数（P1.3）—— 对应 docs/方案设计.md §5.1 / §5.2。

三个纯函数，各自读状态的几个字段、回报一个字符串，由 LangGraph 据此决定下一步去哪个
节点。**这里是全项目唯一一处「哪个词派哪个节点」可以被单测的地方** —— 这段逻辑写在
代码里而不是 prompt 里，正是状态图相对手写编排的结构性优势（§5.1）：它不会漂移。

三条纪律：

1. **只读、不写、不做 IO**。路由函数不碰状态、不自增计数器 —— 计数器归节点管，
   路由只读结果。混在一起会让「为什么又回炉了一轮」无从追查。
2. **阈值从 Settings 取，不硬编码**。每个带闸门的函数留一个可注入的 `max_*` 参数，
   默认走 `get_settings()`：**生产不传、测试不碰环境变量**。
3. **闸门 G1 / G3 在这里落地**（「该不该再循环一轮」天然是控制流判断）。
   **G2 不在这里** —— 它的处置是「前端禁用『继续修改』」，路由看不见 UI，见
   `route_after_confirm` 的说明。
"""

from typing import Literal

from app.core.config import get_settings
from app.graph.state import TravelState

# ---------------------------------------------------------------------------
# 返回值字面量 —— 与 §5.1 表格右列一一对应
# ---------------------------------------------------------------------------

RouteAfterCollect = Literal["ask", "plan"]
RouteAfterReview = Literal["retry", "confirm"]
RouteAfterConfirm = Literal["revise", "eval"]


def route_after_collect(
    state: TravelState,
    *,
    max_ask_rounds: int | None = None,
) -> RouteAfterCollect:
    """需求收集之后：继续追问，还是进入行程生成（§5.1 + **G3**）。

    G3 触发（追问已达上限）时**降级放行**：不再追问，带着缺失字段进入生成，
    由生成节点补默认值并在行程里显式标注假设（§5.2）。
    取舍是清楚的 —— 宁可交付一个「写明了假设」的方案，也不要问到用户失去耐心。

    这条判断**必须在路由里，不能挪进节点**：节点只知道「我这轮问了什么」，
    「还要不要继续问」是控制流决策，只有路由看得见全局状态。

    :param max_ask_rounds: 留空则取 `Settings.max_ask_rounds`（§12）。
    """

    if not state.missing_fields:
        return "plan"

    limit = get_settings().max_ask_rounds if max_ask_rounds is None else max_ask_rounds
    if state.ask_round >= limit:
        return "plan"  # G3：降级放行

    return "ask"


def route_after_review(
    state: TravelState,
    *,
    max_review_retry: int | None = None,
) -> RouteAfterReview:
    """反思审核之后：回炉重生成，还是交给用户确认（§5.1 + **G1**）。

    **出口是 `confirm` 有两种完全不同的含义，调用方必须分清：**

      - `review_passed=True` → 正常通过
      - `review_passed=False` 且重试已达上限 → **G1 降级放行**，带着已知瑕疵进确认页

    第二种情况下，瑕疵由 `state.unresolved_errors` 暴露给前端（§5.2：降级必须透明）。
    路由只报「去哪」，不负责解释为什么 —— 该信息在状态里已经有了，路由再复述一遍
    就又多一份事实来源。

    ⬜ **G5（内容收敛检测）没有实现在这里，原因是它现在实现不了。** 它要比较
    「新旧 `draft_plan` 的相似度」，而 `TravelState.draft_plan` 是**覆盖**语义 ——
    上一轮的草稿已被本轮覆盖，比无可比。要落地得先给状态加一个累加语义的
    `draft_history`，那是 `TravelState` 的契约变更，不属 P1.3。

    :param max_review_retry: 留空则取 `Settings.max_review_retry`（§12）。
    """

    if state.review_passed:
        return "confirm"

    limit = get_settings().max_review_retry if max_review_retry is None else max_review_retry
    if state.retry_count >= limit:
        return "confirm"  # G1：降级放行

    return "retry"


def route_after_confirm(state: TravelState) -> RouteAfterConfirm:
    """用户确认之后：按反馈改行程，还是进入评估（§5.1）。

    **本函数刻意不看 G2（用户修改上限）。** G2 的处置是「前端禁用『继续修改』按钮，
    只允许确认或重置」（§5.2）—— 那是 API 层与 UI 的事。路由是纯读者，看不见 UI
    状态；在它这里自作主张放行，等于把用户**明确拒绝**的方案偷偷推进评估，
    这比多循环一轮糟得多。

    所以「已达 G2 上限却仍在请求 revise」这个情形**根本不该到达这里** ——
    P5 的 chat 入口必须在 `Command(resume=...)` 之前拦掉（§11）。
    万一真的漏到了（客户端绕过 API），图会一路循环到 G4 的 `recursion_limit`
    抛 `GraphRecursionError`，由 API 层兜底成友好提示。**这是刻意的响亮失败**：
    静默放行会让用户以为自己的修改被采纳了。
    """

    return "eval" if state.user_confirmed else "revise"
