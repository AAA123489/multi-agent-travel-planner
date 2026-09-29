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

RouteAfterCollect = Literal["ask", "plan", "reject"]
RouteAfterReview = Literal["retry", "confirm"]
RouteAfterConfirm = Literal["revise", "eval"]


def route_after_collect(
    state: TravelState,
    *,
    max_ask_rounds: int | None = None,
) -> RouteAfterCollect:
    """需求收集之后：继续追问、直接生成，还是就地终止（§5.1 + **G3**）。

    三个出口：

    | 返回 | 何时 | 去哪 |
    |---|---|---|
    | `"reject"` | `stage == "failed"` | `END`（本轮终止，不推进） |
    | `"ask"` | 缺必填项且追问未达上限 | `END`（等用户下一轮输入） |
    | `"plan"` | 不缺，或 G3 触发 | `plan_generate` |

    ## `"reject"` 这条出口是 P4.1 加的

    §5.1 原先只有 `ask` / `plan` 两个。加了它，是因为「需求收集阶段没有可用的
    结果」这件事**此前无处可去**：`route_after_collect` 只读 `missing_fields`，
    而两种情况都会让它判成 `plan` ——

      - **目的地不在已接入范围**（§4.1 要点 5）：字段一个不缺，`missing_fields`
        是空的，于是图一路冲进 `plan_generate`，拿不到任何 POI，最后交出一份
        每个景点都是幻觉的行程；
      - **节点自己崩了**：`@traced_node` 兜底后返回 `{error, stage: failed}`，
        `missing_fields` 保持上一轮的值。第 1 轮它是空的 → 同样冲进 `plan_generate`，
        带着一份**完全空白的需求**。

    两条都是「不该也不能继续往下走」。所以这里复用同一条出口，判据是
    `stage == "failed"` —— 那也是 `traced_node` 失败时写的值。给用户看的话术
    两种情况下都已经在 `state.error` 里了，路由不复述（复述就多一份事实来源）。

    ## G3 触发时**降级放行**，不是终止

    追问已达上限就不再追问，带着缺失字段进入生成，由生成节点补默认值并在行程里
    显式标注假设（§5.2）。取舍是清楚的 —— 宁可交付一个「写明了假设」的方案，
    也不要问到用户失去耐心。

    **注意两者处置相反是有道理的**：G3 缺的是「锦上添花的信息」，兜底就能往下走；
    `reject` 缺的是「能算的东西」，兜底只会产出一份假的行程。

    这条判断**必须在路由里，不能挪进节点**：节点只知道「我这轮问了什么」，
    「还要不要继续问」是控制流决策，只有路由看得见全局状态。

    :param max_ask_rounds: 留空则取 `Settings.max_ask_rounds`（§12）。
    """

    if state.stage == "failed":
        return "reject"

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

    **G5（内容收敛检测）已放弃**，不要往这里补（§5.2 有完整理由，2026-09-28 定案）：
    它要比较「新旧 `draft_plan` 的相似度」，而 `TravelState.draft_plan` 是**覆盖**语义 ——
    上一轮的草稿已被本轮覆盖，比无可比。落地得先给状态加累加语义的 `draft_history`，
    那是 §3.1 的契约变更。而它只能在本函数的 G1 之上再省 1–2 轮 LLM 调用 ——
    **为一个已经很短的环再截短一点，去污染核心状态契约，是性价比最低的一类改动。**

    ⚠ 因此 G1~G4 的编号是**跳号**的（没有 G5）。跳号是刻意的：它是一处「这里删过一个
    闸门」的记录。看到编号不连续就去补一个 G5 回来，是把已做的决策又推翻一遍。

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
