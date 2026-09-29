"""路由函数测试（P1.3 / P1.7）—— 对应 docs/方案设计.md §5.1 / §5.2。

开发流程对 P1 的验收要求是「路由函数测试覆盖**全部分支**」，所以这个文件按分支
逐条写，而不是按函数写。三个函数共 9 个分支，外加三条「注入的默认值真的连到
Settings 上」的接线测试。

**边界值单独测**：G1 / G3 用的是 `>=` 不是 `>`，差一个字符就是「永远多循环一轮」
或者「一次都不循环」。这类 off-by-one 不写边界断言就一定漏。

`fresh_settings` fixture 见 `conftest.py`（图测试也用它，故提为公共 fixture）。
"""


from app.graph.edges import (
    route_after_collect,
    route_after_confirm,
    route_after_review,
)
from app.graph.state import ReviewComment, TravelState


def _error(detail: str = "预算超了") -> ReviewComment:
    return ReviewComment(type="budget", severity="error", detail=detail)


# ===========================================================================
# route_after_collect —— 3 个分支（§5.1 + G3）
# ===========================================================================


def test_collect_plans_when_nothing_missing():
    """需求齐了就走生成 —— 此时追问轮数是多少都不影响判断。"""
    state = TravelState(missing_fields=[], ask_round=99)
    assert route_after_collect(state, max_ask_rounds=3) == "plan"


def test_collect_asks_when_fields_missing_and_budget_left():
    """缺字段且还有追问余额 → 继续问。"""
    state = TravelState(missing_fields=["destination"], ask_round=0)
    assert route_after_collect(state, max_ask_rounds=3) == "ask"


def test_collect_falls_through_at_exact_limit():
    """G3 边界：`ask_round == max_ask_rounds` 就该放行，不是 `max + 1`。

    这是**降级放行**那一侧：宁可交付一个写明了假设的方案，也不要问到用户失去耐心。
    """
    state = TravelState(missing_fields=["destination"], ask_round=3)
    assert route_after_collect(state, max_ask_rounds=3) == "plan"


def test_collect_still_asks_one_below_limit():
    """G3 边界的另一侧：差一轮就还得问。和上一条合起来钉死 `>=`。"""
    state = TravelState(missing_fields=["destination"], ask_round=2)
    assert route_after_collect(state, max_ask_rounds=3) == "ask"


# ===========================================================================
# route_after_review —— 4 个分支（§5.1 + G1）
# ===========================================================================


def test_review_confirms_when_passed():
    state = TravelState(review_passed=True, retry_count=0)
    assert route_after_review(state, max_review_retry=3) == "confirm"


def test_review_retries_when_failed_and_budget_left():
    state = TravelState(review_passed=False, retry_count=1)
    assert route_after_review(state, max_review_retry=3) == "retry"


def test_review_falls_through_at_exact_limit():
    """G1 边界：`retry_count == max_review_retry` 时降级放行。

    出口虽然是 `confirm`，但**语义与「通过」完全不同** —— 前端要靠
    `unresolved_errors` 把隐患显式告诉用户（§5.2：降级必须透明）。
    """
    state = TravelState(review_passed=False, retry_count=3, review_comments=[_error()])
    assert route_after_review(state, max_review_retry=3) == "confirm"


def test_review_still_retries_one_below_limit():
    state = TravelState(review_passed=False, retry_count=2)
    assert route_after_review(state, max_review_retry=3) == "retry"


def test_confirm_means_two_different_things():
    """同样是 `confirm`，两种到达方式的差别必须能从**状态**里读出来。

    路由只报「去哪」，不解释为什么。这条测试守住「路由不复述原因」——
    若哪天有人往返回值里塞信息（返回 `"confirm_degraded"` 之类），
    状态与返回值就成了第二份事实来源。
    """
    passed = TravelState(review_passed=True, retry_count=3, review_comments=[_error()])
    degraded = TravelState(review_passed=False, retry_count=3, review_comments=[_error()])

    # 路由的返回值一模一样 —— 从它这里分不出两者
    assert route_after_review(passed, max_review_retry=3) == "confirm"
    assert route_after_review(degraded, max_review_retry=3) == "confirm"

    # 区分在状态里：passed 时 `unresolved_errors` 刻意返回空
    assert passed.unresolved_errors == []
    assert [c.detail for c in degraded.unresolved_errors] == ["预算超了"]


# ===========================================================================
# route_after_confirm —— 2 个分支（§5.1；G2 不在此处，见函数 docstring）
# ===========================================================================


def test_confirm_goes_to_eval_when_user_approved():
    assert route_after_confirm(TravelState(user_confirmed=True)) == "eval"


def test_confirm_goes_to_revise_when_user_rejected():
    """用户没确认就回生成节点 —— 带上 `user_feedback`（由节点自己去读）。"""
    assert route_after_confirm(TravelState(user_confirmed=False)) == "revise"


def test_confirm_ignores_revision_count():
    """**钉住「G2 不在路由里」这个决定。**

    `user_revision_count` 已达上限时路由**仍然**返回 `revise` —— 在它这里自作主张
    放行，等于把用户明确拒绝的方案偷偷推进评估，比多循环一轮糟得多。
    拦下这个请求是 API 层的职责（§11），路由看不见 UI。
    """
    over_limit = TravelState(user_confirmed=False, user_revision_count=99)
    assert route_after_confirm(over_limit) == "revise"


# ===========================================================================
# 路由的纯度（§5.1 的纪律）
# ===========================================================================


def test_routers_do_not_mutate_state():
    """路由只读。

    计数器自增归节点管。若路由顺手改了状态，「为什么又回炉了一轮」就无从追查 ——
    而这类 bug 在图上表现为「转圈」，最难定位。
    """
    state = TravelState(
        missing_fields=["destination"],
        ask_round=1,
        retry_count=1,
        review_passed=False,
        user_confirmed=False,
    )
    before = state.model_dump()

    route_after_collect(state, max_ask_rounds=3)
    route_after_review(state, max_review_retry=3)
    route_after_confirm(state)

    assert state.model_dump() == before


# ===========================================================================
# 接线 —— 注入的默认值真的连到 Settings 上吗
# ===========================================================================

# 下面三条守的是一类**静默失效**：`max_*` 参数默认是 None，函数体内向
# `get_settings()` 取真实阈值。如果哪天有人在函数体里写了个硬编码兜底
# （`limit = max_retry or 3`），用显式传参写的测试**一条都不会红** ——
# 但生产从此再也不读配置了。所以必须走一遍默认路径。


def test_collect_default_limit_comes_from_settings(fresh_settings):
    """`MAX_ASK_ROUNDS=0` → 一个字段都不许追问。"""
    fresh_settings(MAX_ASK_ROUNDS="0")
    state = TravelState(missing_fields=["destination"], ask_round=0)
    # 若默认值是硬编码的 3，这里会得到 "ask"
    assert route_after_collect(state) == "plan"


def test_review_default_limit_comes_from_settings(fresh_settings):
    """`MAX_REVIEW_RETRY=0` → 一轮都不许回炉。"""
    fresh_settings(MAX_REVIEW_RETRY="0")
    state = TravelState(review_passed=False, retry_count=0)
    # 若默认值是硬编码的 3，这里会得到 "retry"
    assert route_after_review(state) == "confirm"


def test_explicit_limit_overrides_settings(fresh_settings):
    """显式传参优先于 Settings —— 这是测试不必碰环境变量的依据。"""
    fresh_settings(MAX_REVIEW_RETRY="9")
    state = TravelState(review_passed=False, retry_count=3)
    assert route_after_review(state) == "retry"           # 走 Settings：3 < 9
    assert route_after_review(state, max_review_retry=3) == "confirm"  # 走注入：3 >= 3
