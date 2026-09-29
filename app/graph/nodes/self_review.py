"""`self_review` 反思审核节点（P3.1 空壳）—— 对应 §4.3。

## ⚠ 本节点受硬红线 #3 保护，不可被任何「提速 / 省 token」的优化旁路

有同类项目在编排里加了一条「产出超过 80 字就直接结束」的规则，结果审核节点
**从来没被执行过**，形同虚设，而且没人发现（[开发流程.md](../../../docs/开发流程.md) P3）。
对策是 `tests/test_graph.py` 里的两条断言：一条查拓扑（`plan_generate` 之后
必须经过本节点才能到 `user_confirm`），一条查实际轨迹（`node_trace` 里必须出现
`self_review` 且早于 `user_confirm`）。

## P3 空壳的默认行为：**永远不通过**

这是**刻意选的默认值**，不是偷懒。默认通过的话，G1（机器重试上限 → 降级放行）
这条分支就只能靠注入替身才能看到 —— 而那恰好是全图里最容易写错的判断
（差一个 `>` 就是「永远多循环一轮」或「一次都不循环」）。
让默认空壳一路失败，等于让 `walk_graph.py` 一跑就把 G1 摆在眼前。

想看通过的那条路，用 `build_graph(overrides={"self_review": ...})` 换行为 ——
**换的是节点的行为，不是图的连线**，所以结构断言依然成立。

## P4 要在这里做的事

§4.3 的七条程序化校验（闭馆冲突 / 时间冲突 / 通勤超限 / 预算超限 / 路线折返 /
完整性 / 幻觉 POI）落在 `app/graph/validators/`，本节点只负责汇总。
LLM 只做**语义级补充审核**且 severity 固定为 `warn` —— 让 LLM 的产出能阻塞
通过，等于给它一个制造死循环的开关。

**这些校验需要 `plan_struct`，而它的 schema 尚未定义**（见 `plan_generate.py`
的说明）。所以 P3 的审核无从下手，只能报一条「还没实现」。
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import ReviewComment, TravelState

logger = logging.getLogger(__name__)

# 空壳审核报出的那条意见。severity 用 error（而不是 warn）：
# 这样它就会**阻塞通过**，把图推进链路 A 的重试环 —— 这正是默认行为想要的。
STUB_COMMENT = ReviewComment(
    type="other",
    severity="error",
    detail="行程审核规则尚未实现（§4.3 的七条程序化校验要到 P4 才落盘）。",
    suggestion="P4 阶段补齐 validators/ 后，本意见会被真实校验结果取代。",
)


def self_review(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """反思审核。出口见 §5.1 的 `route_after_review`（G1 在路由里）。"""

    passed = False          # ← P3 空壳：见模块 docstring 为何默认不通过
    comments: list[ReviewComment] = [] if passed else [STUB_COMMENT]

    logger.info(
        "审核结果：%s（%d 条意见，其中 error %d 条）",
        "通过" if passed else "不通过",
        len(comments),
        sum(1 for c in comments if c.severity == "error"),
        extra={"thread_id": config.get("configurable", {}).get("thread_id", "-")},
    )

    return {
        "review_comments": comments,      # 覆盖语义，不是追加（§3.3）
        # review_history 是**追加**语义：把本轮快照整份存进去，用于分析收敛性。
        # 注意存的是 list[list[...]] —— 所以这里交一个「装着本次快照的列表」，
        # 而不是把 comments 直接摊平进去（那样历史就散了，读不出「第几轮有哪些意见」）
        "review_history": [comments],
        "review_passed": passed,
        # 只在**不通过**时 +1（§4.3）。通过还加的话，G1 会被正常路径白白消耗
        "retry_count": state.retry_count + (0 if passed else 1),
        "stage": "awaiting_user" if passed else "reviewing",
    }
