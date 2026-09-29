"""建图（P3.3）—— 对应 docs/方案设计.md §2.2 的状态图。

**本文件是全图拓扑的唯一事实来源。** 节点之间有哪几条边、哪个条件边通到哪里，
只在这里写一次；`tests/test_graph.py` 的结构断言读的也是编译产物，
两者对不上就会红。

## 两条回流链路

    链路 A｜机器自省环   self_review → plan_generate   计 retry_count，上限 G1
    链路 B｜用户反馈环   user_confirm → plan_generate   计 user_revision_count，上限 G2

两个环**共用同一个 `plan_generate` 入口**，靠状态里的 `review_comments` /
`user_feedback` 区分这一轮该听谁的。共用而不是各建一个生成节点，是因为
「怎么生成」只有一套逻辑，按来源分叉等于把同一段代码写两遍（然后漂移）。

## 关于 checkpointer

`MemorySaver` 在 langgraph 1.2.7 里只是 `InMemorySaver` 的**向后兼容别名**
（源码注释：`MemorySaver = InMemorySaver  # Kept for backwards compatibility`）。
开发流程 P3.3 写的是 `MemorySaver()`，这里用**本名 `InMemorySaver`** ——
同一个类，但读代码的人不会以为它是个独立的、可能被弃用的东西。

P3 用它只是为了走通 `interrupt()`（它强依赖 checkpointer）。**生产用
`AsyncSqliteSaver`（§7.1），在 P5 的 lifespan 里建**，业务代码零改动 ——
这正是 `checkpointer` 作为参数而不是在函数体里 `new` 一个的原因。
"""

from collections.abc import Mapping

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.edges import (
    route_after_collect,
    route_after_confirm,
    route_after_review,
)
from app.graph.nodes import DEFAULT_NODES, NodeFn, traced_node
from app.graph.state import TravelState

# G4 图递归上限（§5.2）。
#
# 它**不在 §12 的配置清单里**，也就没有对应的环境变量 —— 这是刻意的：
# 这个值是「图失控了，必须停」的最后一道保险，不是需要按部署调参的业务阈值。
# 50 的意义是「远大于任何正常路径的步数」：正常最多
# 1(收集) + 3(生成×重试) + 3(审核) + 1(确认) + 1(评估) ≈ 9 步，
# 而 G1/G2 已经把两个环各自封顶，正常路径不可能逼近 50。
RECURSION_LIMIT = 50


def graph_config(
    session_id: str, *, recursion_limit: int = RECURSION_LIMIT
) -> RunnableConfig:
    """构造调用图时用的运行配置。

    `thread_id` 与 `TravelState.session_id` **保持一致**（§7.1）——
    两者一旦不同源，排查时就得靠人脑做一次映射，而那种映射在半夜三点是不存在的。

    抽成函数而不是让每处调用各写一个 dict：`recursion_limit` 漏写一处，
    那处的 G4 就退回 LangGraph 的默认值 25 —— 一个**不会报错**的静默差异。
    """
    return {"configurable": {"thread_id": session_id}, "recursion_limit": recursion_limit}


def build_graph(
    checkpointer: InMemorySaver | None = None,
    *,
    overrides: Mapping[str, NodeFn] | None = None,
) -> CompiledStateGraph:
    """组装并编译状态图。

    :param checkpointer: 传 `InMemorySaver()` 才能用 `interrupt()`（HITL 强依赖它）。
        不传也能编译，但图跑到 `user_confirm` 会因为没人接住 `interrupt` 而报错。
    :param overrides: 节点名 → 替代实现。**替换的是节点的行为，不是图的连线。**
        结构断言（`self_review` 不可旁路）读的是拓扑，因此不受它影响。

        P3 的四条路径里有三条需要它：默认空壳审核永远不通过（见
        `nodes/self_review.py` 的说明），所以「一次通过」和「用户拒绝一次」
        这两条得把审核换成会通过的实现。

        **未知的节点名直接报错，不静默忽略。** 测试里把 `self_review` 拼成
        `self_reveiw` 是最难查的一类失败：图照样跑、结果照样出来，只是
        断言在别处以一个看似无关的差异失败。

    :raises ValueError: `overrides` 里有不在 `DEFAULT_NODES` 中的节点名。
    """
    unknown = set(overrides or ()) - set(DEFAULT_NODES)
    if unknown:
        raise ValueError(
            f"overrides 里有不存在的节点：{'、'.join(sorted(unknown))}。"
            f"可用节点：{'、'.join(DEFAULT_NODES)}"
        )

    def node_fn(name: str) -> NodeFn:
        """取该节点的实现（替身优先），并统一套上追踪装饰器。

        ⚠ 替身**也必须被装饰**：否则用了替身的那条路径会从 `node_trace` 里
        消失，而 `node_trace` 正是排查「图怎么走的」唯一入口 —— 恰恰在最需要
        它的时候（测试里换了行为、结果不对）它少一段。
        """
        body = (overrides or {}).get(name, DEFAULT_NODES[name])
        return traced_node(name)(body)

    builder = StateGraph(TravelState)

    for name in DEFAULT_NODES:
        builder.add_node(name, node_fn(name))

    builder.add_edge(START, "requirement_collect")

    # G3 在 route_after_collect 里：缺字段且追问未达上限 → "ask" → END 等下一轮
    builder.add_conditional_edges(
        "requirement_collect",
        route_after_collect,
        {"ask": END, "plan": "plan_generate"},
    )

    # 无条件边：生成完必过审核，没有任何绕过的路径（硬红线 #3）
    builder.add_edge("plan_generate", "self_review")

    # G1 在 route_after_review 里：未通过且未达上限 → "retry"，否则 "confirm"
    # ⚠ "confirm" 有两种含义（真通过 / 降级放行），路由只报去向，区分在状态里
    builder.add_conditional_edges(
        "self_review",
        route_after_review,
        {"retry": "plan_generate", "confirm": "user_confirm"},
    )

    # G2 **不在这里**：它的处置是「前端禁用继续修改」，路由看不见 UI。
    # 已达上限却仍在 revise 的请求必须在 API 层拦掉（§5.2 / §11）。
    builder.add_conditional_edges(
        "user_confirm",
        route_after_confirm,
        {"revise": "plan_generate", "eval": "evaluate"},
    )

    builder.add_edge("evaluate", END)

    return builder.compile(checkpointer=checkpointer)
