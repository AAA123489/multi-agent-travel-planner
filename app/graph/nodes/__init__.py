"""五个 Agent 节点（P3.1）—— 「有哪些节点」的唯一清单。

图里实际用了哪几个节点由 `builder.py` 决定；本模块只负责回答「节点叫什么名字、
默认实现是哪个函数」。把这两件事分开，是为了让 `build_graph(overrides=...)`
能有一个明确的替换目标。

**节点函数在这里是未装饰的裸函数**，`@traced_node` 由 `builder.py` 在注册时
统一套上 —— 理由见 `decorators.traced_node` 的 docstring（替身节点也必须被追踪）。

节点名（下面这张表的键）是**契约的一部分**：`node_trace`、SSE 事件、结构断言
全都认这些字符串。改名等于改契约，要同步文档。
"""

from app.graph.nodes.decorators import NodeFn, traced_node
from app.graph.nodes.evaluate import evaluate
from app.graph.nodes.plan_generate import plan_generate
from app.graph.nodes.requirement_collect import requirement_collect
from app.graph.nodes.self_review import self_review
from app.graph.nodes.user_confirm import user_confirm

# 节点名 → 默认实现。**顺序 = §2.2 状态图里从左到右的顺序**，便于对照。
# 用 dict 而不是四个并列常量：builder 需要遍历它来注册，写死四处调用的话
# 加一个节点就要改 builder 的两个地方。
DEFAULT_NODES: dict[str, NodeFn] = {
    "requirement_collect": requirement_collect,
    "plan_generate": plan_generate,
    "self_review": self_review,
    "user_confirm": user_confirm,
    "evaluate": evaluate,
}

__all__ = [
    "DEFAULT_NODES",
    "NodeFn",
    "evaluate",
    "plan_generate",
    "requirement_collect",
    "self_review",
    "traced_node",
    "user_confirm",
]
