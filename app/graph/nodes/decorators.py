"""节点装饰器 `@traced_node`（P3.2）—— 对应 docs/方案设计.md §4 的开篇约定。

五个节点统一由它包裹，负责四件事：**计时、日志、`node_trace` 追加、异常兜底**。
节点本身因此只写业务逻辑，不重复这些样板。

## 一句话结论：异常兜底必须先放行 `GraphBubbleUp`

这是本模块存在的**全部理由**，也是实测踩出来的坑（`scratch/verify_traced_node.py` 的 Q1）：

`interrupt()` 内部是靠**抛异常**（`GraphInterrupt`）来中止节点执行的，而
`GraphInterrupt` 是 `Exception` 的子类。照 §4 的字面要求写「异常兜底」，最自然的
写法就是 `except Exception` —— 于是 `interrupt()` 被当成一次普通失败吞掉，
节点返回 `{"error": "GraphInterrupt: ..."}`，**图不再暂停**。

实测输出（朴素版）：

    interrupts=()   error="GraphInterrupt: (Interrupt(value={'type': 'plan_confirm'}, ...),)"

出了什么事，比「暂停失败」更糟一层：节点没暂停，而是返回了一份
`{"error": ..., "stage": "failed"}`；于是 `user_confirmed` 仍是 `False`，
`route_after_confirm` 判成「用户要改」，**把图又送回生成节点** ——
转一圈回到同一个节点，再吞一次。实测跑出来的是
`langgraph.errors.GraphRecursionError`（撞上 G4 才停）。

也就是说：**一个「暂停」被换成了一个死循环**，而日志里每一次都只写着
「节点执行失败」。用户看到的是转圈到超时，日志里没有一行提到 interrupt。
这正是硬红线 #1 和 §4.4 反复强调 `interrupt()` 的原因 —— 它的失败方式
不是崩溃，是伪装成一次普通的节点故障。

修法就一行：`except GraphBubbleUp: raise` 放在 `except Exception` **之前**。
`GraphBubbleUp` 是 `GraphInterrupt` / `GraphInterrupted` 的共同基类，
放行它就等于「让 LangGraph 的控制流异常穿过去，只兜底真正的业务异常」。

`tests/test_graph.py::test_decorator_does_not_swallow_interrupt` 守着这一行。
用 `scratch/mutate_p3.py` 的变异① 实测：删掉这一行，**8 条测试同时变红**，
其中 4 条路径测试全都以 `GraphRecursionError` 失败。

## 另一条纪律：`node_trace` 的序号从状态里数，不从闭包变量里数

`node_trace` 形如 `["requirement_collect#1", "plan_generate#1", ...]`（§3.2）。
序号靠**数状态里已有的同名条目**得出，而不是在装饰器里维护一个计数器。

原因还是 `interrupt()`：它在恢复时让**节点从头重跑一遍**。任何「执行次数」的
副作用（计数器 +1、往文件写一行、发一次埋点）都会被执行两次。把序号算成
状态的纯函数，重跑时算出的还是同一个值，天然幂等。

实测（同一个探针的 Q2）：暂停时 `node_trace == []`，恢复后 `== ["confirm#1"]` ——
**中断那一轮没被记进去**，因为节点没返回，LangGraph 没提交它的写。这不是 bug，
是 interrupt 语义的直接推论：暂停点之前的一切写入都作废。
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp

from app.graph.state import TravelState

logger = logging.getLogger(__name__)

# 节点统一签名（§4）：吃状态 + 运行配置，吐**状态增量字典**（只含本节点改的字段）。
NodeFn = Callable[[TravelState, RunnableConfig], dict[str, Any]]

# 异常信息写进 `TravelState.error` 时的长度上限。
# 不设上限的话，一个带完整 traceback 文本的上游报错会让整个状态快照膨胀，
# 而 checkpoint 是每个 super-step 都要落盘的。
MAX_ERROR_CHARS = 300


def _next_index(state: TravelState, name: str) -> int:
    """这个节点第几次被执行 —— **从 `node_trace` 数出来，纯读不写**。

    `#` 之前的部分是节点名，之后是序号。用 `split("#")[0]` 而不是 `startswith`
    是因为节点名本身不含 `#`，但序号里可能有（将来若改成 `#3.1` 之类）。

    节点名用 `#` 分隔而不是 `:` 或 `.`：`:` 会在日志里和 `logger.info("%s: ...")`
    的格式串撞上，`.` 与模块名点号混淆。`#` 在两者里都不出现。
    """
    return sum(1 for entry in state.node_trace if entry.split("#")[0] == name) + 1


def traced_node(name: str) -> Callable[[NodeFn], NodeFn]:
    """把节点函数包成「带轨迹、带兜底」的可注册节点。

    :param name: 节点在图里的名字。**显式传而不是取 `fn.__name__`** ——
        图上的节点名是契约的一部分（`node_trace`、SSE 事件、结构断言都认它），
        让它跟着一个可能被重命名的 Python 函数走，等于把契约交给重构决定。

    用法（**包裹动作在 `builder.py`，不在节点模块里**，理由见下）::

        node = traced_node("self_review")(self_review)

    ## 为什么包裹在 builder 而不是写成节点上的 `@装饰器`

    §4 写的是「统一由 `@traced_node("node_name")` 包裹」，本模块把它实现成
    **在注册时统一包裹**。这不是形式差异，是 `build_graph(overrides=...)` 逼出来的：

    测试与 `walk_graph.py` 需要替换某个节点的**行为**（比如让 `self_review`
    这一轮通过），而替换掉的节点**同样需要被追踪** —— 否则 4 条路径里只要有
    一条用了替身，它就从 `node_trace` 里消失，而 `node_trace` 正是排查的入口。

    如果装饰器写在节点模块上，替身函数就得自己在测试里再包一层，写漏一处就
    静默丢轨迹。放在注册点，则「进了图的节点必然被追踪」由构造过程保证。
    """

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        def wrapper(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
            index = _next_index(state, name)
            # 序号在计时开始前就算好 —— 它只依赖进来的状态，与本次执行无关。
            # 放在这里而不是 finally 里：万一 `_next_index` 自己抛了（不会，
            # 但状态是外部传进来的），异常兜底和轨迹追加还能正常工作。
            entry = f"{name}#{index}"
            started = time.perf_counter()

            try:
                result = fn(state, config)
            except GraphBubbleUp:
                # ⚠ 这一行是整个 HITL 机制的命门，不要删、不要挪到下面去。
                # `interrupt()` 靠抛异常暂停图，吞掉它 = 确认页永远不出现。
                raise
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                # 异常写进状态而不是往上抛：与硬红线 #4 同一条姿态 ——
                # 降级而非中断。抛出去会让「图跑到一半停了」变成一个 500，
                # 而状态里留一条 error、stage 置 failed，前端还能显示这是哪一步坏的。
                message = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
                logger.warning(
                    "节点 %s 执行失败（%.0f ms）：%s",
                    entry, elapsed, message,
                    extra={"thread_id": _thread_id(config)},
                )
                return {"error": message, "stage": "failed", "node_trace": [entry]}

            elapsed = (time.perf_counter() - started) * 1000
            logger.info(
                "节点 %s 完成（%.0f ms）",
                entry, elapsed,
                extra={"thread_id": _thread_id(config)},
            )

            # `node_trace` 由装饰器**独占**：节点若返回了同名字段会被这里覆盖。
            # 让两边都能写同一个字段，就会出现「谁后写谁算」的静默竞争 ——
            # reducer 是 `add`，两者会都进列表，轨迹里出现两条同名的条目。
            return {**result, "node_trace": [entry]}

        return wrapper

    return decorator


def _thread_id(config: RunnableConfig | None) -> str:
    """从运行配置里取 thread_id，纯粹为了日志能定位到会话。

    取不到就返回 `-`：**日志函数不该成为新的失败点**。而这不只是防御 ——
    `config` 允许是 `None`（手工单测一个节点时不传），
    且 `configurable` 里没有 `thread_id` 也是合法情形（图没接 checkpointer）。
    """
    if not config:
        return "-"
    return str(config.get("configurable", {}).get("thread_id", "-"))
