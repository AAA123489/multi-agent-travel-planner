"""P3 前置实测：装饰器与 interrupt() 的相互作用，以及图的自省能力。

三个问题，答案都会直接决定 `decorators.py` 和 `builder.py` 怎么写：

Q1. `except Exception` 会不会吞掉 `interrupt()` 抛出的 GraphInterrupt？
    如果会，一个「带异常兜底」的装饰器会让**整个 HITL 机制静默失效** ——
    图不再暂停，而节点返回一个 `{"error": ...}`，看起来只是「审核没通过」。
    这类故障不会报错，只会表现为「确认页永远不出现」。
Q2. `node_trace` 在 interrupt 恢复时会不会被追加两次？（节点从头重跑）
Q3. 编译后的图能不能自省出拓扑？结构断言（不可旁路 self_review）要靠它。

跑法：`.venv/Scripts/python.exe -X utf8 scratch/verify_traced_node.py`
"""

import asyncio
import sys
from pathlib import Path

# 让 `import app` 在没有 PYTHONPATH 的情况下也能用（pytest 靠 pyproject 的
# pythonpath=["."] 解决，裸脚本没有）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.errors import GraphBubbleUp  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

from app.graph.state import TravelState  # noqa: E402

SEP = "=" * 72


def head(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# ---------------------------------------------------------------------------
# Q1 + Q2：装饰器的异常兜底 vs interrupt
# ---------------------------------------------------------------------------


def naive_decorator(fn):
    """**故意写错的那种装饰器** —— §4 要求「异常兜底」，照字面写就是这样。"""

    def wrapper(state: TravelState, config: RunnableConfig) -> dict:
        try:
            return fn(state, config)
        except Exception as exc:  # noqa: BLE001 —— 这就是被检验的写法
            return {"error": f"{type(exc).__name__}: {exc}"}

    return wrapper


def correct_decorator(fn):
    """正确写法：先放行 GraphBubbleUp，再兜底普通异常。"""

    def wrapper(state: TravelState, config: RunnableConfig) -> dict:
        try:
            return fn(state, config)
        except GraphBubbleUp:
            raise                      # ← 关键的一行
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    return wrapper


def make_confirm(body):
    def node(state: TravelState, config: RunnableConfig) -> dict:
        interrupt({"type": "plan_confirm"})
        return body(state, config)

    return node


def make_graph(node_fn, *, with_trace: bool):
    """一个最小两节点图：confirm → done。with_trace 时确认 node_trace 的行为。"""

    def confirm(state: TravelState, config: RunnableConfig) -> dict:
        result = node_fn(state, config)
        if not with_trace:
            return result
        seen = sum(1 for t in state.node_trace if t.split("#")[0] == "confirm")
        return {**result, "node_trace": [f"confirm#{seen + 1}"]}

    def done(state: TravelState, config: RunnableConfig) -> dict:
        return {"stage": "done"}

    b = StateGraph(TravelState)
    b.add_node("confirm", confirm)
    b.add_node("done", done)
    b.add_edge(START, "confirm")
    b.add_edge("confirm", "done")
    b.add_edge("done", END)
    return b.compile(checkpointer=InMemorySaver())


async def probe_swallow() -> None:
    head("Q1 · 异常兜底会不会吞掉 interrupt()")

    for label, deco in (("朴素版（照字面写 except Exception）", naive_decorator),
                        ("修正版（先放行 GraphBubbleUp）", correct_decorator)):
        body = deco(make_confirm(lambda s, c: {"user_confirmed": True}))
        graph = make_graph(body, with_trace=False)
        cfg = {"configurable": {"thread_id": f"q1-{label[:2]}"}}

        # ⚠ **判定暂停用 `snapshot.interrupts`，不能用「有没有抛异常」。**
        # 暂停不是异常：LangGraph 内部接住了它，`ainvoke` 正常返回。
        # 我第一次写这个探针时就栽在这里 —— 两个版本都打印「没抛异常」，
        # 看起来一样，而实际上一个暂停了、一个没有。§7.4 记的就是这条。
        await graph.ainvoke({"session_id": "s"}, cfg)
        snap = await graph.aget_state(cfg)

        if snap.interrupts:
            print(f"{label}：图**正常暂停**。interrupts={snap.interrupts}")
            print("    → HITL 机制完好")
        else:
            print(f"{label}：**图没有暂停**。interrupts=() "
                  f"error={snap.values.get('error')!r}")
            print("    → interrupt 被当成了节点故障；节点返回错误状态而不是暂停，")
            print("      于是 user_confirmed 仍是 False，路由会把人送回生成节点 ——")
            print("      在完整的图里，这表现为转圈到 G4 抛 GraphRecursionError。")


async def probe_double_trace() -> None:
    head("Q2 · interrupt 恢复时 node_trace 会不会追加两次")

    graph = make_graph(
        correct_decorator(make_confirm(lambda s, c: {"user_confirmed": True})),
        with_trace=True,
    )
    cfg = {"configurable": {"thread_id": "q2"}}

    await graph.ainvoke({"session_id": "s"}, cfg)
    snap = await graph.aget_state(cfg)
    print(f"暂停后 node_trace = {snap.values.get('node_trace')}")

    await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    snap = await graph.aget_state(cfg)
    trace = snap.values.get("node_trace")
    print(f"恢复后 node_trace = {trace}")
    print(f"confirm 出现次数 = {sum(1 for t in trace if t.startswith('confirm'))}（应为 1）")
    print(f"user_confirmed = {snap.values.get('user_confirmed')}")


# ---------------------------------------------------------------------------
# Q3：编译后的图能自省出拓扑吗
# ---------------------------------------------------------------------------


async def probe_introspection() -> None:
    head("Q3 · 编译后的图能否自省拓扑（结构断言的基础）")

    def confirm(state: TravelState, config: RunnableConfig) -> dict:
        return {"user_confirmed": True}

    def review(state: TravelState, config: RunnableConfig) -> dict:
        return {"review_passed": True}

    b = StateGraph(TravelState)
    b.add_node("plan_generate", lambda s, c: {})
    b.add_node("self_review", review)
    b.add_node("user_confirm", confirm)
    b.add_conditional_edges(
        "self_review", lambda s: "confirm" if s.review_passed else "retry",
        {"retry": "plan_generate", "confirm": "user_confirm"},
    )
    b.add_edge(START, "plan_generate")
    b.add_edge("plan_generate", "self_review")
    b.add_edge("user_confirm", END)
    graph = b.compile()

    drawn = graph.get_graph()
    print("节点：", sorted(drawn.nodes))
    print("\n边：")
    for edge in drawn.edges:
        cond = "  [条件]" if getattr(edge, "conditional", False) else ""
        print(f"  {edge.source:18} -> {edge.target:18}{cond}")
    print("\n有没有 self_review -> user_confirm 这条直连？",
          any(e.source == "self_review" and e.target == "user_confirm"
              for e in drawn.edges))
    print("drawable 属性存在？", hasattr(drawn, "draw_mermaid"))


async def main() -> None:
    await probe_swallow()
    await probe_double_trace()
    await probe_introspection()
    print(f"\n{SEP}\n完成\n{SEP}")


if __name__ == "__main__":
    asyncio.run(main())
