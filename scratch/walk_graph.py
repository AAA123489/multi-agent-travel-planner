"""P3.4 —— 把四条路径实际走一遍，打印轨迹。

**不需要 API key、不联网**（P3 的节点全是空壳，一个 LLM 都不调）。

跑法（在仓库根目录）：
    .venv/Scripts/python.exe -X utf8 scratch/walk_graph.py

它与 `tests/test_graph.py` 的分工：**测试负责判定，本脚本负责让人看见。**
同样的四条路径，测试里是断言，这里是轨迹打印 —— 排查时你要的是后者
（「图到底怎么走的」），而回归时你要的是前者。

先打印一次图的拓扑，再逐条走。拓扑那一段同时也在验证「编译后的图可以自省」——
P3 的结构断言（反思节点不可旁路）就是靠它。
"""

import asyncio
import sys
from pathlib import Path

# 让 `import app` 在没有 PYTHONPATH 的情况下也能用。
# pytest 靠 pyproject.toml 的 pythonpath=["."] 解决了这件事，裸脚本没有。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.graph.builder import build_graph, graph_config  # noqa: E402
from app.graph.nodes import DEFAULT_NODES  # noqa: E402
from app.graph.state import ReviewComment, TravelRequirement  # noqa: E402

FULL_REQUIREMENT = TravelRequirement(
    origin="北京", destination="成都", days=3, travelers=2, budget=5000.0
)

SEP = "─" * 74


def banner(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def review_passing_after(fail_times: int):
    """审核替身：前 fail_times 轮不通过，之后通过。

    **替换的是节点的行为，不是图的连线** —— 结构断言不受影响。
    默认空壳审核永远不通过（`nodes/self_review.py` 里有说明），所以
    「一次通过」和「用户拒绝一次」这两条路径必须换掉它才走得出来。
    """

    def node(state, config):
        if state.retry_count < fail_times:
            return {
                "review_passed": False,
                "retry_count": state.retry_count + 1,
                "review_comments": [
                    ReviewComment(
                        type="budget",
                        severity="error",
                        detail=f"第 {state.retry_count + 1} 轮审核不通过",
                    )
                ],
                "stage": "reviewing",
            }
        return {"review_passed": True, "review_comments": [], "stage": "awaiting_user"}

    return node


def show(snapshot, label: str) -> None:
    """打印一份快照的关键信息。"""
    values = snapshot.values
    trace = values.get("node_trace", [])
    print(f"\n  【{label}】")
    print(f"    轨迹   : {' → '.join(trace) if trace else '(空)'}")
    print(f"    暂停   : {'是' if snapshot.interrupts else '否'}"
          f"   （.next={snapshot.next}）")
    print(f"    stage  : {values.get('stage')}")
    print(f"    计数   : retry_count={values.get('retry_count', 0)}"
          f"  user_revision_count={values.get('user_revision_count', 0)}"
          f"  ask_round={values.get('ask_round', 0)}")
    print(f"    缺失项 : {values.get('missing_fields', [])}")
    if snapshot.interrupts:
        payload = snapshot.interrupts[0].value
        if isinstance(payload, dict) and payload.get("type") == "plan_confirm":
            print(f"    载荷   : type=plan_confirm  "
                  f"未解决 error {len(payload.get('unresolved_errors', []))} 条")


def show_topology() -> None:
    banner("图拓扑（编译产物自省 —— 结构断言就靠它）")
    drawn = build_graph(InMemorySaver()).get_graph()
    print("\n  节点：", "、".join(sorted(n for n in drawn.nodes
                                       if not n.startswith("__"))))
    print("\n  边：")
    for edge in drawn.edges:
        mark = " [条件]" if edge.conditional else ""
        arrow = "⇢" if edge.conditional else "→"
        print(f"    {edge.source:20} {arrow} {edge.target}{mark}")


async def path1_happy() -> None:
    banner("路径 ① 一次通过：需求完整 → 审核通过 → 用户 approve")
    review = review_passing_after(0)
    graph = build_graph(InMemorySaver(), overrides={"self_review": review})
    config = graph_config("walk-1")

    await graph.ainvoke(
        {"session_id": "walk-1", "user_requirement": FULL_REQUIREMENT}, config
    )
    show(await graph.aget_state(config), "跑到暂停（等着用户确认）")

    await graph.ainvoke(Command(resume={"action": "approve"}), config)
    show(await graph.aget_state(config), "用户 approve 之后")


async def path2_g1() -> None:
    banner("路径 ② 触发机器重试：审核一直不通过 → 撞上 G1 降级放行")
    # 不换任何节点 —— 默认空壳审核永远不通过，正好演示 G1
    graph = build_graph(InMemorySaver())
    config = graph_config("walk-2")

    await graph.ainvoke(
        {"session_id": "walk-2", "user_requirement": FULL_REQUIREMENT}, config
    )
    snapshot = await graph.aget_state(config)
    show(snapshot, "撞上 G1 之后")

    trace = snapshot.values.get("node_trace", [])
    print(f"\n  plan_generate 出现 {sum(1 for t in trace if t.startswith('plan_generate'))} 次"
          f"（G1 上限 3）—— 第 4 轮**没有发生**：闸门在它之前就放行了")
    state = snapshot.values
    print(f"  review_passed={state.get('review_passed')}（**仍是 False**：降级不等于通过）")
    print("  瑕疵通过 interrupt 载荷送到了前端 —— 这就是「降级必须透明」")


async def path3_revise() -> None:
    banner("路径 ③ 用户拒绝一次：链路 B 计数 +1，且 retry_count 清零")
    review = review_passing_after(1)      # 第 1 轮不通过，第 2 轮通过
    graph = build_graph(InMemorySaver(), overrides={"self_review": review})
    config = graph_config("walk-3")

    await graph.ainvoke(
        {"session_id": "walk-3", "user_requirement": FULL_REQUIREMENT}, config
    )
    show(await graph.aget_state(config), "第一次暂停（机器已经回炉过 1 次）")

    await graph.ainvoke(
        Command(resume={"action": "revise", "feedback": "第三天换成博物馆"}), config
    )
    show(await graph.aget_state(config), "用户提了修改意见之后")
    print("\n  ↑ 注意 retry_count 与 user_revision_count 是两个独立的计数器（§2.3）：")
    print("    用户介入后 retry_count 归零，机器重新拿到完整的自省预算 ——")
    print("    所以这一轮它又能「先失败一次再通过」，轨迹里多了一次 plan_generate。")

    await graph.ainvoke(Command(resume={"action": "approve"}), config)
    show(await graph.aget_state(config), "这次 approve，收尾")


async def path4_ask() -> None:
    banner("路径 ④ 追问一轮：第一轮停在追问，第二轮才进生成")
    review = review_passing_after(0)
    graph = build_graph(InMemorySaver(), overrides={"self_review": review})
    config = graph_config("walk-4")

    incomplete = TravelRequirement(destination="成都", days=3)   # 缺 travelers
    await graph.ainvoke(
        {"session_id": "walk-4", "user_requirement": incomplete}, config
    )
    snapshot = await graph.aget_state(config)
    show(snapshot, "第一轮：需求不全")
    print(f"\n    追问话术: {snapshot.values.get('pending_question')}")
    print("    ↑ 这是 P3 的模板占位；§4.1 要求 P4 换成 LLM 生成的自然问句")

    # 第二轮回同一 thread 注入新输入（§7.4 验证过的往返）
    await graph.ainvoke(
        {"user_query": "两个人去", "user_requirement": FULL_REQUIREMENT}, config
    )
    show(await graph.aget_state(config), "第二轮：需求补齐后再入图")


async def main() -> None:
    show_topology()
    await path1_happy()
    await path2_g1()
    await path3_revise()
    await path4_ask()
    banner("四条路径走完")
    print("  默认空壳审核永远不通过 —— 这是刻意的（nodes/self_review.py 有说明）：")
    print("  它让 G1 降级放行成为**默认可见**的行为，而不是要靠注入才看得到的分支。")


if __name__ == "__main__":
    asyncio.run(main())
