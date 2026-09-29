"""图行为测试（P3.5）—— 对应 docs/开发流程.md 的 P3 段。

开发流程对这个文件的要求是「**4 条路径全部符合预期**」，外加强制性的
「**反思审核节点不可被旁路**」结构性断言。所以结构是：

    一、拓扑断言        —— 图**长得对不对**（静态）
    二、四条路径        —— 图**走起来对不对**（动态）
    三、不可旁路        —— 硬红线 #3（静态 + 动态各一条）
    四、装饰器          —— 追踪、兜底、**以及不许吞掉 interrupt()**
    五、装配            —— 替身、配置、未知节点名

为什么拓扑和路径要分开：这是两类完全不同的故障。拓扑错了是**接线**问题
（改一行 builder 就能修），路径错了是**状态语义**问题（reducer、计数器、
interrupt 恢复）。混在一起排查时，你会分不清是线接错了还是节点写错了 ——
而 P4 遇到「模型不听话」时，第一件要做的事正是把这两者分开。
"""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StateSnapshot, interrupt

from app.graph.builder import RECURSION_LIMIT, build_graph, graph_config
from app.graph.nodes import DEFAULT_NODES, NodeFn, traced_node
from app.graph.nodes.decorators import MAX_ERROR_CHARS
from app.graph.nodes.user_confirm import invalid_reason
from app.graph.state import ReviewComment, TravelRequirement, TravelState

# ===========================================================================
# 工具
# ===========================================================================

# 一份「什么都不缺」的需求。路径 ①③ 用它直接跳过需求收集的追问分支。
FULL_REQUIREMENT = TravelRequirement(
    origin="北京", destination="成都", days=3, travelers=2, budget=5000.0
)


def node_names(trace: list[str]) -> list[str]:
    """`["plan_generate#2"]` → `["plan_generate"]`。

    `node_trace` 的条目带 `#N` 序号（§3.2），断言执行顺序时要先把序号剥掉。
    """
    return [entry.split("#")[0] for entry in trace]


def state_of(snapshot: StateSnapshot) -> TravelState:
    """从快照还原出**完整**的 `TravelState`。

    ⚠ 不能直接读 `snapshot.values[...]`：**`values` 只包含被写过的通道**，
    不是一份完整状态。实测（本文件写完后跑出来的）：

        「一次通过」路径里没人写过 `retry_count` → `snapshot.values["retry_count"]`
        抛 `KeyError`，而同一个 `TravelState()` 的默认值是 0。

    这不是 langgraph 的 bug，是通道式状态机的必然结果：没写过的通道就没有值，
    而 Pydantic 的默认值是**模型层**的概念。两者在 `model_validate` 这一步合流。

    **P5 的 API 层要照这个来**：读状态一律先过一遍 `TravelState.model_validate`，
    不要对 `snapshot.values` 做 `.get(..., 默认值)` 式的补救 ——
    那样默认值会散落在十几个调用点，且与 §3.1 的契约重复一遍。
    """
    return TravelState.model_validate(snapshot.values)


def review_failing_times(times: int) -> tuple[NodeFn, list[int]]:
    """造一个「前 `times` 轮不通过，之后通过」的审核替身。

    **返回它每一轮看到的 `retry_count`** —— 这是本文件里最重要的一处设计。
    链路 B 要求「用户介入后 `retry_count` 清零」（§2.3），而清零这件事
    **在图跑完之后是读不出来的**：第二次环跑完，计数器又变回 1 了。
    唯一能直接观测到它的位置，是**下一轮审核进来时读到的那个值**。

    于是：`seen == [0, 1, 0]` 就读作「第 1 轮从 0 开始、第 2 轮从 1 开始
    （说明第 1 轮不通过自增了）、第 3 轮又从 0 开始（说明用户修改后清零了）」。
    顺序本身就是证据，不用去猜。
    """
    seen: list[int] = []

    def node(state: TravelState, config: RunnableConfig) -> dict:
        seen.append(state.retry_count)
        if state.retry_count < times:
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

    return node, seen


def recording(node: NodeFn) -> tuple[NodeFn, list[str]]:
    """包一层，记录**每次执行时读到的 `user_feedback`**。

    为什么需要它：`user_feedback` 是覆盖语义，且**被消费后清空**（§3.2）——
    等图跑到下一次暂停，用户填的意见早就被生成节点读走并抹掉了，直接读状态
    只会得到空串。这不是 bug，是契约；但这样一来「反馈到底有没有送到生成节点」
    就没法在事后观测。包一层在**入口处**记下它读到的值，是唯一能看到传递的办法。
    """
    seen: list[str] = []

    def wrapper(state: TravelState, config: RunnableConfig) -> dict:
        seen.append(state.user_feedback)
        return node(state, config)

    return wrapper, seen


async def run_until_pause(
    graph, config: RunnableConfig, payload: dict
) -> StateSnapshot:
    """跑到图暂停（或结束），返回快照。**判定暂停一律用 `.interrupts`。**

    这里封成一个 helper 不只是省几行：它把「怎么算暂停」这个判断**收在一处**。
    §7.4 的实测结论是 `.next` 在第二次 `interrupt()` 时返回 `()`，
    与「图跑完了」无法区分 —— 若各处自己判断，漏改一处就会有一条测试
    把「暂停」误当成「结束」而**绿着通过**。
    """
    await graph.ainvoke(payload, config)
    return await graph.aget_state(config)


async def resume_with(graph, config: RunnableConfig, payload: dict) -> StateSnapshot:
    """带 resume 值继续跑，返回快照。"""
    await graph.ainvoke(Command(resume=payload), config)
    return await graph.aget_state(config)


# ===========================================================================
# 一、拓扑断言 —— 图长得对不对
# ===========================================================================


def test_graph_has_the_five_nodes(checkpointer):
    drawn = build_graph(checkpointer).get_graph()
    assert set(drawn.nodes) == {
        "__start__", "__end__",
        "requirement_collect", "plan_generate", "self_review", "user_confirm", "evaluate",
    }


def test_every_node_is_wired(checkpointer):
    """每个节点都必须有进有出 —— 孤立节点是「接了一半」的典型症状。

    `add_node` 成功了但忘了连边，图**能编译、能跑**，只是那个节点永远不执行。
    这与硬红线 #3 说的「反思节点形同虚设」是同一类故障，只是成因不同。
    """
    drawn = build_graph(checkpointer).get_graph()
    sources = {e.source for e in drawn.edges}
    targets = {e.target for e in drawn.edges}

    for name in ("requirement_collect", "plan_generate", "self_review",
                 "user_confirm", "evaluate"):
        assert name in sources, f"{name} 没有出边 —— 走进去就出不来了"
        assert name in targets, f"{name} 没有入边 —— 永远到不了"


def test_graph_config_pins_thread_id_and_recursion_limit():
    """G4 的 `recursion_limit` 必须由 `graph_config` 统一带上。

    漏写一处，那处的上限就退回 LangGraph 的默认值（25）—— 一个**不会报错**的
    差异：图在异常路径上会提前抛 `GraphRecursionError`，而日志里看不出原因。
    """
    config = graph_config("session-1")
    assert config["configurable"]["thread_id"] == "session-1"
    assert config["recursion_limit"] == RECURSION_LIMIT


# ===========================================================================
# 二、四条路径（开发流程 P3.4）
# ===========================================================================


async def test_path1_happy_path(checkpointer, fresh_settings):
    """路径 ①：需求完整 → 审核通过 → 用户 approve。**一次都不回炉。**"""
    fresh_settings(MAX_REVIEW_RETRY="3")
    review, _ = review_failing_times(0)          # 一上来就通过
    graph = build_graph(checkpointer, overrides={"self_review": review})
    config = graph_config("path1")

    snapshot = await run_until_pause(
        graph, config,
        {"session_id": "path1", "user_requirement": FULL_REQUIREMENT},
    )

    # 暂停在用户确认，而不是跑完
    assert snapshot.interrupts, "图没有停在 user_confirm"
    assert node_names(snapshot.values["node_trace"]) == [
        "requirement_collect", "plan_generate", "self_review",
    ]

    snapshot = await resume_with(graph, config, {"action": "approve"})
    state = state_of(snapshot)

    assert node_names(state.node_trace) == [
        "requirement_collect", "plan_generate", "self_review", "user_confirm", "evaluate",
    ]
    assert state.retry_count == 0
    assert state.user_confirmed is True
    assert state.final_plan == state.draft_plan
    assert state.stage == "done"
    assert state.eval_result is not None
    # 到 END 了，不再暂停
    assert snapshot.interrupts == ()


async def test_path2_g1_degrades_after_max_retries(checkpointer, fresh_settings):
    """路径 ②：机器重试到 G1 上限后**降级放行**。

    用的是**默认空壳审核**（永远不通过），所以这条测试同时也是「默认图长什么样」
    的记录：它一路撞到 G1 才停。
    """
    fresh_settings(MAX_REVIEW_RETRY="3")
    graph = build_graph(checkpointer)             # 不换任何节点
    config = graph_config("path2")

    snapshot = await run_until_pause(
        graph, config,
        {"session_id": "path2", "user_requirement": FULL_REQUIREMENT},
    )

    state = state_of(snapshot)
    names = node_names(state.node_trace)
    assert names.count("plan_generate") == 3, f"重试轮数不对：{names}"
    assert names.count("self_review") == 3, f"审核轮数不对：{names}"
    # 第 4 轮**没有发生** plan_generate —— 闸门就是在它之前放行的
    assert names[-1] == "self_review"
    assert state.retry_count == 3

    # 降级放行不是「假装通过」：review_passed 仍是 False，瑕疵在状态里看得见
    assert state.review_passed is False
    assert len(state.unresolved_errors) == 1

    # 而且这份瑕疵**必须传到了用户面前**（§5.2：降级必须透明）——
    # 状态里有没有是一回事，前端收没收到是另一回事，这里验的是后者
    payload = snapshot.interrupts[0].value
    assert payload["type"] == "plan_confirm"
    assert len(payload["unresolved_errors"]) == 1
    assert "尚未实现" in payload["unresolved_errors"][0]["detail"]
    assert payload["plan"], "确认载荷里没有行程正文"


async def test_path3_user_revision_resets_retry_count(checkpointer, fresh_settings):
    """路径 ③：用户拒绝一次 → `user_revision_count=1` 且 **`retry_count` 清零**。

    「清零」这件事的观测点见 `review_failing_times` 的说明：整跑完之后
    计数器又变了，只有下一轮审核读到的值能证明它。
    """
    fresh_settings(MAX_REVIEW_RETRY="3")
    review, seen = review_failing_times(1)        # 第 1 轮不通过，第 2 轮通过
    planner, feedback_seen = recording(DEFAULT_NODES["plan_generate"])
    graph = build_graph(
        checkpointer, overrides={"self_review": review, "plan_generate": planner}
    )
    config = graph_config("path3")

    snapshot = await run_until_pause(
        graph, config,
        {"session_id": "path3", "user_requirement": FULL_REQUIREMENT},
    )
    assert snapshot.interrupts
    assert state_of(snapshot).retry_count == 1, "第 1 轮不通过应当自增到 1"

    snapshot = await resume_with(
        graph, config, {"action": "revise", "feedback": "第三天换成博物馆"}
    )
    state = state_of(snapshot)

    assert state.user_revision_count == 1
    assert state.user_confirmed is False
    assert state.stage == "awaiting_user"
    # ⚠ 这里读不到 `stage == "planning"` —— `user_confirm` 在 revise 分支里确实
    # 把它置成了 planning，但**同一次 resume 内**图会继续跑生成 → 审核，
    # 审核通过后又把它改成 awaiting_user，然后才停。
    # 所以 `stage` 是个**瞬态值**：它只在日志里有意义，不要指望在快照里看到
    # 中间那一步。（P4 排查「状态怎么突然变了」时，这是第一个要想起的性质。）
    # `user_feedback` 现在是空的 —— **这是对的**：生成节点读过之后就清掉了
    # （§3.2 的「被消费后清空」），否则下一轮生成还会拿到上一轮的意见，
    # 用户会看到同一句话被反复执行。真正要验的是它**送到过**：
    assert feedback_seen == ["", "", "第三天换成博物馆", ""], (
        f"生成节点各轮读到的 user_feedback 是 {feedback_seen}。\n"
        "期望第 3 次（用户修改后的第一次生成）读到用户的原话，第 4 次又是空 ——\n"
        "后者证明它确实被消费掉了，而不是一直挂在那里。"
    )

    # ↓ 清零的直接证据：第 3 轮审核读到的 retry_count 又回到了 0
    assert seen == [0, 1, 0, 1], (
        f"审核各轮看到的 retry_count 是 {seen}，期望 [0, 1, 0, 1]。\n"
        "读法：第 1 轮的循环从 0 数到 1；用户修改之后，第 2 轮的循环**又从 0 数起**。\n"
        "若是 [0, 1, 1]（第三个值不是 0），说明 user_confirm 没有清零 retry_count。"
    )

    # 清零的**行为后果**：机器重新拿到完整的自省预算，所以第 2 轮循环
    # 一样是「先失败一次、再通过」—— 总计 4 次生成。
    # 反事实：若没清零，第 3 次审核看到的 retry_count 会是 1，
    # 按 `review_failing_times(1)` 的判据直接通过，那就只有 2 次生成。
    names = node_names(state.node_trace)
    assert names.count("plan_generate") == 4, (
        f"清零后机器应当重新获得完整预算（共 4 次生成）：{names}"
    )

    # 收尾：这次 approve
    snapshot = await resume_with(graph, config, {"action": "approve"})
    final = state_of(snapshot)
    assert final.user_confirmed is True
    assert final.stage == "done"
    # 两条链路的计数互不污染（§2.3）：用户改了 1 次，机器重试数不受影响
    assert final.user_revision_count == 1


async def test_path4_ask_then_plan_across_two_runs(checkpointer, fresh_settings):
    """路径 ④：需求缺失 → 第一轮停在追问，**第二轮才进生成**。

    这一条走的是 §7.4 验证过的「跑完 → 注入新输入 → 再次入图」往返，
    也是整个多轮补齐需求的机制基础。
    """
    fresh_settings(MAX_ASK_ROUNDS="3", MAX_REVIEW_RETRY="3")
    review, _ = review_failing_times(0)
    graph = build_graph(checkpointer, overrides={"self_review": review})
    config = graph_config("path4")

    # ---- 第一轮：缺 travelers ----
    incomplete = TravelRequirement(destination="成都", days=3)
    snapshot = await run_until_pause(
        graph, config, {"session_id": "path4", "user_requirement": incomplete}
    )

    assert snapshot.interrupts == (), "追问不是暂停，图应当走到 END"
    assert node_names(snapshot.values["node_trace"]) == ["requirement_collect"]
    assert snapshot.values["missing_fields"] == ["travelers"]
    assert snapshot.values["pending_question"], "停在追问却没生成问题"
    assert snapshot.values["stage"] == "collecting"
    assert snapshot.values["ask_round"] == 1

    # ---- 第二轮：需求补齐，再入同一 thread ----
    snapshot = await run_until_pause(
        graph, config,
        {"user_query": "两个人去", "user_requirement": FULL_REQUIREMENT},
    )

    names = node_names(snapshot.values["node_trace"])
    assert names == [
        "requirement_collect", "requirement_collect",
        "plan_generate", "self_review",
    ], f"两轮的轨迹应当能拼成一条时间线：{names}"
    assert snapshot.values["missing_fields"] == []
    assert snapshot.values["pending_question"] is None
    assert snapshot.values["ask_round"] == 2
    assert snapshot.values["user_query"] == "", "本轮输入应当被节点消费后清空"
    assert snapshot.interrupts, "需求齐了就该走到用户确认并暂停"


# ===========================================================================
# 三、硬红线 #3：反思审核不可被旁路
# ===========================================================================


def test_self_review_cannot_be_rewired_out(checkpointer):
    """**结构断言**：图里不存在一条绕过 `self_review` 的路径。

    真实案例（开发流程 P3）：某同类项目加了一条「产出超过 80 字就直接结束」的
    优化规则，结果反思节点**从来没被执行过**，形同虚设，而且没人发现。

    那条规则如果落在编排层，就表现为「把 `plan_generate` 直接连到 `user_confirm`」
    或者「给 `self_review` 的出口加一条无条件旁路」。下面三条断言分别堵住这两种改法。
    """
    drawn = build_graph(checkpointer).get_graph()

    # ① 生成节点的唯一出口是审核节点，且必须是**无条件边**
    #    （条件边意味着存在一条不经过审核的路径）
    outbound = [e for e in drawn.edges if e.source == "plan_generate"]
    assert len(outbound) == 1, f"plan_generate 有 {len(outbound)} 条出边，应当只有 1 条"
    assert outbound[0].target == "self_review"
    assert not outbound[0].conditional, (
        "plan_generate → self_review 变成了条件边 —— 条件边意味着"
        "存在某种状态可以跳过审核直达下游"
    )

    # ② 用户确认的**唯一**入口来自审核节点
    inbound = [e for e in drawn.edges if e.target == "user_confirm"]
    assert len(inbound) == 1, f"user_confirm 有 {len(inbound)} 个入口"
    assert inbound[0].source == "self_review", (
        f"user_confirm 的入口来自 {inbound[0].source} —— 有一条路径没经过审核"
    )

    # ③ 审核节点自己必须存在且有出边
    assert "self_review" in drawn.nodes
    assert any(e.source == "self_review" for e in drawn.edges)


async def test_self_review_actually_runs_before_user_confirm(checkpointer, fresh_settings):
    """**行为断言**：正常路径下 `node_trace` 里必须出现 `self_review`，且在确认之前。

    与上面那条是配套的：结构断言堵住「改接线」，这条堵住「接线没动但审核被
    短路」—— 比如让 `route_after_review` 永远返回 `"confirm"`。那时拓扑一模一样，
    只有轨迹能看出问题。
    """
    fresh_settings(MAX_REVIEW_RETRY="3")
    graph = build_graph(checkpointer)
    config = graph_config("bypass")

    snapshot = await run_until_pause(
        graph, config,
        {"session_id": "bypass", "user_requirement": FULL_REQUIREMENT},
    )

    # 暂停这一刻：审核必须是最后一道执行的关卡 —— 紧接着才轮到用户确认。
    # （`user_confirm#1` 此刻还不在轨迹里：节点被 interrupt 打断时它的写不会提交，
    #   这是 interrupt 语义的直接推论，见 scratch/verify_traced_node.py 的 Q2。）
    names = node_names(state_of(snapshot).node_trace)
    assert names[-1] == "self_review", f"停在确认前执行的最后一环不是审核：{names}"

    # 收尾后再看完整轨迹 —— 开发流程要求的那条断言是「self_review 出现在
    # user_confirm 之前」，只有在图跑完之后才断得完整。
    snapshot = await resume_with(graph, config, {"action": "approve"})
    names = node_names(state_of(snapshot).node_trace)

    assert "self_review" in names, f"审核节点没被执行：{names}"
    assert names.index("self_review") < names.index("user_confirm"), (
        "审核出现在用户确认之后 —— 把关环节形同虚设"
    )


# ===========================================================================
# 四、装饰器
# ===========================================================================


def test_traced_node_numbers_entries_from_state():
    """序号从**状态**里数，不从闭包变量里数。

    `interrupt()` 恢复时节点会从头重跑。任何「执行次数」的副作用都会被执行两次；
    把序号算成状态的纯函数，重跑时算出的还是同一个值，天然幂等。
    """
    wrapped = traced_node("demo")(lambda state, config: {"stage": "collecting"})

    first = wrapped(TravelState(), {})
    assert first["node_trace"] == ["demo#1"]

    # 状态里已有 1 条同名条目（模拟第二次执行）→ 应当记 #2
    second = wrapped(TravelState(node_trace=["demo#1"]), {})
    assert second["node_trace"] == ["demo#2"]

    # 别的节点的条目不该让序号虚高
    third = wrapped(TravelState(node_trace=["demo#1", "other#1", "other#2"]), {})
    assert third["node_trace"] == ["demo#2"]


def test_traced_node_owns_the_trace_field():
    """`node_trace` 归装饰器独占：节点若返回同名字段会被覆盖。

    两边都能写同一个字段的话，reducer 是 `add` —— 两条同名条目会一起进列表，
    于是「这个节点执行了几次」从此没有答案。
    """
    wrapped = traced_node("demo")(lambda state, config: {"node_trace": ["节点自己写的"]})
    result = wrapped(TravelState(), {})
    assert result["node_trace"] == ["demo#1"]


def test_traced_node_degrades_on_exception_and_keeps_the_trace():
    """异常兜底：降级而非中断（硬红线 #4 的同一条姿态）。

    抛出去会让「图跑到一半停了」变成一个 500；写进状态则前端还能显示是哪一步坏的。
    **失败的节点同样要留痕** —— 否则轨迹里少一段，而那正是最需要它的时候。
    """

    def boom(state, config):
        raise RuntimeError("上游工具炸了")

    result = traced_node("boom")(boom)(TravelState(), {})

    assert result["stage"] == "failed"
    assert "上游工具炸了" in result["error"]
    assert result["node_trace"] == ["boom#1"]


def test_traced_node_truncates_long_errors():
    """超长异常信息要截断：它会进 `error` 字段，而状态是每个 super-step 都落盘的。"""

    def boom(state, config):
        raise RuntimeError("x" * 5000)

    result = traced_node("boom")(boom)(TravelState(), {})
    assert len(result["error"]) <= MAX_ERROR_CHARS


async def test_decorator_does_not_swallow_interrupt(checkpointer):
    """**`interrupt()` 不能被异常兜底吞掉** —— 吞掉 = 整个 HITL 静默失效。

    ⚠ 判定「图暂停了没有」用 `snapshot.interrupts`，**不能用「有没有抛异常」**：
    暂停不是异常，`ainvoke` 会正常返回。实测（`scratch/verify_traced_node.py` Q1）：

        朴素版 `except Exception`：interrupts=()   error="GraphInterrupt: ..."
        修正版 `except GraphBubbleUp: raise`：interrupts=(Interrupt,)  error=None

    朴素版的症状是「确认页永远不出现」，日志里只有一行「节点抛异常了」——
    不崩、不报错，只是安静地不工作。这正是硬红线 #1 要防的那类故障。
    """

    def confirmer(state: TravelState, config: RunnableConfig) -> dict:
        interrupt({"type": "plan_confirm"})
        return {"user_confirmed": True}

    builder = StateGraph(TravelState)
    builder.add_node("confirm", traced_node("confirm")(confirmer))
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", END)
    graph = builder.compile(checkpointer=checkpointer)

    config = graph_config("swallow-test")
    await graph.ainvoke({"session_id": "swallow-test"}, config)
    snapshot = await graph.aget_state(config)

    assert snapshot.interrupts, (
        "图没有暂停 —— 装饰器把 GraphInterrupt 当成普通异常吞掉了。"
        "确认页永远不会出现，而日志里只会写「节点执行失败」。"
    )
    assert snapshot.values.get("error") is None
    assert snapshot.values.get("stage") != "failed"


# ===========================================================================
# 五、装配
# ===========================================================================


def test_unknown_override_raises(checkpointer):
    """替身写错节点名要**响亮地失败**。

    静默忽略的话，图照样编译、照样跑，只是替身没生效 —— 断言会在别处以一个
    看似无关的差异失败（「为什么审核没通过？」），而真正的原因（拼错了名字）
    在任何输出里都不出现。
    """
    with pytest.raises(ValueError, match="self_reveiw"):
        build_graph(checkpointer, overrides={"self_reveiw": lambda s, c: {}})


def test_overrides_do_not_change_topology(checkpointer):
    """替身换的是**行为**，不是**连线** —— 所以结构断言不受它影响。

    这条撑着上面那两条不可旁路断言的效力：如果替身能改拓扑，那结构断言就
    只是在断言「默认图的样子」，而测试用的图可能完全是另一张。
    """
    plain = build_graph(checkpointer).get_graph()
    swapped = build_graph(
        checkpointer, overrides={"self_review": lambda s, c: {}}
    ).get_graph()

    def topology(drawn):
        return sorted((e.source, e.target, bool(e.conditional)) for e in drawn.edges)

    assert topology(plain) == topology(swapped)


async def test_override_node_is_still_traced(checkpointer, fresh_settings):
    """**替身节点也必须进 `node_trace`。**

    装饰器写在节点模块上的话，替身就得在测试里自己再包一层 —— 漏一处就静默
    丢轨迹，而丢的恰好是测试期间那段（最需要轨迹的时候）。
    """
    fresh_settings(MAX_REVIEW_RETRY="3")
    review, _ = review_failing_times(0)
    graph = build_graph(checkpointer, overrides={"self_review": review})
    config = graph_config("traced-override")

    snapshot = await run_until_pause(
        graph, config,
        {"session_id": "traced-override", "user_requirement": FULL_REQUIREMENT},
    )

    trace = snapshot.values["node_trace"]
    assert "self_review#1" in trace, f"替身没被追踪：{trace}"


async def test_empty_revise_feedback_re_asks_instead_of_failing(checkpointer, fresh_settings):
    """非法 resume 值 → **再问一次**，不是 500（§4.4 的异常输入处理）。

    依据 `scratch/verify_interrupt.py` 场景 B 的实测：同一节点内连续 `interrupt()`
    可行，且恢复时第 1 个 `interrupt()` 拿回的仍是上次那个值，所以
    「校验失败 → 再问 → 新值」这条链是确定性的。
    """
    fresh_settings(MAX_REVIEW_RETRY="3")
    review, _ = review_failing_times(0)
    graph = build_graph(checkpointer, overrides={"self_review": review})
    config = graph_config("re-ask")

    snapshot = await run_until_pause(
        graph, config,
        {"session_id": "re-ask", "user_requirement": FULL_REQUIREMENT},
    )
    assert snapshot.interrupts

    # 提意见却不写内容 —— 生成节点将「凭感觉再写一版」，而用户以为意见被采纳了
    snapshot = await resume_with(graph, config, {"action": "revise", "feedback": "   "})

    assert snapshot.interrupts, "非法输入应当再次暂停等输入，而不是放行或抛错"
    assert "input_error" in snapshot.interrupts[0].value
    assert state_of(snapshot).user_revision_count == 0, "非法输入不该计入修改次数"

    # 这次给合法输入，应当能正常收尾
    snapshot = await resume_with(graph, config, {"action": "approve"})
    final = state_of(snapshot)
    assert final.user_confirmed is True
    assert final.stage == "done"


def test_invalid_reason_accepts_valid_input():
    assert invalid_reason({"action": "approve"}) is None
    assert invalid_reason({"action": "revise", "feedback": "换成博物馆"}) is None


@pytest.mark.parametrize(
    ("raw", "hint"),
    [
        ({"action": "whatever"}, "action"),
        ({"action": "revise"}, "修改意见"),
        ({"action": "revise", "feedback": "   "}, "修改意见"),
        ("不是对象", "输入"),
        ({}, "action"),
    ],
)
def test_invalid_reason_rejects_bad_input(raw, hint):
    """坏输入必须给出**能看懂的中文原因**，而且原因里要指向出问题的那个字段。"""
    reason = invalid_reason(raw)
    assert reason is not None
    assert hint in reason, f"报错信息没指向 {hint}：{reason}"
