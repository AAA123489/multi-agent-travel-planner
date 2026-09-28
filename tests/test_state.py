"""TravelState 契约测试（P1.2 / P1.7）。

分两层：

**单元层** —— 派生视图、可变默认值、子模型校验。不依赖 LangGraph。

**集成层**（`test_reducers_inside_real_graph`）—— 在**真实编译的图**上跑两轮，
断言 reducer 语义确实生效。这一层才是关键：`Annotated[list, add]` 写在
Pydantic 模型上能不能被 LangGraph 认出来，是一个**未经验证的假设**。
按本项目方法论（见 P0 探针），假设必须实测，不能靠文档推断 ——
万一不生效，`node_trace` / `review_history` / `messages` 会静默地「只保留最后一轮」，
而这种 bug 要到 P5 前端发现「轨迹只有一行」时才会暴露。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.graph.state import ConfirmInput, ReviewComment, TravelState


def _comment(severity: str, detail: str = "问题") -> ReviewComment:
    return ReviewComment(type="budget", severity=severity, detail=detail)


# ===========================================================================
# 单元层
# ===========================================================================


def test_mutable_defaults_are_not_shared():
    """两个实例的可变字段互不影响。

    这不是废话测试：若把 `Field(default_factory=list)` 写成 `= []`，
    Pydantic 会拦住，但若在别处用类属性持有列表，就会让**所有会话共享同一份轨迹**，
    症状是 A 用户的 node_trace 出现在 B 用户的结果里。
    """
    a, b = TravelState(), TravelState()
    a.missing_fields.append("destination")
    a.node_trace.append("[a] first")
    a.user_requirement.preferences.append("美食")

    assert b.missing_fields == []
    assert b.node_trace == []
    assert b.user_requirement.preferences == []


def test_unresolved_errors_only_errors():
    """只挑 error 级，warn 不算未解决问题（warn 允许降级放行）。"""
    state = TravelState(
        review_passed=False,
        review_comments=[
            _comment("error", "预算超了"),
            _comment("warn", "描述夸大"),
            _comment("error", "第2天闭馆"),
        ],
    )
    assert [c.detail for c in state.unresolved_errors] == ["预算超了", "第2天闭馆"]


def test_unresolved_errors_empty_when_passed():
    """审核通过时无未解决问题 —— 即便列表里还留着历史 error。"""
    state = TravelState(review_passed=True, review_comments=[_comment("error")])
    assert state.unresolved_errors == []


def test_unresolved_errors_is_derived_not_stored():
    """`unresolved_errors` 是派生视图，不是字段。

    钉住这个设计决定：一旦有人把它改成真实字段，就会出现第二份事实来源，
    与 `review_comments` 漂移。断言它不在 `model_fields` 里、也不出现在序列化结果里。
    """
    assert "unresolved_errors" not in TravelState.model_fields
    assert "unresolved_errors" not in TravelState().model_dump()


def test_no_draft_history_field():
    """**G5 已放弃 —— 钉住它的前置条件不存在**（§5.2，2026-09-28 定案）。

    G5（内容收敛检测）要比较「新旧 `draft_plan` 的相似度」，而 `draft_plan` 是
    **覆盖**语义，上一轮草稿已被覆盖，比无可比。要落地它，**第一步必然**是给
    `TravelState` 加一个累加语义的 `draft_history`（`Annotated[list[str], add]`）。

    所以这个字段的存在与否，就是「G5 有没有被重新捡起来」的判据。**哪天这条测试
    变红，说明有人正在加 `draft_history`** —— 那正是提醒你回去读 §5.2 的理由、
    确认这个决策是不是要推翻，而不是顺手把测试改绿。

    与 `test_intercity_backend_is_deliberately_absent`（tests/test_tools.py）
    是同一类钉子：**把「我们刻意没做某件事」写成可执行的断言。**
    设计决定不写成测试，就会被下一个人当作遗漏补回来。
    """
    assert "draft_history" not in TravelState.model_fields
    assert "draft_history" not in TravelState().model_dump()


def test_confirm_input_requires_known_action():
    """resume 值契约：action 只接受 approve / revise。"""
    assert ConfirmInput(action="approve").feedback == ""
    assert ConfirmInput(action="revise", feedback="第2天太赶").feedback == "第2天太赶"
    with pytest.raises(ValueError):
        ConfirmInput(action="maybe")  # type: ignore[arg-type]


# ===========================================================================
# 集成层 —— 在真实图上验证 reducer 语义
# ===========================================================================


def test_reducers_inside_real_graph():
    """在真实编译的图上，验证三种 reducer 语义真的按文档生效。

    两个节点各写一轮，共用一个 thread：

      node_trace       → add 累加：两轮都要在
      review_comments  → 覆盖：只剩第二轮（累加会让 LLM 回头修已修好的问题）
      messages         → add_messages 累加：两条都要在
    """
    observed = {}

    def first(state: TravelState):
        return {
            "node_trace": ["first"],
            "review_comments": [_comment("error", "第一轮的问题")],
            "messages": [HumanMessage(content="我想去成都")],
        }

    def second(state: TravelState):
        # 进第二个节点时，第一个节点的写入应该已经合并完毕
        observed["comments_len_before_second"] = len(state.review_comments)
        # 节点拿到的状态必须支持属性访问（而不是裸 dict）
        observed["attr_access_ok"] = state.review_comments[0].severity == "error"
        return {
            "node_trace": ["second"],
            "review_comments": [_comment("error", "第二轮的问题")],
            "messages": [AIMessage(content="好的")],
        }

    graph = StateGraph(TravelState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    compiled = graph.compile()

    final = compiled.invoke({}, {"configurable": {"thread_id": "t-state"}})

    # --- 前面节点的写入对后继节点可见，且是模型对象而非裸 dict ---
    assert observed["comments_len_before_second"] == 1
    assert observed["attr_access_ok"] is True

    # --- 累加（add）---
    assert final["node_trace"] == ["first", "second"]

    # --- 覆盖（无 reducer）：只剩第二轮，长度为 1 而非 2 ---
    assert len(final["review_comments"]) == 1
    assert final["review_comments"][0].detail == "第二轮的问题"

    # --- 累加（add_messages）---
    assert [m.content for m in final["messages"]] == ["我想去成都", "好的"]

    # --- 派生视图在真实图输出上同样可用 ---
    state = TravelState.model_validate(final)
    assert [c.detail for c in state.unresolved_errors] == ["第二轮的问题"]
