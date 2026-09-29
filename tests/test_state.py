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
from pydantic import ValidationError

from app.graph.nodes.plan_generate import STUB_PLAN_STRUCT
from app.graph.state import (
    ConfirmInput,
    PlanDay,
    PlanItem,
    PlanStruct,
    ReviewComment,
    TravelState,
)


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
# plan_struct 契约（P4.0）
# ===========================================================================
#
# 这些断言的价值不在「Pydantic 能不能校验格式」—— 那不用测。
# 它们钉的是**设计决定**：哪些问题该在结构层拦、哪些该留给 self_review，
# 以及哪些字段是代码填的而不是模型填的。这类决定没有测试就会被下一个人
# 当作遗漏「顺手补上」，而每一条都有明确的反面代价（写在各自的 docstring 里）。


def test_plan_item_normalizes_loose_time_but_rejects_garbage():
    """时分由代码归一化，归不了的不猜。

    「9:00」「9：00」→ `09:00`（纯机械补零，不该为它多烧一轮 LLM）；
    「下午2点」→ **报错**，让它走「只重试 JSON 段」。

    反面代价：若在这里猜（把「下午2点」当 02:00），时间冲突校验会拿着一个
    错数字认真工作 —— 一个「看起来通过了」的假阳性，比报错难查得多。
    """
    assert PlanItem(kind="attraction", name="X", start="9:00").start == "09:00"
    assert PlanItem(kind="attraction", name="X", start="09：00").start == "09:00"
    assert PlanItem(kind="attraction", name="X", start=" 9:00 ").start == "09:00"

    for bad in ("下午2点", "25:00", "9:0", "09:00-11:00", ""):
        with pytest.raises(ValidationError):
            PlanItem(kind="attraction", name="X", start=bad)


def test_plan_item_kind_has_no_transport():
    """`kind` 不含 `transport` —— 通勤段不归模型管。

    通勤时长由 `DistanceTool` 按经纬度算（精度远高于模型估的）。让模型也能
    输出 transport 条目，就会有两个数字（模型的、工具的）都要用，而它们
    必然打架。这条测试是那个决定的钉子（§3.1 的 PlanItemKind 注释）。
    """
    assert PlanItem(kind="attraction", name="X").kind == "attraction"
    with pytest.raises(ValidationError):
        PlanItem(kind="transport", name="从A到B")  # type: ignore[arg-type]


def test_plan_item_poi_id_defaults_to_none():
    """`poi_id` 缺省 None，且这是**幻觉信号**而不是「没填」。

    模型只输出 `name`，`poi_id` 由 `plan_generate` 的代码回查候选清单回填。
    回查不到就留 None —— 于是 self_review 的幻觉校验退化成一次 `is None` 判断，
    确定性的、零成本。

    反面代价：若让模型自己吐 id，它会编出一批**格式正确、库里没有**的 id，
    靠肉眼和靠「格式对不对」都发现不了，只能再查一次库才知道。
    """
    assert PlanItem(kind="attraction", name="宽窄巷子").poi_id is None
    assert PlanItem(kind="attraction", name="宽窄巷子", poi_id="cd-001").poi_id == "cd-001"


def test_plan_struct_does_not_validate_content():
    """**空行程在结构上合法** —— 这是刻意的，不是漏了校验。

    `days == []`、`items == []` 都是「内容不合理」，归 self_review 管。
    若在 Pydantic 层拦掉，失败会表现成 ValidationError，而 §4.2 对它的处置是
    「只重试 JSON 段」—— **真相却是「模型生成了一个空行程」，那该回炉重生成
    整个行程**。两条完全不同的处置，被一个校验混成了同一条路。

    判据：这件事的失败该走哪条处置。

    ⚠️ **必须显式传空列表，不能只断 `PlanStruct()`。** Pydantic 默认
    `validate_default=False` —— **缺省值不参与校验**。所以一个加在 `days` 上的
    `min_length=1` 对 `PlanStruct()` 完全无效，只有显式 `PlanStruct(days=[])`
    才会撞上它。变异测试⑤ 第一次就是这么骗过去的：只断缺省路径，等于没断。
    """
    # 缺省路径
    assert PlanStruct().days == []
    assert PlanDay(day_index=0).items == []

    # **显式空值路径** —— 内容校验若被加回来，撞上的是这一条
    assert PlanStruct(days=[]).days == []
    assert PlanDay(day_index=0, items=[]).items == []
    assert PlanStruct(days=[PlanDay(day_index=0, items=[])]).days[0].items == []


def test_plan_struct_nights_is_derived_from_days():
    """`nights` 是 `days` 的派生属性，且**不等于 hotel 条目数**。

    它回答「要订几晚」（算术：d 天睡 d-1 晚）；完整性校验问的是「行程里提没提
    住宿」（文本）。两个问题、两个答案，不该互相推导 —— 否则模型多写一条
    「第 3 晚也住酒店」就会让预算悄悄翻倍（§8.2 的住宿分项）。

    同时钉住它是派生属性：一旦有人加成字段，就有了第二份事实来源。
    """
    assert "nights" not in PlanStruct.model_fields
    assert PlanStruct().nights == 0
    assert PlanStruct(days=[PlanDay(day_index=0)]).nights == 0
    assert PlanStruct(days=[PlanDay(day_index=i) for i in range(3)]).nights == 2

    # 三天行程 + 三条住宿记录 → 晚数仍是 2，不是 3
    three_days = PlanStruct(
        days=[PlanDay(day_index=i, items=[PlanItem(kind="hotel", name="H")]) for i in range(3)]
    )
    assert three_days.nights == 2


def test_plan_models_are_frozen():
    """三个模型 frozen —— 防「共享可变常量被就地改动」这类跨会话污染。

    `STUB_PLAN_STRUCT` 是模块级常量、被所有 thread 共用。它若不是 frozen，
    某处一个 `stub.days[0].items[0].poi_id = "x"` 就会改到**所有会话**看到的
    那份 —— 症状是「A 会话的占位数据出现在 B 会话里」，而没人会去查常量。
    修改的唯一途径是 `model_copy(update=...)`，这也是 P4.2 回填 poi_id 的方式。
    """
    item = PlanItem(kind="attraction", name="X")
    with pytest.raises(ValidationError):
        item.name = "Y"  # type: ignore[misc]

    fixed = item.model_copy(update={"poi_id": "cd-001"})
    assert fixed.poi_id == "cd-001" and item.poi_id is None


def test_stub_plan_struct_is_well_formed_but_incomplete():
    """P3 的占位 `plan_struct`：**结构合法、内容不合格**。

    结构合法 —— 让 `PlanStruct` 的约束在 P3 的每条路径里都被真实构造一遍，
    否则这些约束要到 P4.2 才有第一行代码碰它们，那时出问题会被误当成
    「接入 LLM 引入的」。

    内容不合格 —— `poi_id` 全为 None，所以 P4.3 的幻觉校验落地后会**正确地**
    把它判成幻觉。让空壳在真校验下表现为「有问题」，而不是伪装成合格行程
    （与 `evaluate` 空壳给 0 分而不是 85 分同一条姿态）。
    """
    assert isinstance(STUB_PLAN_STRUCT, PlanStruct)
    assert STUB_PLAN_STRUCT.days, "占位数据必须真的有一天的内容，否则等于没构造"
    items = [item for day in STUB_PLAN_STRUCT.days for item in day.items]
    assert items, "同上"
    assert all(item.poi_id is None for item in items), "占位数据应当是「会被判为幻觉」的"


def test_travel_state_plan_struct_defaults_to_none():
    """`plan_struct` 缺省 None 而不是空 `PlanStruct()`。

    两者含义不同：None = **还没生成 / 生成失败**；空的 `PlanStruct` = 生成了一个
    空行程。self_review 对前者报「生成失败」（同样走回炉），对后者报「行程为空」——
    两条意见的文案和修复方向都不一样。
    """
    assert TravelState().plan_struct is None
    assert "plan_struct" in TravelState.model_fields


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
