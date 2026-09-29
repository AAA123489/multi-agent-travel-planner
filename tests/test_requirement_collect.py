"""`requirement_collect` 的测试（P4.1）—— 对应 §4.1 / §5.1。

**全部离线**：`tests/conftest.py` 的 autouse `fake_llm` 把每个节点模块里的
`build_llm` 都换成了替身。这里只测「节点怎么跟 LLM 打交道」与「拿回来的东西
怎么处理」，不测「模型答得好不好」—— 后者是 prompt 的事，靠 `scratch/verify_*`
那种真实调用探针看，不是靠断言。

分七层，从里到外：契约 → 纯函数 → 消息 → 失败处置 → 节点 → 路由 → 守卫。
"""

import inspect
import json
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core import llm as llm_module
from app.core.exceptions import LLMError
from app.graph.builder import build_graph, graph_config
from app.graph.edges import route_after_collect
from app.graph.nodes import DEFAULT_NODES
from app.graph.nodes.requirement_collect import (
    FALLBACK_QUESTION,
    MAX_OPTIONAL_ASKS,
    REQUIRED_FIELDS,
    SUPPORTED_DESTINATIONS,
    build_ask_messages,
    build_extract_messages,
    fallback_question,
    merge_delta,
    missing_optional,
    missing_required,
    normalize_start_date,
    render_requirement,
    requirement_collect,
    unsupported_destination,
)
from app.graph.prompts import load_prompt
from app.graph.state import TravelRequirement, TravelRequirementDelta, TravelState


def _delta(**kwargs) -> TravelRequirementDelta:
    return TravelRequirementDelta(**kwargs)


# ===========================================================================
# 一、契约：delta 模型与需求模型必须逐字段对齐
# ===========================================================================


def test_delta_and_requirement_have_the_same_fields():
    """**字段集合必须逐字相同。**

    少一个字段 = 那个字段永远抽不出来，而症状只是「追问里没有它」/「生成时按默认值
    兜底」—— 没有报错、没有异常，只是那个能力悄悄不存在了。多一个字段 = 模型会往
    一个状态里没有的键上写东西，合并时被丢掉，同样无声。

    两个模型相邻放在 `state.py`，就是为了让这条断言看起来像废话 —— 而它成为废话的
    前提，是真的有人比对过。
    """
    assert set(TravelRequirementDelta.model_fields) == set(TravelRequirement.model_fields)


def test_delta_declares_no_required_fields():
    """**全字段可空是 delta 模式的载体，不是风格选择**（§4.1 要点 1 / §12.2）。

    function calling 的参数 schema 直接来自这个模型。只要 `required` 里出现一个名字，
    服务端就会**逼模型给那个字段编一个值** —— 用户没说人数，模型也只能填一个，
    而那个数字会一路流进行程里，再没人分得清它是用户说的还是模型猜的。
    """
    assert TravelRequirementDelta.model_json_schema().get("required", []) == []


def test_every_delta_field_defaults_to_none():
    """缺省值全是 `None`：模型没提供的字段不进 `model_fields_set`，`exclude_unset` 才有东西可排。"""
    empty = TravelRequirementDelta()
    assert empty.model_dump(exclude_unset=True) == {}
    for name in TravelRequirementDelta.model_fields:
        assert getattr(empty, name) is None


def test_delta_fields_carry_descriptions_for_the_model():
    """每个字段都要带 `description` —— 那是模型理解「总预算」「城市名不带市」的唯一来源。

    `TravelRequirement` 的字段注释是写给读代码的人看的，两者内容不同，所以 delta
    不能从它派生（见 `state.py` 里那段说明）。
    """
    missing = [
        name
        for name, field in TravelRequirementDelta.model_fields.items()
        if not field.description
    ]
    assert missing == [], f"这些字段没有 description，模型只能靠字段名猜：{missing}"


# ===========================================================================
# 二、合并：增量语义
# ===========================================================================


def test_merge_applies_only_the_fields_the_model_filled():
    merged = merge_delta(TravelRequirement(origin="北京"), _delta(days=3, travelers=2))
    assert (merged.days, merged.travelers, merged.origin) == (3, 2, "北京")


def test_merge_keeps_old_values_for_fields_the_model_left_alone():
    """delta 模式的全部意义：本轮没提到目的地，**不能把它抹掉**。"""
    base = TravelRequirement(destination="成都", days=3, travelers=2)
    merged = merge_delta(base, _delta(origin="上海"))
    assert (merged.destination, merged.days, merged.origin) == ("成都", 3, "上海")


def test_merge_treats_an_all_null_dump_as_no_change():
    """**反面样本：模型把全部字段都列出来、值全是 `null`。**

    这时每个键都「被 set 过」，光靠 `exclude_unset` 一个都排不掉 —— 合并的结果是把
    用户前几轮说的全部抹掉，然后从第一问重新开始。P4.0 实测 10/10 次模型都没这么干，
    但那是**实测，不是保证**；而它一旦发生，表现是「看起来像开了个新会话」，
    没有人会想到是合并规则干的。这条测试钉的就是那个多出来的 `exclude_none`。
    """
    base = TravelRequirement(destination="成都", days=3, travelers=2)
    dumped = TravelRequirementDelta.model_validate(
        dict.fromkeys(TravelRequirementDelta.model_fields)
    )
    assert dumped.model_dump(exclude_unset=True) != {}, "前提：这些键确实都算 set 过"
    assert merge_delta(base, dumped) == base


def test_merge_appends_preferences_instead_of_replacing_them():
    """第 2 轮说「还想吃点好的」，是往第 1 轮的「人文」上加，不是把它换掉。"""
    base = TravelRequirement(preferences=["人文"])
    assert merge_delta(base, _delta(preferences=["美食"])).preferences == ["人文", "美食"]


def test_merge_dedupes_preferences_when_the_model_echoes_the_whole_list():
    """模型顺手把它看到的完整列表复述一遍是很自然的行为 —— 追加语义必须容忍它。"""
    base = TravelRequirement(preferences=["人文", "美食"])
    merged = merge_delta(base, _delta(preferences=["人文", "美食", "夜生活"]))
    assert merged.preferences == ["人文", "美食", "夜生活"]


def test_merge_leaves_preferences_alone_on_an_empty_list():
    """没提偏好时模型可能回一个空列表 —— 那不该被当成「用户取消全部偏好」。

    理由与 `exclude_none` 那条一致：空列表和「没提」携带的信息一样多。
    """
    base = TravelRequirement(preferences=["人文"])
    assert merge_delta(base, _delta(preferences=[])).preferences == ["人文"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-10-01", "2026-10-01"),
        ("2026/10/1", "2026-10-01"),
        ("2026.10.1", "2026-10-01"),
        ("2026年10月1日", "2026-10-01"),
        (" 2026-10-01 ", "2026-10-01"),
        ("2026-1-5", "2026-01-05"),   # 补零
    ],
)
def test_normalize_start_date_accepts_purely_mechanical_rewrites(raw, expected):
    """纯格式差异归代码 —— 换分隔符、补零都不需要任何外部信息。

    与 `PlanItem._normalize_hhmm` 把「9:00」补成「09:00」是同一条判据：
    归一化需要的信息在谁手上。
    """
    assert normalize_start_date(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "10月1号",          # 缺年份 → 要 today → LLM 的活
        "国庆",             # 同上
        "下周三",           # 同上
        "2026-13-45",       # 格式对、日子不对
        "2026-02-30",
        "2026-10-01 下午",  # 带着没法归一的部分，宁可重问
        "出发日期待定",
    ],
)
def test_normalize_start_date_rejects_anything_it_cannot_normalize_for_sure(raw):
    """**归不了的绝不当成近似值猜出来。**

    猜成「大概是那几天」的日期不会报错 —— 它会在时间冲突校验、闭馆校验里一路畅通
    （那些校验只认格式），直到用户照着它订了机票才暴露。
    """
    assert normalize_start_date(raw) is None


def test_merge_drops_an_unparseable_start_date_so_it_gets_asked_again():
    """归一化失败退化成「多问一轮」，**不是**「带着脏日期往下走」。"""
    merged = merge_delta(TravelRequirement(), _delta(start_date="10月1号"))
    assert merged.start_date is None
    assert "start_date" in missing_optional(merged), "丢掉之后应当回到「可以再问」的状态"


def test_merge_normalizes_a_parseable_start_date():
    merged = merge_delta(TravelRequirement(), _delta(start_date="2026/1/5"))
    assert merged.start_date == "2026-01-05"


# ===========================================================================
# 三、范围判定：判据是数据，不是地理
# ===========================================================================


def test_supported_destinations_match_the_poi_data():
    """白名单与 `data/poi_clean.json` 里实际有的城市必须一致。

    **这是防漂移，不是查重复。** 加了新城市的 POI 却忘了改白名单，症状是
    「数据明明在，系统却说没有」；反过来删了城市不改白名单，症状是「说支持，
    却一个景点都查不出来」。两个方向都要到真正排行程时才暴露。
    """
    poi_file = Path(__file__).resolve().parents[1] / "data" / "poi_clean.json"
    cities = {row["city"] for row in json.loads(poi_file.read_text(encoding="utf-8"))}
    assert set(SUPPORTED_DESTINATIONS) == cities


@pytest.mark.parametrize("city", ["巴黎", "东京", "西安", "成都周边", "成都市"])
def test_unsupported_destination_refuses_and_names_what_it_can_do(city):
    """**判据是「有没有这座城市的 POI 数据」，不是「在不在境外」**（模块 docstring 的偏差①）。

    按关键词判境外会漏掉「西安」—— 那同样一条数据都没有，却会一路走到生成阶段，
    拿不到任何 POI，最后交出一份每个景点都是幻觉的行程。那比直接说「不支持」糟得多。

    「成都市」也在拒绝之列是**刻意的**：城市名归一化是 prompt 的职责（正反例在
    `requirement_collect_extract.md` 里），代码侧只做精确匹配 —— 在查询期再做一遍
    模糊匹配就等于有两处城市名规则（§6.5）。
    """
    message = unsupported_destination(TravelRequirement(destination=city))
    assert message is not None
    for supported in SUPPORTED_DESTINATIONS:
        assert supported in message, "话术里必须写清能做什么，不能只说「不行」"


@pytest.mark.parametrize("city", ["成都", "杭州"])
def test_supported_destination_passes_through(city):
    assert unsupported_destination(TravelRequirement(destination=city)) is None


def test_unknown_destination_is_not_a_rejection_but_a_question():
    """还没抽到目的地 ≠ 不支持 —— 那是「还不知道」，该走追问那条路。

    两者混淆的后果很重：第 1 轮用户只说「两个人」，目的地还是 `None`，
    要是判成「不支持」，整个会话在第一句话就被拒了。
    """
    assert unsupported_destination(TravelRequirement(travelers=2)) is None


# ===========================================================================
# 四、消息：prompt 从哪来、什么放哪一轮
# ===========================================================================


def test_extract_prompt_is_read_from_the_md_file():
    """硬红线 #5：Prompt 放 `*.md`，不塞进 Python 字符串。

    断的是一个**相等关系**，所以把 prompt 抄回 Python 字符串里再改几个字，
    这条测试就会红 —— 它守的是「唯一事实来源」，不是「有没有这段文字」。
    """
    system = build_extract_messages(TravelRequirement(), "去成都", "2026-09-29")[0]
    assert isinstance(system, SystemMessage)
    assert system.content == load_prompt("requirement_collect_extract")


def test_today_goes_in_the_human_turn_not_the_system_prompt():
    """**系统提示必须逐字节相同，与今天是几号无关。**

    服务端 prompt cache 命中靠的是「前缀一样」（P4.0-C 实测：cheap 档 337 个
    input token 里 128 个是 `cache_read`）。把每天都会变的日期塞进系统提示，
    等于每天过零点就把这段缓存整段作废 —— 而这件事在功能上完全看不出来，
    只体现在账单上。
    """
    one = build_extract_messages(TravelRequirement(), "去成都", "2026-09-29")
    two = build_extract_messages(TravelRequirement(), "去成都", "2031-01-01")

    assert one[0].content == two[0].content, "系统提示里混进了会变的东西"
    assert isinstance(one[1], HumanMessage)
    assert "2026-09-29" in one[1].content
    assert "2031-01-01" in two[1].content


def test_extract_payload_carries_the_already_collected_requirement():
    """抽取那一次调用必须看到「已经收集到的需求」，否则它会重复抽旧的字段。"""
    human = build_extract_messages(
        TravelRequirement(destination="成都", days=3), "两个人", "2026-09-29"
    )[1]
    assert "成都" in human.content
    assert "days" in human.content, "渲染里要带字段名，模型靠它回填参数"
    assert "两个人" in human.content


def test_render_requirement_says_so_when_nothing_is_known():
    """空需求不能渲染成空字符串 —— 那在 prompt 里就是一段空白，模型无从判断。"""
    assert "没有" in render_requirement(TravelRequirement())


def test_render_requirement_hides_empty_and_default_values():
    """`preferences=[]` 是默认值，不该出现在「已收集」清单里 —— 那会让模型以为用户提过。"""
    assert "preferences" not in render_requirement(TravelRequirement(destination="成都"))


def test_ask_payload_names_the_missing_fields_in_chinese():
    """问句那一次调用拿到的是**代码算好的**缺失清单，不是让模型自己去比。

    缺失清单**只给中文名，不给参数名**（与「已收集」那一段刻意不同）：
    这里给字段名的唯一效果，是让「days」这个词有更多机会漏进最终问句里 ——
    而 prompt 正在明确要求它别那么写。
    """
    human = build_ask_messages(
        TravelRequirement(destination="成都"), ["days", "travelers"], ["start_date"], "去成都"
    )[1]
    assert "行程天数" in human.content
    assert "出行人数" in human.content
    assert "出发日期" in human.content
    assert "成都" in human.content
    assert "- days" not in human.content, "缺失清单里不该出现参数名"
    assert "destination" in human.content, "「已收集」那一段要给参数名，模型靠它回填"


def test_ask_payload_records_a_silent_round_instead_of_leaving_a_blank():
    """用户这一轮什么都没说时，不能把空字符串塞进 payload —— 那会被读成「他说了空话」。"""
    human = build_ask_messages(TravelRequirement(), ["days"], [], "")[1]
    assert "没有补充" in human.content


def test_ask_prompt_is_read_from_the_md_file():
    system = build_ask_messages(TravelRequirement(), ["days"], [], "玩几天")[0]
    assert system.content == load_prompt("requirement_collect_ask")


# ===========================================================================
# 五、两次 LLM 调用的失败处置**不同**
# ===========================================================================


def test_extraction_failure_propagates_and_loses_nothing_silently(fake_llm):
    """抽取失败**往上抛**，不吞。

    吞掉它意味着节点带着「需求没变」往下走 —— 用户会以为系统没听见，说第二遍；
    或者在 G3 之后拿着空白需求去生成行程。抛出去则落进 `@traced_node` 的兜底，
    由 `route_after_collect` 判成 `reject` 就地终止，用户看得见出了什么事。
    """
    fake_llm.delta = LLMError("上游 500")
    with pytest.raises(LLMError):
        requirement_collect(TravelState(user_query="两个人"), {})


def test_question_failure_falls_back_to_a_template_and_keeps_the_extraction(fake_llm):
    """追问失败**必须吞掉** —— 判据是「失败了损失什么」。

    抽取失败 = 这一轮的信息全丢了；追问失败 = 只是这句话不好听。为了措辞把整个节点
    炸掉，等于让用户白白重说一遍刚才那句话：用一次措辞的失败换一次信息丢失，不划算。
    """
    fake_llm.delta = _delta(destination="成都")
    fake_llm.question = LLMError("话术挂了")

    updates = requirement_collect(TravelState(user_query="去成都"), {})

    assert updates["user_requirement"].destination == "成都", "抽到的字段必须保住"
    assert updates["pending_question"] == fallback_question(["days", "travelers"])


@pytest.mark.parametrize("garbage", ["", "   ", [{"type": "text", "text": "几个人？"}]])
def test_question_with_unusable_content_falls_back_to_the_template(fake_llm, garbage):
    """上游换了模型可能返回结构化块。`str(content)` 的产物是 `[{'type': ...}]`，
    会原样发给用户 —— 宁可退回模板。"""
    fake_llm.delta = _delta(destination="成都")
    fake_llm.question = garbage
    updates = requirement_collect(TravelState(user_query="去成都"), {})
    assert updates["pending_question"] == fallback_question(["days", "travelers"])


def test_fallback_template_uses_chinese_labels_not_field_names():
    """「请提供 days 字段」是很典型的机器腔 —— 兜底话术里也不许出现它。"""
    question = fallback_question(["days", "travelers"])
    assert question == FALLBACK_QUESTION.format(fields="行程天数、出行人数")
    assert "days" not in question


# ===========================================================================
# 六、节点行为
# ===========================================================================


def test_node_skips_the_llm_when_there_is_no_new_input(fake_llm):
    """**空的一轮不调 LLM。**

    模型盯着「已经收集到的需求」会往回猜，把它自己记住的值当成本轮新增再吐一遍 ——
    那正是 delta 模式要避免的事，而且白烧一次调用。此时 delta 为空 = 需求不变，
    下面照常算出还缺什么、照常追问。
    """
    updates = requirement_collect(
        TravelState(user_requirement=TravelRequirement(destination="成都")), {}
    )
    assert fake_llm.structured_schemas == [], "没有新输入却调了抽取"
    assert updates["missing_fields"] == ["days", "travelers"]


def test_node_uses_the_cheap_slot(fake_llm):
    """纯结构化抽取用便宜档（§4.1 表格）。

    槽位全退化到主力档时行为上完全看不出来，只体现在账单上 —— 所以这条得断言。
    """
    requirement_collect(TravelState(user_query="去成都"), {})
    assert fake_llm.slots == ["cheap"]


def test_node_asks_for_the_delta_schema_not_a_full_requirement(fake_llm):
    """节点必须按 **delta** 模式要 schema。要成全量模型的话，模型每轮都会把它
    「以为的完整需求」吐回来，前几轮的用户输入就全变成模型的复述了。"""
    requirement_collect(TravelState(user_query="去成都"), {})
    assert fake_llm.structured_schemas == [TravelRequirementDelta]


def test_node_merges_into_the_existing_requirement(fake_llm):
    fake_llm.delta = _delta(travelers=2)
    state = TravelState(
        user_query="两个人", user_requirement=TravelRequirement(destination="成都", days=3)
    )
    merged = requirement_collect(state, {})["user_requirement"]
    assert (merged.destination, merged.days, merged.travelers) == ("成都", 3, 2)


def test_node_reports_missing_fields_in_a_fixed_order(fake_llm):
    """顺序固定才断言得了，也才让问句每次都一样（set 序会让同样的输入产出不同问句）。"""
    updates = requirement_collect(TravelState(user_query="随便"), {})
    assert updates["missing_fields"] == list(REQUIRED_FIELDS)


def test_node_consumes_the_query_and_bumps_the_round(fake_llm):
    fake_llm.delta = _delta(destination="成都", days=3, travelers=2)
    updates = requirement_collect(TravelState(user_query="去成都三天两个人", ask_round=1), {})
    assert updates["user_query"] == ""
    assert updates["ask_round"] == 2


def test_node_marks_the_stage_and_clears_the_question_when_nothing_is_missing(fake_llm):
    """`pending_question` **每轮都要写**。留上一轮的值会让前端以为这轮还在等回答。"""
    fake_llm.delta = _delta(destination="成都", days=3, travelers=2)
    updates = requirement_collect(
        TravelState(user_query="去成都三天两个人", pending_question="几个人？"), {}
    )
    assert updates["missing_fields"] == []
    assert updates["pending_question"] is None
    assert updates["stage"] == "planning"


def test_node_stages_collecting_while_asking(fake_llm):
    fake_llm.delta = _delta(destination="成都")
    assert requirement_collect(TravelState(user_query="去成都"), {})["stage"] == "collecting"


def test_node_appends_the_question_to_messages_with_a_stable_id(fake_llm):
    """写进 `messages`（§4.1 要点 3）。

    id 取 `ask-{轮次}` 这种**确定性**的值：节点在恢复时会从头重跑，
    随机 id 会让同一句追问在轨迹里出现两遍。
    """
    fake_llm.delta = _delta(destination="成都")
    fake_llm.question = "几个人一起去呀？"
    updates = requirement_collect(TravelState(user_query="去成都"), {})

    appended = updates["messages"]
    assert len(appended) == 1
    assert isinstance(appended[0], AIMessage)
    assert appended[0].content == "几个人一起去呀？"
    assert appended[0].id == "ask-1"

    again = requirement_collect(TravelState(user_query="去成都"), {})["messages"]
    assert again[0].id == appended[0].id, "重跑同一轮不该产生第二条轨迹"


def test_node_does_not_write_messages_when_nothing_is_asked(fake_llm):
    """问齐了就不该再往对话里塞一条空消息。"""
    fake_llm.delta = _delta(destination="成都", days=3, travelers=2)
    assert "messages" not in requirement_collect(TravelState(user_query="去成都三天两人"), {})


def test_node_rejects_an_unsupported_destination_without_asking(fake_llm):
    """拒绝那条路**不该再花一次调用去生成问句** —— 问了也没用。"""
    fake_llm.delta = _delta(destination="巴黎", days=3, travelers=2)
    updates = requirement_collect(TravelState(user_query="想去巴黎玩三天"), {})

    assert updates["stage"] == "failed"
    assert updates["error"] and "成都" in updates["error"]
    assert updates["pending_question"] is None
    assert updates["missing_fields"] == []
    assert fake_llm.chat_calls == [], "已经决定拒绝了，不该再问一句"
    assert updates["user_query"] == "", "被拒绝的一轮同样是消费掉本轮输入"


def test_optional_fields_ride_along_on_a_round_that_is_already_asking(fake_llm):
    """可选字段**搭已经要问的那一轮的顺风车**，从不单独引出新一轮。

    这条区别写在测试里，是因为它反过来说也说得通、而且看起来更「周到」：
    「反正还缺出发日期，那就再问一轮吧」。**不要那么做** —— §5.2 的取舍是
    宁可交付一个写明了假设的方案，也不要问到用户失去耐心。必填齐了就往下走，
    出发日期由生成节点按「最近一个周末」兜底并在行程里标注。

    反过来，当这一轮本来就要追问时，多带一句「顺便，哪天出发？」几乎不花成本，
    却能省掉后面整整一轮 —— 所以它在这条路上。
    """
    fake_llm.delta = _delta(destination="成都")
    updates = requirement_collect(TravelState(user_query="去成都"), {})

    assert updates["missing_fields"] == ["days", "travelers"]
    assert "可选" in fake_llm.chat_calls[-1][1].content
    assert 0 < len(missing_optional(updates["user_requirement"])) <= MAX_OPTIONAL_ASKS


def test_optional_fields_never_justify_an_extra_round(fake_llm):
    """必填齐了就直接进生成 —— 哪怕出发日期、预算都还空着，也不再问一轮。"""
    fake_llm.delta = _delta(destination="成都", days=3, travelers=2)
    updates = requirement_collect(TravelState(user_query="去成都三天两人"), {})

    assert updates["missing_fields"] == []
    assert updates["stage"] == "planning"
    assert fake_llm.chat_calls == [], "必填齐了就不该再问 —— 问了就是白多一轮"


# ===========================================================================
# 七、路由，以及「真的接在图上」
# ===========================================================================


def test_missing_required_ignores_optional_fields():
    """必填判定只看三项 —— 这是 G3 与 §4.1 要点 2 的直接推论。"""
    assert missing_required(TravelRequirement(destination="成都", days=3, travelers=2)) == []


def test_route_rejects_a_failed_collect():
    """`"reject"` 是 P4.1 加的第三条出口（§5.1）。"""
    assert route_after_collect(TravelState(stage="failed")) == "reject"


def test_route_rejects_even_when_nothing_looks_missing():
    """**这条堵的是一个此前就存在的洞。**

    节点崩了的时候，`@traced_node` 返回 `{error, stage: failed}`，而 `missing_fields`
    保持上一轮的值 —— 第 1 轮它是空的，于是路由判成 `plan`，图带着一份**完全空白的需求**
    冲进生成节点。P4.1 之前没有任何一条出口拦得住它。
    """
    assert route_after_collect(TravelState(stage="failed", missing_fields=[])) == "reject"


def test_route_still_degrades_via_g3_instead_of_rejecting():
    """G3 与 reject 的处置**相反**，因为缺的东西不同：
    G3 缺的是「锦上添花的信息」，兜底就能往下走；reject 缺的是「能算的东西」，
    兜底只会产出一份假行程。"""
    stuck = TravelState(stage="collecting", missing_fields=["days"], ask_round=3)
    assert route_after_collect(stuck, max_ask_rounds=3) == "plan"


async def test_the_real_graph_stops_at_the_reject_route(checkpointer, fake_llm):
    """图级验证：目的地不支持时，`plan_generate` 一次都不许被跑到。"""
    fake_llm.delta = _delta(destination="巴黎", days=3, travelers=2)
    config = graph_config("reject-path")
    graph = build_graph(checkpointer)

    snapshot = await graph.ainvoke(
        {"session_id": "reject-path", "user_query": "想去巴黎玩三天，两个人"}, config
    )

    assert [entry.split("#")[0] for entry in snapshot["node_trace"]] == ["requirement_collect"]
    assert snapshot["stage"] == "failed"
    assert snapshot["error"]
    # `snapshot.values` **只含被写过的通道**（P3 踩过的坑②）：没人碰过 `draft_plan`，
    # 它压根不在字典里。所以这里要的是 `get`，不是下标 —— 下标会以 KeyError 的形式
    # 表现成「图的行为不对」，而真相是「这个字段这轮没被写过」，恰恰是我们要的结论。
    assert snapshot.get("draft_plan", "") == "", "拒绝之后不该有任何行程产出"


# ===========================================================================
# 八、守卫：替身必须装在**每一个**会调 LLM 的节点模块上
# ===========================================================================


def _node_modules_holding_build_llm():
    """从 `DEFAULT_NODES` 倒推，不手写清单（手写的清单会漏掉第六个节点）。"""
    modules = {inspect.getmodule(fn) for fn in DEFAULT_NODES.values()}
    return sorted(
        (m for m in modules if m is not None and hasattr(m, "build_llm")),
        key=lambda m: m.__name__,
    )


def test_every_node_module_holding_build_llm_got_the_fake(fake_llm):
    """**这条测试守的是「测试永远离线」这条纪律本身。**

    节点写的是 `from app.core.llm import build_llm` —— 那是往模块全局里放了一份引用。
    替身只打源模块的话，节点拿到的还是原函数，替身**形同虚设，而所有测试照旧是绿的**，
    只是又开始烧真 token。某次重构把 import 方式改掉之后，就会是这个局面。
    """
    modules = _node_modules_holding_build_llm()
    assert modules, "候选模块列表是空的 —— 这条断言自己失效了"
    assert "app.graph.nodes.requirement_collect" in [m.__name__ for m in modules]

    for module in modules:
        assert module.build_llm("cheap") is fake_llm, f"{module.__name__} 还拿着真的 build_llm"

    assert llm_module.build_llm("cheap") is fake_llm, "源模块没被打上，别的路径还会漏出去"


def test_no_real_chat_client_is_ever_constructed(fake_llm, monkeypatch):
    """**反面样本：把 `ChatOpenAI` 换成「一构造就炸」，再跑一遍节点。**

    这条不验行为，验的是**替身确实拦在了真正的出口之前**。只断言「结果对」是不够的 ——
    真调一次 API 也能得出同样的结果，只是慢、要 key、要钱，而这三样都不体现在
    「测试是绿的」上。与 `scratch/verify_llm_factory.py` 的反面样本同一条思路：
    一个不失败的反面样本，说明探针根本没连上被测对象。
    """

    def _boom(*args, **kwargs):
        raise AssertionError("构造了真的 ChatOpenAI —— 替身没拦住")

    monkeypatch.setattr(llm_module, "ChatOpenAI", _boom)
    updates = requirement_collect(TravelState(user_query="去成都三天两人"), {})
    assert updates["ask_round"] == 1, "节点没跑完，说明它半路去建真客户端了"


def test_the_fake_records_what_it_was_asked_for(fake_llm):
    """自测：替身要是没在记录，上面所有「断言调用形状」的测试都会变成空转。"""
    fake_llm.delta = _delta(days=3)
    requirement_collect(TravelState(user_query="三天"), {})
    assert fake_llm.structured_schemas and fake_llm.slots and fake_llm.payloads


def test_today_is_actually_injected_into_the_extract_prompt(fake_llm):
    """§4.1 要点 4：注入 `today`，让 LLM 自己把「10月1号」换算成 ISO。

    代码侧不换算 —— 那需要知道今天几号，而代码再算一遍就是两份事实来源。
    """
    requirement_collect(TravelState(user_query="国庆去成都"), {})
    assert date.today().isoformat() in fake_llm.payloads[0][1].content
