"""工具层契约测试（P1.4 / P1.7）—— 对应 docs/方案设计.md §6。

分三层：

**不变式层** —— `ToolResult` 的两条配套规则。这是本文件最值钱的部分：它守的是
**四个工具实现方的纪律**，而不只是几个字段。硬红线 #4「工具永不抛异常」靠它落地。

**字段边界层** —— `POI` / `OpeningHours` / `DistanceInfo` / `HotelPrice` 的取值约束。
边界值单独测，因为越界数据来自上游抓取（P2 的预处理），而**预处理写错了在运行时
是静默的**：一个 `rating=9.9` 会一路流到评估报告里变成「超出满分」。

**结构层** —— 把 P1.4 的两个设计决定钉成可执行的断言：门面与 Protocol 吃不同的
输入、`IntercityBackend` 刻意不存在。设计决定不写成测试，就会被下一个人顺手改掉。
"""

import inspect

import pytest
from pydantic import ValidationError

from app.tools import base
from app.tools.base import (
    POI,
    DistanceBackend,
    DistanceInfo,
    HotelBackend,
    HotelPrice,
    OpeningHours,
    POIBackend,
    ToolResult,
    TravelTools,
)


def _poi(**overrides) -> POI:
    fields = {
        "id": "cd_0001",
        "name": "武侯祠",
        "city": "成都",
        "type": "景点",
        "tags": ["人文"],
        "lat": 30.6431,
        "lng": 104.0442,
        "rating": 4.5,
        "price": 50.0,
    }
    return POI(**{**fields, **overrides})


# ===========================================================================
# ToolResult 的不变式 —— 硬红线 #4 的落地
# ===========================================================================


def test_success_requires_data():
    """`ok=True` 必须带 data。空列表算 data，`None` 不算。

    没有这条，一个「成功但没有内容」的结果会一路走到前端渲染处才炸。
    """
    with pytest.raises(ValidationError, match="必须带 data"):
        ToolResult[list].success(None)  # type: ignore[arg-type]


def test_success_accepts_empty_list():
    """**空列表是合法的 data** —— 「这个城市没有符合条件的景点」是一个正常结论，
    不是失败。混淆二者会让前端把「查无结果」显示成「系统出错」。
    """
    result = ToolResult[list[POI]].success([])
    assert result.ok is True
    assert result.data == []


def test_failure_requires_error_code():
    """`ok=False` 必须带 error_code。

    失败而不给 code，上层就无法区分「查不到」和「超时」，只能一律显示「出错了」。
    让它**在构造处**失败，比等线上冒出一个没有原因的失败要好。
    """
    with pytest.raises(ValidationError, match="必须带 error_code"):
        ToolResult[list](ok=False)


def test_failure_code_is_a_required_positional_argument():
    """`failure()` 的 `code` 是必填位置参数 —— 忘写是 `TypeError`，当场可见。

    这是刻意设计：比 `ValidationError` 更早暴露（连模型都不用构造）。
    """
    with pytest.raises(TypeError):
        ToolResult[list].failure("查不到")  # type: ignore[call-arg]


def test_success_and_failure_round_trip():
    """两个构造函数的正常路径 —— 顺带钉住 `source` 与 `warnings` 的默认值。"""
    ok = ToolResult[DistanceInfo].success(
        DistanceInfo(km=3.2, minutes=43, mode="taxi", is_estimated=True),
        source="amap",
        warnings=["价格为估算值"],
    )
    assert (ok.ok, ok.source, ok.error_code, ok.warnings) == (
        True,
        "amap",
        None,
        ["价格为估算值"],
    )

    bad = ToolResult[DistanceInfo].failure("上游超时", "timeout", source="amap")
    assert (bad.ok, bad.error, bad.error_code, bad.data) == (False, "上游超时", "timeout", None)


def test_error_code_is_a_closed_set():
    """`error_code` 是闭集 —— 拼错一个不在集合里的值会被拦住。

    闭集的意义是**上层可以穷举分支**：API 层能为五种故障各写一句提示，
    而自由字符串做不到这件事。
    """
    with pytest.raises(ValidationError):
        ToolResult[list](ok=False, error="炸了", error_code="oops")


def test_tool_result_is_generic_over_payload():
    """同一个 `ToolResult` 能收窄到不同载荷类型 —— 四个工具共用一个返回体。"""
    pois = ToolResult[list[POI]].success([_poi()])
    assert isinstance(pois.data[0], POI)
    assert pois.data[0].name == "武侯祠"


# ===========================================================================
# POI 字段边界 —— 越界数据来自预处理，而预处理写错在运行时是静默的
# ===========================================================================


def test_poi_minimal_construction():
    """最小合法构造：经纬度与评分必填，其余可省。"""
    poi = _poi(tags=[], price=None, address=None, opening_hours=None)
    assert poi.tags == []
    assert poi.price is None
    assert poi.price_estimated is False  # 默认：不是估算值


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lat", 91.0),
        ("lat", -91.0),
        ("lng", 181.0),
        ("lng", -181.0),
        ("rating", 5.1),
        ("rating", -0.1),
        ("price", -1.0),
    ],
)
def test_poi_numeric_bounds(field, value):
    """数值范围硬约束。

    `rating=9.9` 这类脏数据不会自己报错，它会静默流进评估报告变成「超出满分」。
    在入口拦掉，比在评估节点里做防御性钳制便宜得多。
    """
    with pytest.raises(ValidationError):
        _poi(**{field: value})


def test_poi_rejects_unknown_type():
    """`type` 是闭集，与 §9.1 预处理第 3 步的类型归一表对应。"""
    with pytest.raises(ValidationError):
        _poi(type="公园")


def test_poi_rejects_unknown_tag():
    """`tags` 同样闭集 —— 自由字符串会让「按 tag 匹配度排序」失去意义。"""
    with pytest.raises(ValidationError):
        _poi(tags=["随便"])


def test_poi_requires_coordinates():
    """经纬度缺失的行在预处理阶段整条丢弃（无法参与距离计算），故此处必填。"""
    with pytest.raises(ValidationError):
        POI(id="x", name="n", city="c", type="景点", rating=4.0)  # type: ignore[call-arg]


# ===========================================================================
# OpeningHours —— 「全天开放」与「解析失败」必须分得开
# ===========================================================================


def test_all_day_and_parse_failure_are_distinguishable():
    """**这是 `all_day` 字段存在的全部理由。**

    两者都表现为 `open is None`：

      - 全天开放 → `all_day=True`，**确定**开着
      - 解析失败 → 全留空，**未知**，只能 warn 后放行

    不区分就等于把「不知道」当成「开着」—— 用户按行程跑过去发现闭馆。
    """
    all_day = OpeningHours(all_day=True, raw="全天开放")
    unparsed = OpeningHours(raw="旺季8:00-18:00（解析失败）")

    assert all_day.open is None and unparsed.open is None  # 表面看一样
    assert all_day.all_day is True
    assert unparsed.all_day is False  # 靠这个分开


def test_opening_hours_keeps_raw_always():
    """`raw` 永远保留（宁缺勿错）—— 结构字段解析不出来时，人还能看原文。"""
    assert OpeningHours(raw="08:30-17:30").raw == "08:30-17:30"


def test_closed_days_uses_zero_indexed_monday():
    """0=周一 … 6=周日，对齐 `datetime.weekday()`。

    这条测试把「0 到底是周一还是周日」钉死 —— 它直接对齐 Python 的 weekday()，
    调用方不需要在判断「今天开不开」时做一次换算式（换算就是 bug 的温床）。
    """
    hours = OpeningHours(closed_days=[0, 6], raw="周一、周日闭馆")
    assert hours.closed_days == [0, 6]

    with pytest.raises(ValidationError):
        OpeningHours(closed_days=[7])  # 越界：一周只有 7 天
    with pytest.raises(ValidationError):
        OpeningHours(closed_days=[-1])


# ===========================================================================
# DistanceInfo / HotelPrice
# ===========================================================================


@pytest.mark.parametrize("mode", ["walk", "public", "taxi", "drive"])
def test_distance_modes_are_the_four_supported(mode):
    """四种交通方式是闭集，与 §3.1 的 `Transport` 别名保持一致。"""
    info = DistanceInfo(km=1.0, minutes=14, mode=mode, is_estimated=True)
    assert info.mode == mode


def test_distance_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        DistanceInfo(km=1.0, minutes=14, mode="fly", is_estimated=True)


def test_distance_rejects_negative_values():
    with pytest.raises(ValidationError):
        DistanceInfo(km=-1.0, minutes=14, mode="walk", is_estimated=True)
    with pytest.raises(ValidationError):
        DistanceInfo(km=1.0, minutes=-14, mode="walk", is_estimated=True)


def test_distance_estimated_flag_semantics():
    """`is_estimated` 是**必填**的，不给默认值。

    mock 永远算不出真实路网距离，高德能 —— 所以「这是估算值」是每个实现都必须
    明确表态的一件事，不该有一个默认值替它表态。
    """
    with pytest.raises(ValidationError):
        DistanceInfo(km=1.0, minutes=14, mode="walk")  # type: ignore[call-arg]


@pytest.mark.parametrize("level", ["budget", "comfort", "luxury"])
def test_hotel_levels_are_the_three_tiers(level):
    """三档对应 §6.3 的 P25 / P50 / P75 分位数。"""
    price = HotelPrice(city="成都", level=level, avg_price=420.0, sample_size=12)
    assert price.level == level
    assert price.is_approx is False


def test_hotel_rejects_unknown_level():
    with pytest.raises(ValidationError):
        HotelPrice(city="成都", level="deluxe", avg_price=420.0, sample_size=12)


def test_hotel_sample_size_allows_zero_but_flags_approx():
    """样本为 0 是合法状态 —— 它表示「该城市没有酒店类 POI」。

    合法，但必须靠 `is_approx=True` 让上游知道这个均价不硬（§6.3 的兜底路径）。
    """
    price = HotelPrice(city="某某市", level="comfort", avg_price=380.0, sample_size=0, is_approx=True)
    assert price.sample_size == 0
    assert price.is_approx is True


# ===========================================================================
# 结构层 —— 把 P1.4 的设计决定钉成断言
# ===========================================================================


def test_facade_and_backend_take_different_inputs():
    """**钉住 P1.4 的分层决定。**

    工具门面吃 id（agent 档下 LLM 只能从 JSON 里给字符串），按域 Protocol 吃 POI
    对象（距离是经纬度的纯函数）。id → 对象的转换**只在门面一处**。

    看着像废话，但它守的是「距离计算不依赖数据层」——若有人图省事把 Protocol
    也改成吃 id，每个 distance backend 都得自带一份 POI 索引，测试会在这里红。
    """
    facade = list(inspect.signature(TravelTools.estimate_distance).parameters)
    backend = list(inspect.signature(DistanceBackend.distance_between).parameters)

    assert facade == ["self", "from_id", "to_id", "mode"]
    assert backend == ["self", "a", "b", "mode"]


@pytest.mark.parametrize(
    ("protocol", "method"),
    [
        (POIBackend, "query_poi"),
        (POIBackend, "get_opening_hours"),
        (DistanceBackend, "distance_between"),
        (HotelBackend, "get_hotel_price"),
        (TravelTools, "poi_query"),
        (TravelTools, "opening_hours"),
        (TravelTools, "estimate_distance"),
        (TravelTools, "hotel_price"),
    ],
)
def test_protocol_methods_return_tool_result(protocol, method):
    """**Protocol 一律返回 `ToolResult`，不是裸值。**

    裸返回值（早先写的 `-> list[POI]`）**表达不了失败** —— 想报「查不到」就只剩
    抛异常一条路，与硬红线 #4 正面冲突。这条测试守住那个修正。
    """
    returns = inspect.signature(getattr(protocol, method)).return_annotation
    assert "ToolResult" in str(returns)


def test_intercity_backend_is_deliberately_absent():
    """**钉住「不预留空 Protocol」这个决定**（§6.4）。

    `TravelState` 里还没有城际交通段这个字段，`plan_struct` 也没定义它 ——
    契约未定义时写下的 Protocol 只能长出假形状，日后还得改。
    所以只留 `INTERCITY_BACKEND` 配置项占位，接口等产品契约定完再补。

    哪天契约定完了，**这条测试会红** —— 那正是提醒你同时改 §6.4 的信号。
    """
    assert not hasattr(base, "IntercityBackend")


def test_node_only_depends_on_the_facade():
    """节点只需要认识 `TravelTools` 一个类型。

    三个按域 Protocol 是 registry 组装时才用到的 —— 这样「POI 用 mock、距离用
    高德」怎么订，节点代码完全看不见，也就不可能出现 backend 分支（§6.4）。
    """
    assert TravelTools is not None
    # 门面暴露的四个方法就是 §6.3 那张表的四个工具，不多不少
    assert {name for name in vars(TravelTools) if not name.startswith("_")} == {
        "poi_query",
        "opening_hours",
        "estimate_distance",
        "hotel_price",
    }
