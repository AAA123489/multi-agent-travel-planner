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
import json

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import ConfigError
from app.tools import base
from app.tools import distance as distance_mod
from app.tools import hotel_price as hotel_mod
from app.tools import opening_hours as hours_mod
from app.tools import poi_query as poi_query_mod
from app.tools.backends.mock_backend import (
    MockDistanceBackend,
    MockHotelBackend,
    MockPOIBackend,
)
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
from app.tools.poi_store import POIStore
from app.tools.registry import build_tools


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


# ===========================================================================
# P2 · 纯函数层 —— 铁律②：测最脆弱的部分，而不是最难 mock 的部分
# ===========================================================================
#
# 这一节全部不需要文件、不需要 backend、不需要网络。它们是「算得对不对」的部分，
# 也是唯一能在 CI 里全速跑的部分。


def test_tag_score_counts_intersection():
    """评分 = 命中标签个数。"""
    poi = _poi(tags=["人文", "亲子", "美食"])
    assert poi_query_mod.tag_score(poi, []) == 0
    assert poi_query_mod.tag_score(poi, ["人文"]) == 1
    assert poi_query_mod.tag_score(poi, ["人文", "美食"]) == 2
    assert poi_query_mod.tag_score(poi, ["自然"]) == 0


def test_rank_prefers_more_matched_tags_over_higher_rating():
    """先比命中数，再比评分。

    这条钉住优先级：一个 4.9 分但只沾一个标签的点，排在 4.2 分沾两个标签的点之后。
    「满足偏好」优先于「口碑好」—— 用户说了想要人文，给美食是答非所问。
    """
    a = _poi(id="a", name="高分少匹配", tags=["人文"], rating=4.9)
    b = _poi(id="b", name="低分多匹配", tags=["人文", "自然"], rating=4.2)

    assert [p.id for p in poi_query_mod.rank([a, b], ["人文", "自然"], 10)] == ["b", "a"]


def test_rank_falls_back_to_rating_when_tags_empty():
    """`tags` 为空 = 没有偏好，所以按评分排 —— 不是「什么都不返回」。"""
    a = _poi(id="a", rating=4.0)
    b = _poi(id="b", rating=4.8)

    assert [p.id for p in poi_query_mod.rank([a, b], [], 10)] == ["b", "a"]


def test_rank_top_k_negative_means_unlimited():
    """`top_k < 0` 明确解释成「不限量」。

    不这么定义的话，`ordered[:-1]` 那种写法会静默丢最后一条 —— 一个负数参数
    换来「少了一个景点」，是那种要查到天亮的 bug。
    """
    pois = [_poi(id=str(i), rating=4.0 + i / 100) for i in range(5)]
    assert len(poi_query_mod.rank(pois, [], 2)) == 2
    assert len(poi_query_mod.rank(pois, [], -1)) == 5


def test_haversine_matches_a_known_pair():
    """武侯祠 → 宽窄巷子：直线约 3.17 km。

    用真实坐标而不是构造坐标：经纬度错位、纬度用错符号、把 `lng` 当 `lat` 传
    —— 这些错误在构造坐标上都测不出来。
    """
    wuhou = _poi(id="w", lat=30.6431, lng=104.0442)
    kuanzhai = _poi(id="k", lat=30.6696, lng=104.0564)
    assert distance_mod.haversine_km(wuhou, kuanzhai) == pytest.approx(3.17, abs=0.05)


def test_haversine_is_not_plain_pythagoras():
    """**Haversine 与平面勾股必须给出不同的答案**，否则这个测试没有意义。

    北纬 30.6° 处一个经度只有约 95.6 km，按 111.32 硬算会把东西向距离放大 16%。
    这条断言守的是「别有人图省事把实现换成勾股」—— 一旦换掉，两点距离立刻变大。
    """
    a = _poi(id="a", lat=30.6, lng=104.0)
    b = _poi(id="b", lat=30.6, lng=104.1)  # 纯东西向

    haversine = distance_mod.haversine_km(a, b)
    pythagoras = 0.1 * 111.32  # 把经度当纬度算的错法

    assert haversine < pythagoras * 0.95


def test_estimate_applies_road_factor_and_is_always_estimated():
    """`is_estimated` **恒为 True** —— mock 永远算不出真实路网距离。

    前端据此标一个「约」字。这个标记比距离本身更重要：它是「这个数字能不能
    当真」的载体，一旦默认成 False，用户会拿估算值做决策。
    """
    a = _poi(id="a", lat=30.6431, lng=104.0442)
    b = _poi(id="b", lat=30.6696, lng=104.0564)
    info = distance_mod.estimate(a, b, "taxi")

    assert info.is_estimated is True
    assert info.km == pytest.approx(distance_mod.haversine_km(a, b) * 1.3, abs=0.01)


@pytest.mark.parametrize(
    ("mode", "speed"),
    [("walk", 4.5), ("public", 15.0), ("taxi", 22.0), ("drive", 25.0)],
)
def test_estimate_uses_the_documented_speed_table(mode, speed):
    """四种方式的速度是 §6.3 定死的。改速度 = 改契约，这条会红。"""
    assert distance_mod.MODE_SPEED_KMH[mode] == speed

    a = _poi(id="a", lat=30.6, lng=104.0)
    b = _poi(id="b", lat=30.7, lng=104.0)  # 正北 0.1° ≈ 11.12 km
    info = distance_mod.estimate(a, b, mode)

    expected = max(1, round(info.km / speed * 60))
    assert info.minutes == expected


def test_estimate_never_reports_zero_minutes_for_distinct_places():
    """两个不同的地方不报「0 分钟」。

    50 米外说「0 分钟到达」更像 bug 而不是事实。**真正重合时返回 0** ——
    那不是兜底，那是真的。
    """
    near_a = _poi(id="a", lat=30.6, lng=104.0)
    near_b = _poi(id="b", lat=30.6001, lng=104.0)
    assert distance_mod.estimate(near_a, near_b, "walk").minutes == 1

    same = _poi(id="c", lat=30.6, lng=104.0)
    assert distance_mod.estimate(near_a, same, "walk").minutes == 0
    assert distance_mod.estimate(near_a, same, "walk").km == 0.0


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], 25.0, 1.75),
        ([1.0, 2.0, 3.0, 4.0], 50.0, 2.5),
        ([1.0, 2.0, 3.0, 4.0], 75.0, 3.25),
        ([1.0, 2.0, 3.0, 4.0], 0.0, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 100.0, 4.0),
        ([7.0], 25.0, 7.0),
        ([400.0, 100.0, 300.0], 50.0, 300.0),  # 输入乱序也要先排序
    ],
)
def test_percentile_matches_numpy_linear(values, q, expected):
    """分位数用**线性插值**，与 `numpy.percentile` 默认算法一致。

    特意对齐 numpy：P8/P9 做数据分析时脚本大概率用 numpy 复核这里的数字，
    算法不一致会出现「代码算 420、脚本算 460」这种要查半天的差异。
    手算的期望值写在上面的参数表里 —— 不调 numpy 生成，否则等于用被测对象
    验被测对象。
    """
    assert hotel_mod.percentile(values, q) == pytest.approx(expected)


def test_price_for_level_falls_back_below_min_sample():
    """样本 < 3 → 用全局经验值兜底，且 `is_approx=True`。

    **兜底不是失败**：预算估算是行程生成的必需输入，「这个城市没有足够数据」
    不该让用户拿不到行程。降级 + 透明标记，是 §5.2 默认失败姿态的具体样子。
    """
    price = hotel_mod.price_for_level([268.0, 520.0], "杭州", "comfort")

    assert price.is_approx is True
    assert price.avg_price == hotel_mod.HOTEL_FALLBACK_PRICE["comfort"]
    assert price.sample_size == 2  # 样本数如实报告，不是报 0


def test_price_for_level_uses_percentile_at_min_sample():
    """**恰好 3 条就该算分位数** —— 边界在 `< MIN_SAMPLE_SIZE`，不是 `<=`。"""
    price = hotel_mod.price_for_level([268.0, 520.0, 1180.0], "成都", "luxury")

    assert price.is_approx is False
    assert price.avg_price == pytest.approx(850.0)  # P75
    assert price.sample_size == 3


def test_fallback_prices_are_declared_per_level():
    """三档兜底值齐全且递增 —— 经济 < 舒适 < 豪华。

    递增这条不是废话：三档反了的话，「升级到豪华」会算出更低的预算，
    而每个数字单看都「像个价格」。
    """
    assert set(hotel_mod.HOTEL_FALLBACK_PRICE) == {"budget", "comfort", "luxury"}
    assert (
        hotel_mod.HOTEL_FALLBACK_PRICE["budget"]
        < hotel_mod.HOTEL_FALLBACK_PRICE["comfort"]
        < hotel_mod.HOTEL_FALLBACK_PRICE["luxury"]
    )


def test_split_known_keeps_request_order_and_dedupes():
    """`split_known` 保留请求顺序、去重、把缺失单独挑出来。

    保留顺序：调用方给的顺序就是它想展示的顺序。重排会让「第 3 天上午的景点」
    在结果里乱序。去重：同一个 id 报两次「查不到」是噪音。
    """
    hours = OpeningHours(open="08:00", close="18:00", raw="08:00-18:00")
    known, missing = hours_mod.split_known(
        ["a", "b", "a", "c", "b"], {"b": hours}
    )

    assert list(known) == ["b"]
    assert missing == ["a", "c"]


# ===========================================================================
# P2 · POIStore 的容错 —— 硬红线 #4 在数据层的延伸
# ===========================================================================
#
# 这一节的核心命题只有一句：**坏数据不许变成异常**。四种坏法各测一条。
# 每条都用 tmp_path 造一个真文件，不 mock 文件系统 —— 要测的正是
# 「读文件这一步出问题时会怎样」。


def _write(tmp_path, content: str):
    path = tmp_path / "poi.json"
    path.write_text(content, encoding="utf-8")
    return path


_VALID_ROW = {
    "id": "x1",
    "name": "某景点",
    "city": "成都",
    "type": "景点",
    "tags": ["人文"],
    "lat": 30.6,
    "lng": 104.0,
    "rating": 4.5,
    "price": 0.0,
}


def test_store_missing_file_is_not_an_exception(tmp_path):
    """文件不存在 → `load_error` 是 `not_found`，**不抛异常**。

    这一条是整个容错设计的起点：文件缺失是最常见的部署事故，而它一旦抛异常，
    「工具永不抛异常」这条纪律就得在每个调用点重复检查一遍。
    """
    store = POIStore.load(tmp_path / "nope.json")

    assert store.ok is False
    assert store.load_error is not None
    assert store.load_error[1] == "not_found"
    assert store.size == 0
    assert store.in_city("成都") == []


def test_store_malformed_json_is_upstream_error(tmp_path):
    """JSON 语法坏 → `upstream_error`。与「文件不存在」分开报 —— 修法完全不同。"""
    store = POIStore.load(_write(tmp_path, "{ 这不是 JSON"))

    assert store.ok is False
    assert store.load_error is not None
    assert store.load_error[1] == "upstream_error"


@pytest.mark.parametrize("content", ['{"a": 1}', '"字符串"', "42"])
def test_store_top_level_must_be_a_list(tmp_path, content):
    """顶层不是数组 → `upstream_error`。

    一个对象 / 字符串 / 数字也能被 `json.loads` 吃下去，然后在逐条校验时变成
    「一条都活不下来」。提前拦住，报错才说得清是「文件形状不对」。
    """
    store = POIStore.load(_write(tmp_path, content))

    assert store.ok is False
    assert store.load_error is not None
    assert store.load_error[1] == "upstream_error"


def test_store_skips_bad_row_and_keeps_the_rest(tmp_path):
    """**一条脏数据不该让整个城市全查不出来。**

    这条测试是这一节里最值钱的：它守的是「逐条容错」而不是「整批放弃」。
    坏法用的是真实会出现的两种 —— 评分越界（抓取脏值）、缺经纬度（预处理漏网）。
    """
    rows = [
        _VALID_ROW,
        {**_VALID_ROW, "id": "x2", "rating": 9.9},  # 评分越界
        {**_VALID_ROW, "id": "x3", "lat": None},  # 坐标缺失
        {**_VALID_ROW, "id": "x4", "name": "另一景点"},
    ]
    store = POIStore.load(_write(tmp_path, json.dumps(rows, ensure_ascii=False)))

    assert store.ok is True  # 有活下来的，就不是整批失败
    assert store.size == 2
    assert sorted(store.by_id) == ["x1", "x4"]
    assert len(store.warnings) == 2
    assert all("跳过" in w for w in store.warnings)


def test_store_warning_names_the_field(tmp_path):
    """坏行警告要点出**哪个字段**、**第几条** —— 否则等于没说。"""
    rows = [_VALID_ROW, {**_VALID_ROW, "id": "x2", "rating": 9.9}]
    store = POIStore.load(_write(tmp_path, json.dumps(rows, ensure_ascii=False)))

    assert len(store.warnings) == 1
    assert "第 1 条" in store.warnings[0]
    assert "rating" in store.warnings[0]


def test_store_keeps_first_on_duplicate_id(tmp_path):
    """重复 id 保留先出现的那条 + warning。

    不处理的话 `by_id` 与 `by_city` 会指向两条不同的记录 —— 同一个 id
    在两条路径上查出不同的东西，是最难查的一类漂移。
    """
    rows = [
        {**_VALID_ROW, "name": "先出现的"},
        {**_VALID_ROW, "name": "后出现的"},
    ]
    store = POIStore.load(_write(tmp_path, json.dumps(rows, ensure_ascii=False)))

    assert store.size == 1
    assert store.get("x1") is not None
    assert store.get("x1").name == "先出现的"
    assert any("重复" in w for w in store.warnings)


def test_store_all_rows_bad_is_reported_not_hidden(tmp_path):
    """有数据但一条都没活下来 → `upstream_error`，**不能表现成空 store**。

    空 store 对上层意味着「这个城市没有景点」，而真相是「文件配错了 /
    schema 变了」。两者都是空列表，但一个是正常结论、一个是事故。
    """
    rows = [{"id": "x1"}, {"id": "x2"}]  # 两条都不是合法 POI
    store = POIStore.load(_write(tmp_path, json.dumps(rows)))

    assert store.ok is False
    assert store.load_error is not None
    assert store.load_error[1] == "upstream_error"


def test_store_empty_file_is_ok_not_error(tmp_path):
    """`[]` 是合法的空知识库 —— 一条数据都没有，但文件本身没问题。

    与上一条对照着看：**「空」和「坏」必须分开**。空文件是 `ok=True` 的空 store。
    """
    store = POIStore.load(_write(tmp_path, "[]"))

    assert store.ok is True
    assert store.size == 0


def test_store_exposes_city_candidates(tmp_path):
    """`cities()` 给得出候选 —— 城市名写错时，报错信息里要能列出有什么。

    排序是**码点序**（`sorted` 对 str 的默认行为），不是拼音序 ——
    成(U+6210) 在 杭(U+676D) 之前。断言写成码点序是刻意的：说明这里不承诺
    「像给人看的顺序」，只承诺「稳定可复现」。要拼音序得装 pypinyin，
    为一句候选提示引入一个依赖不划算。
    """
    rows = [_VALID_ROW, {**_VALID_ROW, "id": "x2", "city": "杭州"}]
    store = POIStore.load(_write(tmp_path, json.dumps(rows, ensure_ascii=False)))

    assert store.cities() == ["成都", "杭州"]


# ===========================================================================
# P2 · 三个 mock backend —— 一致性 + 失败面
# ===========================================================================


def _store_with(rows: list[dict]) -> POIStore:
    """直接构造 store，**不走文件** —— 单测不该为了造数据先写一个临时文件。"""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(list[POI])
    pois = adapter.validate_python(rows)
    return POIStore(
        by_id={p.id: p for p in pois},
        by_city={c: [p for p in pois if p.city == c] for c in {p.city for p in pois}},
    )


def test_store_with_helper_is_equivalent_to_loading(tmp_path):
    """反证 `_store_with` 不是自欺欺人：它与真加载得到同样的索引。

    测试辅助函数也会写错。如果它构造出的 store 与 `POIStore.load` 行为不同，
    下面所有基于它的测试都在测一个不存在的东西。
    """
    rows = [_VALID_ROW, {**_VALID_ROW, "id": "x2", "city": "杭州"}]
    from_file = POIStore.load(_write(tmp_path, json.dumps(rows, ensure_ascii=False)))
    in_memory = _store_with(rows)

    assert set(from_file.by_id) == set(in_memory.by_id)
    assert from_file.cities() == in_memory.cities()
    assert from_file.size == in_memory.size


@pytest.mark.parametrize(
    ("backend_cls", "protocol"),
    [
        (MockPOIBackend, POIBackend),
        (MockDistanceBackend, DistanceBackend),
        (MockHotelBackend, HotelBackend),
    ],
)
def test_mock_backend_satisfies_its_protocol(backend_cls, protocol):
    """三个 mock 结构上满足各自的 Protocol。"""
    args = () if backend_cls is MockDistanceBackend else (_store_with([_VALID_ROW]),)
    assert isinstance(backend_cls(*args), protocol)


@pytest.mark.parametrize(
    ("backend_cls", "protocol"),
    [
        (MockPOIBackend, POIBackend),
        (MockDistanceBackend, DistanceBackend),
        (MockHotelBackend, HotelBackend),
    ],
)
def test_mock_backend_defines_protocol_methods_itself(backend_cls, protocol):
    """**方法必须定义在自己身上，不能靠继承。**

    这条测的是 base.py 里记下的那个实测结论：显式继承 Protocol 会让漏实现的
    方法**静默返回 `None`**。也就是说「class MockPOIBackend(POIBackend): pass」
    能通过上面的 isinstance 检查（方法名在），却在调用时返回 None。

    所以把「方法定义在自己 `__dict__` 里」写成断言 —— 漏实现一个方法时，
    变红的是这里，而不是生产环境里一句莫名其妙的 `NoneType has no attribute`。
    """
    missing = [name for name in protocol.__protocol_attrs__ if name not in backend_cls.__dict__]
    assert missing == [], f"{backend_cls.__name__} 没有自己实现：{missing}"


def test_mock_poi_backend_surfaces_store_failure():
    """store 载入失败 → 查询返回 `ok=False`，**错误原文来自 store**。

    错误信息不在这里重编：`POIStore` 已经把「文件不存在 / JSON 坏」区分好了，
    重写一遍就是第二份措辞，两边迟早对不上。
    """
    store = POIStore.empty("POI 数据文件不存在：xxx", "not_found")
    result = MockPOIBackend(store).query_poi("成都", ["人文"])

    assert result.ok is False
    assert result.error_code == "not_found"
    assert "不存在" in (result.error or "")


def test_mock_poi_backend_empty_city_is_ok_with_candidates():
    """城市查不到 → `ok=True` + 空列表 + warn 列出已载入的城市。

    空结果不是失败（P1.4 的不变式）。但要 warn：**「城市名写错了」和
    「这个城市真没有景点」在返回值里长得一模一样**，给个候选列表能让人一眼
    看出是哪种。
    """
    store = _store_with([_VALID_ROW])
    result = MockPOIBackend(store).query_poi("火星", ["人文"])

    assert result.ok is True
    assert result.data == []
    assert any("成都" in w for w in result.warnings)


def test_mock_distance_backend_rejects_unknown_mode_without_raising():
    """非法交通方式 → `invalid_param`，**不抛 `KeyError`**。

    类型上 `TravelMode` 已经限死了四个值，但工具对**任何**输入都不能抛异常
    （硬红线 #4），包括从 JSON 反序列化进来的脏值 —— 而 `MODE_SPEED_KMH[mode]`
    对未知 key 正是抛 `KeyError`。这条测试守的就是那个兜底。
    """
    a = _poi(id="a", lat=30.6, lng=104.0)
    b = _poi(id="b", lat=30.7, lng=104.0)
    result = MockDistanceBackend().distance_between(a, b, "rocket")  # type: ignore[arg-type]

    assert result.ok is False
    assert result.error_code == "invalid_param"


def test_mock_hotel_backend_ignores_null_prices():
    """`price=None` 表示「不知道」，**不是「0 元」**。

    把 None 当 0 参与分位数，P25 会被直接拉到 0 —— 整档均价失真，
    而数字看上去仍然「像个价格」。这条测的正是那个失真。
    """
    rows = [
        {**_VALID_ROW, "id": "h1", "type": "酒店", "price": None},
        {**_VALID_ROW, "id": "h2", "type": "酒店", "price": 300.0},
        {**_VALID_ROW, "id": "h3", "type": "酒店", "price": 500.0},
        {**_VALID_ROW, "id": "h4", "type": "酒店", "price": 700.0},
    ]
    result = MockHotelBackend(_store_with(rows)).get_hotel_price("成都", "budget")

    assert result.ok is True
    assert result.data is not None
    assert result.data.sample_size == 3  # 有价的 3 条，null 那条不算
    assert result.data.avg_price == pytest.approx(400.0)  # P25 of [300,500,700]
    assert result.data.is_approx is False


def test_mock_hotel_backend_warns_when_falling_back():
    """兜底时必须同时给出**机器看得懂的标记**和**人看得懂的话**。

    `is_approx` 进评估指标（§4.5 的 budget_fit），warning 进日志与界面。
    """
    rows = [{**_VALID_ROW, "id": "h1", "type": "酒店", "price": 300.0}]
    result = MockHotelBackend(_store_with(rows)).get_hotel_price("成都", "comfort")

    assert result.data is not None
    assert result.data.is_approx is True
    assert any("经验值" in w for w in result.warnings)


# ===========================================================================
# P2 · registry 分发 —— 未实现的后端要响亮地失败
# ===========================================================================


def _settings(**overrides) -> Settings:
    """构造一个**不读 .env** 的 Settings（显式传值，不调 from_env）。"""
    return Settings(llm_api_key="sk-test", **overrides)


@pytest.fixture
def tools(tmp_path):
    """一个装配好的门面，指向仓库里那份真实数据。

    刻意**不用临时文件**：P2 的验收标准就是「这份数据能被查出来」，
    用临时数据测等于绕过了要验的东西。数据文件本身的形状在下一节单独测。
    """
    return build_tools(_settings())


def test_registry_satisfies_the_facade_protocol(tools):
    """组装出来的门面满足 `TravelTools`。"""
    assert isinstance(tools, TravelTools)


def test_registry_exposes_exactly_the_four_tools(tools):
    """门面暴露的就四个方法与四个工具对象，不多不少。"""
    assert {n for n in vars(type(tools)) if not n.startswith("_")} == {
        "poi_query",
        "opening_hours",
        "estimate_distance",
        "hotel_price",
        "tools",
    }
    assert len(tools.tools) == 4


def test_registry_defaults_to_mock(tools):
    """默认配置下四个域全部落 mock —— P2 阶段这是唯一的实现。

    用**返回值的 `source`** 判断而不是读 registry 的内部字段：`source` 是写给
    上层的契约（§6.2），内部字段是实现细节。这样断言的是「用户看到的来源是
    mock」，而不是「某个私有属性等于某个字符串」。
    """
    sources = {
        tools.poi_query("成都", ["人文"]).source,
        tools.opening_hours(["cd_0001"]).source,
        tools.estimate_distance("cd_0001", "cd_0002", "walk").source,
        tools.hotel_price("成都", "comfort").source,
    }
    assert sources == {"mock"}


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("POI_BACKEND", "amap"),
        ("DISTANCE_BACKEND", "amap"),
        ("HOTEL_BACKEND", "aigohotel"),
    ],
)
def test_unimplemented_backend_raises_config_error(env_name, value):
    """配了没实现的后端 → **启动就失败**，不静默降级成 mock。

    静默降级会让「我明明配了高德」无从察觉，一直到演示时才发现距离是假的。
    报错里要写清是哪个环境变量，以及计划什么时候上（否则用户不知道该改配置
    还是该等）。
    """
    with pytest.raises(ConfigError, match=env_name):
        build_tools(_settings(**{env_name: value}))


def test_intercity_backend_is_rejected_too():
    """城际交通域同样拒绝 —— 它不是「换个数据源」，是「这个功能还不存在」。

    `TravelState` 里没有任何字段装城际交通段（§6.6），所以这里必须拦住。
    """
    with pytest.raises(ConfigError, match="INTERCITY_BACKEND"):
        build_tools(_settings(INTERCITY_BACKEND="variflight"))


def test_demo_mode_short_circuits_before_the_unsupported_check(monkeypatch):
    """`DEMO_MODE=true` 时即使配了没实现的后端也能起来。

    因为 `effective_backend` 把四个域全短路成 mock 了，`POI_BACKEND=amap`
    根本不会生效 —— 这正是这个开关的用途：演示防翻车。
    **但它必须走 `effective_backend` 而不是直接读字段**，否则这里会误报。
    """
    tools = build_tools(_settings(DEMO_MODE="true", POI_BACKEND="amap", HOTEL_BACKEND="aigohotel"))
    assert isinstance(tools, TravelTools)


def test_store_failure_reaches_the_tools(tmp_path):
    """数据文件坏了 → 每个工具都返回 `ok=False`，进程照常起来。

    这是**运行期**故障而不是配置写错，所以不走 `ConfigError` 那条路：启动期有
    `validate_paths()` 守文件存在，运行期坏了则降级成 `ok=False` 由上层处理。
    """
    tools = build_tools(_settings(POI_DATA_PATH=str(tmp_path / "nope.json")))

    assert tools.poi_query("成都", ["人文"]).ok is False
    assert tools.hotel_price("成都", "comfort").ok is False


# ===========================================================================
# P2 · 端到端 —— 拿仓库里那份真实数据跑通四个工具
# ===========================================================================


def test_real_data_file_loads(tools):
    """`data/poi_clean.json` 能被装配层读进来，且 30 条全部合法。"""
    result = tools.poi_query("成都", [], top_k=-1)
    assert result.ok is True
    assert len(result.data) == 15

    hangzhou = tools.poi_query("杭州", [], top_k=-1)
    assert hangzhou.ok is True
    assert len(hangzhou.data) == 15


def test_acceptance_wuhou_to_kuanzhai_is_three_to_six_km(tools):
    """**P2 验收标准之一**：武侯祠 → 宽窄巷子落在 3~6 km（和地图对得上）。

    这条不是普通单测，它是「假数据的经纬度大致真实」这个要求的可执行版本
    （开发流程 P2 假数据要求②）—— 经纬度抄错时，第一个红的就是它。
    """
    result = tools.estimate_distance("cd_0001", "cd_0002", "taxi")

    assert result.ok is True
    assert result.data is not None
    assert 3.0 <= result.data.km <= 6.0


def test_acceptance_all_four_tools_work_on_real_data(tools):
    """四个工具在真实数据上都能跑通，且返回的都是 `ok=True`。"""
    assert tools.poi_query("成都", ["人文"], top_k=5).ok is True
    assert tools.opening_hours(["cd_0001", "cd_0004"]).ok is True
    assert tools.estimate_distance("cd_0001", "cd_0002", "walk").ok is True
    assert tools.hotel_price("成都", "comfort").ok is True


def test_unknown_id_is_a_tool_result_not_an_exception(tools):
    """**P2 验收标准之二**：脏数据/坏 id 触发 `ok=False`，不是抛异常。"""
    result = tools.estimate_distance("cd_0001", "不存在", "taxi")

    assert result.ok is False
    assert result.error_code == "not_found"
    assert "不存在" in (result.error or "")


def test_opening_hours_missing_entries_are_warned_and_absent(tools):
    """查不到的开放时间：**不在 dict 里** + 有 warning。

    「在 dict 里但值是 None」会让每个调用点判两次 —— 而这两件事表达的是同一件
    事：不知道。所以用「缺席 + 警告」表达。
    """
    result = tools.opening_hours(["cd_0001", "根本没有这个 id"])

    assert result.ok is True
    assert list(result.data) == ["cd_0001"]
    assert any("未知" in w for w in result.warnings)


def test_real_data_exercises_all_opening_hours_shapes(tools):
    """真实数据必须**恰好覆盖**开放时间的四种形态（P2 假数据要求③）。

    只有四种都在，审核节点（P4）的四条分支才有真实输入可跑：
      全天开放 → all_day；正常区间 → open/close；闭馆日 → closed_days；
      解析失败 → 结构留空但 raw 在（§9.1 第 6 步：宁缺勿错）

    挂在真实数据上而不是构造数据上，是因为**这条同时守着「数据没被改坏」** ——
    有人把那条解析失败的记录顺手「修好」，这里就会红。
    """
    result = tools.opening_hours(
        ["cd_0001", "cd_0002", "cd_0004", "hz_0007"]
    )
    hours = result.data
    assert hours is not None

    assert hours["cd_0001"].open == "08:00"  # 正常区间
    assert hours["cd_0002"].all_day is True  # 全天开放
    assert hours["hz_0007"].closed_days == [0]  # 周一闭馆

    unparsed = hours["cd_0004"]  # 分季调整 → 解析失败
    assert unparsed.open is None and unparsed.close is None
    assert unparsed.all_day is False
    assert unparsed.raw  # 原文必须留着


def test_real_data_has_an_unpriced_hotel(tools):
    """数据里必须有**一条无价酒店**，它让 `杭州` 的样本数低于阈值。

    这是「故意留脏数据」在酒店域的落点：它让兜底路径由真实数据触发，
    而不是只在单测里被构造出来。改掉它，这条会红。
    """
    result = tools.hotel_price("杭州", "comfort")

    assert result.data is not None
    assert result.data.sample_size == 1
    assert result.data.is_approx is True


def test_search_url_points_at_a_real_page(tools):
    """铁律③：URL 指向**真实可点开的搜索页**，不是 `example.com` 死链。"""
    poi = tools.poi_query("成都", ["人文"], top_k=1).data[0]
    url = poi.search_url

    assert url.startswith("https://www.amap.com/search?")
    assert "query=" in url and "city=" in url
    assert "example.com" not in url


def test_search_url_is_derived_not_stored():
    """`search_url` 是派生属性，**不进 `model_fields`**。

    存一份就是第二份事实来源：数据里改个名，URL 还留着旧的，而没有人会去测
    「这条 URL 和这条 name 还对得上吗」。与 `unresolved_errors` 同一个处理方式。
    """
    assert "search_url" not in POI.model_fields
    assert "search_url" not in _poi().model_dump()
