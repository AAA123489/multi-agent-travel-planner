"""本地 JSON 后端（P2）—— 三个域各一个，**不是一个大类**（§6.4）。

为什么按域拆而不是 `MockBackend` 一把抓：配置是按域选的（`POI_BACKEND=mock` +
`HOTEL_BACKEND=aigohotel` 是合法组合），接口自然也要按域拆。塞成一个大类，
每个实现都得为不属于自己的方法写一堆 `not_implemented`。

三个后端共用一份 `POIStore`（同一份内存索引，不重复载入）—— **共用数据，
不共用接口**。

**三个类都不继承 `POIBackend` / `DistanceBackend` / `HotelBackend`，这是刻意的。**
已实测：显式继承 Protocol 会让漏实现的方法静默返回 `None`，反而削弱检查
（详见 `base.py` 里的实测记录）。结构化实现 + `tests/test_tools.py` 的一致性
断言，才是真的守得住。
"""

from app.tools.base import (
    POI,
    DistanceInfo,
    HotelLevel,
    HotelPrice,
    OpeningHours,
    POITag,
    ToolResult,
    TravelMode,
)
from app.tools.distance import MODE_SPEED_KMH, estimate
from app.tools.hotel_price import MIN_SAMPLE_SIZE, price_for_level
from app.tools.poi_query import rank
from app.tools.poi_store import POIStore

SOURCE = "mock"

# 酒店类 POI 的 type 取值。写成常量而不是散落的字面量：预处理脚本、
# 归一化规则、这里三处都依赖它，散着写迟早有一处打错成「宾馆」而静默查不到。
HOTEL_TYPE = "酒店"


def _load_failure(store: POIStore) -> ToolResult | None:
    """store 载入失败时，原样翻成一个 `ok=False`。

    **失败原因不在这里重编** —— `POIStore` 已经把「文件不存在 / JSON 坏 /
    顶层不是数组」区分好了，重写一遍就是第二份措辞，两边迟早对不上。
    """
    if store.load_error is None:
        return None
    message, code = store.load_error
    return ToolResult.failure(message, code, source=SOURCE)


class MockPOIBackend:
    """景点域：查询 + 开放时间。两者读同一份数据，所以必须在同一个 backend 里。"""

    def __init__(self, store: POIStore) -> None:
        self._store = store

    def query_poi(self, city: str, tags: list[POITag], top_k: int = 20) -> ToolResult[list[POI]]:
        failure = _load_failure(self._store)
        if failure is not None:
            return failure

        pois = self._store.in_city(city)
        if not pois:
            # 空结果**不是失败**（P1.4 的不变式：空列表是合法 data）。
            # 但要 warn —— 「城市名写错了」和「这个城市真没有景点」在返回值里
            # 长得一模一样，给个候选列表能让人一眼看出是哪种。
            known = "、".join(self._store.cities())
            return ToolResult.success(
                [],
                source=SOURCE,
                warnings=[f"没有「{city}」的数据。已载入的城市：{known}"] if known else [],
            )

        return ToolResult.success(rank(pois, tags, top_k), source=SOURCE)

    def get_opening_hours(self, poi_ids: list[str]) -> ToolResult[dict[str, OpeningHours]]:
        """**只返回查到的**，缺的不编 —— 「哪些缺了」由门面统一成 warning。

        在这里也加一条 warning 会让门面和 backend 各报一次，同一件事出现两遍。
        """
        failure = _load_failure(self._store)
        if failure is not None:
            return failure

        found: dict[str, OpeningHours] = {}
        for poi_id in poi_ids:
            poi = self._store.get(poi_id)
            if poi is not None and poi.opening_hours is not None:
                found[poi_id] = poi.opening_hours

        return ToolResult.success(found, source=SOURCE)


class MockDistanceBackend:
    """距离域：纯计算，**不需要 store**（距离是经纬度的函数，不是数据的函数）。

    这也是 §6.4 分层的收益之一：`AmapBackend` 换成真实路径规划时，替换的只有
    这一个类，`DistanceTool` 与所有调用点一行不动。
    """

    def distance_between(self, a: POI, b: POI, mode: TravelMode) -> ToolResult[DistanceInfo]:
        if mode not in MODE_SPEED_KMH:
            # 类型上 `TravelMode` 已经限死了四个值，这里是**兜底的那一层**：
            # 硬红线 #4 要求工具对任何输入都不抛异常，包括从 JSON 反序列化进来的
            # 脏值。`MODE_SPEED_KMH[mode]` 会抛 KeyError —— 那就是一次违规。
            return ToolResult.failure(
                f"不支持的交通方式：{mode}（可用：{'、'.join(MODE_SPEED_KMH)}）",
                "invalid_param",
                source=SOURCE,
            )

        return ToolResult.success(estimate(a, b, mode), source=SOURCE)


class MockHotelBackend:
    """酒店域：从知识库里的酒店类 POI 算分位数。"""

    def __init__(self, store: POIStore) -> None:
        self._store = store

    def get_hotel_price(self, city: str, level: HotelLevel) -> ToolResult[HotelPrice]:
        failure = _load_failure(self._store)
        if failure is not None:
            return failure

        # 只取有价的。**price=None 表示「不知道」，不是「0 元」** ——
        # 把它当 0 参与分位数会把 P25 直接拉到 0，整档均价失真。
        prices = [
            poi.price
            for poi in self._store.in_city(city)
            if poi.type == HOTEL_TYPE and poi.price is not None
        ]

        price = price_for_level(prices, city, level)

        # 样本不足时把话说清楚。`is_approx` 是给机器看的标记，
        # warning 是给人看的 —— 评估报告里两个都要有（§4.5 的 budget_fit）。
        warnings = (
            [
                f"「{city}」{level} 档只有 {price.sample_size} 条酒店价格样本"
                f"（少于 {MIN_SAMPLE_SIZE} 条），已改用全局经验值 {price.avg_price:.0f} 元/晚"
            ]
            if price.is_approx
            else []
        )
        return ToolResult.success(price, source=SOURCE, warnings=warnings)
