"""距离估算（P2）—— 门面 `TravelTools.estimate_distance` 的实现（§6.3 / §6.4）。

这个文件是**两层接口之间唯一的转换点**：门面吃 `from_id, to_id`（LLM 只能给
字符串），Protocol 的 `distance_between` 吃 `POI`（距离是经纬度的纯函数）。
转换在这里做，`DistanceBackend` 的实现方永远不需要认识 id，也就不需要自带一份
POI 索引（§6.4）。

距离算法：**Haversine 直线距离 × 1.3 路网系数**。
1.3 是「实际道路里程比直线远多少」的经验系数，不是精确值 —— 所以
`is_estimated` 恒为 `True`，前端据此标一个「约」字。高德走真实路径规划时才是
`False`。**这个标记比距离本身更重要**：它是「这个数字能不能当真」的载体。
"""

import math

from app.tools.base import POI, DistanceBackend, DistanceInfo, ToolResult, TravelMode
from app.tools.poi_store import POIStore

# 直线距离 → 路网里程的经验系数。取值依据：城市路网的实际里程约为直线的
# 1.2~1.4 倍（受河流、环线、单行影响）。取中位 1.3。
ROAD_FACTOR = 1.3

# 各交通方式的平均速度（km/h）。取的是**含等待与换乘的城市平均**，不是最高速：
# public 的 15 里含候车与步行到站，taxi 的 22 含拥堵。
MODE_SPEED_KMH: dict[TravelMode, float] = {
    "walk": 4.5,
    "public": 15.0,
    "taxi": 22.0,
    "drive": 25.0,
}

# 地球平均半径（km）。Haversine 用它把角度差换成距离。
EARTH_RADIUS_KM = 6371.0


def haversine_km(a: POI, b: POI) -> float:
    """两点的球面直线距离（km）。

    用 Haversine 而不是平面勾股：经度的实际长度随纬度收缩，在成都（北纬 30.6°）
    一个经度只有约 95.6 km，直接按 111.32 算会把东西向距离系统性放大 16%。
    纬度越高错得越多 —— 用平面公式做的「估算」在哈尔滨会离谱到没法解释。
    """
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    d_lat = lat2 - lat1
    d_lng = math.radians(b.lng - a.lng)

    h = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    # 用 asin(sqrt(h)) 而不是 atan2：h 因浮点误差可能略大于 1，atan2 版本对此更稳，
    # 但 asin 版本更短且这里 h 不会溢出（坐标已由 POI 限定在合法范围）。先夹一下更保险。
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def estimate(a: POI, b: POI, mode: TravelMode) -> DistanceInfo:
    """直线距离 × 路网系数，再按该方式的速度折成耗时。

    **耗时至少 1 分钟**（除非两点完全重合）：两个不同的地方，说「0 分钟到达」
    比说「1 分钟」更容易被当成 bug。真正重合时返回 0，那是事实。
    """
    km = haversine_km(a, b) * ROAD_FACTOR
    if km == 0:
        return DistanceInfo(km=0.0, minutes=0, mode=mode, is_estimated=True)

    minutes = max(1, round(km / MODE_SPEED_KMH[mode] * 60))
    return DistanceInfo(km=round(km, 2), minutes=minutes, mode=mode, is_estimated=True)


class DistanceTool:
    """门面用的工具对象 —— **id → POI 的转换在此，且只在此**（§6.4）。

    查不到 id 时返回 `not_found` 而不是抛 `KeyError`。这是硬红线 #4 的直接体现：
    调用方拿到的是 `ok=False`，与「超时」「上游挂了」走同一条分支。
    """

    name = "estimate_distance"
    description = "估算两个场所之间的距离和交通耗时，支持步行、公共交通、打车、自驾四种方式。"

    def __init__(self, backend: DistanceBackend, store: POIStore) -> None:
        self._backend = backend
        self._store = store

    def __call__(self, from_id: str, to_id: str, mode: TravelMode) -> ToolResult[DistanceInfo]:
        start = self._store.get(from_id)
        end = self._store.get(to_id)

        # 两个 id 分开报 —— 只说「有个 id 查不到」会让调用方自己去二分
        unknown = [pid for pid, poi in ((from_id, start), (to_id, end)) if poi is None]
        if unknown:
            return ToolResult.failure(
                f"场所不存在：{'、'.join(unknown)}",
                "not_found",
                warnings=[f"当前可用城市：{'、'.join(self._store.cities())}"]
                if self._store.cities()
                else [],
            )

        assert start is not None and end is not None  # 上面已排空，给类型检查器看
        return self._backend.distance_between(start, end, mode)
