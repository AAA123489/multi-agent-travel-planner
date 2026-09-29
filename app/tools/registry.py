"""后端分发与门面组装（P2）—— §6.4 的 `registry.py`。

**这里做两件事，且只做两件事：**

1. **按域选后端**：每个域读各自的 `*_BACKEND` 配置，经 `Settings.effective_backend()`
   （不是直接读字段 —— DEMO_MODE 的短路逻辑只在那一个地方）。
2. **组装成门面**：把四个工具对象拼成节点看到的那一个 `TravelTools`。

**Agent 节点代码里因此不出现任何 backend 分支** —— 节点只 import `TravelTools`
这一个类型，「POI 用 mock、距离用高德」怎么订，它完全看不见。

**未实现的后端要响亮地失败，不静默降级。** `POI_BACKEND=amap` 时进程直接起不来，
而不是悄悄用 mock 数据 —— 后者会让「我明明配了高德」这件事无从察觉，
一直到演示时才发现距离是假的。这是本项目的默认姿态：降级必须透明，
而配置层面的降级（你没实现）与运行层面的降级（上游挂了）是两回事。
"""

from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigError
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
    HotelLevel,
    HotelPrice,
    OpeningHours,
    POIBackend,
    POITag,
    ToolResult,
    TravelMode,
)
from app.tools.distance import DistanceTool
from app.tools.hotel_price import HotelPriceTool
from app.tools.opening_hours import OpeningHoursTool
from app.tools.poi_query import PoiQueryTool
from app.tools.poi_store import POIStore

# 该域已实现的后端 → 构造函数。表驱动而不是 if 链：加数据源只改这一处，
# 不会漏掉某个分支（与 config.py 的 DOMAIN_SPEC 同一个思路）。
#
# ⚠ 三个「第二实现」都还没写（§6.4：AmapBackend 属第二阶段）。它们**刻意不出现在
# 表里** —— 与其放一个 `NotImplementedError` 占位，不如让「查不到」和
# 「还没实现」共用一个显式的报错，措辞里能写清「哪个阶段上」。
_PLANNED = {
    ("poi", "amap"): "P8 数据接入阶段",
    ("distance", "amap"): "P8 数据接入阶段",
    ("hotel", "aigohotel"): "P8 数据接入阶段",
    ("intercity", "variflight"): "契约定完之后（§6.6：TravelState 还没有装城际交通段的字段）",
}


def _unsupported(domain: str, backend: str, env_name: str) -> ConfigError:
    """未实现 / 未知后端的统一报错。

    报**环境变量名**而不是域名字段名 —— 与 `config.py` 同一条约定：用户手里
    拿的是 `.env`，报 `poi_backend` 他还得自己做一次名字翻译。
    """
    if (domain, backend) in _PLANNED:
        return ConfigError(
            f"{env_name}={backend} 尚未实现（计划在 {_PLANNED[(domain, backend)]} 落地）。"
            f"请改用 mock，或等该阶段完成。"
        )
    return ConfigError(f"{env_name}={backend} 不是已知的后端。")


def _build_poi_backend(settings: Settings, store: POIStore) -> POIBackend:
    name = settings.effective_backend("poi")
    if name == "mock":
        return MockPOIBackend(store)
    raise _unsupported("poi", name, "POI_BACKEND")


def _build_distance_backend(settings: Settings) -> DistanceBackend:
    name = settings.effective_backend("distance")
    if name == "mock":
        return MockDistanceBackend()
    raise _unsupported("distance", name, "DISTANCE_BACKEND")


def _build_hotel_backend(settings: Settings, store: POIStore) -> HotelBackend:
    name = settings.effective_backend("hotel")
    if name == "mock":
        return MockHotelBackend(store)
    raise _unsupported("hotel", name, "HOTEL_BACKEND")


def _reject_intercity(settings: Settings) -> None:
    """城际交通域**只允许 mock**，且它连 mock 实现都没有。

    `TravelState` 里没有任何字段装城际交通段、`plan_struct` 也没定义它
    （§6.6）—— 所以 `INTERCITY_BACKEND=variflight` 不是「换个数据源」，
    而是「这个功能还不存在」。此时必须 `ConfigError`，不能静默忽略：
    静默忽略会让用户以为自己配上了城际交通。
    """
    name = settings.effective_backend("intercity")
    if name != "mock":
        raise _unsupported("intercity", name, "INTERCITY_BACKEND")


def load_store(settings: Settings) -> POIStore:
    """载入 POI 索引。**失败不抛异常** —— 返回一个 `load_error` 非空的 store。

    为什么不像其他配置问题那样直接 `raise ConfigError`：数据文件坏了属于
    **运行期**故障，不是配置写错了。启动期有 `validate_paths()` 守文件是否存在；
    真到运行期发现坏了，正确的姿态是「工具返回 ok=False → 上层降级」，
    而不是让整个进程起不来。这是硬红线 #4 在装配层的延续。
    """
    return POIStore.load(settings.poi_file)


class ToolRegistry:
    """`TravelTools` 门面的实现 —— **节点只认这一个类型**（§6.4）。

    四个方法与四个工具对象**一一对应，只做转发**：门面这一层没有任何业务逻辑，
    唯一的例外是 `estimate_distance` 的 id → POI 转换，而那件事在 `DistanceTool`
    内部（§6.4：转换只在门面一处）。

    写成一长串显式转发而不是 `__getattr__` 动态代理，是为了让「门面到底暴露了
    哪四个方法」在源码里一眼可数 —— 动态代理会让它取决于被代理对象的属性，
    读代码时看不出来，而 `test_node_only_depends_on_the_facade` 也没法守。
    """

    def __init__(
        self,
        *,
        poi_query: PoiQueryTool,
        opening_hours: OpeningHoursTool,
        distance: DistanceTool,
        hotel_price: HotelPriceTool,
    ) -> None:
        self._poi_query = poi_query
        self._opening_hours = opening_hours
        self._distance = distance
        self._hotel_price = hotel_price

    # --- §6.3 的四个工具，方法名与 TravelTools Protocol 逐字一致 ---

    def poi_query(
        self, city: str, tags: list[POITag], top_k: int = 20
    ) -> ToolResult[list[POI]]:
        return self._poi_query(city, tags, top_k)

    def opening_hours(self, poi_ids: list[str]) -> ToolResult[dict[str, OpeningHours]]:
        return self._opening_hours(poi_ids)

    def estimate_distance(
        self, from_id: str, to_id: str, mode: TravelMode
    ) -> ToolResult[DistanceInfo]:
        return self._distance(from_id, to_id, mode)

    def hotel_price(self, city: str, level: HotelLevel) -> ToolResult[HotelPrice]:
        return self._hotel_price(city, level)

    # --- 给 P5 的 lifespan / 调试用 ---

    @property
    def tools(self) -> list[Any]:
        """四个工具对象（不是门面方法）—— P4 的 agent 档要把它们交给 LLM。

        返回列表而不是四个属性：P4 绑定时是遍历，逐个取会写成四行重复代码。
        用 `Any` 而不是联合类型：这四个类没有共同基类（故意的，见 base.py 的
        实测记录），硬造一个基类只为标注类型不划算。
        """
        return [self._poi_query, self._opening_hours, self._distance, self._hotel_price]


def build_tools(settings: Settings | None = None) -> ToolRegistry:
    """按配置组装门面。**P5 的 lifespan 调它一次并把结果挂在 `app.state`。**

    这里不做 `@lru_cache`：缓存会让测试之间互相串（换一个 settings 却拿到上一个
    store）。要单例就在 lifespan 里持一份 —— 缓存的边界应当与生命周期一致。
    """
    settings = settings or get_settings()

    _reject_intercity(settings)
    store = load_store(settings)

    poi_backend = _build_poi_backend(settings, store)
    distance_backend = _build_distance_backend(settings)
    hotel_backend = _build_hotel_backend(settings, store)

    return ToolRegistry(
        poi_query=PoiQueryTool(poi_backend),
        opening_hours=OpeningHoursTool(poi_backend),
        distance=DistanceTool(distance_backend, store),
        hotel_price=HotelPriceTool(hotel_backend),
    )
