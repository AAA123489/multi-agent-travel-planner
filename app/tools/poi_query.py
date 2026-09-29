"""景点查询（P2）—— 门面 `TravelTools.poi_query` 的实现（§6.3）。

排序规则本身是**纯函数**（`rank`），单测直接喂 `POI` 列表即可，不需要文件、
不需要 backend、不需要 LLM —— 这是 [开发流程.md](../../docs/开发流程.md) 铁律②
要求的那种「测最脆弱的部分，而不是最难 mock 的部分」。

**关于评分用「命中标签数」而不是加权：** 标签之间没有天然权重，硬编一组
（人文 0.8、美食 0.6……）是在没有依据的地方假装有依据。命中数相同再看评分，
规则简单、可解释、可复现。真要加权，得先有评估数据支撑，那是 P9 的事。
"""

from app.tools.base import POI, POIBackend, POITag, ToolResult


def tag_score(poi: POI, tags: list[POITag]) -> int:
    """命中的标签个数（取交集大小）。

    `tags` 为空时恒为 0 —— 此时排序退化成「按评分」，「不带偏好」应当等价于
    「没有偏好」，而不是「什么都不返回」。
    """
    return len(set(poi.tags) & set(tags))


def rank(pois: list[POI], tags: list[POITag], top_k: int) -> list[POI]:
    """按 (命中标签数, 评分) 降序取前 `top_k`。

    两个键都降序：先满足偏好，偏好内再挑好的。`top_k < 0` 视为不限量 ——
    与其让它静默返回空（`[:-1]` 那种经典坑），不如明确成「不限」。
    """
    ordered = sorted(pois, key=lambda poi: (tag_score(poi, tags), poi.rating), reverse=True)
    return ordered if top_k < 0 else ordered[:top_k]


class PoiQueryTool:
    """门面用的工具对象。

    **P2 阶段它很薄，这是正常的** —— 它现在只做转发。它的位置是为 P4 的 agent 档
    留的接缝：那时 `name` / `description` / 参数模型要交给 LLM 当 tool schema，
    而这三个东西属于工具，不属于数据源（高德与 mock 的 PoI 工具对外是同一个工具）。

    查询语义归 backend（§6.4 的 `POIBackend.query_poi` 就是这么定的：高德有自己的
    相关性排序，mock 有上面的 `rank`）。在这里再排一次序 = 两处排序规则，
    迟早不一致。
    """

    name = "poi_query"
    description = "按城市和偏好标签查询当地的景点、餐厅、购物等场所，返回评分较高的若干条。"

    def __init__(self, backend: POIBackend) -> None:
        self._backend = backend

    def __call__(
        self, city: str, tags: list[POITag], top_k: int = 20
    ) -> ToolResult[list[POI]]:
        """转发给 backend。失败原样返回 `ok=False` —— 工具层不吞错、不重试。"""
        return self._backend.query_poi(city, tags, top_k)
