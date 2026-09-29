"""酒店均价（P2）—— 门面 `TravelTools.hotel_price` 的实现（§6.3）。

只回答「这个城市这个档次住一晚大概多少钱」，**不返回具体酒店** ——
住哪家由生成节点决定，工具只管预算。两件事问的其实是两个问题：
「什么价位」和「哪几家」，混在一个返回值里会让预算估算被酒店列表牵着走。

**分位数而不是均值**：一个城市里有一家 5000 元的度假村，均值会被它单条拉高，
而 P50 不会。房价分布是长尾的，均值在这里不是好统计量。
"""

from app.tools.base import HotelBackend, HotelLevel, HotelPrice, ToolResult

# 三档对应的分位数（§6.3）。budget 取 P25、luxury 取 P75 ——
# 问「最便宜的档大概多少」时，中位数偏高；问「最贵的档」时中位数偏低。
LEVEL_PERCENTILE: dict[HotelLevel, float] = {
    "budget": 25.0,
    "comfort": 50.0,
    "luxury": 75.0,
}

# ⚠ **样本不足时的全局经验值 —— 这是写死的估计，不是实测数据。**
#
# 来源：2024-2025 国内主要城市酒店均价的量级（经济型连锁 ~200-300、
# 中端连锁 ~400-500、五星/度假 ~800-1500 元/晚），取整到档位中值。
#
# **它必须靠 `is_approx=True` 让上层知道这个数字不硬**（§4.5 的 budget_fit
# 会用到它）。用这一组数兜底等于承认「我不知道这个城市多少钱」——
# 如果它被当成真实均价报给用户，用户按虚价做的决策就是假的。
HOTEL_FALLBACK_PRICE: dict[HotelLevel, float] = {
    "budget": 250.0,
    "comfort": 450.0,
    "luxury": 900.0,
}

# 少于这个样本数就不算分位数，直接兜底。3 是「能看出分布形状」的下限，
# 1~2 个样本算出来的「P25/P75」只是其中一个值本身，没有统计意义。
MIN_SAMPLE_SIZE = 3


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数，**与 `numpy.percentile` 的默认算法一致**（`linear`）。

    特意对齐 numpy 是刻意的：P8 接入真实数据、P9 做评估时，分析脚本大概率用
    numpy 复核这里的数字。两边算法一致，数字就对得上；不一致（比如换成
    `lower` / `nearest`）会出现「代码算出 420、脚本算出 460」这种查半天的差异。

    `values` 必须非空、`q` 在 [0, 100]。调用方保证 —— 本函数是内部纯函数，
    不做防御性校验（校验在 `price_for_level` 那条边界上做）。
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    # 位置 = q% × (n-1)，落在 [floor, ceil] 之间做线性插值
    position = (q / 100.0) * (len(ordered) - 1)
    low = int(position)                      # 向下取整
    high = min(low + 1, len(ordered) - 1)    # q=100 时 low 已是最后一项
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def price_for_level(
    prices: list[float], city: str, level: HotelLevel
) -> HotelPrice:
    """把一批房价（已去掉空值）折成该档次的均价。

    **样本不足不是失败** —— 返回兜底价并置 `is_approx=True`，`ok` 仍为 True。
    理由：预算估算是行程生成的**必需输入**，没有它整条链就断了；而「这个城市
    没有足够酒店数据」不该让用户拿不到行程。降级 + 透明标记，是 §5.2 的默认
    失败姿态在这里的具体样子。
    """
    if len(prices) < MIN_SAMPLE_SIZE:
        return HotelPrice(
            city=city,
            level=level,
            avg_price=HOTEL_FALLBACK_PRICE[level],
            sample_size=len(prices),
            is_approx=True,
        )

    return HotelPrice(
        city=city,
        level=level,
        avg_price=round(percentile(prices, LEVEL_PERCENTILE[level]), 2),
        sample_size=len(prices),
        is_approx=False,
    )


class HotelPriceTool:
    """门面用的工具对象。转发给 backend。"""

    name = "hotel_price"
    description = "查询某个城市某个档次（经济/舒适/豪华）的酒店平均每晚价格。"

    def __init__(self, backend: HotelBackend) -> None:
        self._backend = backend

    def __call__(self, city: str, level: HotelLevel) -> ToolResult[HotelPrice]:
        return self._backend.get_hotel_price(city, level)
