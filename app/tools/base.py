"""工具层契约（P1.4）—— 对应 docs/方案设计.md §6。

这个文件把三件事定死：

1. **统一返回体 `ToolResult`** —— 工具永不抛异常给上层（硬红线 #4），
   一切失败都包成 `ok=False`。上层拿到 `ok=False` 的策略是**降级而非中断**。
2. **数据模型** `POI` / `OpeningHours` / `DistanceInfo` / `HotelPrice`。
   字段形状与 §9.1 的 POI 知识库 schema 一一对应。
3. **四个 Protocol** —— 三个按域的 backend 接口，加一个给节点用的门面 `TravelTools`。

**为什么 Protocol 要按域拆**（§6.4）：按域的配置必然要求按域的接口。
没有任何单个 backend 能同时实现「查景点 + 算距离 + 报酒店价」——
高德管不了酒店价，AIGOHOTEL 管不了景点。硬塞进一个协议的结果是
每个实现都得为不属于自己的方法写一堆 `not_implemented`。

**为什么门面与 Protocol 是两层**：门面的 `estimate_distance` 吃 id（LLM 只能给
字符串），Protocol 的 `distance_between` 吃 `POI`（距离是经纬度的纯函数）。
两层的名字刻意不同 —— 看一个调用点就能判断它在哪一层。
"""

from typing import Annotated, Generic, Literal, Protocol, TypeVar, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, Field, model_validator

# ============================================================================
# 字面量别名 —— 与 §9.1 预处理第 3 步的类型归一表一一对应，改这里 = 改预处理脚本
# ============================================================================

POIType = Literal["景点", "酒店", "餐厅", "交通枢纽", "购物"]
POITag = Literal["人文", "自然", "美食", "亲子", "夜生活", "购物", "演出"]
TravelMode = Literal["walk", "public", "taxi", "drive"]
HotelLevel = Literal["budget", "comfort", "luxury"]
ErrorCode = Literal["not_found", "invalid_param", "timeout", "upstream_error", "not_implemented"]

# 0=周一 … 6=周日，对齐 datetime.weekday()。做成别名是为了让「0 到底是周一还是
# 周日」这件事出现在**类型**里，而不是散在注释里等人踩
Weekday = Annotated[int, Field(ge=0, le=6)]


# ============================================================================
# 数据模型
# ============================================================================


class OpeningHours(BaseModel):
    """开放时间 —— **预处理归一化的结果，不是上游原文**（§9.1 第 6 步）。

    「`open` 是 None」有两种截然不同的可能，靠 `all_day` 区分：

      - `all_day=True` → 全天开放，**确定**开着
      - 全部留空       → 解析失败，**未知**

    两者都表现为 `open=None`，但审核节点对它们的处理不同：前者可直接判定当天
    开放，后者只能 warn 后放行。不区分就等于把「不知道」当成「开着」。
    """

    open: str | None = Field(default=None, description="HH:MM；未知或全天开放时为 None")
    close: str | None = Field(default=None, description="HH:MM；同上")
    closed_days: list[Weekday] = Field(default_factory=list, description="闭馆日，0=周一…6=周日")
    all_day: bool = Field(default=False, description="全天开放 —— 与「解析失败」区分开")
    raw: str = Field(default="", description="上游原文，永远保留（宁缺勿错）")


class POI(BaseModel):
    """POI 知识库的一条（§9.1 统一 schema）。

    **必填 / 可空的划分依据是预处理阶段的保证**，不是随手定的：

      - `lat` / `lng` —— 缺失的行在预处理时**整条丢弃**（无法参与距离计算）→ 必填
      - `rating`      —— 缺失的填该城市该类型中位数 → 必填
      - `price`       —— 只有酒店会真缺（景点缺失填 0 并标 `price_estimated`）→ 可空
    """

    id: str
    name: str
    city: str
    type: POIType
    tags: list[POITag] = Field(default_factory=list)
    address: str | None = None
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    rating: float = Field(ge=0, le=5)
    price: float | None = Field(
        default=None, ge=0, description="门票 / 房价 / 餐饮人均；None = 未知"
    )
    price_estimated: bool = Field(default=False, description="价格为估算值，前端要标注")
    opening_hours: OpeningHours | None = None

    @property
    def search_url(self) -> str:
        """在地图上搜索这个场所的链接（开发流程 P2 铁律③）。

        **做成派生视图而不是字段**，理由与 `TravelState.unresolved_errors` 同源：
        它是 `name` + `city` 的函数，存一份就是第二份事实来源 —— 数据里改个名，
        URL 还留着旧的，而没有人会去测「这条 URL 和这条 name 还对得上吗」。

        拼的是**真实搜索页**而不是死链：演示时点进去能看到真东西（铁律③）。
        """
        return f"https://www.amap.com/search?query={quote(self.name)}&city={quote(self.city)}"


class DistanceInfo(BaseModel):
    """两点间距离与耗时。

    `is_estimated` 的含义要精确：**True = 这不是真实路网距离**。
    mock 后端靠 Haversine × 1.3 路网系数算，永远为 True；高德走真实路径规划才是
    False。前端据此决定要不要标一个「约」字。
    """

    km: float = Field(ge=0)
    minutes: int = Field(ge=0)
    mode: TravelMode
    is_estimated: bool


class HotelPrice(BaseModel):
    """城市 × 档次的酒店均价（§6.3）。

    mock 取酒店类 POI 的分位数：budget=P25 / comfort=P50 / luxury=P75。
    `sample_size < 3` 时退到全局经验值并置 `is_approx=True` ——
    样本太少时中位数没有意义，必须让上游知道这个数字不硬。
    """

    city: str
    level: HotelLevel
    avg_price: float = Field(ge=0)
    sample_size: int = Field(ge=0)
    is_approx: bool = Field(default=False)


# ============================================================================
# 统一返回体
# ============================================================================

T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    """所有工具的返回值都长这样（§6.2）。

    **铁律：工具永不抛异常给上层** —— 失败一律包成 `ok=False`。
    上层拿到 `ok=False` 的策略是**降级而非中断**：距离算不出来就按直线估算并标注，
    酒店查不到就用默认均价。
    """

    ok: bool
    data: T | None = None
    error: str | None = Field(default=None, description="给人看的中文描述，会进日志")
    error_code: ErrorCode | None = None
    source: str = Field(default="mock", description="mock | amap | aigohotel | cache")
    elapsed_ms: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list, description="如「价格为估算值」")

    @model_validator(mode="after")
    def _check_shape(self) -> "ToolResult[T]":
        """把「ok 与载荷不配套」挡在构造处。

        没有这条，`ToolResult(ok=False)` 能构造出来 —— 而它既没 error 也没
        error_code，上层只能显示一句「未知错误」。让它在源头就构造失败，
        比等线上冒出一个没有原因的失败要好。

        这里 raise 的 ValueError 是**编程错误**（模型用错了），不是上游失败，
        与硬红线 #4 不冲突 —— 那条红线管的是「工具不把外部故障抛给上层」。
        """
        if self.ok and self.data is None:
            raise ValueError("ok=True 必须带 data（空列表也算 data，None 不算）")
        if not self.ok and self.error_code is None:
            raise ValueError("ok=False 必须带 error_code，否则上层无从分支")
        return self

    @classmethod
    def success(
        cls,
        data: T,
        *,
        source: str = "mock",
        warnings: list[str] | None = None,
        elapsed_ms: int = 0,
    ) -> "ToolResult[T]":
        """成功。data 是必填位置参数 —— 忘传直接就是 TypeError。"""
        return cls(
            ok=True, data=data, source=source, warnings=warnings or [], elapsed_ms=elapsed_ms
        )

    @classmethod
    def failure(
        cls,
        error: str,
        code: ErrorCode,
        *,
        source: str = "mock",
        warnings: list[str] | None = None,
        elapsed_ms: int = 0,
    ) -> "ToolResult[T]":
        """失败。**`code` 是必填位置参数，这是刻意的。**

        失败而不给 error_code，上层就无法区分「查不到」和「超时」，只能一律显示
        「出错了」。做成必填参数，忘写就是一个当场可见的错误。
        """
        return cls(
            ok=False,
            error=error,
            error_code=code,
            source=source,
            warnings=warnings or [],
            elapsed_ms=elapsed_ms,
        )


# ============================================================================
# 按域 Protocol —— 每个只管一个数据域（§6.4）
# ============================================================================

# ⚠ **实现方不要继承这三个 Protocol，只要结构上长得一样就行。**
#
# 已实测（Python 3.12）：显式继承 Protocol 会让**漏实现的方法静默返回 `None`** ——
#
#     class Impl(POIBackend):
#         pass               # 忘了实现 query_poi
#     Impl().query_poi(...)  # → None，不报错
#
# 也就是说继承反而**削弱**了检查：漏实现一个方法，症状是「工具返回 None」
# 而不是「启动就报错」，与「绿不等于过」是同一类陷阱。结构化实现没这个问题 ——
# 少写一个方法，`isinstance` 断言当场变红（见 tests/test_tools.py 的一致性测试）。
#
# `@runtime_checkable` 是为了让那个断言能跑。注意它只检查**方法名存不存在**，
# 不检查签名 —— 所以签名由 `test_protocol_methods_return_tool_result` 那一组
# 反射测试单独守。


@runtime_checkable
class POIBackend(Protocol):
    """景点域。

    查询与开放时间**必须在同一个 backend 里** —— 它们读同一份数据源，拆成两个
    协议只会让实现方重复持有一份索引。
    """

    def query_poi(self, city: str, tags: list[POITag], top_k: int = 20) -> ToolResult[list[POI]]:
        """按城市 + 标签查景点，按 tag 匹配度 × 评分排序。"""
        ...

    def get_opening_hours(self, poi_ids: list[str]) -> ToolResult[dict[str, OpeningHours]]:
        """批量取开放时间。**查不到的条目直接不出现在 dict 里**，并往 warnings 加一条。"""
        ...


@runtime_checkable
class DistanceBackend(Protocol):
    """距离域。**入参是 POI 对象，不是 id。**

    距离是经纬度的纯函数，无 I/O 也因此没有失败面 —— 不该为了把 id 换成坐标而
    反过来依赖数据层。id → 对象的转换由门面负责（§6.4）。
    """

    def distance_between(self, a: POI, b: POI, mode: TravelMode) -> ToolResult[DistanceInfo]:
        """两点距离与耗时。mode 决定速度：walk 4.5 / public 15 / taxi 22 / drive 25 km/h。"""
        ...


@runtime_checkable
class HotelBackend(Protocol):
    """酒店域。只报均价，不返回具体酒店 —— 住哪家由生成节点决定，工具只管预算。"""

    def get_hotel_price(self, city: str, level: HotelLevel) -> ToolResult[HotelPrice]:
        """城市 × 档次均价。样本不足时退全局经验值并置 `is_approx=True`。"""
        ...


# ============================================================================
# 门面 —— registry 组装出来交给节点，节点只 import 这一个类型
# ============================================================================


@runtime_checkable
class TravelTools(Protocol):
    """节点看到的四个工具（§6.3 那张表）。

    **节点只 import 这一个类型**，不 import 上面三个 Protocol。这样「POI 用 mock、
    距离用高德」这类组合怎么订，节点代码完全看不见，也就不可能出现 backend 分支。

    由 registry 实现：持有三个按域 backend，方法逐个转发。唯一的实质逻辑是
    `estimate_distance` 的 id → POI 转换 —— 查不到就返回
    `ToolResult.failure(..., "not_found")`。

    命名与 Protocol 那层**刻意不同**（`poi_query` vs `query_poi`、
    `estimate_distance` vs `distance_between`）：看一个调用点就能判断它在哪一层。
    """

    def poi_query(self, city: str, tags: list[POITag], top_k: int = 20) -> ToolResult[list[POI]]:
        """同 `POIBackend.query_poi`，直接转发。"""
        ...

    def opening_hours(self, poi_ids: list[str]) -> ToolResult[dict[str, OpeningHours]]:
        """同 `POIBackend.get_opening_hours`，直接转发。"""
        ...

    def estimate_distance(
        self, from_id: str, to_id: str, mode: TravelMode
    ) -> ToolResult[DistanceInfo]:
        """吃 id 不吃 POI —— agent 档下 LLM 只能从 JSON arguments 里给字符串。

        本方法是两层之间**唯一的转换点**：查表把两个 id 换成 `POI`，再调
        `DistanceBackend.distance_between`。任一 id 查不到 → `not_found`。
        """
        ...

    def hotel_price(self, city: str, level: HotelLevel) -> ToolResult[HotelPrice]:
        """同 `HotelBackend.get_hotel_price`，直接转发。"""
        ...
