"""TravelState 及全部子模型 —— 五个 Agent 节点共享的状态契约（P1.2）。

对应 docs/方案设计.md §3。**这是全项目的地基**：5 个节点之间不互相调用，
全靠往这块共同状态上写字段、读字段。字段形状或 reducer 语义改错，
下游每个节点都要返工。

三个关键约定：

1. **覆盖 vs 累加**（§3.3）：代表「当前轮次最新事实」的字段用覆盖语义，
   需要审计轨迹的另开 `*_history` 字段用累加语义。
   典型反例：`review_comments` 若改成累加，第 2 轮审核时 LLM 会同时看到
   「上一轮已修好的问题」和「本轮新问题」，于是回头去修已修好的部分 ——
   改 A 忘 B、改 B 又坏 A，永远收敛不了。

2. **`Annotated[list, add]` 的字段无法用 `return []` 清空**（§3.3 的警告）。
   凡是需要「消费后清空」的字段一律不加 reducer。本文件里只有
   `messages`、`review_history`、`node_trace` 三个是累加的。

3. **派生值不设字段**：载荷需要但状态里没有的值（如 G1 的 `unresolved_errors`）
   做成派生视图，避免第二份事实来源漂移。见 `TravelState.unresolved_errors`。

4. **结构校验归 Pydantic，内容校验归 `self_review`**（P4.0 定，见 `PlanStruct`）。
   判据是「这件事的失败该走哪条处置」：格式错 → 重试 JSON 段；内容不合理
   → 回炉重生成。在 Pydantic 里拦内容问题，会把后者的失败伪装成前者。
"""

from operator import add
from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# 字面量类型别名 —— 集中定义，供节点、校验器、prompt 渲染共用
# ---------------------------------------------------------------------------

ReviewType = Literal[
    "time_conflict",   # 时段重叠 / 单日总时长超可用时长
    "distance",        # 相邻点通勤超限
    "budget",          # 估算总和超预算
    "closed",          # 闭馆冲突
    "routing",         # 路线折返
    "completeness",    # 缺用餐时段 / 缺住宿 / 未回应明确要求
    "other",
]

Severity = Literal["error", "warn"]
"""error 必须修复（阻塞 review_passed）；warn 仅建议优化。"""

Pace = Literal["relaxed", "moderate", "intense"]

Transport = Literal["public", "taxi", "drive", "walk"]

Stage = Literal[
    "collecting",     # 需求收集中
    "planning",       # 行程生成中
    "reviewing",      # 反思审核中
    "awaiting_user",  # 等待用户确认（图停在 interrupt）
    "evaluating",     # 评估中
    "done",
    "failed",
]

PlanItemKind = Literal["attraction", "meal", "hotel"]
"""行程条目的类别。

**刻意没有 `transport`。** 通勤段由 `DistanceTool` 按相邻点位的经纬度算出来
（精度远高于模型估的），让模型输出一条「从 A 打车到 B 40 分钟」只会多一个
可编造的对象，而它和工具算出的那个数字必然打架 —— 那时要以谁为准？

**也没有城际交通**：`Transport` 字面量只覆盖市内，城际段的契约未定义
（§6.6）。`INTERCITY_BACKEND` 配置项先留着占位，功能等契约定完再上。"""

# "HH:MM" 的 24 小时制。**用 pattern 而不是只写 str**：`start`/`end` 是
# 时间冲突校验的唯一输入，而 §4.1 的日期教训（P4.0 实测：「10月1号」原样返回）
# 说明模型不会自觉归一化格式。让它在这里撞墙 → 走 §4.2 的「只重试 JSON 段」，
# 好过让 `"下午2点"` 一路流到校验函数里被静默误解析。
HHMM_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"


# ---------------------------------------------------------------------------
# 子模型
# ---------------------------------------------------------------------------


class TravelRequirement(BaseModel):
    """结构化出行需求。

    **全部字段可空**：由 `requirement_collect` 逐轮增量填充，任一时刻都可能只填了一部分。
    判「信息是否已足够」不看本模型是否完整，而看 `TravelState.missing_fields` 是否为空。
    """

    origin: str | None = None          # 出发城市
    destination: str | None = None     # 目的地城市
    start_date: str | None = None      # 出发日期，归一化为 ISO8601 (YYYY-MM-DD)
    days: int | None = None            # 行程天数
    budget: float | None = None        # 总预算（元）
    travelers: int | None = None       # 出行人数
    preferences: list[str] = Field(default_factory=list)  # 人文/美食/自然/亲子/夜生活
    pace: Pace | None = None
    transport: Transport | None = None


class ReviewComment(BaseModel):
    """单条反思审核意见。"""

    type: ReviewType
    severity: Severity
    day_index: int | None = None          # 问题落在第几天（0-based），全局问题为 None
    detail: str                           # 问题描述。重生成时作为约束喂回 LLM
    suggestion: str | None = None         # 修复建议


class EvalResult(BaseModel):
    """评估报告（§8）。

    分数均为 0-100。**本模型不做范围校验**——Judge 模型偶尔会给出 105 这类越界值，
    此时应由 §8 的计算代码钳制并记 warn，而不是让 Pydantic 抛 ValidationError
    把整个 evaluate 节点打断（硬红线 #4 的同一条思路：降级而非中断）。
    """

    requirement_match: float              # 需求匹配度（LLM-as-Judge）
    feasibility: float                    # 行程可行性（程序化）
    budget_fit: float                     # 预算符合度（程序化）
    total_score: float                    # 加权总分，见 §8.3
    passed: bool
    dimensions_detail: dict = Field(default_factory=dict)
    judge_reason: str = ""
    estimated_cost: float | None = None
    elapsed_ms: int = 0


class ConfirmInput(BaseModel):
    """`user_confirm` 节点 interrupt 的 resume 值契约（§4.4）。

    由前端提交、API 层校验后交给 `Command(resume=...)`。
    这是**前端 ↔ 图**之间唯一的输入契约，两侧必须同时改。
    """

    action: Literal["approve", "revise"]
    feedback: str = ""                    # action == "revise" 时必填（校验在节点内做）


# ---------------------------------------------------------------------------
# plan_struct —— 机器可读的行程（§4.2 的第二输出段）
# ---------------------------------------------------------------------------
#
# 下面三个模型一律 **frozen**。它们是值对象，不是可变容器：
#
#   - 唯一预期的修改方式是 `model_copy(update=...)` —— 而 P4.2 的流程恰好就是
#     「模型输出 → 回填 poi_id → 得到新对象」，天然是复制管道。
#   - 更要紧的是防一类真 bug：模块级常量若是一个可变的 Pydantic 模型，
#     下游任何一处原地改动都会**跨会话污染**（同一个对象被所有 thread 共用）。
#     P3 的 `STUB_DRAFT_PLAN` 是个 str，没有这个问题；换成模型就有了。


class PlanItem(BaseModel):
    """行程里的一个条目（景点 / 用餐 / 住宿）。

    字段是从**下游消费者倒推**出来的，不是凭空设计的（每个字段都有主）：

      - `start` / `end` → §4.3 的时间冲突、闭馆冲突；§8.2 的市内交通时长
      - `kind`          → §4.3 的完整性（有没有用餐/住宿）、§8.2 的门票分项
      - `poi_id`        → §4.3 的幻觉检测、路线折返（坐标）；
                          §8.2 的门票单价（`POI.price`）
      - `name`          → 给人看；同时是 `poi_id` 回查失败时的**唯一线索**

    ⚠️ **`poi_id` 由 `plan_generate` 的代码回填，不是 LLM 输出。**
    模型只负责给出 `name`，代码拿着候选清单回查（先精确匹配、再归一化匹配）。
    这样做把「幻觉检测」变成确定性的、零成本的：**回查不到 → `poi_id` 留 None
    → self_review 直接判为幻觉 POI**。

    让模型直接吐 id 是反过来的 —— 它会编出一批**格式正确、库里没有**的 id，
    而那些 id 看上去和真的没区别，校验只能靠再查一次库才发现。
    """

    model_config = ConfigDict(frozen=True)

    kind: PlanItemKind
    name: str
    poi_id: str | None = Field(
        default=None,
        description="由 plan_generate 回填；None = 候选清单里没有这个名字（疑似幻觉）",
    )
    start: str | None = Field(default=None, pattern=HHMM_PATTERN, description="HH:MM")
    end: str | None = Field(default=None, pattern=HHMM_PATTERN, description="HH:MM")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _normalize_hhmm(cls, value: object) -> object:
        """把「9:00」「9：00」这类等价写法先归一到 `HH:MM`，再做格式校验。

        **这里与 §4.1 的日期处理刻意相反，判据是「归一化需要的信息在谁手上」：**

          - 日期 → 交给 LLM。把「10月1号」变成 `2026-10-01` 需要知道今天几号，
            而 `today` 是注入 prompt 的、LLM 手里就有 —— 代码再做一次就是两份实现。
          - 时分 → 交给代码。补个零不需要任何外部信息，为它多烧一轮 LLM 不划算。

        **归不了的绝不在这里猜。** 「下午2点」原样返回、撞上 `pattern` 报错，
        然后由 §4.2 的「只重试 JSON 段」处理 —— 猜错（把「下午2点」当成 02:00）
        会让时间冲突校验拿着一个错数字认真工作，比直接失败糟得多。
        """
        if not isinstance(value, str):
            return value
        text = value.strip().replace("：", ":")   # 全角冒号
        head, sep, tail = text.partition(":")
        if sep and head.isdigit() and len(head) == 1:
            return f"0{head}:{tail}"
        return text


class PlanDay(BaseModel):
    """行程的一天。`items` 的**顺序即当天游览顺序** —— 通勤校验按相邻对取。"""

    model_config = ConfigDict(frozen=True)

    day_index: int = Field(ge=0, description="0-based，与 `ReviewComment.day_index` 同一套")
    items: list[PlanItem] = Field(default_factory=list)


class PlanStruct(BaseModel):
    """每日点位顺序（§4.2）。

    **只做结构校验，不做内容校验** —— 这条分工要守住：

      - Pydantic 管的：`day_index` 非负、`start`/`end` 是 `HH:MM`、
        `kind` 在字面量集合内。
      - `self_review` 管的：行程**空不空**、有没有用餐时段、有没有闭馆冲突……

    所以这里**刻意没有「每天至少一个条目」这类校验**。`days == []` 在结构上
    合法，它是内容问题 —— 若在这里 raise，失败会表现成「JSON 解析失败」，
    而真相是「模型生成了一个空行程」，两者该走完全不同的处置
    （前者重试 JSON 段，后者回炉重生成整个行程）。
    """

    model_config = ConfigDict(frozen=True)

    days: list[PlanDay] = Field(default_factory=list)

    @property
    def nights(self) -> int:
        """住宿晚数（§8.2 的住宿分项要用）。

        **派生属性，不设字段。** 它是 `days` 的函数（`d` 天行程睡 `d-1` 晚），
        存一份就是第二份事实来源 —— 与 `TravelState.unresolved_errors`、
        `POI.search_url` 同一处理方式。

        ⚠️ 它**不是**「hotel 类条目的数量」。两个数字回答的是不同问题：
        住宿分项问「要订几晚」（算术），完整性校验问「行程里提没提住宿」
        （文本）。它们不该相等，也不该互相推导 —— 让它们各自独立，
        才不会因为模型多写了一条「第 3 晚也住酒店」而让预算悄悄翻倍。
        """
        return max(0, len(self.days) - 1)


# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------


class TravelState(BaseModel):
    """五个节点共享的唯一状态对象（§3.1）。

    节点统一签名 `def node(state: TravelState, config: RunnableConfig) -> dict`，
    返回**状态增量字典**（只含本节点修改的字段），由 LangGraph 按字段的
    reducer 语义合并回状态。
    """

    # ---------- 会话标识 ----------
    session_id: str = ""

    # ---------- 输入 ----------
    user_query: str = ""                  # 本轮用户原始输入。每轮覆写，节点消费后返回 "" 清空
    user_requirement: TravelRequirement = Field(default_factory=TravelRequirement)
    missing_fields: list[str] = Field(default_factory=list)   # 为空 → 可进入生成
    pending_question: str | None = None                    # 非空 → 本轮以追问结束

    # ---------- 产物 ----------
    draft_plan: str = ""                  # 行程草稿（Markdown，给人看）
    plan_struct: PlanStruct | None = None  # 每日点位顺序（给机器校验）
    review_comments: list[ReviewComment] = Field(default_factory=list)  # **覆盖**语义
    review_history: Annotated[list[list[ReviewComment]], add] = Field(default_factory=list)
    review_passed: bool = False
    user_feedback: str = ""               # **覆盖**语义，被 plan_generate 消费后清空
    user_confirmed: bool = False
    final_plan: str = ""
    eval_result: EvalResult | None = None

    # ---------- 循环控制 ----------
    retry_count: int = 0                  # 链路 A：机器自省计数（G1）
    user_revision_count: int = 0          # 链路 B：用户修改计数（G2），独立上限
    ask_round: int = 0                    # 追问轮数（G3）
    stage: Stage = "collecting"
    error: str | None = None

    # ---------- 轨迹 ----------
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    node_trace: Annotated[list[str], add] = Field(default_factory=list)

    @property
    def unresolved_errors(self) -> list[ReviewComment]:
        """降级放行时仍未解决的 error 级意见（G1，§5.2）。

        **派生视图，故意不设独立字段。** 原因：`review_comments` 用的是覆盖语义，
        任何时刻都恰好持有「本轮全部意见」，再存一份 error 子集就是第二份事实来源，
        迟早与主列表漂移（改了一个忘了另一个，是状态类 bug 的经典成因）。

        什么时候非空：`review_passed=False` 且重试已达上限 —— 即闸门 G1 触发，
        带着已知瑕疵降级放行。此时前端必须把它显式展示给用户（§5.2：降级必须透明）。
        """
        if self.review_passed:
            return []
        return [c for c in self.review_comments if c.severity == "error"]
