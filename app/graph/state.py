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
"""

from operator import add
from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

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
    plan_struct: dict = Field(default_factory=dict)   # 每日点位顺序（JSON，给机器校验）
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
