"""`requirement_collect` 需求收集节点（P4.1 接入真 LLM）—— 对应 §4.1。

## 它做三件事

1. **抽取**：便宜档 + `TravelRequirementDelta` 结构化输出，把本轮 `user_query`
   里的**新增/修改**字段抽出来，增量合并进 `user_requirement`。
2. **判缺**：必填三项（`destination` / `days` / `travelers`）还缺哪些。
3. **追问**：缺东西时让 LLM 写一句自然中文问句，同时写进 `messages`。

## 三处刻意的分工 —— 判据都是「这件事需要的信息在谁手上」

| 事情 | 归谁 | 为什么 |
|---|---|---|
| 「10月1号」→ `2026-10-01` | **LLM** | 要算出这个日期，必须知道今天几号，而 `today` 是注入 prompt 的 |
| `2026/1/5` → `2026-01-05` | **代码** | 纯机械，补零不需要任何外部信息，为它多烧一轮 LLM 不划算 |
| 「缺不缺必填项」 | **代码** | 路由（§5.1）要读它，而路由必须是可单测的纯函数 |

**归不了的绝不猜。** 「忙完这阵子就去」换算不出来 → `start_date` 按**缺失**处理 →
下一轮再问一次。猜成「下个月 1 号」会让后面的时间闭合校验拿着一个错数字认真工作，
而错日期不会报错，只会算出一份排不下的行程。

（这条分工与 `PlanItem._normalize_hhmm` 是同一套判据的两次应用：那边把
「9:00」→`09:00` 归代码、「下午2点」原样返回撞 pattern；这边把纯格式归代码、
需要 `today` 的归 LLM。两处 docstring 互相引用。）

## 与 §4.1 的两处偏差（都已回填文档）

**① 要点 5 的「境外拦截」被泛化成「目的地不在已接入范围」。**
真正的约束是「有没有这座城市的 POI 数据」，境外只是它的子集。按关键词判境外会漏掉
「想去西安」—— 那同样一条数据都没有，却会一路走到生成阶段，拿不到任何 POI，
最后交出一份每个景点都是幻觉的行程。**那比直接说「不支持」糟得多。**

**② 本节点对 `messages` 是只写不读的。**
§4.1 的输入清单里有 `messages`，但 P4.1 之前**没有任何地方写 HumanMessage**
（用户轮要由 P5 的 API 层写进状态）。这时读 `messages`，读到的会是一串
单向的自问自答 —— 上一轮我们问的那句，后面跟着的还是我们问的那句。
拿这种上下文去生成追问，模型会以为用户在自言自语。

等 P5 把用户轮补上，这里再开读取（届时把最近若干条一起交给问句那一次调用）。
**抽取那一次调用则永远不该读它**：历史里带着用户前几轮说的旧值，
模型会把它们当成本轮新增再输出一遍，而 delta 合并会照单全收。
「已经收集到的需求」那份渲染就是历史的权威摘要，一份就够。
"""

import logging
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.core.exceptions import AppError
from app.core.llm import LLMClient, build_llm
from app.graph.nodes.decorators import thread_id_of
from app.graph.prompts import load_prompt
from app.graph.state import TravelRequirement, TravelRequirementDelta, TravelState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 契约常量
# ---------------------------------------------------------------------------

# 必填三项（§4.1 要点 2）：缺了它们行程根本排不出来。
# 其余字段（origin / start_date / budget / preferences / pace / transport）
# 缺失**不阻塞** —— 由生成节点按默认值兜底并在行程里标注假设，或由 G3 降级放行。
REQUIRED_FIELDS: tuple[str, ...] = ("destination", "days", "travelers")

# 目的地白名单 —— **判据是数据，不是地理**（见模块 docstring 的偏差①）。
# 与 `data/poi_clean.json` 里实际有的城市必须一致，由
# `test_supported_destinations_match_the_poi_data` 守着：加了新城市的 POI
# 却忘了改这里，那条测试会红。
SUPPORTED_DESTINATIONS: tuple[str, ...] = ("成都", "杭州")

# 字段名 → 中文名。**两个用途**：渲染给 LLM 看（它要按参数名回填），
# 以及退化问句。报英文字段名（「请提供 days」）是很典型的机器腔。
FIELD_LABELS: dict[str, str] = {
    "destination": "目的地城市",
    "days": "行程天数",
    "travelers": "出行人数",
    "origin": "出发城市",
    "start_date": "出发日期",
    "budget": "预算",
    "preferences": "偏好",
    "pace": "行程节奏",
    "transport": "市内交通",
}

# 可选字段的追问优先级 + 上限。**这是产品判断，不是技术限制**：
# 可选字段有六个，全列出来问句就不像人话了；而 preferences / pace / transport
# 属于「有更好、没有就按常规来」，问了多半得到一句「都行」——
# 让生成节点按默认值兜底更省事。留下来的三个都会实质改变行程：
# 日期决定闭馆与天气、出发地决定城际段与距离、预算决定住宿档次。
OPTIONAL_ASK_ORDER: tuple[str, ...] = ("start_date", "origin", "budget")
MAX_OPTIONAL_ASKS = 2

# ⚠ 只在**追问话术那一次调用失败时**兜底，不是主路径。见 `ask_question`。
FALLBACK_QUESTION = "为了帮你排出合适的行程，还想确认一下：{fields}？"

_EXTRACT_PROMPT = "requirement_collect_extract"
_ASK_PROMPT = "requirement_collect_ask"


# ---------------------------------------------------------------------------
# 纯函数：判定与归一化
# ---------------------------------------------------------------------------


def missing_required(requirement: TravelRequirement) -> list[str]:
    """列出还没填的必填项，**顺序固定**（按 `REQUIRED_FIELDS`，不是 set 序）。

    顺序固定是为了可断言：话术里字段的先后每次都一样，测试才能直接比字符串。
    用 set 的话同样的输入会产出不同的问句，而那是纯随机噪声。
    """
    return [name for name in REQUIRED_FIELDS if getattr(requirement, name) is None]


def missing_optional(requirement: TravelRequirement) -> list[str]:
    """可以顺带追问的**可选**字段（至多 `MAX_OPTIONAL_ASKS` 条）。

    ⚠️ **两条纪律：**

    1. **它不进 `missing_fields`。** 那个字段是路由契约 —— §5.1 的
       `route_after_collect` 读它判「该不该继续问」，只能装必填项。把可选字段混进去，
       「用户没提预算」会变成「不能进入生成」，与 §4.1 要点 2 直接冲突。

    2. **它只在本来就要追问的那一轮里出现**，从不单独引出新一轮。必填项齐了就直接
       进生成 —— 哪怕出发日期还空着。反过来那个做法看着更「周到」，但它与 §5.2
       的取舍相反：宁可交付一个写明了假设的方案，也不要问到用户失去耐心。
       而在一轮本来就有的追问里多带一句「顺便哪天出发？」几乎不花成本，
       却能省掉后面整整一轮。

       所以本函数的调用点在 `if missing:` 里面 —— 那个位置就是这条纪律本身，
       搬出去就会悄悄多问无数轮。
    """
    empty = [name for name in OPTIONAL_ASK_ORDER if getattr(requirement, name) is None]
    return empty[:MAX_OPTIONAL_ASKS]


def unsupported_destination(requirement: TravelRequirement) -> str | None:
    """目的地不在已接入范围时给出给用户看的话术，否则 `None`。

    目的地还没抽出来（`None`）时返回 `None` —— 那不是「不支持」，
    是「还不知道」，该走追问那条路。

    城市名走**精确匹配**，与 `POIStore.in_city` 保持一致（§6.5：「成都 / 成都市」
    的归一化是预处理阶段的职责，在查询期再做一遍就等于有两处城市名规则，迟早不一致）。
    模型侧的归一化由 prompt 的正反例负责，不在这里补救 —— 否则改一处措辞
    和改一处代码会分别生效，谁也说不清最终是哪条规则赢了。
    """
    city = requirement.destination
    if city is None or city in SUPPORTED_DESTINATIONS:
        return None
    return (
        f"抱歉，我目前只能规划{'、'.join(SUPPORTED_DESTINATIONS)}"
        f"——「{city}」的景点数据还没有接入，排不出靠得住的行程。"
        "把目的地换成这两座城市之一，我马上开始。"
    )


# `2026/1/5`、`2026.1.5`、`2026年1月5日` 这类**纯格式差异**用翻译表拉平。
# 不处理「10月1号」—— 它缺年份，而年份要靠 `today` 推，那是 LLM 的活（见模块 docstring）。
_DATE_TRANSLATION = str.maketrans(
    {"年": "-", "月": "-", "日": "", "号": "", "/": "-", ".": "-"}
)


def normalize_start_date(value: str | None) -> str | None:
    """把可机械归一的日期写法拉成 `YYYY-MM-DD`；拉不动的返回 `None`。

    **返回 `None` 的意思是「按缺失处理」，不是「出错了」。** 调用方据此丢掉这个
    字段，于是它下一轮会被再问一次 —— 归一化失败退化成多问一句，
    而不是带着一个脏日期往下走。

    它会拒绝的东西，和它接受的东西一样重要：

    - `10月1号` → `None`（缺年份，需要 `today` → LLM 的活）
    - `国庆` → `None`（同上）
    - `2026-13-45` → `None`（格式对、日子不对，交给 `date()` 判）
    - `2026-10-01 下午` → `None`（带着没法归一的部分，宁可重问）

    ⚠️ **绝不做「尽力而为」的解析。** 把一个含混的字符串猜成某个日期，
    它会在后面一路畅通无阻（时间冲突校验、闭馆校验都只认格式），
    直到用户照着它订了机票才暴露。
    """
    if value is None:
        return None
    text = value.strip().translate(_DATE_TRANSLATION)
    parts = text.split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    year, month, day = (int(part) for part in parts)
    try:
        return date(year, month, day).isoformat()   # 顺带把 2026-1-5 补成 2026-01-05
    except ValueError:
        return None


def _dedup(items: list[str]) -> list[str]:
    """保序去重（`dict.fromkeys` 也行，但这里要的是「第一次出现的位置」）。

    需要它是因为模型**有可能**把它看到的完整列表原样吐回来 —— 人类轮里给了
    「已经收集到的需求」，它顺手复述一遍是很自然的行为。追加语义如果不带去重，
    第 2 轮就会出现两个「人文」。
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def merge_delta(base: TravelRequirement, delta: TravelRequirementDelta) -> TravelRequirement:
    """把本轮增量并进已有需求（§4.1 要点 1）。

    ## `exclude_unset` **和** `exclude_none` 都要

    `exclude_unset` 是 delta 语义的载体：只取模型**填过**的字段，没填的保留旧值。
    它成立的前提是 `TravelRequirementDelta` 全字段可空 —— 全可空时 function calling
    的 JSON Schema `required` 为空，模型才可能「只填想填的」（§4.1 要点 1 / §12.2）。

    `exclude_none` 是防一类具体的灾难：**模型把全部字段都 dump 出来、值全是 `null`。**
    这时每个键都「被 set 过」，`exclude_unset` 一个都排不掉，合并的结果是把用户
    前几轮说的全部抹掉，然后重新从第一问开始问。P4.0 实测 10/10 次模型都只返回
    本轮提到的字段 —— 但那是**实测，不是保证**，而它失效的表现恰好是「看起来像
    新会话」，最难联想到是合并规则干的。

    多带一个 `exclude_none` 会不会丢掉什么？不会：`TravelRequirement` 里
    **「明确没有」和「还不知道」是同一个状态**（都是 `None`），所以「明确置空」
    这件事在当前契约里不携带任何信息，排掉它没有损失。

    ## `preferences` 单独处理：**追加**，不是覆盖

    其余字段都是标量，「本轮说了什么就是什么」——覆盖是对的。列表不是：
    第 2 轮用户说「还想吃点好的」，是往第 1 轮的「人文」上**加**，不是把它换掉。

    另一种做法是让 delta 里装「累计后的完整列表」、代码照样覆盖。**不选它**，
    因为那要模型自己去做合并（它得看到旧列表、判断哪些是新的、一个不漏地抄回来），
    而代码做同一件事是确定的、零成本的。模型只要报「这轮提到了什么」，
    剩下的是集合运算。
    """
    updates = delta.model_dump(exclude_unset=True, exclude_none=True)

    raw_date = updates.pop("start_date", None)
    if raw_date is not None:
        normalized = normalize_start_date(raw_date)
        if normalized is None:
            # **不把原值打进日志** —— 与 `llm._parse_tool_call` 同一条姿态：
            # 只报「哪件事没成」，不报内容。要查具体值可以去看 LLM 调用记录。
            logger.warning("start_date 不是可机械归一的日期，按缺失处理（下一轮会再问）")
        else:
            updates["start_date"] = normalized

    fresh_preferences = list(updates.pop("preferences", None) or [])

    merged = base.model_copy(update=updates)
    if fresh_preferences:
        merged = merged.model_copy(
            update={"preferences": _dedup([*base.preferences, *fresh_preferences])}
        )
    return merged


# ---------------------------------------------------------------------------
# 拼给 LLM 的消息
# ---------------------------------------------------------------------------


def render_requirement(requirement: TravelRequirement) -> str:
    """把已收集的需求渲染成给 LLM 看的中文清单（**历史的权威摘要**）。

    字段名和中文名都写上（`- 行程天数（days）：3`）：模型要靠参数名回填，
    而中文名让它不必做一次「days 是什么」的翻译 —— 这一层翻译 LLM 做得来，
    但每一次调用都要做一遍，没有收益。

    **同时给字段名与中文名还有第二个作用**：这份渲染和 prompt 里的字段表
    用的是同一套名字，模型只要照着抄就不会写错键。
    """
    filled = requirement.model_dump(exclude_none=True, exclude_defaults=True)
    if not filled:
        return "（还没有收集到任何信息）"
    lines = []
    for name, value in filled.items():
        shown = "、".join(str(item) for item in value) if isinstance(value, list) else value
        lines.append(f"- {FIELD_LABELS.get(name, name)}（{name}）：{shown}")
    return "\n".join(lines)


def build_extract_messages(
    requirement: TravelRequirement, query: str, today: str
) -> list[BaseMessage]:
    """抽取那次调用的输入。

    **`today` 放在 human 轮，不放系统提示。** 系统提示因此在所有会话、所有日子里
    **逐字节相同** —— 服务端的 prompt cache 命中的就是它（P4.0-C 实测：cheap 档
    337 个 input token 里 128 个是 `cache_read`）。把每天都会变的日期塞进系统提示，
    等于每天一过零点就把这段缓存整段作废。

    `query` 原样带上，不做任何裁剪或转义：它是用户的话，任何改写都是在替用户说话。
    """
    human = (
        f"今天的日期是 {today}。\n\n"
        f"### 已经收集到的需求\n{render_requirement(requirement)}\n\n"
        f"### 用户这一轮说\n{query}"
    )
    return [
        SystemMessage(content=load_prompt(_EXTRACT_PROMPT)),
        HumanMessage(content=human),
    ]


def build_ask_messages(
    requirement: TravelRequirement,
    missing: list[str],
    optional: list[str],
    query: str,
) -> list[BaseMessage]:
    """追问那次调用的输入（§4.1 要点 3）。

    缺什么由**代码**算好再告诉模型（`missing` / `optional`），不让它自己去比 ——
    「信息够不够」是控制流判断，路由要读，必须可单测（§5.1）。模型这一趟只负责
    **措辞**，不负责判断。
    """
    lines = [
        f"### 已经收集到的需求\n{render_requirement(requirement)}",
        "",
        "### 还缺的必填信息（必须问到）",
        *[f"- {FIELD_LABELS.get(name, name)}" for name in missing],
    ]
    if optional:
        lines += [
            "",
            "### 可以顺带问一句（可选，问不到也没关系）",
            *[f"- {FIELD_LABELS.get(name, name)}" for name in optional],
        ]
    lines += ["", f"### 用户这一轮说\n{query or '（用户这一轮没有补充新内容）'}"]
    return [
        SystemMessage(content=load_prompt(_ASK_PROMPT)),
        HumanMessage(content="\n".join(lines)),
    ]


def fallback_question(missing: list[str]) -> str:
    """退化问句 —— 只在 LLM 那次调用失败/返回空时用。

    P3 的主路径就是它，P4.1 之后退居兜底。**留着的理由**：一句机器腔的追问，
    比「用户重说一遍刚才那句话」好得多。
    """
    labels = "、".join(FIELD_LABELS.get(name, name) for name in missing)
    return FALLBACK_QUESTION.format(fields=labels)


# ---------------------------------------------------------------------------
# 两次 LLM 调用
# ---------------------------------------------------------------------------


def extract_delta(
    llm: LLMClient,
    requirement: TravelRequirement,
    query: str,
    today: str,
) -> TravelRequirementDelta:
    """本轮结构化抽取（§4.1 要点 1）。

    **失败直接往上抛，不吞。** 抽取失败意味着这一轮用户说的话我们一个字也没接住；
    吞掉它会让节点带着「需求没变」的状态继续走 —— 下一轮要么问出同一句话
    （用户会以为系统没听见），要么在 G3 之后拿着空白需求去生成行程。
    抛出去则落进 `@traced_node` 的兜底：`stage=failed` + `error`，
    由 `route_after_collect` 判成 `reject` 就地终止，**用户看得见出了什么事**。

    另外，客户端侧已经有重试（`llm_max_retries=3`），能走到这里说明不是偶发抖动。
    """
    chain = llm.structured(TravelRequirementDelta)
    return chain.invoke(build_extract_messages(requirement, query, today))


def ask_question(llm: LLMClient, messages: list[BaseMessage], missing: list[str]) -> str:
    """让 LLM 写一句自然中文追问（§4.1 要点 3）。

    ## 与 `extract_delta` 相反：**这里失败必须吞掉**

    两处的处置不同，判据是「失败了损失什么」：抽取失败 = 这一轮的信息全丢了；
    追问话术失败 = 只是这句话不好听。因为后者把整个节点炸掉，等于让用户
    **白白重说一遍刚才那句话** —— 用一次措辞的失败换一次信息丢失，不划算。

    追问的每一次调用都在「已经抽到字段」之后发生，所以这里的兜底是有意义的：
    该保住的已经保住了。

    ⚠️ 兜底问句是模板化的（`FALLBACK_QUESTION`），这是**刻意留着的欠债**：
    它只在 LLM 那次调用挂掉时出现，而那时「像人一样说话」已经不是首要问题了。
    """
    try:
        reply = llm.chat(messages)
    except AppError as exc:
        # 工厂的契约是「任何上游异常都包成 AppError 子类」，所以这里兜得住。
        logger.warning("追问话术生成失败（%s），退化成模板问句", type(exc).__name__)
        return fallback_question(missing)

    content = reply.content
    # 不是 str 说明上游返回了结构化块（换模型时可能出现）。退回模板而不是
    # `str(content)` —— 后者的产物是 `[{'type': 'text', ...}]`，会原样发给用户。
    if not isinstance(content, str) or not content.strip():
        logger.warning("追问话术返回了空内容，退化成模板问句")
        return fallback_question(missing)
    return content.strip()


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------


def requirement_collect(state: TravelState, config: RunnableConfig) -> dict[str, Any]:
    """需求收集。出口见 §5.1 的 `route_after_collect`（G3 在路由里）。"""

    llm = build_llm("cheap")
    query = state.user_query.strip()

    if query:
        delta = extract_delta(
            llm, state.user_requirement, query, date.today().isoformat()
        )
    else:
        # **没有新输入就不调 LLM。** 空的一轮会让模型盯着「已经收集到的需求」
        # 往回猜，把它自己记住的值当成本轮新增再输出一遍 —— 那正是 delta 模式
        # 要避免的事，而且白烧一次调用。此时 delta 为空 = 需求不变，
        # 下面照常算出还缺什么、照常追问。
        delta = TravelRequirementDelta()
        logger.info("本轮没有新的用户输入，跳过抽取")

    requirement = merge_delta(state.user_requirement, delta)

    # `ask_round` 每进一次本节点就 +1（§4.1 输出清单）。
    # **不是「问了几次」，是「收集了几轮」** —— 最后一轮需求齐了也会 +1。
    # 两种计法都能工作，但 G3 的判据 `ask_round >= MAX_ASK_ROUNDS` 只在
    # 「每轮都 +1」下与「最多问 MAX 次」等价（路由看的是自增**之后**的值）。
    ask_round = state.ask_round + 1

    rejection = unsupported_destination(requirement)
    if rejection is not None:
        logger.info(
            "目的地不在已接入范围，终止本轮",
            extra={"thread_id": thread_id_of(config)},
        )
        # `stage="failed"` 是路由的判据（`route_after_collect` → "reject"）。
        # 话术放 `error`：那是状态里唯一「给用户看的失败原因」字段，
        # 前端有现成的位置渲染它，不需要为这一条新增字段。
        return {
            "user_requirement": requirement,
            "missing_fields": [],
            "pending_question": None,
            "user_query": "",
            "ask_round": ask_round,
            "stage": "failed",
            "error": rejection,
        }

    missing = missing_required(requirement)
    optional = missing_optional(requirement)

    updates: dict[str, Any] = {
        "user_requirement": requirement,
        "missing_fields": missing,
        # 每轮都写，不留上一轮的值：上一轮的追问已经问过了，留在这里会让前端
        # 以为这轮还在等用户回答（§4.1 输出清单里它是每轮都出现的字段）。
        "pending_question": None,
        "user_query": "",          # 消费掉本轮输入（§3.2：节点消费后清空）
        "ask_round": ask_round,
        # 还在追问 → collecting；问齐了 → 交给生成节点，状态进入 planning
        "stage": "collecting" if missing else "planning",
    }

    if missing:
        question = ask_question(
            llm, build_ask_messages(requirement, missing, optional, query), missing
        )
        updates["pending_question"] = question
        # `messages` 是 `add_messages`（累加语义），返回一条就是追加一条。
        # id 取 `ask-{轮次}` 这种**确定性**的值：节点在恢复时会从头重跑，
        # 生成随机 id 会让同一句追问在轨迹里出现两遍。
        updates["messages"] = [AIMessage(content=question, id=f"ask-{ask_round}")]

    logger.info(
        "需求收集第 %d 轮：缺失 %s",
        ask_round, missing or "无",
        extra={"thread_id": thread_id_of(config)},
    )

    return updates
