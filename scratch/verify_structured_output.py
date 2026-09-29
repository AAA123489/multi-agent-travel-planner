"""P4.0 前置实测：`with_structured_output` 在 DeepSeek 上到底怎么用。

## 一、这轮实测最大的发现：**三种 method 全部不可用**

§4.1 写的是「用 `llm.with_structured_output(TravelRequirementDelta)`」。照字面写，
**一个字段都抽不出来** —— 实测（langchain-openai 1.6.4 + `deepseek-v4-flash`）：

| method | 实际报错 | 能否修 |
|---|---|---|
| 默认（不传）= `json_schema` | `This response_format type is unavailable now` | ❌ 服务端不支持 |
| `json_mode` | `Prompt must contain the word 'json' in some form` | ⚠ 能修，但见下方张力 |
| `function_calling` | `Thinking mode does not support this tool_choice` | ✅ 能修，见下 |

**`with_structured_output` 的默认 method 是 `json_schema`** —— 它走的是
`response_format={"type":"json_schema"}`，而 DeepSeek 不认这个类型。所以
「不传 method」和「显式传 json_schema」是同一个失败。

## 二、`function_calling` 为什么被拦：thinking mode

`with_structured_output(method="function_calling")` 会**强制**
`tool_choice={"type":"function","function":{"name":...}}` 来逼模型走指定的函数。
而 `deepseek-v4-flash` **默认开着 thinking（推理）模式**，thinking 模式不支持强制
`tool_choice` —— 于是 400。

**修法是关掉 thinking，而关它的参数形状只有一个是对的**（都已实测）：

| 参数形状 | 结果 |
|---|---|
| `{"thinking": {"type": "disabled"}}` | ✅ **真正生效** |
| `{"thinking": False}` | ❌ 422：`invalid type: boolean`，`expected string` |
| `{"enable_thinking": False}` | ⚠ 请求被接受，**thinking 仍在** —— 配了没用 |
| `{"chat_template_kwargs": {"enable_thinking": False}}` | ⚠ 同上，配了没用 |

后两种最危险：**它们不报错**。只有拿 `function_calling` 当试金石才看得出来
thinking 有没有真关掉 —— 直接对话两种都会成功（thinking 开着也答普通问题）。

## 三、附带发现：thinking 默认烧 token

不关 thinking 时的 `response_metadata` 里有 `reasoning_tokens: 147` ——
「10 月 1 号去杭州玩 3 天」这种一句话输入，光推理就烧了 147 个 token。

**这印证了 §12 的双模型槽位设计**，并给它加了一条实现细节：
需求抽取（便宜档）这类纯结构化抽取任务应当**关掉 thinking**；行程生成 /
反思审核（主力档）是否也关，要在 P4.2 单独测质量差异后再定 ——
**不能因为省 token 就把主力档的推理能力关掉。**

## 四、三个可选路径与选型

| 路径 | 做法 | 取舍 |
|---|---|---|
| **A · function_calling + 关 thinking** | `with_structured_output(S, method="function_calling")` + `extra_body` | ✅ **选它**。直接拿 Pydantic 对象 |
| B · json_mode | 同上 + prompt 里必须出现 "json" 这个词 | ⚠ 与 §4.2「**绝对不要输出 JSON**」正面冲突 |
| C · 手动工具调用 | `bind_tools([S], tool_choice="auto")` 再自己解析 | 多一层手写解析；P0.3 已证 `tool_calls` 格式标准，可行但没必要 |

**选 A。** B 的排除理由是那条冲突：json_mode 要求 prompt 含 "json"，而这条路正是
§4.2 花了一整条纪律去禁止的东西（模型的格式模仿会让它把 JSON 吐给用户）。
为了走一条更差的路而把 prompt 写脏，不划算。

## 五、五个问题的实测

Q1. 三种 method 哪个能用 → **只有 `function_calling` + 关 thinking**。
Q2. **没提到的字段：省略 key 还是给 null？** —— 决定 delta 模式能否落地。
Q3. 「禁止编造」的实际效果 —— 编出来的话，§4.1 的追问逻辑一次都不会触发。
Q4. 嵌套结构能否正常返回（未来 `plan_struct` 就是嵌套的）。
Q5. `temperature=0.3` 下的稳定性 —— 决定测试能不能断言具体值。

**探针里的模型是草稿，不进 `app/`** —— `TravelRequirementDelta` 的最终形状
由这份实测结果决定（P4.0-B）。

跑法：`.venv/Scripts/python.exe -X utf8 scratch/verify_structured_output.py`
需要 `.env` 里的 LLM_API_KEY，会真实联网。**密钥在任何情况下都不打印。**
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Literal

# 让 `import app` 在没有 PYTHONPATH 的情况下也能用（pytest 靠 pyproject 的
# pythonpath=["."] 解决，裸脚本没有）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.core.config import get_settings  # noqa: E402

SEP = "=" * 72

# Q2 每变体跑多少次。取 10 是因为要看的不是「会不会」，而是「多常会」——
# 跑 1 次抽到省略 key 就下结论，是把运气当契约。
Q2_ROUNDS = 10

# 唯一真正生效的「关掉 thinking」参数形状（实测四种，见模块 docstring）。
# **不要改成 `{"enable_thinking": False}` 之类** —— 服务端会收下但不生效，
# 症状是 function_calling 继续报 thinking mode 错误，而直接对话看不出区别。
DISABLE_THINKING = {"thinking": {"type": "disabled"}}


def head(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# ---------------------------------------------------------------------------
# 探针用的草稿模型 —— 刻意做成「全部可空 + 注意默认值」
# ---------------------------------------------------------------------------


class RequirementDelta(BaseModel):
    """§4.1 的 `TravelRequirementDelta` 草稿形状。

    **注意 `preferences` 的类型选择**：正式契约里 `TravelRequirement.preferences`
    是 `list[str] = Field(default_factory=list)`，但草稿这里写成 `list[str] | None`
    —— 因为本次要观察的正是「未提供时模型给什么」，而一个非 Optional 的 list
    字段会把 `null` 变成 ValidationError，把「模型给了 null」这个信号吞掉。
    这是**为了观测而故意放宽**，不是建议的契约形状。
    """

    origin: str | None = None
    destination: str | None = None
    start_date: str | None = None
    days: int | None = None
    budget: float | None = None
    travelers: int | None = None
    preferences: list[str] | None = None
    pace: Literal["relaxed", "moderate", "intense"] | None = None
    transport: Literal["public", "taxi", "drive", "walk"] | None = None


class DayItem(BaseModel):
    """Q4 的嵌套结构 —— 未来 `plan_struct` 的缩小版。"""

    name: str
    start: str | None = None
    end: str | None = None
    kind: Literal["attraction", "meal", "hotel"] = "attraction"


class DayPlan(BaseModel):
    day_index: int
    items: list[DayItem]


class Itinerary(BaseModel):
    days: list[DayPlan]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def make_llm(*, thinking: bool = False) -> ChatOpenAI:
    """建 LLM。**默认关 thinking**，因为它是 `function_calling` 能用的前提。

    :param thinking: 主力档若确实需要推理能力，传 True —— 但那时
        `with_structured_output(method="function_calling")` 就不可用，
        要改用 `bind_tools(tool_choice="auto")`（路径 C）。这个取舍留给 P4.2。
    """
    s = get_settings()
    extra = {} if thinking else {"extra_body": DISABLE_THINKING}
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature,
        timeout=s.llm_timeout,
        max_retries=s.llm_max_retries,
        **extra,
    )


def structured(llm: ChatOpenAI, schema, method: str = "function_calling"):
    """统一的结构化输出入口。

    **必须显式传 `method`，而且只能是 `function_calling`。** 默认值是
    `json_schema`，在 DeepSeek 上直接 400（见模块 docstring 的实测表）。
    """
    return llm.with_structured_output(schema, method=method)


def describe_binding(chain, label: str) -> str:
    """看一个结构化输出对象实际绑定了 `tools` 还是 `response_format`。"""
    bound = getattr(chain, "first", None)
    kwargs = getattr(bound, "kwargs", {}) or {}
    kind = "tools" if "tools" in kwargs else ("response_format" if "response_format" in kwargs else "?")
    print(f"  [{label}] 走 {kind}")
    return kind


def probe_methods(llm_factory, schema) -> None:
    """Q1：三种 method 的可用性实测 —— **这一条就炸了**。

    保留成可重跑的实验而不是只写结论：换模型后必须重跑，因为「哪个 method 能用」
    完全取决于服务端支持什么，跟代码无关。
    """
    head("Q1 · 三种 method 在 DeepSeek 上哪个真的能用")

    results: dict[str, str] = {}

    # 默认 method（不传）—— §4.1 字面写法的实际下场
    plain = llm_factory(thinking=True)
    describe_binding(plain.with_structured_output(schema), "默认（不传 method）")

    for method in ("json_schema", "json_mode", "function_calling"):
        thinking_off = llm_factory(thinking=False)
        try:
            chain = thinking_off.with_structured_output(schema, method=method)
        except Exception as exc:  # noqa: BLE001 —— 构造期失败也要看清
            results[method] = f"构造失败：{type(exc).__name__}: {exc}"
            print(f"\nmethod={method}：{results[method]}")
            continue

        describe_binding(chain, f"method={method}（thinking 关）")
        try:
            chain.invoke("10月1号去杭州玩3天")
            results[method] = "✅ 可用"
        except Exception as exc:  # noqa: BLE001
            results[method] = f"❌ {str(exc).replace(chr(10), ' ')[:110]}"
        print(f"      调用结果：{results[method]}")

    print("\n结论：")
    for method, verdict in results.items():
        print(f"  {method:18} {verdict}")
    usable = [m for m, v in results.items() if v.startswith("✅")]
    print(f"\n→ 可用：{usable or '（无）'}")
    print("  **§4.1 的 `with_structured_output(TravelRequirementDelta)` 不能照字面写**")
    print("  —— 默认走 json_schema，在 DeepSeek 上全部 400。必须显式 function_calling。")

    # thinking 的开关形状也要每次重验：错形状不报错，静默失效
    print("\n关 thinking 的参数形状（错的那种不报错，只看直接对话看不出区别）：")
    for shape in ({"thinking": {"type": "disabled"}}, {"enable_thinking": False}):
        probe = llm_factory(thinking=False)
        probe = probe.model_copy(update={"extra_body": shape})
        try:
            probe.with_structured_output(schema, method="function_calling").invoke("10月1号去杭州玩3天")
            print(f"  ✅ {shape} —— 真的关掉了")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {shape} —— 没生效：{str(exc).replace(chr(10), ' ')[:80]}")


def probe_delta_semantics(llm: ChatOpenAI) -> None:
    """Q2：**决定性问题** —— 没提到的字段，模型是省略还是给 null。

    两个 prompt 变体，各自跑 `Q2_ROUNDS` 次，统计：
      - 省略 key 的次数（`exclude_unset` 可用）
      - 显式给 null 的次数（`exclude_unset` 失效，会抹掉已有值）

    输入刻意设计成「已有 destination=成都，本轮只说了天数」——
    这样 `destination` 就是一个**必须被保住**的字段。
    """
    head("Q2 · 没提到的字段：省略 key 还是给 null？（决定性问题）")

    current = {"destination": "成都", "origin": "北京", "travelers": 2}
    user_input = "再玩 3 天吧"

    variants = {
        "A · 明确要求省略": (
            "你是一个旅行需求抽取器。\n"
            "下面给你「已收集的需求」和「用户本轮输入」。\n"
            "**只输出本轮新增或修改的字段。没有提到的字段不要出现在输出里。**\n"
            "无把握的字段留 null。不要编造。"
        ),
        "B · 不作要求": (
            "你是一个旅行需求抽取器。\n"
            "下面给你「已收集的需求」和「用户本轮输入」。\n"
            "只输出本轮新增或修改的字段。"
        ),
    }

    for label, system in variants.items():
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "已收集的需求：{current}\n用户本轮输入：{user_input}")]
        )
        chain = prompt | structured(llm, RequirementDelta)

        omitted = 0
        explicit_null = 0
        call_errors = 0
        samples: list[set[str]] = []

        for _ in range(Q2_ROUNDS):
            try:
                result = chain.invoke({"current": current, "user_input": user_input})
            except Exception as exc:  # noqa: BLE001 —— 探针要看清失败长什么样
                call_errors += 1
                if call_errors == 1:
                    print(f"  第 1 次调用就失败：{type(exc).__name__}: {exc}")
                continue

            fields_set = set(result.model_fields_set)
            samples.append(fields_set)
            if "destination" in fields_set:
                explicit_null += 1
                print(f"  ⚠ 模型把 destination 也标成已设置，值为 {result.destination!r}")
            else:
                omitted += 1

        print(f"\n变体 {label}（成功 {len(samples)}/{Q2_ROUNDS} 次）：")
        print(f"  省略 destination 的次数 = {omitted}")
        print(f"  带上 destination 的次数 = {explicit_null}")
        if call_errors:
            print(f"  调用异常 {call_errors} 次")
        if samples:
            union = set().union(*samples)
            print(f"  各轮 model_fields_set 的并集：{sorted(union)}")
            print(f"  逐轮 fields_set：{[sorted(s) for s in samples]}")

    print("\n判读：")
    print("  · 变体 A 显著优于 B → 一行 prompt 能解决，写进 §4.1 的 prompt 要点")
    print("  · 两者都常带 null   → 必须走代码层方案（exclude_none / 只合并非空值）")
    print("  · 两者都基本省略     → exclude_unset 直接可用")


def probe_no_fabrication(llm: ChatOpenAI) -> None:
    """Q3：「禁止编造」的实际效果。

    输入只说「想去成都玩」—— 三个必填项里缺 `days` 和 `travelers`。
    如果模型把这两个编出来，那 §4.1 的「必填判定」就形同虚设：
    `missing_fields` 永远为空，追问逻辑一次都不会触发。
    """
    head("Q3 · 「禁止编造」真的拦得住吗")

    system = (
        "你是一个旅行需求抽取器。只抽取用户**明确表达**的信息。\n"
        "无把握的字段留 null，**禁止编造**。编造出发地会导致后续距离计算全错。\n"
        "只输出本轮新增或修改的字段。"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "已收集的需求：{{}}\n用户本轮输入：{user_input}")]
    )
    chain = prompt | structured(llm, RequirementDelta)

    runs = 5
    fabricated = Counter()
    ok = 0
    for i in range(runs):
        try:
            result = chain.invoke({"user_input": "想去成都玩"})
        except Exception as exc:  # noqa: BLE001
            print(f"  第 {i + 1} 次失败：{type(exc).__name__}: {exc}")
            continue
        ok += 1
        print(f"  第 {i + 1} 次：{result.model_dump(exclude_unset=True)}")
        for field in ("days", "travelers", "origin", "start_date"):
            if getattr(result, field) is not None:
                fabricated[field] += 1

    print(f"\n{ok}/{runs} 次成功；被编造出来的字段：{dict(fabricated) or '（无，全部留空）'}")
    print("判读：必填三项（destination/days/travelers）里编出 days 或 travelers")
    print("      → `missing_fields` 永远为空，§4.1 的追问逻辑一次都不会触发。")


def probe_nested(llm: ChatOpenAI) -> None:
    """Q4：嵌套结构能不能正常返回 —— 未来 `plan_struct` 就是嵌套的。"""
    head("Q4 · 嵌套结构（list[嵌套模型]）能否正常返回")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是行程生成器。只使用用户提到的景点，禁止编造。"),
            (
                "human",
                "用 2 天游成都，第一天去宽窄巷子和武侯祠，第二天去大熊猫基地。"
                "请给每天的点位顺序。",
            ),
        ]
    )
    chain = prompt | structured(llm, Itinerary)

    try:
        result = chain.invoke({})
    except Exception as exc:  # noqa: BLE001
        print(f"失败：{type(exc).__name__}: {exc}")
        return

    print(f"返回 {len(result.days)} 天")
    for day in result.days:
        names = [f"{it.name}({it.kind})" for it in day.items]
        print(f"  第 {day.day_index} 天：{names}")
    print("\n判读：能拿到嵌套结构 → `plan_struct` 可以直接用 Pydantic 模型接。")


def probe_stability(llm: ChatOpenAI) -> None:
    """Q5：同一输入跑 3 次是否一致 —— 决定测试能不能断言具体值。"""
    head("Q5 · temperature=0.3 下的稳定性")

    prompt = ChatPromptTemplate.from_messages(
        [("system", "只抽取用户明确表达的信息，无把握留 null。"), ("human", "{q}")]
    )
    chain = prompt | structured(llm, RequirementDelta)

    q = "10月1号去杭州玩3天，两个人，预算6000"
    seen = []
    for i in range(3):
        try:
            result = chain.invoke({"q": q})
        except Exception as exc:  # noqa: BLE001
            print(f"  第 {i + 1} 次失败：{type(exc).__name__}: {exc}")
            continue
        dump = result.model_dump(exclude_unset=True)
        seen.append(dump)
        print(f"  第 {i + 1} 次：{dump}")

    if len(seen) >= 2:
        print(f"\n三次结果是否完全一致：{len({str(s) for s in seen}) == 1}")
        print("判读：不一致 → P4.1 的单测不能断言具体抽取值，")
        print("      只能断言「结构正确 + 明确表达的字段被抽到」。")


def main() -> None:
    s = get_settings()
    print(f"模型：{s.llm_model}    base_url：{s.llm_base_url}")
    print(f"API key 已配置：{bool(s.llm_api_key)}（**不打印内容**）")
    if not s.llm_api_key:
        print("\n⚠ .env 里没有 LLM_API_KEY，探针无法运行。")
        return

    probe_methods(make_llm, RequirementDelta)

    llm = make_llm()          # thinking 关 —— function_calling 的前提
    probe_delta_semantics(llm)
    probe_no_fabrication(llm)
    probe_nested(llm)
    probe_stability(llm)

    print(f"\n{SEP}\n完成\n{SEP}")


if __name__ == "__main__":
    main()
