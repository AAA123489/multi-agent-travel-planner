"""P4.1 探针：`requirement_collect` 的 prompt 真的做到了它承诺的事吗？

**这是需要 API key、需要联网的一侧。** 另一半（节点怎么用 LLM、拿回来的东西怎么
合并）在 `tests/test_requirement_collect.py`，那半边全离线。

分工的理由：单测能证明「节点把 `today` 放进了 human 轮」，**证明不了「模型据此把
『10月1号』算成了 `2026-10-01`」** —— 那是 prompt 的效力，只有真打一次才知道。

## 它首先验的是 P4.0 留下的那个失败项

P4.0 实测：输入「10月1号」，返回的 `start_date` **就是** `'10月1号'`，
没有归一化（§12.2）。§4.1 要点 4 那句「由 LLM 结合注入的 today 换算」当时不成立。
P4.1 补了「注入 today + 正反例」，**这件事到底有没有被修好，本探针说了算**。
所以下面把日期那几条单列成一组 —— 它们不是顺带验的，是这一阶段的主要待验项。

## 每个正例都配一个「不该发生的事」

「抽到了 destination」不算通过，「**没抽到 origin**」才算 —— 前者模型编一个也能过。
禁止编造这件事，只能靠断言「没出现」来验。

跑法：`.venv/Scripts/python.exe -X utf8 scratch/verify_requirement_collect.py`
**改 prompt 或换模型都要重跑**（结论与模型名绑定，同 §12.1 / §12.2 的约定）。
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import SystemMessage  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm import build_llm  # noqa: E402
from app.graph.nodes.requirement_collect import (  # noqa: E402
    ask_question,
    build_ask_messages,
    build_extract_messages,
    extract_delta,
    fallback_question,
)
from app.graph.state import TravelRequirement, TravelRequirementDelta  # noqa: E402

FAILURES: list[str] = []
TODAY = date.today().isoformat()
NEXT_YEAR = date.today().year + 1


def report(ok: bool, label: str, detail: str = "") -> None:
    print(f"{'✅' if ok else '❌'} {label}{(' —— ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def safe(exc: BaseException) -> str:
    """异常文本，**抹掉密钥**（硬红线 #6：探针的终端输出也是日志）。"""
    text = f"{type(exc).__name__}: {exc}"
    key = get_settings().llm_api_key
    return text.replace(key, "***") if len(key) >= 8 else text


def extract(query: str, base: TravelRequirement | None = None) -> TravelRequirementDelta:
    """真调一次抽取，返回模型给出的 delta。异常直接抛给调用方去 report。"""
    llm = build_llm("cheap")
    return extract_delta(llm, base or TravelRequirement(), query, TODAY)


# ---------------------------------------------------------------------------
# 一、P4.0 的失败项：日期归一化
# ---------------------------------------------------------------------------


def check_dates() -> None:
    print("\n— 日期归一化（P4.0 实测失败的那一条）—")

    cases = [
        ("10月1号", f"{date.today().year}-10-01", "只有月日 → 按今年算"),
        ("国庆去成都", f"{date.today().year}-10-01", "节日 → 换算"),
        ("2027年3月5日去杭州", "2027-03-05", "有年月日 → 直接换算"),
    ]
    for query, expected, why in cases:
        try:
            delta = extract(query)
        except Exception as exc:  # noqa: BLE001 —— 探针要的是「红在哪」，不是干净退出
            report(False, f"「{query}」", safe(exc))
            continue
        report(
            delta.start_date == expected,
            f"「{query}」→ {delta.start_date!r}",
            f"期望 {expected}（{why}）",
        )

    # 横跨年末的写法：今年已过的日子应当落到明年。今天取 3 月 5 号做输入，
    # 只要今天在 3 月 5 号之后，就该是明年 —— 而「今天是几号」这件事只有模型知道，
    # 正是把它交给 LLM 的原因。
    past_month_day = "1月2号"
    already_passed = date.today() > date(date.today().year, 1, 2)
    if already_passed:
        try:
            delta = extract(f"{past_month_day}出发")
        except Exception as exc:  # noqa: BLE001
            report(False, f"「{past_month_day}」跨年", safe(exc))
        else:
            report(
                delta.start_date == f"{NEXT_YEAR}-01-02",
                f"「{past_month_day}」（今年已过）→ {delta.start_date!r}",
                f"期望 {NEXT_YEAR}-01-02",
            )

    # 换算不出来的必须留空 —— 猜一个出来比留空糟得多（§4.1 要点 4）
    try:
        delta = extract("等忙完这阵子就去成都，大概秋天吧")
    except Exception as exc:  # noqa: BLE001
        report(False, "「忙完这阵子」留空", safe(exc))
    else:
        report(
            delta.start_date is None,
            f"「忙完这阵子」→ start_date={delta.start_date!r}",
            "说不清的日期必须留空，让它退回「再问一轮」",
        )


# ---------------------------------------------------------------------------
# 二、禁止编造：只能靠断言「没出现」来验
# ---------------------------------------------------------------------------


def check_no_fabrication() -> None:
    print("\n— 禁止编造（编造出发地会让后续距离计算全错）—")

    try:
        delta = extract("想去成都玩")
    except Exception as exc:  # noqa: BLE001
        report(False, "「想去成都玩」", safe(exc))
        return

    report(delta.destination == "成都", f"destination={delta.destination!r}")
    untouched = {
        name: getattr(delta, name)
        for name in ("origin", "days", "travelers", "start_date", "budget")
    }
    report(
        all(value is None for value in untouched.values()),
        f"没提到的字段全部留空 —— {untouched}",
        "任何一个被填上都说明模型在猜",
    )


def check_delta_mode_omits_old_values() -> None:
    print("\n— delta 模式：本轮没提到的字段不该被复述 —")

    base = TravelRequirement(destination="成都", days=3, travelers=2)
    try:
        delta = extract("换成杭州吧", base)
    except Exception as exc:  # noqa: BLE001
        report(False, "「换成杭州吧」", safe(exc))
        return

    report(delta.destination == "杭州", f"destination={delta.destination!r}")
    # **关键**：模型看到了「已收集：成都/3天/2人」，它不该把这些再输出一遍。
    # 输出一遍本身无害（合并后值相同），但它意味着 delta 语义没被理解 ——
    # 而同样是这个行为，在用户改口的那一轮就会把新值顶掉（见下一条）。
    echoed = [
        name
        for name in ("days", "travelers")
        if name in delta.model_dump(exclude_unset=True)
    ]
    report(not echoed, f"没有复述未改动的字段 —— 复述了：{echoed or '无'}")


def check_delta_does_not_resurrect_old_values() -> None:
    print("\n— 改口那一轮：旧值不许把新值顶回去 —")

    base = TravelRequirement(destination="成都", days=3, travelers=2)
    try:
        delta = extract("算了，还是去杭州，玩 5 天", base)
    except Exception as exc:  # noqa: BLE001
        report(False, "「还是去杭州，玩 5 天」", safe(exc))
        return

    report(
        delta.destination == "杭州" and delta.days == 5,
        f"destination={delta.destination!r} days={delta.days!r}",
        "期望杭州 / 5",
    )


# ---------------------------------------------------------------------------
# 三、城市名写法与范围
# ---------------------------------------------------------------------------


def check_city_spelling_and_scope() -> None:
    print("\n— 城市名写法与范围 —")

    for query, expected in [("想去成都市玩三天", "成都"), ("去杭州市", "杭州")]:
        try:
            delta = extract(query)
        except Exception as exc:  # noqa: BLE001
            report(False, f"「{query}」", safe(exc))
            continue
        report(delta.destination == expected, f"「{query}」→ {delta.destination!r}", f"期望 {expected}")

    # **支持范围之外的目的地必须照实填，不许改写成成都** —— 节点要靠这个值
    # 才判得出「不支持」，模型这里一「热心」，用户收到的是杭州的行程还莫名其妙。
    for query in ["想去巴黎玩五天", "去西安看兵马俑"]:
        try:
            delta = extract(query)
        except Exception as exc:  # noqa: BLE001
            report(False, f"「{query}」", safe(exc))
            continue
        report(
            delta.destination not in (None, "成都", "杭州"),
            f"「{query}」→ destination={delta.destination!r}",
            "必须照实填，不能改写成支持范围内的城市",
        )


# ---------------------------------------------------------------------------
# 四、追问话术
# ---------------------------------------------------------------------------


def check_question_phrasing() -> None:
    print("\n— 追问话术（§4.1 要点 3：不许模板化）—")

    llm = build_llm("cheap")
    requirement = TravelRequirement(destination="成都")
    try:
        question = ask_question(
            llm,
            build_ask_messages(requirement, ["days", "travelers"], ["start_date"], "想去成都"),
            ["days", "travelers"],
        )
    except Exception as exc:  # noqa: BLE001
        report(False, "生成追问", safe(exc))
        return

    print(f"   模型写的问句：{question}")
    report(question != fallback_question(["days", "travelers"]), "没有落回兜底模板")
    report(len(question) <= 80, f"长度 {len(question)} ≤ 80")
    leaked = [name for name in ("days", "travelers", "字段") if name in question]
    report(not leaked, f"没有漏出参数名 —— 漏了：{leaked or '无'}")
    report(
        "天" in question and ("人" in question or "位" in question),
        "把「几天」「几个人」都问到了",
    )


# ---------------------------------------------------------------------------
# 五、反面样本
# ---------------------------------------------------------------------------


def check_negative() -> None:
    """**探针必须证明自己连着服务端，而且证明的是「prompt 在起作用」。**

    第一版这里做的是「把系统提示整段换成『你是只会说好的的助手，永远不要调工具』」，
    然后断言结果必须变。**它没变 —— 模型照样把三个字段抽对了。** 原因是：
    human 轮里已经给全了信息（「想去成都玩三天，两个人」），schema 里也写全了字段
    描述，模型光靠这两样就能填对。整段换掉系统提示，换掉的是一份它本来就多余的说明。

    所以「换掉 prompt」本身不是一个有效的反面样本 —— 有效的反面样本要
    **换掉某一条具体的指令，再断言那条指令对应的行为跟着反过来**。
    下面两条针对的都是 schema **管不了**的事（猜不猜、换不换算），
    所以它们一变，就只能是 prompt 变的。

    （这个道理与 P4.0-C 那两条反面样本相通：反面样本不失败，说明探针没连上被测对象。
    只是「被测对象」比想象中细 —— 不是「prompt 这个文件」，是「里面某一条指令」。）
    """
    print("\n— 反面样本：把指令反过来说，对应的行为必须跟着反过来 —")

    llm = build_llm("cheap")
    chain = llm.structured(TravelRequirementDelta)
    messages = build_extract_messages(TravelRequirement(), "10月1号想去成都玩三天，两个人", TODAY)

    normal = chain.invoke(messages).model_dump(exclude_unset=True)
    print(f"   正常 prompt → {normal}")

    inverted = SystemMessage(
        content=(
            "你是需求提取助手。严格遵守以下两条：\n"
            "1. 用户没有提到的字段，也要按一般人的常见情况填上（默认出发地填「北京」）。\n"
            "2. start_date 原样照抄用户的说法，不要做任何日期换算。\n"
            "仍然要调用工具。"
        )
    )
    broken = chain.invoke([inverted, messages[1]]).model_dump(exclude_unset=True)
    print(f"   反向 prompt → {broken}")

    report(
        broken.get("origin") is not None,
        f"「没提到也要填」这条指令生效了 —— origin={broken.get('origin')!r}",
        "没生效说明模型根本没读这段系统提示",
    )
    report(
        broken.get("start_date") == "10月1号",
        f"「不许换算日期」这条指令生效了 —— start_date={broken.get('start_date')!r}",
        "两处都照着反向指令走了，才说明正向的那两条（禁止编造 / 换算日期）也是它们做到的",
    )


def main() -> None:
    print(f"今天 = {TODAY}（prompt 里的日期换算全部以此为准）")
    print(f"抽取档 = {get_settings().llm_model_cheap}")

    check_dates()
    check_no_fabrication()
    check_delta_mode_omits_old_values()
    check_delta_does_not_resurrect_old_values()
    check_city_spelling_and_scope()
    check_question_phrasing()
    check_negative()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"❌ {len(FAILURES)} 项未通过：")
        for name in FAILURES:
            print(f"   - {name}")
        raise SystemExit(1)
    print("✅ 全部通过：prompt 承诺的事，服务端确实做到了。")


if __name__ == "__main__":
    main()
