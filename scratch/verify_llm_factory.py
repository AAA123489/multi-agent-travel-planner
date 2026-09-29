"""P4.0-C 探针：`app/core/llm.py` 的工厂发出去的请求，DeepSeek 真的接受吗？

**这是需要 API key、需要联网的一侧。** 另一半（工厂封住的三件事有没有被封住）
在 `tests/test_llm.py`，那半边离线。

分工的理由：单测能证明「我们发了 `method="function_calling"` 和 thinking:disabled」，
**证明不了「服务端接受」** —— 后者只有真打一次才知道。反过来，真打一次能证明
「这次成功了」，**证明不了「换个人写还会成功」**。

## 为什么探针里必须有「反面样本」

只在正例上跑通，等于没跑：探针自己写错了（比如 `structured()` 其实没连上网、
或者异常被吞了），输出与「全对」一模一样。所以下面 `check_negative()` 故意用
**P4.0 踩出来的那两种错误写法**去调用，要求它们**必须失败**。
它们失败了，正例的成功才算数。

跑法：`.venv/Scripts/python.exe -X utf8 scratch/verify_llm_factory.py`
**换模型必须重跑**（结论与模型名绑定，同 §12.1 的约定）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.llm import DISABLE_THINKING, THINKING_BODY, _usage_from, build_llm  # noqa: E402

FAILURES: list[str] = []


def report(ok: bool, label: str, detail: str = "") -> None:
    print(f"{'✅' if ok else '❌'} {label}{(' —— ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def safe(exc: BaseException) -> str:
    """异常文本，**抹掉密钥**（硬红线 #6：探针的终端输出也是日志）。"""
    text = f"{type(exc).__name__}: {exc}"
    key = get_settings().llm_api_key
    return text.replace(key, "***") if len(key) >= 8 else text


# ---------------------------------------------------------------------------
# 用来接结构化输出的模型（P4.1 的 TravelRequirementDelta / P4.2 的 PlanStruct
# 还没落地，这里用最小形状代替 —— 本探针验的是**投递路径**，不是 schema 设计）
# ---------------------------------------------------------------------------


class Delta(BaseModel):
    destination: str | None = None
    days: int | None = None
    travelers: int | None = None


class Day(BaseModel):
    day_index: int
    items: list[str] = []


class Plan(BaseModel):
    days: list[Day] = []


class Capture(BaseCallbackHandler):
    """把自己挂到 invoke 的 config 上，收 token 用量。"""

    def __init__(self) -> None:
        super().__init__()
        self.usages: list[dict[str, int]] = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.usages.append(_usage_from(response))


def usage_of(client, chain, payload) -> tuple[Any, dict[str, int]]:
    """invoke 并回传这次调用的 token 用量（按 invoke 级 config 挂回调，走公开 API）。"""
    capture = Capture()
    result = chain.invoke(payload, config={"callbacks": [capture]})
    return result, (capture.usages[-1] if capture.usages else {})


# ---------------------------------------------------------------------------
# 正例
# ---------------------------------------------------------------------------


def check_cheap_structured() -> None:
    """便宜档：thinking 关 + `function_calling`。这是 §4.1 的主路径。"""
    print("\n[1] cheap 档 · structured（thinking 关 → function_calling）")
    try:
        result, usage = usage_of(
            None,
            build_llm("cheap").structured(Delta),
            "10月1号出发，去成都玩3天，两个人",
        )
    except Exception as exc:
        report(False, "cheap.structured 调用成功", safe(exc)[:200])
        return

    report(True, "cheap.structured 调用成功", repr(result)[:120])
    # `.get(..., 0)`：thinking 关着时这个键**整个不出现**（实测），不是 0
    report(
        usage.get("reasoning_tokens", 0) == 0,
        "cheap 档没有推理 token（thinking 确实关掉了）",
        f"usage={usage}",
    )
    report(result.destination is not None, "抽到了 destination", repr(result.destination))


def check_main_structured() -> None:
    """主力档：thinking **开** → 必须自动绕到 `bind_tools`，否则就是那 400。

    这一条是整个工厂存在的理由：thinking 与 `function_calling` 不兼容，
    而调用方（P4.2 的 plan_generate）不该知道这件事。
    """
    print("\n[2] main 档 · structured（thinking 开 → 自动走 bind_tools）")
    try:
        result, _ = usage_of(
            None,
            build_llm("main").structured(Plan),
            "排一个成都两天行程，每天两个景点：宽窄巷子、武侯祠、锦里、杜甫草堂",
        )
    except Exception as exc:
        report(False, "main.structured 调用成功（若报 thinking/tool_choice 相关 400，"
                      "说明又走回了 function_calling）", safe(exc)[:240])
        return

    report(True, "main.structured 调用成功", repr(result)[:160])
    report(len(result.days) >= 1, "返回了非空 days", f"days={len(result.days)}")


def check_main_chat() -> None:
    """主力档纯文本：thinking 开着，且 token 账单上看得见。"""
    print("\n[3] main 档 · chat（thinking 开 → reasoning_tokens > 0）")
    try:
        message = build_llm("main").chat([HumanMessage(content="用一句话介绍成都。")])
    except Exception as exc:
        report(False, "main.chat 调用成功", safe(exc)[:200])
        return

    report(True, "main.chat 调用成功", repr(str(message.content))[:100])
    raw_usage = dict(getattr(message, "usage_metadata", None) or {})
    parsed = _usage_from(
        LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={})
    )
    report(
        parsed.get("reasoning_tokens", 0) > 0,
        "main 档有推理 token（thinking 确实开着）",
        f"解析出的 usage={parsed} 原始={raw_usage}",
    )


# ---------------------------------------------------------------------------
# 反面样本 —— 探针的自检
# ---------------------------------------------------------------------------


def check_negative() -> None:
    """**用 P4.0 踩出来的两种错误写法去调，要求它们必须 400。**

    它们不失败，就说明这个探针根本没有真的打到服务端（或异常被吞了），
    那么上面三个 ✅ 一文不值。
    """
    print("\n[4] 反面样本（必须失败，否则本探针无效）")
    settings = get_settings()

    def raw(*, method: str | None, extra_body: Any) -> ChatOpenAI:
        kwargs: dict[str, Any] = {"api_key": settings.llm_api_key}
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=0,
            timeout=settings.llm_timeout,
            max_retries=0,
            **kwargs,
        )

    cases = [
        ("不传 method（默认 json_schema）+ 关 thinking", raw(method=None, extra_body=DISABLE_THINKING)),
        (
            "thinking 开着 + function_calling",
            raw(method="function_calling", extra_body=THINKING_BODY[True]),
        ),
    ]
    for label, client in cases:
        chain = (
            client.with_structured_output(Delta)
            if "不传 method" in label
            else client.with_structured_output(Delta, method="function_calling")
        )
        try:
            chain.invoke("去成都")
        except Exception as exc:
            report(True, f"反例按预期失败：{label}", safe(exc).splitlines()[0][:120])
        else:
            report(False, f"反例**没有**失败：{label}", "本探针可能根本没连上服务端")


def main() -> int:
    settings = get_settings()
    # 只打「配没配」，不打值 —— 终端输出也是日志
    print(
        f"模型={settings.llm_model} 端点={settings.llm_base_url} "
        f"密钥={'已配' if settings.llm_api_key else '未配'} max_retries={settings.llm_max_retries}"
    )
    if not settings.llm_api_key:
        print("未配置 LLM_API_KEY，无法运行本探针。")
        return 2

    check_cheap_structured()
    check_main_structured()
    check_main_chat()
    check_negative()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"❌ {len(FAILURES)} 项未通过：")
        for item in FAILURES:
            print(f"   - {item}")
        return 1
    print("✅ 全部通过：工厂封住的两条纪律在服务端确实生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
