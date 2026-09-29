"""P4.0-C 变异测试 —— 验证 `app/core/llm.py` 那批断言不是摆设。

**这个脚本存在的理由**：`tests/test_llm.py` 里超过一半的断言断的是**配置是否被封住**
（method 是不是 `function_calling`、thinking 的开关形状对不对、哪个槽位温度是 0）。
这类断言最容易写成摆设 —— 它看起来在测行为，实际只是把常量抄了一遍：
把实现改掉它照样绿，因为它比较的是「文档里那句」和「代码里那句」是否同一个字符串。

判据只有一个：**改坏一处实现，必须有一条测试变红。** 绿着通过的变异 = 那条断言没在
观察代码。这与 P1/P2/P3 的十六次实测同一条方法论。

跑法：
    .venv/Scripts/python.exe -X utf8 scratch/mutate_p40c.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LLM = "app/core/llm.py"

# (说明, 文件, 原文片段, 替换成什么, 期望哪条测试挂)
MUTATIONS = [
    (
        "① STRUCTURED_METHOD 退回那个 100% 400 的默认值",
        LLM,
        'STRUCTURED_METHOD = "function_calling"',
        'STRUCTURED_METHOD = "json_schema"',
        "test_structured_method_constant_is_not_the_broken_default",
    ),
    (
        "② structured_route 判反 —— thinking 开着仍走 function_calling",
        LLM,
        'return "bind_tools" if thinking else "function_calling"',
        'return "function_calling" if thinking else "bind_tools"',
        "test_structured_route_matches_the_measured_matrix",
    ),
    (
        "③ 关 thinking 换成那个「被接受但不生效」的写法",
        LLM,
        'DISABLE_THINKING: dict[str, Any] = {"thinking": {"type": "disabled"}}',
        'DISABLE_THINKING: dict[str, Any] = {"enable_thinking": False}',
        "test_thinking_body_uses_the_only_shape_that_works",
    ),
    (
        "④ 关 thinking 换成另一个静默失效的写法",
        LLM,
        'DISABLE_THINKING: dict[str, Any] = {"thinking": {"type": "disabled"}}',
        'DISABLE_THINKING: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}',
        "test_thinking_body_uses_the_only_shape_that_works",
    ),
    (
        "⑤ 建客户端时压根不传 extra_body（thinking 全开）",
        LLM,
        "        extra_body=THINKING_BODY[thinking],\n",
        "",
        "test_build_llm_sends_the_thinking_switch_it_claims_to",
    ),
    (
        "⑥ judge 的温度改成跟随全局（丢掉了「评估可复现」）",
        LLM,
        '"judge": SlotSpec(model_field="llm_judge_model", thinking=False, temperature=0.0),',
        '"judge": SlotSpec(model_field="llm_judge_model", thinking=False),',
        "test_judge_temperature_is_pinned_to_zero_not_inherited",
    ),
    (
        "⑦ cheap 槽位指向主力模型（双模型槽位变成摆设）",
        LLM,
        '"cheap": SlotSpec(model_field="llm_model_cheap", thinking=False),',
        '"cheap": SlotSpec(model_field="llm_model", thinking=False),',
        "test_slot_specs_cover_every_slot_and_point_at_real_settings_fields",
    ),
    (
        "⑧ 槽位表里写错一个配置项名（第一次用该槽位时才 AttributeError）",
        LLM,
        '"judge": SlotSpec(model_field="llm_judge_model"',
        '"judge": SlotSpec(model_field="llm_model_judge"',
        "test_slot_specs_cover_every_slot_and_point_at_real_settings_fields",
    ),
    (
        "⑨ 便宜档悄悄开着 thinking（纯烧 token，无任何症状）",
        LLM,
        '"cheap": SlotSpec(model_field="llm_model_cheap", thinking=False),',
        '"cheap": SlotSpec(model_field="llm_model_cheap", thinking=True),',
        "test_thinking_defaults_match_the_p4_0_decision",
    ),
    (
        "⑩ redact 的短串保护去掉（空密钥会把正文搅碎）",
        LLM,
        "        if len(secret) >= 8:",
        "        if True:",
        "test_redact_skips_short_secrets_instead_of_shredding_the_text",
    ),
    (
        "⑪ 上游异常不再包成 LLMError（API 层拿不到错误码）",
        LLM,
        "        except AppError:\n            raise  # 已经是我们的异常（如「模型没调工具」），别包第二层\n"
        "        except Exception as exc:\n"
        "            raise LLMError(\n"
        '                f"LLM 调用失败（槽位 {self.slot}）",\n'
        "                detail=redact(f\"{type(exc).__name__}: {exc}\", self._secret),\n"
        "            ) from exc\n",
        "        except Exception as exc:\n            raise exc\n",
        "test_chat_wraps_upstream_errors_and_redacts_the_secret",
    ),
    (
        "⑫ 结构化那条路不套错误包装（异常裸奔到节点层）",
        LLM,
        "        return cast(\"Runnable[Any, ModelT]\", self._guard(chain, schema.__name__))",
        "        return cast(\"Runnable[Any, ModelT]\", chain)",
        "test_structured_surfaces_the_same_wrapped_error",
    ),
    (
        "⑬ 用量解析只认 §12.2 那个旧键名（thinking 判据永远读不到）",
        LLM,
        '        for key in ("reasoning", "reasoning_tokens"):',
        '        for key in ("reasoning_tokens",):',
        "test_usage_from_reads_langchain_usage_metadata",
    ),
    (
        "⑭ 用量解析在拿不到数据时抛异常（观测钩子打断主流程）",
        LLM,
        "    raw: dict[str, Any] = {}\n    try:\n"
        "        raw = dict(getattr(response.generations[0][0].message, \"usage_metadata\", None) or {})\n"
        "    except Exception:\n        raw = {}\n",
        "    raw: dict[str, Any] = {}\n"
        "    raw = dict(getattr(response.generations[0][0].message, \"usage_metadata\", None) or {})\n",
        "test_usage_from_returns_empty_instead_of_raising_on_garbage",
    ),
    (
        "⑮ 污染解析失败时的 -1 兜底（配不上耗时记成 0，「很快」的假好消息）",
        LLM,
        "        return int((time.monotonic() - started) * 1000) if started is not None else -1",
        "        return int((time.monotonic() - started) * 1000) if started is not None else 0",
        "test_usage_callback_reports_missing_timing_as_negative_one",
    ),
    (
        "⑯ 回调把 prompt 原文写进日志（硬红线 #6 泄露）",
        LLM,
        '            "LLM 调用完成：槽位=%s 耗时=%dms 输入=%s 输出=%s 推理=%s",\n'
        "            self.slot,\n",
        '            "LLM 调用完成：槽位=%s 耗时=%dms 输入=%s 输出=%s 推理=%s prompts=%s",\n'
        "            self.slot,\n            str(kwargs.get(\"prompts\", \"\")),\n",
        "test_usage_callback_logs_numbers_and_never_the_prompt_text",
    ),
    (
        "⑰ 节点扫描器的禁用清单清空（架构约束失去守护）",
        "tests/test_llm.py",
        'BANNED_CALLS = frozenset({"ChatOpenAI", "with_structured_output", "bind_tools"})',
        "BANNED_CALLS = frozenset()",
        "test_the_scanner_actually_catches_offences",
    ),
    (
        "⑱ 扫描目录被指错（扫了一个没有节点文件的目录）",
        "tests/test_llm.py",
        'NODES_DIR = Path(__file__).resolve().parents[1] / "app" / "graph" / "nodes"',
        'NODES_DIR = Path(__file__).resolve().parents[1] / "app" / "core"',
        "test_nodes_never_build_their_own_chat_openai（目录自检）",
    ),
    (
        "⑲ extra_body 不再被扫描器拦（第 4 条纪律失去守护）",
        "tests/test_llm.py",
        'BANNED_KEYWORDS = frozenset({"extra_body"})',
        "BANNED_KEYWORDS = frozenset()",
        "test_the_scanner_actually_catches_offences",
    ),
]


def run_pytest() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout


def failures_of(out: str) -> list[str]:
    """列出所有变红的测试。

    **只取最后一行是不够的** —— 一条变异常常同时打红好几条测试，只打印一行会
    让「实际挂的」与「期望挂的」看起来对不上，读的人会误以为有偏差。
    """
    return [
        line.split(" - ")[0].replace("FAILED ", "").strip()
        for line in out.splitlines()
        if line.startswith("FAILED ")
    ]


def main() -> None:
    print("先跑一遍基线 —— 必须全绿，否则变异结果没意义\n")
    code, out = run_pytest()
    print(f"基线：{'全绿 ✓' if code == 0 else '**已经红了，先修**'}")
    print(out.strip().splitlines()[-1] if out.strip() else "")
    if code != 0:
        return

    results = []
    for label, rel, old, new, expect in MUTATIONS:
        path = ROOT / rel
        original = path.read_text(encoding="utf-8")
        if old not in original:
            results.append((label, "**片段没匹配上**", []))
            print(f"\n{label}\n  ⚠ 没找到要替换的片段，跳过")
            continue

        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            code, out = run_pytest()
        finally:
            path.write_text(original, encoding="utf-8")

        caught = code != 0
        hits = failures_of(out)
        results.append((label, "变红 ✓" if caught else "**依然全绿 ✗**", hits))
        print(f"\n{label}\n  {'变红 ✓' if caught else '**依然全绿 ✗**'}（期望：{expect}）")
        for name in hits:
            print(f"    - {name}")

    print("\n" + "=" * 74)
    print("汇总")
    print("=" * 74)
    bad = 0
    for label, verdict, caught in results:
        print(f"  {label}")
        print(f"     {verdict}（{len(caught)} 条测试变红）")
        if "依然全绿" in verdict or "没匹配" in verdict:
            bad += 1
    print(f"\n{len(results) - bad}/{len(results)} 条变异被测试抓住")
    if bad:
        print(f"⚠ {bad} 条没抓住 —— 那些断言是摆设，要补测试")


if __name__ == "__main__":
    main()
