"""P3 变异测试 —— 改一行代码，看测试会不会变红。

**这是唯一能证明「测试真的看得见这段代码」的手段。** 覆盖率数字做不到：
它只说这一行被执行过，不说「把它改坏会不会有人发现」。
本项目已实测过七次「测试全绿但其实没覆盖」，所以每个阶段收口都跑一遍。

每条变异：备份原文件 → 精确替换一处 → 跑 pytest → 还原 → 记录结果。
**任何一条绿着通过，都说明那条断言是摆设。**

跑法：
    .venv/Scripts/python.exe -X utf8 scratch/mutate_p3.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (说明, 文件, 原文片段, 替换成什么, 期望哪条测试挂)
MUTATIONS = [
    (
        "① 装饰器的异常兜底吞掉 interrupt()",
        "app/graph/nodes/decorators.py",
        "            except GraphBubbleUp:\n"
        "                # ⚠ 这一行是整个 HITL 机制的命门，不要删、不要挪到下面去。\n"
        "                # `interrupt()` 靠抛异常暂停图，吞掉它 = 确认页永远不出现。\n"
        "                raise\n",
        "",
        "test_decorator_does_not_swallow_interrupt（以及所有路径）",
    ),
    (
        "② user_confirm 忘了清零 retry_count",
        "app/graph/nodes/user_confirm.py",
        '        "retry_count": 0,\n',
        "",
        "test_path3_user_revision_resets_retry_count",
    ),
    (
        "③ 把「生成 → 审核」改成「生成 → 确认」（旁路反思节点）",
        "app/graph/builder.py",
        '    builder.add_edge("plan_generate", "self_review")\n',
        "",
        "test_self_review_cannot_be_rewired_out + 全部路径",
    ),
    (
        "④ 默认空壳审核改成「永远通过」",
        "app/graph/nodes/self_review.py",
        "    passed = False",
        "    passed = True",
        "test_path2_g1_degrades_after_max_retries",
    ),
    (
        "⑤ route_after_collect 的 ask 分支接到 plan_generate",
        "app/graph/builder.py",
        '        {"ask": END, "plan": "plan_generate"},',
        '        {"ask": "plan_generate", "plan": "plan_generate"},',
        "test_path4_ask_then_plan_across_two_runs",
    ),
    (
        "⑥ graph_config 漏掉 recursion_limit（G4 静默退回默认 25）",
        "app/graph/builder.py",
        '    return {"configurable": {"thread_id": session_id}, "recursion_limit": recursion_limit}',
        '    return {"configurable": {"thread_id": session_id}}',
        "test_graph_config_pins_thread_id_and_recursion_limit",
    ),
    (
        "⑦ 装饰器的 node_trace 序号改成永远 1（不再从状态里数）",
        "app/graph/nodes/decorators.py",
        "    return sum(1 for entry in state.node_trace if entry.split(\"#\")[0] == name) + 1",
        "    return 1",
        "test_traced_node_numbers_entries_from_state / test_path4",
    ),
    (
        "⑧ 审核只在通过时调用（retry_count 恒不自增 → 退化成死循环）",
        "app/graph/nodes/self_review.py",
        '        "retry_count": state.retry_count + (0 if passed else 1),',
        '        "retry_count": state.retry_count,',
        "test_path2 会因 G4 recursion_limit 抛 GraphRecursionError",
    ),
    (
        "⑨ user_confirm 不再拦截「revise 但不填意见」",
        "app/graph/nodes/user_confirm.py",
        '    if decision.action == "revise" and not decision.feedback.strip():',
        "    if False:",
        "test_empty_revise_feedback_re_asks_instead_of_failing + 参数化的 invalid_reason 用例",
    ),
]


def run_pytest() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout


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
            results.append((label, "片段没匹配上", expect))
            print(f"\n{label}\n  ⚠ 没找到要替换的片段，跳过")
            continue

        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        try:
            code, out = run_pytest()
        finally:
            path.write_text(original, encoding="utf-8")

        tail = [line for line in out.strip().splitlines() if line.strip()][-1]
        verdict = "变红 ✓" if code != 0 else "**依然全绿 ✗**"
        results.append((label, f"{verdict} — {tail}", expect))
        print(f"\n{label}\n  {verdict}\n  {tail}\n  （期望挂：{expect}）")

    print("\n" + "=" * 74)
    print("汇总")
    print("=" * 74)
    bad = 0
    for label, verdict, expect in results:
        print(f"  {label}\n     {verdict}")
        if "依然全绿" in verdict or "没匹配" in verdict:
            bad += 1
    print(f"\n{len(results) - bad}/{len(results)} 条变异被测试抓住")
    if bad:
        print(f"⚠ {bad} 条没抓住 —— 那些断言是摆设，要补测试")


if __name__ == "__main__":
    main()
