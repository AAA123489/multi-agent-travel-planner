"""P4.1 变异测试 —— 验证 `requirement_collect` 那批断言不是摆设。

**判据只有一个：改坏一处实现，必须有一条测试变红。** 绿着通过的变异 = 那条断言
没在观察代码。P1 三次、P2 四次、P3 九次、P4.0-C 十九次，这是第五批。

从这里起的变异测试比前几批多一层要求：**它自己会联网。** 被变异的是
「测试怎么拦住 LLM」这件事，改坏了自然就放真调用出去了。所以本脚本在子进程里
装了一个 `sitecustomize.py` 把 socket 断掉 —— 变异跑飞时留下的是失败，不是账单。

跑法：
    .venv/Scripts/python.exe -X utf8 scratch/mutate_p41.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NODE = "app/graph/nodes/requirement_collect.py"
STATE = "app/graph/state.py"
EDGES = "app/graph/edges.py"
BUILDER = "app/graph/builder.py"
CONFTEST = "tests/conftest.py"

# 子进程里禁网。**这不是保险，是必需**：⑪ 号变异（替身只打源模块）会把节点放回
# 真实调用路径上去，而 `.env` 里有真有 key、机器也有网 —— 于是一次「验证测试有效」
# 的动作会变成一次真实 API 调用，还可能是绿的。
_NO_NETWORK = '''
import socket
def _blocked(*args, **kwargs):
    raise RuntimeError("变异测试期间禁止联网")
socket.create_connection = _blocked
'''

# (说明, 文件, 原文片段, 替换成什么, 期望哪条测试挂)
MUTATIONS = [
    (
        "① 合并时不再排掉 null（模型全字段 dump 一次就把需求清空）",
        NODE,
        "    updates = delta.model_dump(exclude_unset=True, exclude_none=True)",
        "    updates = delta.model_dump(exclude_unset=True)",
        "test_merge_treats_an_all_null_dump_as_no_change",
    ),
    (
        "② preferences 改成覆盖（第 2 轮提到新偏好就把第 1 轮的顶掉）",
        NODE,
        '        merged = merged.model_copy(\n'
        '            update={"preferences": _dedup([*base.preferences, *fresh_preferences])}\n'
        "        )",
        "        merged = merged.model_copy(update={\"preferences\": fresh_preferences})",
        "test_merge_appends_preferences_instead_of_replacing_them",
    ),
    (
        "③ 追加时不去重（模型复述完整列表 → 「人文」出现两次）",
        NODE,
        'update={"preferences": _dedup([*base.preferences, *fresh_preferences])}',
        'update={"preferences": [*base.preferences, *fresh_preferences]}',
        "test_merge_dedupes_preferences_when_the_model_echoes_the_whole_list",
    ),
    (
        "④ 日期只查形状、不查是不是真日子（2026-02-30 照样收下）",
        NODE,
        "    try:\n"
        "        return date(year, month, day).isoformat()   "
        "# 顺带把 2026-1-5 补成 2026-01-05\n"
        "    except ValueError:\n"
        "        return None",
        '    return f"{year:04d}-{month:02d}-{day:02d}"',
        "test_normalize_start_date_rejects_anything_it_cannot_normalize_for_sure",
    ),
    (
        "⑤ 日期做「尽力而为」的解析（「10月1号」被猜成 0010-01-01）",
        NODE,
        "    parts = text.split(\"-\")\n"
        "    if len(parts) != 3 or not all(part.isdigit() for part in parts):\n"
        "        return None",
        '    parts = (text.split("-") + ["1", "1"])[:3]\n'
        "    if not all(part.isdigit() for part in parts):\n"
        "        return None",
        "test_normalize_start_date_rejects_anything_it_cannot_normalize_for_sure",
    ),
    (
        "⑥ 范围判定永远放行（「想去巴黎」一路走到生成）",
        NODE,
        "    if city is None or city in SUPPORTED_DESTINATIONS:\n        return None",
        "    if city is None or city in SUPPORTED_DESTINATIONS or True:\n        return None",
        "test_unsupported_destination_refuses_and_names_what_it_can_do",
    ),
    (
        "⑦ 白名单加上一座没有数据的城市（判据从「有数据」退回「我想支持」）",
        NODE,
        'SUPPORTED_DESTINATIONS: tuple[str, ...] = ("成都", "杭州")',
        'SUPPORTED_DESTINATIONS: tuple[str, ...] = ("成都", "杭州", "巴黎")',
        "test_supported_destinations_match_the_poi_data",
    ),
    (
        "⑧ 抽取失败被吞掉（这一轮用户说的话悄悄丢掉）",
        NODE,
        "    chain = llm.structured(TravelRequirementDelta)\n"
        "    return chain.invoke(build_extract_messages(requirement, query, today))",
        "    chain = llm.structured(TravelRequirementDelta)\n"
        "    try:\n"
        "        return chain.invoke(build_extract_messages(requirement, query, today))\n"
        "    except AppError:\n"
        "        return TravelRequirementDelta()",
        "test_extraction_failure_propagates_and_loses_nothing_silently",
    ),
    (
        "⑨ 追问失败往上抛（为了措辞把整轮信息一起炸掉）",
        NODE,
        "    except AppError as exc:\n"
        "        # 工厂的契约是「任何上游异常都包成 AppError 子类」，所以这里兜得住。\n"
        '        logger.warning("追问话术生成失败（%s），退化成模板问句", type(exc).__name__)\n'
        "        return fallback_question(missing)",
        "    except AppError:\n        raise",
        "test_question_failure_falls_back_to_a_template_and_keeps_the_extraction",
    ),
    (
        "⑩ 追问返回内容不做检查（空串/结构化块原样发给用户）",
        NODE,
        "    if not isinstance(content, str) or not content.strip():",
        "    if False:",
        "test_question_with_unusable_content_falls_back_to_the_template",
    ),
    (
        "⑪ 没有新输入也调抽取（模型把自己记住的值当本轮新增吐回来）",
        NODE,
        "    if query:",
        "    if True:",
        "test_node_skips_the_llm_when_there_is_no_new_input",
    ),
    (
        "⑫ 必填齐了不写 pending_question（留上一轮的值，前端以为还在等回答）",
        NODE,
        '        "pending_question": None,\n'
        '        "user_query": "",          # 消费掉本轮输入（§3.2：节点消费后清空）',
        '        "user_query": "",          # 消费掉本轮输入（§3.2：节点消费后清空）',
        "test_node_marks_the_stage_and_clears_the_question_when_nothing_is_missing",
    ),
    (
        "⑬ 追问消息的 id 变成随机（恢复重跑时同一句追问记两遍）",
        NODE,
        'updates["messages"] = [AIMessage(content=question, id=f"ask-{ask_round}")]',
        'updates["messages"] = [AIMessage(content=question, id=f"ask-{id(question)}")]',
        "test_node_appends_the_question_to_messages_with_a_stable_id",
    ),
    (
        "⑭ 拒绝之后仍留一句待答问句",
        NODE,
        '            "missing_fields": [],\n            "pending_question": None,',
        '            "missing_fields": [],\n            "pending_question": "抱歉，暂时排不了。",',
        "test_node_rejects_an_unsupported_destination_without_asking",
    ),
    (
        "⑮ 可选字段不进「本来就要问」的那一轮（白丢一次省一轮的机会）",
        NODE,
        "    optional = missing_optional(requirement)",
        "    optional = []",
        "test_optional_fields_ride_along_on_a_round_that_is_already_asking",
    ),
    (
        "⑯ 可选字段自己引出一轮追问（问到用户失去耐心）",
        NODE,
        "    if missing:",
        "    if missing or optional:",
        "test_optional_fields_never_justify_an_extra_round",
    ),
    (
        "⑰ delta 里出现一个必填字段（服务端开始逼模型编值）",
        STATE,
        '    origin: str | None = Field(default=None, description="出发城市")',
        '    origin: str = Field(description="出发城市")',
        "test_delta_declares_no_required_fields",
    ),
    (
        "⑱ delta 少一个字段（那个字段永远抽不出来）",
        STATE,
        '    pace: Pace | None = Field(default=None, description="relaxed / moderate / intense")\n',
        "",
        "test_delta_and_requirement_have_the_same_fields",
    ),
    (
        "⑲ 结构化 schema 被要求成全量需求而不是 delta",
        NODE,
        "    chain = llm.structured(TravelRequirementDelta)",
        "    chain = llm.structured(TravelRequirement)",
        "test_node_asks_for_the_delta_schema_not_a_full_requirement",
    ),
    (
        "⑳ 便宜档换成主力档（双槽位变摆设，只体现在账单上）",
        NODE,
        '    llm = build_llm("cheap")',
        '    llm = build_llm("main")',
        "test_node_uses_the_cheap_slot",
    ),
    (
        "㉑ today 挪进系统提示（每天过零点作废整段 prompt 缓存）",
        NODE,
        "        SystemMessage(content=load_prompt(_EXTRACT_PROMPT)),\n"
        '        HumanMessage(content=human),\n'
        "    ]",
        '        SystemMessage(content=f"{load_prompt(_EXTRACT_PROMPT)}\\n今天是 {today}。"),\n'
        '        HumanMessage(content=human),\n'
        "    ]",
        "test_today_goes_in_the_human_turn_not_the_system_prompt",
    ),
    (
        "㉒ prompt 抄回 Python 字符串（硬红线 #5 失效）",
        NODE,
        "        SystemMessage(content=load_prompt(_EXTRACT_PROMPT)),",
        '        SystemMessage(content="你是旅行规划助理，处在需求收集阶段。"),',
        "test_extract_prompt_is_read_from_the_md_file",
    ),
    (
        "㉓ 路由不看 stage（崩掉的节点带着空白需求冲进生成）",
        EDGES,
        '    if state.stage == "failed":\n        return "reject"\n\n',
        "",
        "test_route_rejects_a_failed_collect",
    ),
    (
        "㉔ reject 那条边接到生成节点上",
        BUILDER,
        '{"ask": END, "plan": "plan_generate", "reject": END},',
        '{"ask": END, "plan": "plan_generate", "reject": "plan_generate"},',
        "test_the_real_graph_stops_at_the_reject_route",
    ),
    (
        "㉕ 替身只打源模块（节点拿着真 build_llm，测试静默联网）",
        CONFTEST,
        "    for module in _llm_consumer_modules():",
        "    for module in [llm_module]:",
        "test_every_node_module_holding_build_llm_got_the_fake",
    ),
]


def run_pytest() -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "sitecustomize.py").write_text(_NO_NETWORK, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([tmp, str(ROOT)])}
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            errors="replace", env=env,
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


def summary_line(out: str) -> str:
    """没有 FAILED 行时的兜底说明。

    **「变红了」有可能是收集期就崩了，一条测试都没跑** —— 这时 `failures_of`
    返回空列表，汇总里那一行会长成 `变红 ✓（0 条测试变红）`，看上去像一次
    干净的捕获。它确实是红的（证明变异有效），但红的不是断言，
    是 import 失败；不写清楚，下一个读的人会把两者当同一回事。

    （实测来源：⑰ 把 delta 的一个字段改成必填，pydantic 直接就构造不出类，
    conftest 导入即崩。）
    """
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[-1] if lines else "（没有任何输出）"


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
        for name in hits[:6]:
            print(f"    - {name}")
        if len(hits) > 6:
            print(f"    …… 另有 {len(hits) - 6} 条")
        if caught and not hits:
            print(f"    ⚠ 一条测试都没跑到（收集期就崩了）：{summary_line(out)}")

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
