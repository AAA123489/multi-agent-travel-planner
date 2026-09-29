"""P4.0 变异测试 —— 专门验证 `plan_struct` 契约的那批新断言。

**这个脚本存在的理由**：P4.0 新增的 8 条测试断的都是**设计决定**（哪些问题在
结构层拦、`poi_id` 谁填、`nights` 是不是派生……），而不是格式细节。设计决定的
断言最容易写成摆设 —— 把实现改掉它照样绿，因为它断的是「文档里写着的那句话」
而不是「代码真的这么做了」。

每条变异：备份原文件 → 精确替换一处 → 跑 pytest → 还原 → 记录结果。
**任何一条绿着通过，都说明那条断言是摆设。**

跑法：
    .venv/Scripts/python.exe -X utf8 scratch/mutate_p4.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATE = "app/graph/state.py"

# (说明, 文件, 原文片段, 替换成什么, 期望哪条测试挂)
MUTATIONS = [
    (
        "① 时分不再归一化（「9:00」直接撞 pattern）",
        STATE,
        '        if not isinstance(value, str):\n'
        '            return value\n'
        '        text = value.strip().replace("：", ":")   # 全角冒号\n'
        '        head, sep, tail = text.partition(":")\n'
        '        if sep and head.isdigit() and len(head) == 1:\n'
        '            return f"0{head}:{tail}"\n'
        '        return text\n',
        "        return value\n",
        "test_plan_item_normalizes_loose_time_but_rejects_garbage",
    ),
    (
        "② 归一化开始「猜」—— 把无法识别的原样返回改成兜底 02:00",
        STATE,
        '        head, sep, tail = text.partition(":")\n'
        '        if sep and head.isdigit() and len(head) == 1:\n'
        '            return f"0{head}:{tail}"\n'
        '        return text\n',
        '        head, sep, tail = text.partition(":")\n'
        '        if sep and head.isdigit() and len(head) == 1:\n'
        '            return f"0{head}:{tail}"\n'
        '        return "02:00"\n',
        "test_plan_item_normalizes_loose_time_but_rejects_garbage（垃圾值应报错）",
    ),
    (
        "③ `kind` 里把 transport 加回来",
        STATE,
        'PlanItemKind = Literal["attraction", "meal", "hotel"]',
        'PlanItemKind = Literal["attraction", "meal", "hotel", "transport"]',
        "test_plan_item_kind_has_no_transport",
    ),
    (
        "④ `poi_id` 的缺省值从 None 改成空串（幻觉信号与「没填」混淆）",
        STATE,
        '    poi_id: str | None = Field(\n'
        '        default=None,\n',
        '    poi_id: str | None = Field(\n'
        '        default="",\n',
        "test_plan_item_poi_id_defaults_to_none",
    ),
    (
        "⑤ 给 PlanStruct 加上内容校验（空 days 直接构造失败）",
        STATE,
        '    model_config = ConfigDict(frozen=True)\n\n'
        '    days: list[PlanDay] = Field(default_factory=list)\n',
        '    model_config = ConfigDict(frozen=True)\n\n'
        '    days: list[PlanDay] = Field(default_factory=list, min_length=1)\n',
        "test_plan_struct_does_not_validate_content",
    ),
    (
        "⑥ `nights` 改成按 hotel 条目数算（第二份事实来源）",
        STATE,
        '        return max(0, len(self.days) - 1)',
        '        return sum(1 for d in self.days for i in d.items if i.kind == "hotel")',
        "test_plan_struct_nights_is_derived_from_days",
    ),
    (
        "⑦ PlanItem 不再 frozen",
        STATE,
        '    model_config = ConfigDict(frozen=True)\n\n    kind: PlanItemKind',
        '    kind: PlanItemKind',
        "test_plan_models_are_frozen",
    ),
    (
        "⑧ TravelState.plan_struct 缺省从 None 改成空 PlanStruct",
        STATE,
        '    plan_struct: PlanStruct | None = None  # 每日点位顺序（给机器校验）',
        '    plan_struct: PlanStruct | None = Field(default_factory=PlanStruct)',
        "test_travel_state_plan_struct_defaults_to_none",
    ),
    (
        "⑨ 占位 plan_struct 换成一份「看起来合格」的内容（poi_id 填上）",
        "app/graph/nodes/plan_generate.py",
        'PlanItem(kind="attraction", name="（P4 占位景点）", start="09:00", end="11:00"),',
        'PlanItem(kind="attraction", name="（P4 占位景点）", poi_id="cd-001",\n'
        '                     start="09:00", end="11:00"),',
        "test_stub_plan_struct_is_well_formed_but_incomplete",
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
