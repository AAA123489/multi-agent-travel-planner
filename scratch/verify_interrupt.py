# -*- coding: utf-8 -*-
"""
P0.2 —— 验证 langgraph interrupt() 的恢复语义

对应 docs/开发流程.md 的 P0：
  Q1  interrupt() 恢复后，节点函数里 interrupt() 之前的代码会不会再跑一次？
      → 决定 docs/方案设计.md §5.4「interrupt 前必须无副作用」是硬约束还是软建议
  Q2  图被 interrupt 暂停时，stream 是怎么结束的？最后一块长什么样？
      → 决定 §9.3「interrupt 的收尾」代码怎么写
  Q3  get_state(config).next 在「暂停」和「跑完」两种状态下分别返回什么？
      → 决定 §10.2 的模式判定（resume 还是新输入）靠不靠得住
  Q4  同一节点内连续两次 interrupt() 会怎样？
      → 决定 §5.4「resume 值校验失败时再次 interrupt」是否可行

运行：python scratch/verify_interrupt.py
环境：langgraph 1.2.7 / Python 3.12（无需 API key，不联网）

验证结论写在文件末尾的「结论」段，跑完请对照 docs/方案设计.md 回填。
"""

import sys  # 导入 sys 模块，用于处理标准输出和运行环境相关信息。

try:  # 尝试设置控制台输出编码，避免中文输出出现乱码。
    sys.stdout.reconfigure(encoding="utf-8")  # 把标准输出编码改为 UTF-8，确保中文字符正常显示。
except Exception:  # 兼容部分运行环境没有 reconfigure() 方法的情况。
    pass  # 直接忽略异常，保证脚本不会因为环境差异而中断。

from operator import add  # 导入 add，用于对列表型状态做累加聚合。
from typing import Annotated, TypedDict  # 导入 TypedDict 与 Annotated，用于定义状态结构和聚合字段。

from langgraph.checkpoint.memory import InMemorySaver  # 导入内存检查点组件，用于保存线程执行状态。
from langgraph.graph import START, END, StateGraph  # 导入图的入口、出口和状态图构造器。
from langgraph.types import Command, interrupt  # 导入 Command 和 interrupt，用于恢复执行和触发中断。

SEP = "=" * 72  # 定义统一分隔线，便于输出各个验证场景的标题。


def banner(title: str) -> None:  # 定义标题打印函数，用于输出场景标题栏。
    print(f"\n{SEP}\n{title}\n{SEP}")  # 按统一格式打印分隔线、标题和分隔线。


# ============================================================================
# 场景 A：基本 interrupt / resume
# ============================================================================

class StateA(TypedDict):  # 定义场景 A 的状态结构，只有一个 answer 字段。
    answer: str  # answer 保存节点最终返回值。


LOG_A: list[str] = []  # 记录场景 A 的执行日志，用于观察恢复后是否重跑节点代码。


def node_a(state: StateA):  # 定义场景 A 的核心节点函数。
    # ↓↓↓ Q1 数这一行出现了几次：2 次 = 从头重跑，1 次 = 从断点继续  # Q1 的关键观察点：恢复之后，节点前代码是否再次执行。
    LOG_A.append("[A] 进入 node_a（interrupt 之前的代码）")  # 记录进入节点之前代码的执行。
    got = interrupt({"ask": "你的名字？", "from": "node_a"})  # 触发中断，等待外部恢复值。
    LOG_A.append(f"[A] interrupt() 返回 {got!r}（interrupt 之后的代码）")  # 记录恢复后继续执行的后续逻辑。
    return {"answer": got}  # 返回恢复值，完成节点状态更新。


def build_a():  # 构建场景 A 的状态图。
    g = StateGraph(StateA)  # 创建一个 StateGraph，状态类型为 StateA。
    g.add_node("node_a", node_a)  # 注册 node_a 节点到图中。
    g.add_edge(START, "node_a")  # 连接起点到 node_a。
    g.add_edge("node_a", END)  # 连接 node_a 到终点 END。
    return g.compile(checkpointer=InMemorySaver())  # 编译图并启用内存检查点，支持 interrupt / resume。


def run_scenario_a():  # 执行场景 A 的完整验证流程。
    banner("场景 A：基本 interrupt / resume")  # 打印场景 A 的标题。
    LOG_A.clear()  # 清空日志，避免旧数据干扰当前验证。
    app = build_a()  # 构造当前场景的应用实例。
    cfg = {"configurable": {"thread_id": "A"}}  # 设置线程 ID，隔离不同场景状态。

    print("① 首次 stream({}, cfg) —— 图会跑到 interrupt 处暂停")  # 输出第一次执行说明。
    try:  # 捕获首次执行时可能的异常。
        for chunk in app.stream({}, cfg):  # 初次调用 stream 时，图会在 interrupt 处暂停。
            print("     chunk:", chunk)  # 输出流式 chunk，便于观察暂停行为。
    except Exception as e:  # 捕获异常并记录。
        print(f"     ✗ 抛异常 {type(e).__name__}: {e}")  # 输出异常类型和信息，便于定位问题。

    snap = app.get_state(cfg)  # 获取当前线程状态快照，观察暂停态。
    print("\n② get_state() —— 暂停态")  # 输出暂停状态说明。
    print(f"     .next       = {snap.next!r}")  # 打印 next 字段，观察中断时的状态。
    print(f"     .values     = {snap.values!r}")  # 打印 values 字段，观察图状态数据。
    print(f"     .interrupts = {getattr(snap, 'interrupts', '<无此属性>')!r}")  # 输出中断信息，判断当前线程是否正在暂停。
    print(f"     .tasks      = {snap.tasks!r}")  # 输出任务状态，补充状态信息。

    print("\n③ stream(Command(resume='张三'), cfg) —— 恢复执行")  # 输出恢复执行说明。
    try:  # 捕获恢复过程中的异常。
        for chunk in app.stream(Command(resume="张三"), cfg):  # 使用 resume 参数恢复图执行。
            print("     chunk:", chunk)  # 输出恢复后的 chunk 流。
    except Exception as e:  # 如果恢复失败则记录异常。
        print(f"     ✗ 抛异常 {type(e).__name__}: {e}")  # 打印恢复阶段异常信息。

    snap2 = app.get_state(cfg)  # 获取恢复后的状态快照。
    print("\n④ get_state() —— 跑完态")  # 输出完成态说明。
    print(f"     .next       = {snap2.next!r}")  # 打印恢复后的 next 字段。
    print(f"     .values     = {snap2.values!r}")  # 打印恢复后的值状态。

    print("\n⑤ 观测记录 LOG_A：")  # 输出关键日志记录。
    for line in LOG_A:  # 逐行打印日志记录。
        print("     ", line)  # 逐项显示执行日志。

    enters = sum(1 for x in LOG_A if "进入 node_a" in x)  # 统计 node_a 实际被调用多少次。
    print(f"\n>>> Q1 结论：node_a 函数体共被进入 {enters} 次")  # 输出第一问的结论。
    print(f"     {enters} == 2  → 恢复时【从头重跑】，§5.4 的「纯函数」约束是硬约束")  # 若等于 2，则说明恢复时会从头重跑。
    print(f"     {enters} == 1  → 恢复时【从断点继续】，§5.4 可以放宽")  # 若等于 1，则说明恢复时从断点继续。


# ============================================================================
# 场景 B：同一节点内连续两次 interrupt
# ============================================================================

class StateB(TypedDict):  # 定义场景 B 的状态结构。
    answer: str  # answer 保存最终返回值。


LOG_B: list[str] = []  # 记录场景 B 的执行日志，用于分析连续 interrupt 行为。


def node_b(state: StateB):  # 定义场景 B 的节点函数。
    LOG_B.append("[B] 进入 node_b")  # 记录进入节点时刻。
    first = interrupt({"ask": "第一次提问", "idx": 1})  # 第一次触发 interrupt，等待恢复值。
    LOG_B.append(f"[B] 第 1 个 interrupt() 返回 {first!r}")  # 记录第一次恢复值。

    if first != "ok":  # 如果第一次恢复值不合法，则继续触发第二次 interrupt。
        second = interrupt({"ask": "第一次输入不合法，请重新输入", "idx": 2})  # 第二次 interrupt，模拟校验失败后的重问。
        LOG_B.append(f"[B] 第 2 个 interrupt() 返回 {second!r}")  # 记录第二次恢复值。
        return {"answer": second}  # 返回第二次恢复值，结束节点。

    return {"answer": first}  # 第一次值合法时直接返回它。


def build_b():  # 构建场景 B 的状态图。
    g = StateGraph(StateB)  # 创建图对象。
    g.add_node("node_b", node_b)  # 注册节点 node_b。
    g.add_edge(START, "node_b")  # 连接起点到 node_b。
    g.add_edge("node_b", END)  # 连接 node_b 到 END。
    return g.compile(checkpointer=InMemorySaver())  # 编译图并启用检查点。


def run_scenario_b():  # 执行场景 B 的验证流程。
    banner("场景 B：同一节点内连续两次 interrupt（验证 §5.4 的异常输入处理）")  # 打印场景标题。
    LOG_B.clear()  # 清空上一轮日志。
    app = build_b()  # 创建当前场景的 app。
    cfg = {"configurable": {"thread_id": "B"}}  # 配置线程 ID。

    print("① 首次 stream —— 应停在 node_b 的第 1 个 interrupt")  # 输出第一次执行说明。
    try:  # 捕获第一次执行异常。
        for chunk in app.stream({}, cfg):  # 第一次流式调用应在第一个 interrupt 处暂停。
            print("     chunk:", chunk)  # 输出流式 chunk。
    except Exception as e:  # 统一处理异常。
        print(f"     ✗ 抛异常 {type(e).__name__}: {e}")  # 输出异常信息。

    print("\n② resume 一个【不合法】的值 'bad' —— 应触发第 2 个 interrupt")  # 输出第二次恢复说明。
    try:  # 捕获第二次恢复的异常。
        for chunk in app.stream(Command(resume="bad"), cfg):  # 模拟非法输入 bad，触发第二个 interrupt。
            print("     chunk:", chunk)  # 输出恢复后的 chunk。
    except Exception as e:  # 如果恢复失败则记录异常。
        print(f"     ✗ 抛异常 {type(e).__name__}: {e}")  # 输出异常细节。
    snap = app.get_state(cfg)  # 获取当前状态，查看图是否仍处于暂停态。
    print(f"     .next = {snap.next!r}   ← ⚠️ 实测是 (), 图确实暂停了但 .next 为空！")  # 说明 .next 在中断时可能为空。
    print(f"     .interrupts = {snap.interrupts!r}")  # 输出中断列表，确认中断状态。

    print("\n③ resume【合法】值 'ok' —— 走完")  # 输出第三次恢复说明。
    try:  # 捕获最终恢复阶段异常。
        for chunk in app.stream(Command(resume="ok"), cfg):  # 使用合法值 ok 恢复并继续执行。
            print("     chunk:", chunk)  # 输出最终 chunk。
    except Exception as e:  # 处理异常。
        print(f"     ✗ 抛异常 {type(e).__name__}: {e}")  # 输出异常信息。
    snap = app.get_state(cfg)  # 获取最终状态快照。
    print(f"     .next = {snap.next!r}   .values = {snap.values!r}")  # 输出最终状态字段。

    print("\n④ 观测记录 LOG_B：")  # 输出日志记录。
    for line in LOG_B:  # 逐行读取日志。
        print("     ", line)  # 输出记录内容。

    got_ok = any("'ok'" in x for x in LOG_B)  # 判断日志中是否出现合法恢复值 ok。
    print(f"\n>>> Q4 结论：{'可行' if got_ok else '不可行'}")  # 输出第四问结论。
    print("     注意看第 1 个 interrupt() 在第二次恢复时返回的是 'bad' 还是别的值 ——")  # 提醒观察恢复值在多次中断中的流转。
    print("     这决定「校验失败后再次 interrupt」时能不能保留上一次的输入。")  # 说明这个结论对设计的影响。


# ============================================================================
# 场景 C：状态判定矩阵 + 图跑完后能否用新输入再入图
#   —— 直接模拟本项目的「追问 → 下一轮再入图」结构（P3 路径 ④）
# ============================================================================

class StateC(TypedDict):  # 定义场景 C 的状态结构，包括 user_query 和历史记录。
    user_query: str  # 当前轮次的用户输入文本。
    history: Annotated[list[str], add]  # 历史记录会通过 add 聚合式累积。


def collect(state: StateC):  # 处理收集用户输入的节点。
    q = state.get("user_query") or ""  # 安全获取当前用户输入，空则为 ""。
    return {"user_query": "", "history": [f"收到: {q}"]}  # 清空当前输入并追加到历史中。


def confirm(state: StateC):  # 定义确认节点，等待用户确认。
    decision = interrupt({"ask": "确认行程？", "history": state["history"]})  # 挂起执行，等待外部确认值。
    return {"history": [f"用户: {decision}"]}  # 将用户确认值写回状态历史。


def route_c(state: StateC):  # 根据历史状态路由到确认或结束。
    # 摹拟真实路由：信息不够就回 END 等下一轮，够了才进确认  # 说明真实业务中的条件判断思路。
    return "confirm" if len(state["history"]) >= 2 else "ask"  # 若历史满足要求则进入 confirm，否则回 END。


def build_c():  # 构造场景 C 的状态图。
    g = StateGraph(StateC)  # 创建图对象。
    g.add_node("collect", collect)  # 注册 collect 节点。
    g.add_node("confirm", confirm)  # 注册 confirm 节点。
    g.add_edge(START, "collect")  # 从起点到 collect。
    g.add_conditional_edges("collect", route_c, {"confirm": "confirm", "ask": END})  # 根据历史长度判断是否继续到确认。
    g.add_edge("confirm", END)  # confirm 执行完后走向 END。
    return g.compile(checkpointer=InMemorySaver())  # 编译图并启用内存检查点。


def probe(app, cfg, label):  # 状态探针，用于输出当前图的暂停/结束状态。
    snap = app.get_state(cfg)  # 获取当前状态快照。
    nxt, ints = snap.next, snap.interrupts  # 读取 next 和 interrupts 字段。
    print(f"     [{label}]")  # 打印标签。
    print(f"        .next       = {nxt!r}")  # 打印 next 字段。
    print(f"        .interrupts = {ints!r}")  # 打印 interrupts 字段。
    print(f"        .values     = {snap.values!r}")  # 打印 values 字段。
    # 这才是可靠的「图是否暂停」判定  # 说明真正的暂停判定应看 .interrupts。
    print(f"        >>> 暂停判定 has_interrupt = {bool(ints)}")  # 将 interrupts 转成布尔值，直观判定是否暂停。


def run_scenario_c():  # 执行场景 C 的完整验证流程。
    banner("场景 C：状态判定矩阵 + 跑完后再次入图（模拟追问链路）")  # 输出场景标题。
    app = build_c()  # 构造 app 实例。
    cfg = {"configurable": {"thread_id": "C"}}  # 设置线程 ID。

    print("① 第 1 轮：只给一句话（信息不足 → 应走 ask 分支到 END）")  # 输出第一轮说明。
    for chunk in app.stream({"user_query": "我想去成都"}, cfg):  # 第一轮只给一个输入，信息不足，应走 END。
        print("     chunk:", chunk)  # 打印 chunk。
    probe(app, cfg, "第1轮跑完（图已到 END，非暂停）")  # 检查当前状态，确认图已结束。

    print("\n② 第 2 轮：同一 thread 注入新输入（历史应保留 → 走 confirm → 暂停）")  # 输出第二轮说明。
    for chunk in app.stream({"user_query": "玩3天"}, cfg):  # 第二轮在同一 thread 中继续注入新输入。
        print("     chunk:", chunk)  # 打印 chunk。
    probe(app, cfg, "第2轮暂停（在 confirm 处 interrupt）")  # 检查图是否在 confirm 处暂停。

    print("\n③ resume 确认 → 图跑完")  # 输出恢复说明。
    for chunk in app.stream(Command(resume="满意"), cfg):  # 模拟用户确认，恢复图继续执行。
        print("     chunk:", chunk)  # 打印最终 chunk。
    probe(app, cfg, "第3轮跑完")  # 检查图已结束。

    print("\n④ 图跑完后，再注入新输入会怎样？（模拟用户开新一段对话）")  # 输出新轮输入说明。
    try:  # 捕获第四轮输入可能出现的异常。
        for chunk in app.stream({"user_query": "再来一次"}, cfg):  # 再次注入新输入，模拟新一轮对话。
            print("     chunk:", chunk)  # 打印新一轮的 chunk。
    except Exception as e:  # 处理异常。
        print(f"     ✗ 抛异常 {type(e).__name__}: {e}")  # 打印异常详细信息。
    probe(app, cfg, "第4轮")  # 检查新一轮后的状态。

    print("\n>>> Q3 结论：判定「图是否暂停」要用 .interrupts 而不是 .next")  # 输出关键结论。
    print("     .next 在「节点中途 interrupt」的某些情况下会返回 ()，与「图已跑完」无法区分。")  # 解释 .next 的歧义。
    print("     .interrupts 非空 ⟺ 图处于暂停态，两种情况都能正确区分。")  # 提出可靠判定规则。


# ============================================================================

if __name__ == "__main__":  # 入口函数：当脚本作为主程序执行时运行下面的代码。
    print(f"Python  {sys.version.split()[0]}")  # 输出 Python 版本，方便记录脚本运行环境。
    run_scenario_a()  # 执行场景 A：验证基础 interrupt / resume 语义。
    run_scenario_b()  # 执行场景 B：验证连续 interrupt 与重试行为。
    run_scenario_c()  # 执行场景 C：验证状态判定和新轮回流逻辑。
    banner("跑完了。把上面标 >>> 的结论回填进 docs/方案设计.md 的 §5.4 / §9.3 / §10.2")  # 输出最终提示，提醒回填设计文档。