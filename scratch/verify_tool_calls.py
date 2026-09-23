# -*- coding: utf-8 -*-
"""
P0.3 —— 验证 LLM 返回的是标准 tool_calls，还是把调用意图吐成了文本（DSML）

对应 docs/开发流程.md 的 P0.3：
  为什么必须验：选型是「OpenAI 兼容协议 + 配置驱动切模型」。真正依赖 tool_calls 协议的有三处——
  requirement_collect 的 with_structured_output、evaluate 的 Judge 打分、PLAN_TOOL_MODE=agent 档
  的 LLM 自主调用（默认的 deterministic 档下工具由节点代码确定性调用，不走协议，见方案设计 §4.2）。
  有实测案例表明，DeepSeek 在部分场景下**不返回标准 tool_calls 字段，而是把调用意图吐成 DSML 格式的文本**。
  这种失败**不会报错** —— LLM 把「我要查武侯祠」当正文讲给用户听，而不是真的去查数据。
  结果就是 feasibility 指标莫名掉分，而你查不出原因。

验什么（5 个场景，每个对应一处会真实踩到的坑）：

  A  非流式 · 单工具        → 最基本的判据：tool_calls 非空 + 结构标准
  B  流式 · 单工具          → 我们最终走 SSE，真实调用路径是流式。delta 拼装后还是不是标准结构？
  C  非流式 · 多工具并行    → 决定 messages 里 tool 结果的配对能不能按 index 来
  D  回灌 tool 结果再问一次 → 验证「伪造 id 补回 messages」能否被 API 接受（防御层的核心风险）
  E  强制 tool_choice=none  → 主动制造「模型想调工具但协议层不许」的处境，看 content 有没有文本标记

运行：
    # 1) 项目根目录建 .env（已在 .gitignore 里），写这三行：
    #       LLM_API_KEY=sk-xxxxxxxx
    #       LLM_BASE_URL=https://api.deepseek.com/v1
    #       LLM_MODEL=deepseek-v4-flash
    # 2) .venv/Scripts/python.exe scratch/verify_tool_calls.py
    #    只跑其中几个场景：python scratch/verify_tool_calls.py A B C

环境：openai 3.19.0 直连，需要 API key、需要联网。
      **刻意不走 langchain** —— langchain-openai 会做一层包装与归一化，可能把不标准的输出吃掉，
      那样我们就看不到线上的真相。这里要的是原始报文。

验证结论写在文件末尾的「>>> 结论」段，跑完请对照 docs/方案设计.md 回填（§12.1 LLM 调用协议实测）。
"""

import json  # 解析 tool_calls 里的 arguments，判断是不是合法 JSON。
import os  # 读取环境变量中的 API key / base_url / 模型名。
import sys  # 处理标准输出编码与命令行参数。

try:  # 尝试设置控制台输出编码，避免中文和 ▁ 这类特殊符号乱码。
    sys.stdout.reconfigure(encoding="utf-8")  # 把标准输出编码改为 UTF-8，Windows 控制台必需。
except Exception:  # 兼容部分运行环境没有 reconfigure() 方法的情况。
    pass  # 直接忽略异常，保证脚本不会因为环境差异而中断。

from openai import OpenAI  # 导入官方 SDK，用最原始的方式发请求、看原始响应。

SEP = "=" * 72  # 统一分隔线，便于在长输出里定位各场景。

# 文本形式的「调用意图泄漏」特征词。命中任意一个 → content 里混进了伪协议。
# 注意 `dsml` 用小写比对（content.lower()），能同时抓住 DSML / dsml / Dsml。
TEXT_LEAK_MARKERS = (
    "<tool_call>", "</tool_call>",  # 最常见的伪标签形式
    "<function_call>", "</function_call>",  # 另一种伪标签外壳
    "dsml",  # DeepSeek 自家格式的标记
    "tool▁calls", "｜tool▁calls｜",  # 带 U+2581 下划线的全角标记（DeepSeek 特有）
    "<|tool_calls|>", "<|invoke|>",  # 管道符风格的特判 token
    "antml:invoke",  # 部分模型抄 Anthropic 模板时会漏出这个
)

RESULTS: list[tuple[str, str, str]] = []  # 汇总表：(场景, 判定, 说明)，跑完统一打印

# 场景 A 的原始响应，场景 D 要用它当上下文，所以存在模块级。
SCENARIO_A_MSG: dict | None = None  # 归一化后的 assistant 消息（含 tool_calls）


def banner(title: str) -> None:  # 打印场景标题栏。
    print(f"\n{SEP}\n{title}\n{SEP}")  # 按统一格式打印分隔线、标题和分隔线。


def record(scenario: str, verdict: str, note: str) -> None:  # 记录一个场景的判定结果。
    RESULTS.append((scenario, verdict, note))  # 追加到汇总表，最后一起打印。
    print(f"\n>>> [{scenario}] {verdict} —— {note}")  # 立刻回显，跑的时候就能看到。


# ============================================================================
# 配置与客户端
# ============================================================================

def load_config() -> dict | None:  # 读取 LLM 配置，缺 key 时给出可直接照抄的指引。
    try:  # dotenv 没装也不该让脚本崩掉，直接把异常咽了走 os.environ。
        from dotenv import load_dotenv  # 延迟导入，避免顶层依赖失败导致整个脚本不可用。
        from pathlib import Path  # 用于定位项目根目录。
        env_path = Path(__file__).resolve().parents[1] / ".env"  # 按脚本位置推项目根，而不是靠当前工作目录。
        # 显式传路径：从别的目录调用时也能找到 .env，否则会静默变成「没配 key」。
        load_dotenv(env_path)  # 载入项目根目录的 .env。
        print(f"已加载 .env：{env_path}（存在={env_path.exists()}）")  # 回显实际路径，配置问题一眼可见。
    except ImportError:  # 环境里没有 python-dotenv 时降级。
        print("（提示：未安装 python-dotenv，只读系统环境变量）")  # 说明降级行为。

    api_key = os.environ.get("LLM_API_KEY", "").strip()  # 必填项，取不到就得提示用户。
    if not api_key:  # 没有 key 就无法继续，给出明确的自救步骤。
        print("\n" + SEP)  # 打印分隔线，让提示更醒目。
        print("✗ 未找到 LLM_API_KEY，脚本无法运行。")  # 明确指出缺失项。
        print("  在项目根目录建一个 .env（已在 .gitignore 中，不会入库），写入：")  # 给出操作路径。
        print("      LLM_API_KEY=sk-xxxxxxxx")  # 示例：密钥。
        print("      LLM_BASE_URL=https://api.deepseek.com/v1")  # 示例：DeepSeek 端点。
        print("      LLM_MODEL=deepseek-v4-flash")  # 示例：模型名，与方案设计 §12 默认值一致。
        print("  然后重跑：.venv/Scripts/python.exe scratch/verify_tool_calls.py")  # 给出重跑命令。
        print(SEP)  # 收尾分隔线。
        return None  # 用 None 表示配置不完整。

    cfg = {  # 组装配置字典。
        "api_key": api_key,  # 密钥，绝不打印明文。
        "base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").strip(),  # 端点，按 §12 默认值。
        "model": os.environ.get("LLM_MODEL", "deepseek-v4-flash").strip(),  # 模型名，按 §12 默认值。
        "timeout": float(os.environ.get("LLM_TIMEOUT", "60")),  # 单次调用超时，按 §12 默认值。
    }  # 配置组装完毕。
    print(f"base_url = {cfg['base_url']}")  # 回显端点，方便确认没连错环境。
    print(f"model    = {cfg['model']}")  # 回显模型名，结论要跟模型版本绑定。
    print(f"api_key  = {mask_key(cfg['api_key'])}   ← 脱敏，红线要求密钥永不进日志")  # 只打印掩码形式。
    return cfg  # 返回可用配置。


def mask_key(key: str) -> str:  # 密钥脱敏函数，对应方案设计 §14 的日志红线。
    if len(key) <= 10:  # 太短的 key 直接整体打码，避免反向推出内容。
        return "*" * len(key)  # 全掩码。
    return f"{key[:6]}...{key[-4:]}"  # 保留头尾各几位，够定位是哪个 key 即可。


def make_client(cfg: dict) -> OpenAI:  # 构造直连客户端。
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=cfg["timeout"])  # 标准 OpenAI 兼容用法。


# ============================================================================
# 工具定义 —— 签名照抄方案设计 §6.3，P4 可以几乎原样搬过去
# ============================================================================

WEATHER_TOOL = {  # 场景 A/B/E 用的最小工具：参数最简单，排除「模型不会填复杂参数」的干扰。
    "type": "function",  # 声明这是 function 类型工具。
    "function": {  # 工具描述体。
        "name": "get_weather",  # 工具名，小写下划线风格。
        "description": "查询指定城市今天的天气。当用户询问天气、气温、是否下雨时必须调用。",  # 描述里写死触发条件，降低模型不调用的概率。
        "parameters": {  # 参数 schema，标准 JSON Schema。
            "type": "object",  # 顶层固定为 object。
            "properties": {  # 逐字段声明。
                "city": {"type": "string", "description": "城市名，如「成都」"},  # 城市参数。
            },  # properties 结束。
            "required": ["city"],  # city 必填。
        },  # parameters 结束。
    },  # function 结束。
}  # WEATHER_TOOL 结束。

POI_TOOL = {  # 场景 C 用的第二个工具：对应 §6.3 的 poi_query，用来验并行调用。
    "type": "function",  # function 类型。
    "function": {  # 描述体。
        "name": "poi_query",  # 与 §6.3 一致的工具名。
        "description": "按城市和标签查询景点列表。用户想找景点、美食、人文类去处时必须调用。",  # 触发条件写清楚。
        "parameters": {  # 参数 schema。
            "type": "object",  # 顶层 object。
            "properties": {  # 字段声明。
                "city": {"type": "string", "description": "城市名"},  # 城市。
                "tags": {  # 标签数组。
                    "type": "array",  # 数组类型。
                    "items": {"type": "string"},  # 元素为字符串。
                    "description": "标签，如 人文 / 美食 / 自然",  # 取值说明。
                },  # tags 结束。
                "top_k": {"type": "integer", "description": "返回条数，默认 5"},  # 条数。
            },  # properties 结束。
            "required": ["city"],  # city 必填，其余可选 —— 顺便看模型会不会漏填可选参数。
        },  # parameters 结束。
    },  # function 结束。
}  # POI_TOOL 结束。

SYSTEM_PROMPT = (  # 系统提示，模仿真实节点的口吻。
    "你是一个旅行规划助手。需要外部数据时，必须调用提供的工具获取，"
    "不要凭记忆编造天气或景点信息。"
)  # 提示词结束。


# ============================================================================
# 响应解析与打印 —— 非流式和流式共用同一套输出，方便肉眼对比
# ============================================================================

def normalize_message(msg) -> dict:  # 把 SDK 的 ChatCompletionMessage 拍平成纯字典。
    return {  # 统一结构：role / content / tool_calls 三项。
        "role": msg.role,  # 角色，正常是 assistant。
        "content": msg.content,  # 正文，DSML 泄漏就发生在这里。
        "tool_calls": [  # 逐条拍平工具调用。
            {  # 单条工具调用的标准形状。
                "index": i,  # 序号，从 0 开始（非流式响应的 SDK 对象没有 index，这里自己补）。
                "id": tc.id,  # 调用 id，回灌 tool 结果时必须原样带回。
                "type": tc.type,  # 正常是 "function"。
                "name": tc.function.name,  # 函数名。
                "arguments": tc.function.arguments,  # 参数字符串（注意：是字符串，不是 dict）。
            }  # 单条结束。
            for i, tc in enumerate(msg.tool_calls or [])  # 没有 tool_calls 时退化成空列表。
        ],  # tool_calls 结束。
    }  # 字典结束。


def scan_text_leak(content: str | None) -> list[str]:  # 在正文里找伪协议标记。
    if not content:  # 空正文不可能有标记。
        return []  # 直接返回空。
    low = content.lower()  # 统一小写，兼容 DSML/dsml 各种写法。
    hits = [m for m in TEXT_LEAK_MARKERS if m.lower() in low]  # 逐个特征词做子串匹配。
    # 兜底启发式：正文里同时出现 "name" 和 "arguments" 且长得像 JSON，多半是调用意图漏成了文本。
    if not hits and '"arguments"' in low and '"name"' in low:  # 判断是否疑似 JSON 化的调用意图。
        hits.append("疑似 JSON 化的 tool_call（含 name+arguments 字段）")  # 记一条可疑线索。
    return hits  # 返回命中的特征词列表。


def dump_message(label: str, norm: dict) -> None:  # 打印一条归一化后的消息，这是本脚本的核心观察点。
    print(f"\n[{label}]")  # 打印标签行。
    print(f"  role     = {norm['role']!r}")  # 打印角色。
    content = norm["content"]  # 取出正文。
    if content is None or content == "":  # 正文为空是「干净」的表现之一。
        print("  content  = <空>")  # 明确标出空，而不是打印 None 让人猜。
    else:  # 有正文就原样打出来。
        print(f"  content  = {content!r}   ← 正文原文，重点看这里有没有伪协议标记")  # repr 形式能看到转义和不可见字符。
        hits = scan_text_leak(content)  # 扫描泄漏特征。
        if hits:  # 命中就高亮报警。
            print(f"  ⚠️ 正文中命中疑似伪协议标记：{hits}")  # 打印命中的特征词。
        else:  # 没命中则说明正文干净。
            print("  ✓ 正文中未发现伪协议标记")  # 正面结论。

    calls = norm["tool_calls"]  # 取出工具调用列表。
    print(f"  tool_calls 数量 = {len(calls)}")  # 数量是第一个判据：0 就是没调用。
    for tc in calls:  # 逐条详查。
        print(f"    [{tc['index']}] id        = {tc['id']!r}")  # id 为空会直接导致后续配对断裂。
        print(f"         type      = {tc['type']!r}")  # 非 "function" 说明协议不对。
        print(f"         name      = {tc['name']!r}")  # 函数名必须能在我们的工具表里找到。
        print(f"         arguments = {tc['arguments']!r}")  # 原始字符串，别急着 parse。
        try:  # 试着解析参数。
            parsed = json.loads(tc["arguments"] or "{}")  # 正常情况是合法 JSON 字符串。
            print(f"         → JSON 解析成功：{parsed}")  # 成功则打印结构化结果。
        except Exception as e:  # 解析失败是重要发现：参数被截断或本身就是伪格式。
            print(f"         ✗ JSON 解析失败：{type(e).__name__}: {e}")  # 打印失败原因。


def judge_tool_calls(norm: dict, label: str) -> tuple[str, str]:  # 对一个响应下三向判定。
    calls = norm["tool_calls"]  # 取出调用列表。
    hits = scan_text_leak(norm["content"])  # 扫正文标记。
    if calls:  # 有调用 → 再查结构完整性。
        problems = []  # 收集结构问题。
        for tc in calls:  # 逐条检查。
            if not tc["id"]:  # id 缺失。
                problems.append(f"[{tc['index']}] 缺 id")  # 记录问题。
            if tc["type"] != "function":  # 类型不对。
                problems.append(f"[{tc['index']}] type={tc['type']!r} 不是 function")  # 记录问题。
            if not tc["name"]:  # 函数名缺失。
                problems.append(f"[{tc['index']}] 缺 name")  # 记录问题。
            try:  # 参数必须是合法 JSON，否则回灌 messages 时 API 会拒。
                json.loads(tc["arguments"] or "{}")  # 只做验证，不用结果。
            except Exception:  # 解析失败。
                problems.append(f"[{tc['index']}] arguments 不是合法 JSON")  # 记录问题。
        if problems:  # 有结构缺陷。
            return "⚠️ 部分通过", f"tool_calls 非空但有结构缺陷：{problems}"  # 需要防御层补齐。
        return "✅ 通过", f"tool_calls 结构标准（{len(calls)} 条），正文无伪协议标记"  # 最理想的结果。
    if hits:  # 没调用，但正文里有标记 → 就是文档警告的那种静默失败。
        return "⚠️ 需防御层", f"tool_calls 为空，但正文里出现伪协议标记：{hits}"  # 结论：必须加归一化层。
    return "❌ 未通过", "tool_calls 为空，正文也没有伪协议标记 —— 模型只是没调用工具"  # 成因不同：模型或 prompt 的问题。


# ============================================================================
# 场景 A：非流式 · 单工具
# ============================================================================

def run_scenario_a(client: OpenAI, cfg: dict) -> None:  # 场景 A：最基础的判据。
    global SCENARIO_A_MSG  # 需要把结果存到模块级，供场景 D 复用。
    banner("场景 A：非流式 · 单工具（最基本的判据）")  # 打印标题。
    messages = [  # 构造两轮对话。
        {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示。
        {"role": "user", "content": "成都今天天气怎么样？"},  # 必然触发工具调用的问题。
    ]  # messages 结束。
    try:  # 网络和鉴权错误不该让整个脚本崩掉。
        resp = client.chat.completions.create(  # 发起非流式请求。
            model=cfg["model"],  # 模型名。
            messages=messages,  # 对话内容。
            tools=[WEATHER_TOOL],  # 只给一个工具。
            tool_choice="auto",  # 交给模型自己决定。
            temperature=0,  # 探针用 0，减少随机性带来的噪音。
        )  # 请求结束。
    except Exception as e:  # 捕获所有异常。
        print(f"✗ 调用失败：{type(e).__name__}: {e}")  # 打印异常，便于判断是 key 错还是端点错。
        record("A 非流式单工具", "⛔ 调用失败", f"{type(e).__name__}: {e}")  # 记入汇总表。
        return  # 提前退出本场景。

    print(f"finish_reason = {resp.choices[0].finish_reason!r}   ← 正常应是 'tool_calls'")  # finish_reason 是最快的旁证。
    norm = normalize_message(resp.choices[0].message)  # 归一化后统一打印。
    dump_message("场景 A 原始响应", norm)  # 输出观察点。
    SCENARIO_A_MSG = norm  # 存起来给场景 D 用。
    verdict, note = judge_tool_calls(norm, "A")  # 下判定。
    record("A 非流式单工具", verdict, note)  # 记录结果。


# ============================================================================
# 场景 B：流式 —— 我们最终走 SSE，这条才是线上的真实路径
# ============================================================================

def run_scenario_b(client: OpenAI, cfg: dict) -> None:  # 场景 B：delta 拼装后的结构。
    banner("场景 B：流式 · 单工具（SSE 真实路径，delta 拼装）")  # 打印标题。
    messages = [  # 与场景 A 同构，保证对比干净。
        {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示。
        {"role": "user", "content": "杭州今天天气怎么样？"},  # 问另一个城市，避免命中缓存式的记忆。
    ]  # messages 结束。
    acc: dict[int, dict] = {}  # index -> 拼接中的工具调用，流式必须按 index 归并。
    content_parts: list[str] = []  # 正文片段，流式下 content 是逐 token 来的。
    finish_reason = None  # 记录结束原因。
    chunk_count = 0  # 统计 chunk 数，顺便验证流确实是分片的。
    try:  # 捕获流式调用异常。
        stream = client.chat.completions.create(  # 发起流式请求。
            model=cfg["model"],  # 模型名。
            messages=messages,  # 对话内容。
            tools=[WEATHER_TOOL],  # 同一个工具。
            tool_choice="auto",  # 同样交给模型决定。
            temperature=0,  # 同样用 0。
            stream=True,  # 打开流式。
        )  # 请求结束。
        for chunk in stream:  # 逐个消费 chunk。
            chunk_count += 1  # 计数。
            if not chunk.choices:  # 有些 chunk 只有 usage 没有 choices。
                continue  # 跳过。
            choice = chunk.choices[0]  # 取第一个候选。
            delta = choice.delta  # 取增量对象。
            if delta.content:  # 有正文增量。
                content_parts.append(delta.content)  # 累加。
            for tc in (delta.tool_calls or []):  # 处理工具调用增量。
                slot = acc.setdefault(  # 按 index 取或新建槽位。
                    tc.index,  # delta 里的 index 是拼装的关键。
                    {"index": tc.index, "id": "", "type": "function", "name": "", "arguments": ""},  # 槽位初值。
                )  # setdefault 结束。
                if tc.id:  # id 通常只在第一个 delta 里出现。
                    slot["id"] = tc.id  # 记下 id。
                if tc.type:  # 类型同理。
                    slot["type"] = tc.type  # 记下类型。
                if tc.function and tc.function.name:  # 函数名可能被切开分片下发。
                    slot["name"] += tc.function.name  # 用累加而不是覆盖。
                if tc.function and tc.function.arguments:  # 参数几乎必然是分片的。
                    slot["arguments"] += tc.function.arguments  # 累加拼装。
            if choice.finish_reason:  # 记录结束原因。
                finish_reason = choice.finish_reason  # 保存。
    except Exception as e:  # 捕获流式异常。
        print(f"✗ 调用失败：{type(e).__name__}: {e}")  # 打印异常。
        record("B 流式单工具", "⛔ 调用失败", f"{type(e).__name__}: {e}")  # 记入汇总表。
        return  # 提前退出。

    norm = {  # 把拼装结果整理成与 normalize_message 完全一致的结构。
        "role": "assistant",  # 角色固定。
        "content": "".join(content_parts) or None,  # 正文片段拼接，空串归成 None。
        "tool_calls": [acc[i] for i in sorted(acc)],  # 按 index 排序，保证顺序稳定。
    }  # 组装结束。
    print(f"chunk 数 = {chunk_count}   finish_reason = {finish_reason!r}")  # 旁证：流确实是分片来的。
    dump_message("场景 B 流式拼装结果", norm)  # 输出观察点。
    verdict, note = judge_tool_calls(norm, "B")  # 下判定。
    if verdict == "✅ 通过":  # 流式和非流式结构一致的话，P5/P6 的拼装代码就是可复用的。
        note += "；流式拼装结果与非流式结构一致"  # 补充结论。
    record("B 流式单工具", verdict, note)  # 记录结果。


# ============================================================================
# 场景 C：非流式 · 一次要调两个工具（并行调用）
# ============================================================================

def run_scenario_c(client: OpenAI, cfg: dict) -> None:  # 场景 C：并行调用的配对风险。
    banner("场景 C：非流式 · 一次触发两个工具（并行调用）")  # 打印标题。
    messages = [  # 构造一个同时需要两个工具的问题。
        {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示。
        {"role": "user", "content": "成都今天天气怎么样？另外推荐 2 个成都的人文景点。"},  # 一个提问命中两类工具。
    ]  # messages 结束。
    try:  # 捕获异常。
        resp = client.chat.completions.create(  # 发起请求。
            model=cfg["model"],  # 模型名。
            messages=messages,  # 对话内容。
            tools=[WEATHER_TOOL, POI_TOOL],  # 给两个工具。
            tool_choice="auto",  # 交给模型决定调几个。
            temperature=0,  # 保持确定性。
        )  # 请求结束。
    except Exception as e:  # 异常处理。
        print(f"✗ 调用失败：{type(e).__name__}: {e}")  # 打印异常。
        record("C 并行调用", "⛔ 调用失败", f"{type(e).__name__}: {e}")  # 记入汇总表。
        return  # 提前退出。

    norm = normalize_message(resp.choices[0].message)  # 归一化。
    dump_message("场景 C 原始响应", norm)  # 输出观察点。
    n = len(norm["tool_calls"])  # 拿调用条数。
    verdict, note = judge_tool_calls(norm, "C")  # 先做通用判定。
    if n >= 2:  # 真出现了并行调用。
        ids = [tc["id"] for tc in norm["tool_calls"]]  # 取所有 id。
        if len(set(ids)) == len(ids) and all(ids):  # id 唯一且都非空 → 配对可做。
            record("C 并行调用", "✅ 通过", f"单次响应返回 {n} 条 tool_calls，id 唯一非空，可按 index 配对")  # 正面结论。
            return  # 结束本场景。
        record("C 并行调用", "⚠️ 需防御层", f"返回 {n} 条 tool_calls 但 id 有重复或缺失：{ids}")  # 结论：要对齐 id。
        return  # 结束本场景。
    record("C 并行调用", verdict, f"只返回 {n} 条 tool_calls（未触发并行）—— {note}")  # 结论：并行不是必然，别把配对逻辑写成假设两条。
    print("    注意：这不是协议错误，只是模型这次选择串行。若要强验并行，可加 prompt 明确要求一次问完。")  # 补充说明，避免误判。


# ============================================================================
# 场景 D：把 tool 结果回灌，再问一次 —— 防御层的核心风险在这
# ============================================================================

def run_scenario_d(client: OpenAI, cfg: dict) -> None:  # 场景 D：回灌配对。
    banner("场景 D：回灌 tool 结果再问一次（验证 id 配对是否被 API 接受）")  # 打印标题。
    if not SCENARIO_A_MSG:  # 场景 A 没跑成或没调用成功。
        print("⏭ 跳过：场景 A 没有产出可用的 assistant 消息。")  # 说明跳过原因。
        record("D 回灌配对", "⏭ 跳过", "依赖场景 A 的成功输出")  # 记入汇总表。
        return  # 直接退出。

    assistant_msg = {"role": "assistant", "content": SCENARIO_A_MSG["content"] or ""}  # 先摆角色与正文。
    fabricated = []  # 记录哪些 id 是伪造的。
    tool_calls_payload = []  # 回灌时要带上的 tool_calls。
    for tc in SCENARIO_A_MSG["tool_calls"]:  # 逐条处理。
        call_id = tc["id"]  # 取原始 id。
        if not call_id:  # id 缺失正是文档里说的那个坑。
            call_id = f"dsml_{tc['index']}"  # 按文档规定的方案伪造一个。
            fabricated.append(call_id)  # 记录一下。
        tool_calls_payload.append(  # 组装标准形状。
            {
                "id": call_id,  # 关键字段：必须与 tool 消息里的 tool_call_id 完全一致。
                "type": "function",  # 固定值。
                "function": {"name": tc["name"], "arguments": tc["arguments"]},  # 原样带回。
            }
        )  # 追加结束。
    assistant_msg["tool_calls"] = tool_calls_payload  # 挂到 assistant 消息上。

    if fabricated:  # 有伪造 id 就明确报警。
        print(f"⚠️ 检测到 {len(fabricated)} 条缺失 id，已伪造：{fabricated}   ← 这正是 §P0.3 预案的第一种情况")  # 提示当前处在防御层分支。
    else:  # id 齐全。
        print("✓ 原始 id 齐全，无需伪造，直接原样回灌")  # 说明走的是正常分支。

    messages = [  # 拼完整的上下文。
        {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示。
        {"role": "user", "content": "成都今天天气怎么样？"},  # 与场景 A 相同的问题，保证语义连贯。
        assistant_msg,  # 刚才那条 assistant（含 tool_calls）。
    ]  # 前半段结束。
    for tc in tool_calls_payload:  # 每条 tool_call 都要配一条 tool 消息。
        messages.append(  # 追加配对消息。
            {
                "role": "tool",  # 角色固定为 tool。
                "tool_call_id": tc["id"],  # 必须与上面的 id 严格一致，否则 API 报 400。
                "content": json.dumps({"ok": True, "data": "晴，22℃，微风"}, ensure_ascii=False),  # 模拟工具返回体。
            }
        )  # 追加结束。

    try:  # 这一步是真正要验的：API 会不会接受这组配对。
        resp = client.chat.completions.create(  # 第二次调用。
            model=cfg["model"],  # 模型名。
            messages=messages,  # 含 assistant + tool 的完整上下文。
            tools=[WEATHER_TOOL],  # 工具定义要再传一次。
            tool_choice="none",  # 强制出文字，看它能不能基于工具结果作答。
            temperature=0,  # 保持确定性。
        )  # 请求结束。
    except Exception as e:  # 配对不合法时 API 会直接报 400。
        print(f"✗ 回灌被拒绝：{type(e).__name__}: {e}")  # 打印异常原文，里面通常写明哪个字段不对。
        record("D 回灌配对", "❌ 未通过", f"API 拒绝对话配对：{type(e).__name__} —— 归一化层必须伪造合法 id")  # 记入汇总表。
        return  # 退出。

    final = resp.choices[0].message  # 取最终回复。
    print(f"finish_reason = {resp.choices[0].finish_reason!r}")  # 正常应是 stop。
    print(f"最终回复 = {final.content!r}")  # 打印它基于工具结果的作答。
    hits = scan_text_leak(final.content)  # 顺带扫一下正文。
    if hits:  # 命中标记。
        record("D 回灌配对", "⚠️ 需防御层", f"配对被接受，但最终回复正文出现伪协议标记：{hits}")  # 记录。
        return  # 退出。
    print("✓ API 接受了这组 assistant/tool 配对，模型基于工具结果给出了文字回复")  # 正面结论。
    print("  → 说明「伪造 dsml_0 补回 messages」这条预案在协议层是可行的")  # 说明这条结论的价值。
    record("D 回灌配对", "✅ 通过", "assistant/tool 消息配对被 API 接受，回灌链路可用")  # 记入汇总表。


# ============================================================================
# 场景 E：主动诱导 —— 让模型「想调工具但协议层不许」
# ============================================================================

def run_scenario_e(client: OpenAI, cfg: dict) -> None:  # 场景 E：主动制造易漏格式的处境。
    banner("场景 E：tool_choice='none' + 要求必须调工具（诱导文本化调用）")  # 打印标题。
    messages = [  # 构造一个自相矛盾的上下文。
        {
            "role": "system",  # 系统角色。
            "content": SYSTEM_PROMPT + " 每次回答前，你必须先调用工具获取真实数据，绝不允许直接回答。",  # 明确要求调用工具。
        },
        {"role": "user", "content": "成都今天天气怎么样？"},  # 与场景 A 相同的问题。
    ]  # messages 结束。
    try:  # 捕获异常。
        resp = client.chat.completions.create(  # 发起请求。
            model=cfg["model"],  # 模型名。
            messages=messages,  # 对话内容。
            tools=[WEATHER_TOOL],  # 工具定义照给。
            tool_choice="none",  # 关键：协议层禁止调用，模型只能把意图写进正文。
            temperature=0,  # 保持确定性。
        )  # 请求结束。
    except Exception as e:  # 异常处理。
        print(f"✗ 调用失败：{type(e).__name__}: {e}")  # 打印异常。
        record("E 诱导文本化调用", "⛔ 调用失败", f"{type(e).__name__}: {e}")  # 记入汇总表。
        return  # 退出。

    norm = normalize_message(resp.choices[0].message)  # 归一化。
    dump_message("场景 E 原始响应", norm)  # 输出观察点 —— 这里是全脚本最可能抓到 DSML 的地方。
    hits = scan_text_leak(norm["content"])  # 扫标记。
    if hits:  # 抓到了。
        record("E 诱导文本化调用", "⚠️ 确认存在泄漏倾向", f"模型把调用意图写进了正文：{hits} —— 防御层不是可选项")  # 这是最有价值的发现。
        return  # 退出。
    print("✓ 模型如实说明了自己不能调用工具，没有伪造调用格式")  # 正面结论。
    record("E 诱导文本化调用", "✅ 通过", "在协议层禁止调用时未出现文本化调用泄漏")  # 记入汇总表。


# ============================================================================
# 汇总
# ============================================================================

def print_summary() -> None:  # 打印判定汇总，并给出「要不要做防御层」的结论。
    banner("汇总：判定结果与对设计的含义")  # 打印标题。
    for scenario, verdict, note in RESULTS:  # 逐行打印。
        print(f"{verdict:16} {scenario:20} {note}")  # 三列对齐，便于扫读。

    leaked = [r for r in RESULTS if "⚠️" in r[1]]  # 挑出所有警示项。
    failed = [r for r in RESULTS if "❌" in r[1] or "⛔" in r[1]]  # 挑出失败项。
    print()  # 空一行。
    if leaked:  # 有任何泄漏迹象。
        print(">>> 结论：⚠️ 必须做防御层。")  # 明确结论。
        print("    在 app/core/llm.py 里加一层归一化：正则解析文本形式的调用意图，")  # 给出实现位置。
        print("    并**伪造 id（如 dsml_0）补回 messages**，否则 assistant/tool 消息配对会断，API 直接报错。")  # 强调关键点。
        print("    这层要单独写单测（预案详见方案设计 §12.1）。")  # 提醒配套测试。
    elif failed:  # 没有泄漏但判定失败 —— 是模型或 prompt 的问题，不是协议问题。
        print(">>> 结论：协议层干净，但有场景未通过 —— 属于模型能力 / prompt 问题，先别做防御层。")  # 明确结论。
        print("    对策方向：把工具 description 写得更具强制性，或在 prompt 里写明触发条件。")  # 给可行方向。
    else:  # 全部通过。
        print(">>> 结论：✅ 全部通过，ToolResult 链路直接用标准协议。")  # 明确结论。
        print("    app/core/llm.py 不需要归一化层，只保留 token / 耗时埋点即可，复杂度省下来了。")  # 说明省了什么。
    print("\n跑完请把上面的结论回填进 docs/方案设计.md（§12.1 LLM 调用协议实测）。")  # 提醒回填文档。


def parse_scenarios() -> list[str]:  # 解析命令行参数，支持只跑部分场景。
    picked = [a.upper() for a in sys.argv[1:] if a.upper() in {"A", "B", "C", "D", "E"}]  # 只认 A~E。
    return picked or ["A", "B", "C", "D", "E"]  # 缺省跑全部。


# ============================================================================

if __name__ == "__main__":  # 主入口。
    print(f"Python {sys.version.split()[0]}")  # 打印 Python 版本，结论要跟环境绑定。
    config = load_config()  # 载入配置。
    if config is None:  # 配置不全就没必要继续。
        sys.exit(1)  # 用非零退出码明确表示失败。

    client_ = make_client(config)  # 建客户端。
    wanted = parse_scenarios()  # 决定跑哪些场景。
    print(f"本次运行场景：{wanted}")  # 回显，避免以为跑了全部。

    # D 依赖 A 的输出，所以只要点了 D 就顺带把 A 跑掉。
    if "D" in wanted and "A" not in wanted:  # 判断依赖。
        wanted = ["A"] + wanted  # 把 A 插到最前面。
        print("（场景 D 依赖 A 的输出，已自动补跑 A）")  # 说明为什么多跑了一个。

    runners = {  # 场景名到执行函数的映射。
        "A": run_scenario_a,  # 非流式单工具。
        "B": run_scenario_b,  # 流式单工具。
        "C": run_scenario_c,  # 并行调用。
        "D": run_scenario_d,  # 回灌配对。
        "E": run_scenario_e,  # 诱导文本化调用。
    }  # 映射结束。
    for key in wanted:  # 按顺序执行。
        runners[key](client_, config)  # 调用对应场景。

    print_summary()  # 打印汇总。
