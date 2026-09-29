"""LLM 工厂契约测试（P4.0-C）。

**全部离线** —— 不联网、不需要 API key。真实调用的验证在
`scratch/verify_llm_factory.py`（那才是「DeepSeek 真的接受这个请求」的证据），
本文件管的是另一半：**工厂封住的三件事有没有被封住**。

为什么这半边非测不可：P4.0 踩出来的两个坑都属于**不报错的失效** ——
`with_structured_output` 不写 method 只是 400，`enable_thinking=False` 甚至
连 400 都没有、请求被正常接受而 thinking 照开。这两件事靠「跑一下看看」验不出来，
只能靠对着**发出去的请求内容**断言。
"""

import ast
import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError
from app.core.llm import (
    DISABLE_THINKING,
    ENABLE_THINKING,
    REDACTED,
    SLOT_SPECS,
    STRUCTURED_METHOD,
    THINKING_BODY,
    LLMClient,
    TokenUsageCallback,
    _parse_tool_call,
    _usage_from,
    build_llm,
    redact,
    reset_llm_cache,
    structured_route,
)

# 测试里出现的「密钥」一律是**编的**。
# 硬红线：永不把真实密钥写进断言 —— pytest 的断言自省会把它原文打印到终端，
# 失败路径就是一条泄露路径（P1.6 已踩过一次）。
FAKE_KEY = "sk-fake-0000-1111-2222"


@pytest.fixture(autouse=True)
def _fresh_llm_cache():
    """每个测试前后都清客户端缓存。

    不加这个的话，`test_build_llm_...` 里换掉 settings 的那几条会因为**上一条测试
    已经填过缓存**而拿到旧对象 —— 那时断言失败的原因看着像「工厂没按新配置建」，
    实际是缓存命中。**测试间共享状态是造假的经典来源。**
    """
    reset_llm_cache()
    yield
    reset_llm_cache()


class _StubChat:
    """假客户端：记录被调了哪个方法，并返回一个能跑出结果的 Runnable。

    鸭子类型即可 —— `LLMClient` 只用到 `invoke` / `with_structured_output` /
    `bind_tools` / `model_name` / `extra_body` 五个成员。
    """

    def __init__(self, boom: Exception | None = None, structured_result: BaseModel | None = None):
        self.calls: list[str] = []
        self.model_name = "stub-model"
        self.extra_body = None
        self._boom = boom
        self._structured_result = structured_result or _City(city="成都")

    def with_structured_output(self, schema, **kwargs):
        self.calls.append(f"with_structured_output(method={kwargs.get('method')})")
        return RunnableLambda(lambda _: self._raise_or(self._structured_result))

    def bind_tools(self, tools, **kwargs):
        self.calls.append(f"bind_tools(tool_choice={kwargs.get('tool_choice')})")
        name = getattr(tools[0], "__name__", "tool")
        return RunnableLambda(
            lambda _: self._raise_or(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": self._structured_result.model_dump(),
                            "id": "call_1",
                        }
                    ],
                )
            )
        )

    def invoke(self, messages, **kwargs):
        self.calls.append("invoke")
        return self._raise_or(AIMessage(content="好的"))

    def _raise_or(self, value):
        if self._boom is not None:
            raise self._boom
        return value


class _City(BaseModel):
    city: str


def _client(stub: _StubChat, *, thinking: bool) -> LLMClient:
    return LLMClient(stub, slot="cheap", thinking=thinking, temperature=0.3)


# ===========================================================================
# 槽位表
# ===========================================================================


def test_slot_specs_cover_every_slot_and_point_at_real_settings_fields():
    """三个槽位齐、字段名互不相同、且都真的存在于 `Settings` 上。

    最后一条不是废话：槽位表是**字符串映射**，打错一个字母的后果是
    `getattr(settings, "llm_model_chap")` 在**第一次调用该槽位时**抛 AttributeError
    —— 也就是「需求收集这个节点一跑就炸」，而那时没人会怀疑到工厂的字符串上。
    """
    assert set(SLOT_SPECS) == {"cheap", "main", "judge"}

    fields = [spec.model_field for spec in SLOT_SPECS.values()]
    assert len(set(fields)) == 3, "三个槽位必须指向三个不同的配置项，否则「双模型槽位」是假的"
    for name in fields:
        assert name in Settings.model_fields, f"Settings 上没有配置项 {name}"


def test_thinking_defaults_match_the_p4_0_decision():
    """便宜档关 thinking（定案）；主力档开着（P4.2 对比后再定）。

    主力档这一条**不是永久契约**，是一条有意的当前值 —— P4.2 若测出「关掉推理
    质量没差」，就该把它改成 False，并同步 §12.2。把它写成断言是为了让那次改动
    是**有意识的**：改这里的人会先看到这段 docstring。
    """
    assert SLOT_SPECS["cheap"].thinking is False
    assert SLOT_SPECS["judge"].thinking is False
    assert SLOT_SPECS["main"].thinking is True


def test_judge_temperature_is_pinned_to_zero_not_inherited():
    """judge 的 temperature=0 是**槽位属性**，不是「全局恰好是 0」。

    §8.2 要求 LLM-as-Judge 用 temperature=0 —— 评估要可复现，同一个行程评两次
    分数不一样的话，§8.4 的批量评估就没法拿来比。若让节点自己传，迟早有一个
    调用点忘了；而症状是「分数偶尔抖一下」，没人会往温度上想。

    `temperature=None` 那一侧同样要钉：它表示「跟随全局」，抄成具体数字就丢了
    这层含义（改 `.env` 后缓存里还是旧值，两者长得一模一样）。
    """
    assert SLOT_SPECS["judge"].temperature == 0.0
    assert SLOT_SPECS["cheap"].temperature is None
    assert SLOT_SPECS["main"].temperature is None


# ===========================================================================
# 关 thinking 的参数形状（§12.2 实测表）
# ===========================================================================


def test_thinking_body_uses_the_only_shape_that_works():
    """`{"thinking": {"type": "..."}}` —— **唯一有效的形状**（§12.2 实测）。

    另外两种常见写法（`enable_thinking` / `chat_template_kwargs`）请求会被接受、
    thinking 照开。所以这条不是在测「字典对不对」，是在拦一类**没有症状的失效**。
    """
    assert DISABLE_THINKING == {"thinking": {"type": "disabled"}}
    assert ENABLE_THINKING == {"thinking": {"type": "enabled"}}
    assert THINKING_BODY[False] == DISABLE_THINKING
    assert THINKING_BODY[True] == ENABLE_THINKING

    # 反面写法一个都不许出现
    for body in THINKING_BODY.values():
        assert "enable_thinking" not in body
        assert "chat_template_kwargs" not in body
        assert set(body) == {"thinking"}
        assert set(body["thinking"]) == {"type"}


def test_build_llm_sends_the_thinking_switch_it_claims_to():
    """端到端把开关钉到**最终发出的 `extra_body`** 上。

    只测 `THINKING_BODY` 那张表是不够的 —— 表对了但 `ChatOpenAI(...)` 那行写错
    （比如漏传、或传成 `enable_thinking`）时，上一条测试照样绿。
    """
    assert build_llm("cheap").extra_body == DISABLE_THINKING
    assert build_llm("judge").extra_body == DISABLE_THINKING
    assert build_llm("main").extra_body == ENABLE_THINKING


# ===========================================================================
# 结构化输出的两条路
# ===========================================================================


def test_structured_route_matches_the_measured_matrix():
    """(thinking, 结构化输出) 的合法组合表 —— 写反了就是那 100% 的 400。

    §12.2 实测：

      thinking 关 → `function_calling` ✅ ／ thinking 开 → ❌ 400，只能 `bind_tools`
    """
    assert structured_route(thinking=False) == "function_calling"
    assert structured_route(thinking=True) == "bind_tools"


def test_structured_method_constant_is_not_the_broken_default():
    """`method` 常量必须是 `function_calling`。

    `with_structured_output` 的默认是 `json_schema`，而 DeepSeek 不支持它 ——
    所以「不传 method」与「传 json_schema」是同一个失败。这一点最容易漏：
    代码看起来没写错，只是没写全。
    """
    assert STRUCTURED_METHOD == "function_calling"


def test_structured_calls_with_structured_output_when_thinking_is_off():
    stub = _StubChat()
    client = _client(stub, thinking=False)

    result = client.structured(_City).invoke("去成都")

    assert stub.calls == [f"with_structured_output(method={STRUCTURED_METHOD})"]
    assert result == _City(city="成都")


def test_structured_falls_back_to_bind_tools_when_thinking_is_on():
    """thinking 开着时必须绕开 `with_structured_output` —— 走了就是 400。"""
    stub = _StubChat()
    client = _client(stub, thinking=True)

    result = client.structured(_City).invoke("去成都")

    assert stub.calls == ["bind_tools(tool_choice=auto)"]
    assert result == _City(city="成都")


def test_structured_returns_the_parsed_model_on_both_routes():
    """**两条路给调用方的是同一个东西** —— 这是 `structured()` 存在的全部意义。

    若两条路返回值不同（一条给模型、一条给 `AIMessage`），节点就得自己判走了哪条，
    等于把这个判断泄漏到了每个调用点 —— 那正是工厂要消灭的东西。
    """
    for thinking in (False, True):
        client = _client(_StubChat(), thinking=thinking)
        result = client.structured(_City).invoke("任意输入")
        assert isinstance(result, _City), f"thinking={thinking} 这条路返回的不是模型实例"
        assert result.city == "成都"


def test_parse_tool_call_rejects_a_model_that_answered_without_calling_the_tool():
    """模型没调工具 → `LLMError`，而不是 `IndexError`。

    thinking 开着走 `bind_tools` 时，模型偶尔直接回文本而不调工具。裸的
    `calls[0]` 会抛 `IndexError`，被 API 层当成服务器 bug（500）；
    包装成 `LLMError` 才有可能按 §11 的错误码给出「换条路 / 重试」的语义。
    """
    message = AIMessage(content="我不调用工具")

    with pytest.raises(LLMError) as excinfo:
        _parse_tool_call(_City).invoke(message)

    assert "没有调用工具" in str(excinfo.value)
    assert excinfo.value.code == "llm_error"


def test_parse_tool_call_rejects_args_that_do_not_match_the_schema():
    """工具参数结构不对 → `LLMError`，且 **detail 只说字段名与错误类型、不说值**。

    参数里是用户输入。`detail` 是要进日志的，把原文捎进去就是一次泄露
    （硬红线 #6）。

    注意 Pydantic 报的是**缺失的 `city`**，而不是那个多余的 `wrong_field`
    —— 默认模式下多余字段被静默忽略。所以这里断言的是 `city`：
    写测试时的直觉（「它会说 wrong_field 不认识」）是错的，实测才知道。
    """
    message = AIMessage(
        content="",
        tool_calls=[{"name": "_City", "args": {"wrong_field": "13800138000"}, "id": "c1"}],
    )

    with pytest.raises(LLMError) as excinfo:
        _parse_tool_call(_City).invoke(message)

    assert "不合规字段：city(" in (excinfo.value.detail or "")
    assert "13800138000" not in (excinfo.value.detail or "")


# ===========================================================================
# 脱敏
# ===========================================================================


def test_redact_removes_the_secret_from_upstream_error_text():
    """上游报错里可能带着整条 URL / header —— 直接写进日志等于泄露密钥。"""
    text = f"401 Client Error for url: https://api.deepseek.com?token={FAKE_KEY}"

    cleaned = redact(text, FAKE_KEY)

    assert FAKE_KEY not in cleaned
    assert REDACTED in cleaned


def test_redact_skips_short_secrets_instead_of_shredding_the_text():
    """短于 8 位的 secret 一律跳过。

    不跳的话，`LLM_API_KEY=""`（未配置时的默认值）会让 `replace("", "***")`
    在**每个字符之间**插一个 `***`，把报错信息搅成一团 —— 比不脱敏还糟。
    """
    text = "报错里带着 12345678 这个串"

    assert redact(text, "") == text
    assert redact(text, "abc") == text
    # 边界：正好 8 位就动手了，所以「跳过」的判据是 < 8 而不是 <= 8
    assert redact(text, "1234567") == text
    assert redact(text, "12345678") == "报错里带着 *** 这个串"


def test_chat_wraps_upstream_errors_and_redacts_the_secret():
    """`chat()` 把上游异常包成 `LLMError`，且 detail 已脱敏。

    不包的话 API 层拿不到 `code`，只能一律 500；不脱敏的话密钥进日志。
    """
    stub = _StubChat(boom=RuntimeError(f"401 for https://api.deepseek.com?key={FAKE_KEY}"))
    client = LLMClient(stub, slot="cheap", thinking=False, temperature=0.3, secret=FAKE_KEY)

    with pytest.raises(LLMError) as excinfo:
        client.chat([])

    assert FAKE_KEY not in (excinfo.value.detail or "")
    assert REDACTED in (excinfo.value.detail or "")
    assert excinfo.value.code == "llm_error"


def test_structured_surfaces_the_same_wrapped_error():
    """结构化那条路也要包 —— 它跑的是别人的 Runnable，不套一层就漏出去了。"""
    stub = _StubChat(boom=RuntimeError(f"boom {FAKE_KEY}"))
    client = LLMClient(stub, slot="main", thinking=True, temperature=0.3, secret=FAKE_KEY)

    with pytest.raises(LLMError):
        client.structured(_City).invoke("任意输入")


def test_llm_error_is_not_double_wrapped():
    """已经是 `AppError` 的异常原样抛出，别包成「LLM 调用失败」。

    包两层会让「模型没调工具」这条**已知的、可处置的**失败退化成一句笼统的
    调用失败，节点就再也分不出该重试还是该降级。
    """
    original = LLMError("模型没调工具")
    client = LLMClient(_StubChat(boom=original), slot="main", thinking=True, temperature=0.3)

    with pytest.raises(LLMError) as excinfo:
        client.structured(_City).invoke("任意输入")

    assert excinfo.value is original


# ===========================================================================
# token / 耗时埋点
# ===========================================================================


def test_usage_from_reads_langchain_usage_metadata():
    """langchain-core 归一化后的真实形状 —— **推理 token 叫 `reasoning`，不是
    `reasoning_tokens`**（P4.0-C 实测，`scratch/verify_llm_factory.py` 打出来的原文）。

    §12.2 那张表里的 `reasoning_tokens` 是原始 openai SDK 的字段名（那条探针直连
    SDK）。**同一个量在两套命名下不同名** —— 只认一个的话，另一个永远是「缺失」，
    而「缺失」与「thinking 关着」长得一模一样，又一个不报错的失效。
    """
    message = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 34,
            "total_tokens": 46,
            "input_token_details": {"cache_read": 3},
            "output_token_details": {"reasoning": 7},
        },
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={})

    assert _usage_from(result) == {
        "input_tokens": 12,
        "output_tokens": 34,
        "total_tokens": 46,
        "reasoning_tokens": 7,
        "cached_tokens": 3,
    }


def test_usage_from_accepts_both_reasoning_key_names():
    """两套命名都认。只认一套 = 那个量在另一条路径上永远缺失。"""
    for key in ("reasoning", "reasoning_tokens"):
        message = AIMessage(
            content="x",
            usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
                            "output_token_details": {key: 9}},
        )
        result = LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={})
        assert _usage_from(result)["reasoning_tokens"] == 9, key


def test_usage_from_reports_no_reasoning_key_when_thinking_is_off():
    """thinking 关着时**这个键整个不出现**，不是 0。

    所以判「关没关」必须写 `.get("reasoning_tokens", 0) == 0` ——
    直接取键会把「关着」误判成失败（P4.0-C 的探针第一次就是这么红的）。
    """
    message = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={})

    assert "reasoning_tokens" not in _usage_from(result)
    assert _usage_from(result).get("reasoning_tokens", 0) == 0


def test_usage_from_falls_back_to_openai_style_token_usage():
    """`usage_metadata` 缺失时退回 `llm_output["token_usage"]`（OpenAI 原始命名）。

    两套命名都要认：`reasoning_tokens` 正是 §12.2 判定 thinking 关没关掉的证据，
    解析不到它，那条判据就没了。
    """
    result = LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 6,
                "total_tokens": 11,
                "completion_tokens_details": {"reasoning_tokens": 3},
            }
        },
    )

    assert _usage_from(result) == {
        "input_tokens": 5,
        "output_tokens": 6,
        "total_tokens": 11,
        "reasoning_tokens": 3,
    }


def test_usage_from_returns_empty_instead_of_raising_on_garbage():
    """挖不到就返回空 dict。**一个纯观测钩子不该有能力打断主流程。**"""
    assert _usage_from(LLMResult(generations=[], llm_output=None)) == {}
    assert _usage_from(LLMResult(generations=[[]], llm_output={})) == {}
    assert _usage_from("这不是 LLMResult") == {}


def test_usage_callback_logs_numbers_and_never_the_prompt_text(caplog):
    """回调**只记数字**。

    `on_llm_start` 收到的 `prompts` 就是用户原始输入（可能含真实姓名、手机号）。
    把它写进日志就是一次泄露（硬红线 #6），而日志会被收集、转发、贴进 issue。
    """
    caplog.set_level(logging.INFO, logger="app.core.llm")
    handler = TokenUsageCallback("cheap")
    message = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 8, "output_tokens": 9, "total_tokens": 17},
    )

    handler.on_llm_start({}, ["我的手机号是 13800138000，帮我订成都的行程"], run_id="r1")
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={}),
        run_id="r1",
        metadata={"thread_id": "t1"},
    )

    assert "13800138000" not in caplog.text
    assert "槽位=cheap" in caplog.text
    assert "输入=8" in caplog.text
    assert "输出=9" in caplog.text


def test_usage_callback_reports_missing_timing_as_negative_one(caplog):
    """配不上耗时记 -1 而不是 0 —— 0 看起来像「很快」，是个假的好消息。"""
    caplog.set_level(logging.INFO, logger="app.core.llm")
    handler = TokenUsageCallback("main")

    handler.on_llm_end(LLMResult(generations=[], llm_output={}), run_id="never-started")

    assert "耗时=-1ms" in caplog.text


# ===========================================================================
# 工厂：缓存、密钥占位、配置来源
# ===========================================================================


def test_build_llm_picks_the_model_from_its_own_slot():
    settings = get_settings()

    assert build_llm("cheap").model_name == settings.llm_model_cheap
    assert build_llm("main").model_name == settings.llm_model
    assert build_llm("judge").model_name == settings.llm_judge_model


def test_build_llm_temperature_follows_the_slot_then_the_global():
    settings = get_settings()

    assert build_llm("judge").temperature == 0.0
    assert build_llm("cheap").temperature == settings.llm_temperature
    assert build_llm("main").temperature == settings.llm_temperature


def test_build_llm_is_cached_per_settings_not_per_slot(monkeypatch):
    """缓存键 = 全部相关设置。

    两次拿到同一对象（连接池复用），**改配置后拿到新对象**（无需手动失效）。
    「按 slot 缓存 + 手动失效」那种写法漏清一次的症状是「改了配置不生效」，
    且只在测试里看得见。
    """
    first = build_llm("cheap")
    assert build_llm("cheap") is first

    patched = Settings(llm_api_key=FAKE_KEY, llm_model_cheap="换成别的模型")
    monkeypatch.setattr("app.core.llm.get_settings", lambda: patched)

    rebuilt = build_llm("cheap")
    assert rebuilt is not first
    assert rebuilt.model_name == "换成别的模型"


def test_build_llm_constructs_without_an_api_key(monkeypatch):
    """没配密钥时**照样构造得出来**。

    `ChatOpenAI(api_key="")` 会直接抛 `OpenAIError: Missing credentials` ——
    那样「导入模块」「跑离线单测」都得先有一个真密钥。真正的门是
    `Settings.validate_startup()`（P5 的 lifespan 里调）：**进程起不来**，
    好过在所有离线检查里先炸一次。
    """
    monkeypatch.setattr(
        "app.core.llm.get_settings", lambda: Settings(llm_api_key="", llm_model="m")
    )

    client = build_llm("main")

    assert client.model_name == "m"


def test_build_llm_never_logs_the_api_key(caplog):
    """构造日志里不许出现密钥（硬红线 #6）。"""
    caplog.set_level(logging.INFO, logger="app.core.llm")
    reset_llm_cache()

    build_llm("main")

    assert get_settings().llm_api_key not in caplog.text
    assert "构造 LLM 客户端" in caplog.text


# ===========================================================================
# 结构性钉子
# ===========================================================================

NODES_DIR = Path(__file__).resolve().parents[1] / "app" / "graph" / "nodes"

BANNED_CALLS = frozenset({"ChatOpenAI", "with_structured_output", "bind_tools"})
BANNED_KEYWORDS = frozenset({"extra_body"})

# 五个节点 + 装饰器。**按名字列出来而不是数个数**：只数个数的话，
# 扫描目录被指错（比如指到 `tests/`）时文件数照样够，扫描测试会静默失去守护。
# （这是 ⑱ 号变异实测出来的 —— 第一版写的是 `len(paths) >= 5`，被它溜过去了。）
EXPECTED_NODE_MODULES = frozenset(
    {
        "decorators",
        "evaluate",
        "plan_generate",
        "requirement_collect",
        "self_review",
        "user_confirm",
    }
)


def _llm_offences(source: str, where: str) -> list[str]:
    """在**源码**里找一手构造 / 一处配置 LLM 客户端的调用。

    走 `ast` 而不是字符串匹配，是因为字符串匹配分不清「用了」和「警告别人别用」
    —— 节点 docstring 里正写着这条纪律，匹配法会被自己的告诫绊倒，
    而人只会去改措辞（文档价值被侵蚀），不会去改扫描器。
    走 AST 之后，docstring 是 `Constant` 节点，天然看不见。

    覆盖三件事，对应 P4.0 踩出来的三条：
      - `ChatOpenAI(...)` —— 绕过工厂 = 可能忘了关 thinking
      - `.with_structured_output(...)` —— 不传 `method` 就是那 100% 的 400
      - `.bind_tools(...)` —— 工厂已经按 thinking 自动选路了，节点再手写一遍是重复实现
      - `extra_body=` —— thinking 开关的唯一正确形状也封在工厂里
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in BANNED_CALLS:
            found.append(f"{where}:{node.lineno} 调用了 {name}()")
        for keyword in node.keywords:
            if keyword.arg in BANNED_KEYWORDS:
                found.append(f"{where}:{node.lineno} 传了 {keyword.arg}=")
    return found


def test_the_scanner_actually_catches_offences():
    """**扫描器本身先被扫一遍。**

    P1~P3 已实测过十六次「全绿 ≠ 覆盖到位」—— 一个 `ast.walk` 写错、判据写错的
    扫描测试，输出上和「一处违规都没有」一模一样。这两条正例就是它的变异测试。
    """
    assert _llm_offences("x = ChatOpenAI(model='m')", "s") == ["s:1 调用了 ChatOpenAI()"]
    assert _llm_offences("x.y.with_structured_output(S)", "s") == [
        "s:1 调用了 with_structured_output()"
    ]
    assert _llm_offences("c = ChatOpenAI(m='m', extra_body={})", "s") == [
        "s:1 调用了 ChatOpenAI()",
        "s:1 传了 extra_body=",
    ]
    # 反例：提及这些名字的**文档**不算违规，这才是走 AST 的理由
    assert _llm_offences('"""不要写 with_structured_output 或 ChatOpenAI"""', "s") == []
    assert _llm_offences("x = 别的工厂()", "s") == []


def test_nodes_never_build_their_own_chat_openai():
    """节点层不许一手构造 / 配置 LLM 客户端 —— 一律走 `build_llm()`。

    **这条测试值一整个 P4.0。** 两个坑（默认 method 是 `json_schema`、关 thinking
    的写法静默失效）都是「在某处多写一行 `ChatOpenAI(...)` 就会重新踩上」，
    而踩上时**没有任何症状** —— 只在某个节点的调用上莫名 400，那时没人会怀疑到
    「工厂外多建了一个客户端」上。

    与 `test_intercity_backend_is_deliberately_absent`（tests/test_tools.py）同类：
    **把架构约束写成可执行的断言**，而不是靠 code review 的记忆。
    """
    assert NODES_DIR.is_dir(), f"节点目录不存在：{NODES_DIR}"
    paths = sorted(NODES_DIR.glob("*.py"))
    missing = EXPECTED_NODE_MODULES - {path.stem for path in paths}
    assert not missing, f"扫描目录里缺这些节点模块：{sorted(missing)}（目录指错了？）"

    offenders = [
        offence
        for path in paths
        for offence in _llm_offences(path.read_text(encoding="utf-8"), path.name)
    ]

    assert offenders == [], (
        "节点层出现了一手构造 LLM 客户端的代码 —— 一律改走 app.core.llm.build_llm()。"
        f"违规处：{offenders}"
    )


def test_validation_error_is_wired_into_the_tool_call_parser():
    """`_parse_tool_call` 是靠 `ValidationError` 识别「参数结构不对」的。

    若哪天 Pydantic 的异常类型变了（或有人把 `pydantic.ValidationError`
    换成 `ValueError`），那个分支会静默失效 —— 参数不合规会以 Pydantic 原生
    异常的形态冒到节点层，不再被包成 `LLMError`。这条把两者的绑定钉住。
    """
    with pytest.raises(ValidationError):
        _City(wrong_field="x")
