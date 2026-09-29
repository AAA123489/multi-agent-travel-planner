"""LLM 工厂 —— 对应 §12（双模型槽位）与 §12.2（结构化输出实测）。

## 这个文件存在的唯一理由：把「踩了不会报错」的坑封在一处

P4.0 实测出两条，**都属于不报错的失效** —— 写错了不抛异常，只是在某个调用上
莫名 400，或者悄悄烧 token：

1. `with_structured_output(Schema)` 的默认 method 是 `json_schema`，
   DeepSeek 上 **100% 400**（`This response_format type is unavailable now`）。
   必须显式 `method="function_calling"`。
2. 关 thinking 的参数形状**只有一个是对的**：`extra_body={"thinking": {"type": "disabled"}}`。
   写成 `{"enable_thinking": False}` 或 `{"chat_template_kwargs": {...}}`，请求会被
   **接受**、thinking 照开 —— 症状是 `function_calling` 路径报「thinking mode 不支持
   强制 tool_choice」，而那时没人会怀疑到关 thinking 的那一行。

**所以节点代码不许自己建 `ChatOpenAI`。** 两条纪律都收在这里，漏一处就在每个节点
各踩一遍，且踩了没有症状。`tests/test_llm.py::test_nodes_never_build_own_chat_openai`
是一条结构性测试，专门钉这件事。

## 三个槽位（§12）

| 槽位 | 谁用 | 模型 | thinking | temperature |
|---|---|---|---|---|
| `cheap` | `requirement_collect` | `LLM_MODEL_CHEAP` | 关（P4.0 定案） | 全局 |
| `main` | `plan_generate` / `self_review` | `LLM_MODEL` | **开**（P4.2 对比后定） | 全局 |
| `judge` | `evaluate` 的 LLM-as-Judge（§8.2） | `LLM_JUDGE_MODEL` | 关 | **0** |

`judge` 的 `temperature=0`、`cheap` 的关 thinking，都是**槽位属性、不暴露成参数**。
理由同上：让每个节点自己传，迟早有一个忘了传，而症状分别是「评估分数每次都不一样」
和「白烧推理 token」—— 两条都不会有人往调用参数上想。

## `structured()` 为什么内部分两条路（§12.2 的矩阵）

| thinking | `with_structured_output(method="function_calling")` | `bind_tools(tool_choice="auto")` + 手动解析 |
|---|---|---|
| **关** | ✅ | ✅ |
| **开** | ❌ 400 | ✅ |

主力档要保留推理能力就正撞上这个矩阵。`structured()` 按本槽位的 thinking 自动选路，
**调用方拿到的是同一个东西** —— 一个 invoke 出来就是 Pydantic 模型的 Runnable。
"""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, TypeVar, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.exceptions import AppError, LLMError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

LLMSlot = Literal["cheap", "main", "judge"]

# ---------------------------------------------------------------------------
# 两条实测定下的常量 —— 改动前先跑 scratch/verify_structured_output.py
# ---------------------------------------------------------------------------

DISABLE_THINKING: dict[str, Any] = {"thinking": {"type": "disabled"}}
ENABLE_THINKING: dict[str, Any] = {"thinking": {"type": "enabled"}}

# thinking 的开关形状**只有一个是对的**（§12.2 实测表）。
#
# `{"thinking": False}` → 422（expected string）；另外两种常见写法（`enable_thinking`、
# `chat_template_kwargs`）**被接受但不生效** —— 那是最难查的一类失效。
#
# 打开那一侧写 `enabled` 而不是省略 `extra_body`：实测两者行为一致（都是 thinking 开），
# 但**「一致」与「被忽略」在这里无法区分**，因为服务端默认就是开。显式写出来的价值是
# 表达意图 —— 万一服务端把默认翻成关，省略写法会静默丢掉推理能力，而显式写法不会。
# `enabled` 已验证是服务端认识的取值（它没被 422 挡下，一路走到 tool_choice 校验）。
THINKING_BODY: dict[bool, dict[str, Any]] = {
    False: DISABLE_THINKING,
    True: ENABLE_THINKING,
}

STRUCTURED_METHOD = "function_calling"
"""结构化输出走的 method。**必须是它** —— 默认的 `json_schema` 在 DeepSeek 上 400。"""

StructuralRoute = Literal["function_calling", "bind_tools"]

REDACTED = "***"


@dataclass(frozen=True)
class SlotSpec:
    """一个模型槽位的全部固定属性。

    `temperature=None` 表示「跟随全局 `LLM_TEMPERATURE`」。用 None 而不是直接把
    全局值抄进来，是为了让「跟随全局」这件事本身可断言 —— 抄进来的话，
    改了 `.env` 但缓存里还是旧值，两者长得一模一样。
    """

    model_field: str
    thinking: bool
    temperature: float | None = None


SLOT_SPECS: dict[LLMSlot, SlotSpec] = {
    "cheap": SlotSpec(model_field="llm_model_cheap", thinking=False),
    "main": SlotSpec(model_field="llm_model", thinking=True),
    "judge": SlotSpec(model_field="llm_judge_model", thinking=False, temperature=0.0),
}


# ---------------------------------------------------------------------------
# 脱敏与用量
# ---------------------------------------------------------------------------


def redact(text: str, *secrets: str) -> str:
    """把密钥从文本里抹掉（硬红线 #6）。

    上游报错里**可能带着整条请求 URL 或 header** —— 直接把 `str(exc)` 塞进日志，
    等于把密钥写进日志文件，而日志文件是要被收集、转发、贴进 issue 的。

    `< 8` 位的 secret 一律跳过：那多半是空串或占位符，替换它会把整段文本搅碎
    （空串的 `replace` 会在每个字符间插一个 `***`）。真密钥远比 8 位长。
    """
    for secret in secrets:
        if len(secret) >= 8:
            text = text.replace(secret, REDACTED)
    return text


def _usage_from(response: LLMResult) -> dict[str, int]:
    """从 `LLMResult` 里挖 token 用量。挖不到就返回空 dict，**绝不抛异常**。

    一个纯观测用的钩子不该有能力打断主流程 —— 回调里抛异常会被 LangChain
    降级成 warning，但会让同一批里后续回调不再执行。

    要 `reasoning_tokens` 是因为它就是 §12.2 判定「thinking 到底关没关掉」的
    唯一证据。**光看「调用成功」验不出来** —— thinking 开着也能正常答普通问题
    （这正是那两种错误写法能骗过人的原因）。

    ⚠️ **thinking 关着时这个键整个不出现，不是 0。** 判「关没关」要写成
    `usage.get("reasoning_tokens", 0) == 0`；写成 `== 0` 直接取键，会把「关着」
    误判成失败（P4.0-C 的探针第一次就是这么红的）。

    两条来源分开取、**不是一条 try 包下来的**：`generations` 为空（流式未回、
    或失败）时，第一条来源会抛 `IndexError`，若与第二条同一个 try，回退路径
    就永远走不到 —— 而那恰好是最需要它的时候。
    """
    raw: dict[str, Any] = {}
    try:
        raw = dict(getattr(response.generations[0][0].message, "usage_metadata", None) or {})
    except Exception:
        raw = {}
    if not raw:
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        raw = dict(token_usage) if isinstance(token_usage, dict) else {}

    usage: dict[str, int] = {}
    # 同一个量在两套命名下各出现一次（langchain-core 的 / OpenAI 原始的），
    # 先写的那个赢 —— 所以顺序是「标准命名在前」。
    for source, target in (
        ("input_tokens", "input_tokens"),
        ("prompt_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        if target not in usage and isinstance(raw.get(source), int):
            usage[target] = raw[source]

    # ⚠️ 两套命名在这里**同名不同义**，是 P4.0-C 实测踩出来的：
    #   - 原始 openai SDK（§12.1 / §12.2 的探针走的那条）→ `reasoning_tokens`
    #   - 经 langchain-core 归一化 → **`reasoning`**（短名）
    # **只认其中一个的话**，另一个形状下的推理 token 永远「缺失」，而「缺失」看起来
    # 和「thinking 关着」一模一样 —— 又一个不报错的失效（P4.0-C 第一版就是只认旧名）。
    details = raw.get("output_token_details") or raw.get("completion_tokens_details")
    if isinstance(details, dict):
        for key in ("reasoning", "reasoning_tokens"):
            if isinstance(details.get(key), int):
                usage["reasoning_tokens"] = details[key]
                break

    # 缓存命中的 token 单独记：它便宜一个数量级，混在 input_tokens 里看不出省钱效果
    cache = raw.get("input_token_details") or raw.get("prompt_tokens_details")
    if isinstance(cache, dict):
        for key in ("cache_read", "cached_tokens"):
            if isinstance(cache.get(key), int):
                usage["cached_tokens"] = cache[key]
                break

    return usage


class TokenUsageCallback(BaseCallbackHandler):
    """每次 LLM 调用的 token 与耗时埋点。

    **只记数字，不记正文。** prompt 里有用户的原始输入（可能含真实姓名、手机号），
    `on_llm_start` 收到的 `prompts` 正是那些原文 —— 把它写进日志就是一次泄露
    （硬红线 #6）。正文要看，去看 `node_trace` 与本轮状态，那里是**有意**给用户看的。
    """

    def __init__(self, slot: LLMSlot) -> None:
        super().__init__()
        self.slot = slot
        self._started: dict[str, float] = {}

    def on_llm_start(self, serialized: Any, prompts: list[str], **kwargs: Any) -> None:
        self._started[str(kwargs.get("run_id"))] = time.monotonic()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        elapsed_ms = self._pop_elapsed(kwargs.get("run_id"))
        usage = _usage_from(response)
        logger.info(
            "LLM 调用完成：槽位=%s 耗时=%dms 输入=%s 输出=%s 推理=%s",
            self.slot,
            elapsed_ms,
            usage.get("input_tokens", "-"),
            usage.get("output_tokens", "-"),
            usage.get("reasoning_tokens", "-"),
            extra={
                "thread_id": (kwargs.get("metadata") or {}).get("thread_id", "-"),
                "llm": {"slot": self.slot, "elapsed_ms": elapsed_ms, **usage},
            },
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._pop_elapsed(kwargs.get("run_id"))
        # 只记类型不记原文：异常串里可能带上游 URL / header（同 redact 的理由），
        # 而这里拿不到 settings，抹不干净就不写。
        logger.warning(
            "LLM 调用失败：槽位=%s 错误类型=%s",
            self.slot,
            type(error).__name__,
            extra={"thread_id": (kwargs.get("metadata") or {}).get("thread_id", "-")},
        )

    def _pop_elapsed(self, run_id: Any) -> int:
        """取本次调用的耗时；配对不上返回 -1（而不是 0 —— 0 看起来像「很快」）。"""
        started = self._started.pop(str(run_id), None)
        return int((time.monotonic() - started) * 1000) if started is not None else -1


# ---------------------------------------------------------------------------
# 结构化输出的两条路
# ---------------------------------------------------------------------------


def structured_route(thinking: bool) -> StructuralRoute:
    """(thinking, 结构化输出) 的合法组合查表 —— §12.2 实测矩阵的代码化。

    **抽成纯函数是为了让它可被变异测试。** 直接写在 `if` 里也能跑，但那样写反了
    只有真实 400 才发现得了；这里 `tests/test_llm.py` 对着它断言，
    改一行就跑红。
    """
    return "bind_tools" if thinking else "function_calling"


def _parse_tool_call(schema: type[ModelT]) -> Runnable[Any, ModelT]:
    """把 `bind_tools` 返回的 `AIMessage` 解析成模型实例。

    **这里没有归一化层** —— §12.1 已实测 `tool_calls` 格式标准（`id` / `name` /
    `arguments` 齐全，参数是合法 JSON），所以只剩「取第一条」这一件事。

    三种失败各自抛 `LLMError`，**且都不把参数原文带进消息**：参数里是用户输入。
    `detail` 只放结构信息（哪个字段不合规），不放值。
    """

    def _extract(message: Any) -> ModelT:
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            raise LLMError(
                f"模型没有调用工具，拿不到 {schema.__name__}。"
                "thinking 开着时走的是 bind_tools 路径，模型偶尔会直接回文本而不调工具。"
            )

        args = calls[0].get("args")
        if isinstance(args, str):
            # 有的实现把参数给成 JSON 字符串而不是 dict
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise LLMError(f"工具参数不是合法 JSON，无法构造 {schema.__name__}") from exc

        try:
            return schema.model_validate(args)
        except ValidationError as exc:
            # 只报「哪个字段、哪种不合规」，**不报值** —— 值是用户输入（见 redact）。
            detail = "、".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<根>'}({err['type']})"
                for err in exc.errors()
            )
            raise LLMError(
                f"工具参数不符合 {schema.__name__} 的结构", detail=f"不合规字段：{detail}"
            ) from exc

    return RunnableLambda(_extract)


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class LLMClient:
    """一个模型槽位的客户端。

    **不暴露底层的 `ChatOpenAI`**（`_client` 私有），只给三个出口：

      - `chat(messages)` → `AIMessage`：纯文本生成
      - `structured(schema)` → invoke 出来就是 `schema` 实例的 Runnable
      - `slot` / `thinking` / `temperature`：只读，供日志与测试断言

    这不是封装洁癖。裸 `ChatOpenAI` 一旦露出去，`client._client.with_structured_output(S)`
    这种写法就会出现 —— 而它正是那个 **100% 400 的写法**，且看起来完全正常，
    code review 时不会有人停下来。
    """

    def __init__(
        self,
        client: BaseChatModel,
        *,
        slot: LLMSlot,
        thinking: bool,
        temperature: float,
        secret: str = "",
    ) -> None:
        self._client = client
        self._secret = secret
        self.slot = slot
        self.thinking = thinking
        self.temperature = temperature

    # ---------- 只读视图 ----------

    @property
    def model_name(self) -> str:
        """底层模型名。给日志与测试看 —— **不是「可以拿它建新客户端」的许可**。"""
        return getattr(self._client, "model_name", "") or ""

    @property
    def extra_body(self) -> dict[str, Any]:
        """实际发出去的 `extra_body` —— **thinking 开关就藏在这里**。

        之所以要暴露它，是因为那是本项目唯一一处「配错了完全没症状」的配置：
        `{"enable_thinking": False}` 请求会被接受、thinking 照开。有一个能直接
        断言「发出去的到底是什么」的入口，才能把它钉成测试。
        """
        return dict(getattr(self._client, "extra_body", None) or {})

    # ---------- 出口一：纯文本 ----------

    def chat(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """纯文本生成。返回**原始 `AIMessage`** —— 调用方常要读它的 `usage_metadata`。"""
        return cast(AIMessage, self._call(lambda: self._client.invoke(list(messages))))

    # ---------- 出口二：结构化 ----------

    def structured(self, schema: type[ModelT]) -> Runnable[Any, ModelT]:
        """结构化输出。`invoke(...)` 出来直接是 `schema` 的实例。

        两条路**自动选**（`structured_route`），调用方不需要知道走了哪条 ——
        这正是这个方法存在的意义。
        """
        if structured_route(self.thinking) == "function_calling":
            chain: Runnable = self._client.with_structured_output(
                schema, method=STRUCTURED_METHOD
            )
        else:
            chain = self._client.bind_tools([schema], tool_choice="auto") | _parse_tool_call(
                schema
            )
        return cast("Runnable[Any, ModelT]", self._guard(chain, schema.__name__))

    # ---------- 内部 ----------

    def _call(self, fn: Any) -> Any:
        try:
            return fn()
        except AppError:
            raise  # 已经是我们的异常（如「模型没调工具」），别包第二层
        except Exception as exc:
            raise LLMError(
                f"LLM 调用失败（槽位 {self.slot}）",
                detail=redact(f"{type(exc).__name__}: {exc}", self._secret),
            ) from exc

    def _guard(self, chain: Runnable, what: str) -> Runnable:
        """给结构化链套一层错误包装。

        没有它的话，`with_structured_output` 内部的失败会以**上游原始异常**的形态
        冒到节点层 —— 那串原文里可能带着密钥（同 `redact` 的理由），而且 API 层
        拿不到 `LLMError.code`，只能一律按 500 处理。
        """
        return RunnableLambda(lambda value: self._call(lambda: chain.invoke(value)))


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

_NOT_CONFIGURED = "sk-not-configured"
"""`LLM_API_KEY` 为空时的占位密钥。

`ChatOpenAI(api_key="")` 会直接抛 `OpenAIError: Missing credentials` —— 那样
「导入模块」「跑单元测试」都会要求一个真实密钥。占位值让**构造**不需要密钥，
真正的门是 `Settings.validate_startup()`（P5 的 lifespan 里调），
而那才是应该失败的地方：启不来进程，而不是在某个节点里 401。
"""


def build_llm(slot: LLMSlot = "main") -> LLMClient:
    """按槽位取客户端。**节点代码只用这个，不自己建 `ChatOpenAI`。**"""
    settings = get_settings()
    spec = SLOT_SPECS[slot]
    return _cached_client(
        slot=slot,
        model=getattr(settings, spec.model_field),
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=spec.temperature if spec.temperature is not None else settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        thinking=spec.thinking,
    )


@lru_cache(maxsize=8)
def _cached_client(
    *,
    slot: LLMSlot,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    thinking: bool,
) -> LLMClient:
    """客户端缓存。**缓存键就是全部相关设置** —— 改了 `.env` 自然得到新客户端，
    不需要谁记得去清缓存。

    这是与「按 slot 缓存 + 手动失效」的关键差别：后者漏清一次，症状是
    「改了配置不生效」，而且只在测试里能看出来。

    缓存的意义在连接池：`ChatOpenAI` 底下是一个 httpx Client，每次新建
    = 每次重开 TCP + TLS。一个请求要打 5~10 次 LLM，这个开销不能忽略。

    顺带在这唯一一次执行里打一行配置日志 —— 三个槽位封了什么，启动后一看日志
    就知道，不用去读代码猜。**不含密钥**（硬红线 #6）。
    """
    logger.info(
        "构造 LLM 客户端：槽位=%s 模型=%s thinking=%s 温度=%s",
        slot,
        model,
        "开" if thinking else "关",
        temperature,
        extra={"llm": {"slot": slot, "model": model, "thinking": thinking}},
    )
    chat = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key or _NOT_CONFIGURED,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        callbacks=[TokenUsageCallback(slot)],
        # thinking 的开关只此一处。**开着时也要显式写**，理由见 THINKING_BODY 注释。
        extra_body=THINKING_BODY[thinking],
    )
    return LLMClient(
        chat, slot=slot, thinking=thinking, temperature=temperature, secret=api_key
    )


def reset_llm_cache() -> None:
    """清客户端缓存。**测试用**，生产不需要（见 `_cached_client` 的缓存键说明）。"""
    _cached_client.cache_clear()
