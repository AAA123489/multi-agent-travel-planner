"""pytest 全局配置与 fixture。

按 docs/方案设计.md §13 在这里放跨文件共用的东西：`fresh_settings`（配置隔离）、
`checkpointer`（会话隔离）、`fake_llm`（**网络隔离**）。
"""

import inspect
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from app.core import llm as llm_module
from app.core.config import get_settings
from app.graph.nodes import DEFAULT_NODES
from app.graph.state import TravelRequirementDelta


class FakeLLM:
    """离线 LLM 替身 —— 让节点测试永不联网（§13 说的「Mock LLM」）。

    ## 它为什么是 autouse 的

    P4.1 之前，`tests/test_graph.py` 的路径④会传 `user_query`，而那时的
    `requirement_collect` 是个空壳，不碰 LLM。P4.1 把真 LLM 接进去之后，
    **那条测试立刻变成一次真实 API 调用** —— 耗时从 0.6 秒涨到 4.2 秒、
    消耗 token、需要 `.env` 里有一把真 key。它跑得通，所以没有任何人会发现，
    直到某台机器上没有 key，或者账单对不上。

    P1/P2/P3 都写着「完全不碰 LLM，无需 API key」。那句话要么靠纪律维持，
    要么靠构造维持 —— **这里选后者**：替身 autouse，忘不掉。
    需要什么返回值就自己赋：`fake_llm.delta = TravelRequirementDelta(days=3)`。

    ## 记录了什么

    `structured_schemas` 让测试能断言「节点确实按 delta 模式要了那个 schema」——
    否则「节点悄悄改成全量抽取」和「节点没接 LLM」在输出上长得一样。
    `payloads` 让测试能断言 prompt 里真的塞进了 `today`。
    """

    def __init__(self) -> None:
        # 默认：结构化调用抽不到任何东西（对路径测试足够 —— 需求已在状态里），
        # 自由对话返回一句固定问句。赋成 Exception 实例则让该次调用抛异常。
        self.delta: TravelRequirementDelta | BaseException = TravelRequirementDelta()
        self.question: str | BaseException = "你打算去哪儿、玩几天、几个人一起？"
        self.slots: list[str] = []
        self.structured_schemas: list[type] = []
        self.payloads: list[Any] = []
        self.chat_calls: list[list[Any]] = []

    def structured(self, schema: type) -> RunnableLambda:
        self.structured_schemas.append(schema)
        return RunnableLambda(self._respond)

    def _respond(self, payload: Any) -> TravelRequirementDelta:
        self.payloads.append(payload)
        if isinstance(self.delta, BaseException):
            raise self.delta
        return self.delta

    def chat(self, messages: list[Any]) -> AIMessage:
        self.chat_calls.append(list(messages))
        if isinstance(self.question, BaseException):
            raise self.question
        return AIMessage(content=self.question)


def _llm_consumer_modules() -> list[ModuleType]:
    """所有可能持有 `build_llm` 这个名字的模块。

    **从 `DEFAULT_NODES` 倒推，不手写清单**：手写的话，加第六个节点时漏掉一行，
    症状是「那个节点的测试悄悄联网了」—— 而它照样是绿的。
    这与 `tests/test_llm.py` 的 AST 扫描器是同一条理由：清单会漂，倒推不会。
    """
    modules = {inspect.getmodule(fn) for fn in DEFAULT_NODES.values()}
    modules.add(llm_module)
    return sorted((m for m in modules if m is not None), key=lambda m: m.__name__)


def patch_build_llm(fake: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """把每个模块里的 `build_llm` 换成返回替身的假函数。

    **同时打 `app.core.llm.build_llm` 和各节点模块自己的那个名字。**
    节点写的是 `from app.core.llm import build_llm`，那是往模块全局里放了一份
    引用 —— 只打源模块的话，节点拿到的还是原函数，替身形同虚设。
    （反过来，`tests/test_llm.py` 也是按名字 import 的，所以它**不会**被这里影响，
    那批测试仍然在真的测工厂。）

    返回被替换的模块名，供调用方断言「确实打到了东西」。
    """
    def _fake_build(slot: str = "main") -> FakeLLM:
        # 记下槽位，「节点有没有用便宜档」才断言得出来 —— 双模型槽位
        # 全都退化到主力档时，行为上完全看不出来，只是账单变贵
        fake.slots.append(slot)
        return fake

    patched: list[str] = []
    for module in _llm_consumer_modules():
        if hasattr(module, "build_llm"):
            monkeypatch.setattr(module, "build_llm", _fake_build)
            patched.append(module.__name__)
    return patched


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """自动装上的离线 LLM 替身 —— 见 `FakeLLM` 的说明。

    断言「至少打到了一个模块」是**防这个 fixture 自己静默失效**：某次重构把
    import 方式改掉之后，`patched` 会变成空列表，而那时所有测试照样是绿的，
    只是又开始烧真 token 了。
    """
    fake = FakeLLM()
    patched = patch_build_llm(fake, monkeypatch)
    assert patched, (
        "没有任何模块被装上 LLM 替身 —— 测试会去调真实服务。"
        f"候选：{[m.__name__ for m in _llm_consumer_modules()]}"
    )
    return fake


@pytest.fixture
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """改环境变量后让 `get_settings` 重新读取，测完还原。

    `get_settings` 是 `@lru_cache` 单例 —— 不清缓存的话，最先跑的那个测试读到的
    配置会被后面所有测试复用，`monkeypatch` 看起来「没生效」，而你会去怀疑
    被测函数写错了。（这就是 CLAUDE.md 里那句 `get_settings.cache_clear()` 的由来。）

    用法::

        def test_x(fresh_settings):
            fresh_settings(MAX_REVIEW_RETRY="0")

    **为什么这个 fixture 在图测试里是必需的，而不只是方便**：路由函数的阈值
    默认走 `get_settings()`，也就意味着它会读开发机上的 `.env`。一个把
    `MAX_REVIEW_RETRY` 调成 5 的人跑 P3 的验收测试，会看到「重试次数是 5 不是 3」
    这种在自己机器上永远复现不了别人却能复现的失败。
    """

    def _apply(**env: str) -> None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()


@pytest.fixture
def checkpointer() -> InMemorySaver:
    """一个干净的内存 checkpointer。

    **每个测试一份**，不做会话级共享：checkpointer 是按 `thread_id` 存状态的，
    共享会让上一个测试留下的话题被下一个测试接上 —— 而那看起来像图的行为错了。
    与 `build_tools()` 刻意不加 `@lru_cache` 是同一条理由。
    """
    return InMemorySaver()
