"""pytest 全局配置与 fixture。

按 docs/方案设计.md §13，后续阶段还会在这里加：
    - Mock LLM（P4 起，让节点测试不消耗 token）
"""

from collections.abc import Callable, Iterator

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import get_settings


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
