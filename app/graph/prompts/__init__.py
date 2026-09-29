"""Prompt 文件目录 + 载入器（P4.1）—— 硬红线 #5 的落地点。

**Prompt 一律放本目录的 `*.md`，不塞进 Python 字符串。** 理由不是洁癖：

- prompt 的改动是**反复比对**的产物（换个措辞、加个反例、看效果变没变）。
  `.md` 能 diff、能整段复制出去调、能被不写 Python 的协作者直接改；
- 写成 f-string 之后，读它得先穿过一层 Python 缩进与转义 —— 改一个字的成本
  远高于改一个 `.md`，于是 prompt 就不会被反复改了。那正是它退化的原因。

## 本版没有占位符替换

P4.1 的两份 prompt **都是全静态的**：`today` 这类每轮都变的东西放在 human 轮，
不放系统提示里 —— 那样系统提示在多轮之间**逐字节相同**，服务端的 prompt cache
才命中得了（P4.0-C 实测：cheap 档 337 个 input token 里 128 个是 `cache_read`）。
把每天都会变的日期塞进系统提示，等于每天早上把缓存整段作废。

需要替换时再加模板机制（`string.Template`，不要用 `str.format` —— prompt 里
一定有 JSON 示例，`{}` 会把 format 打爆）。**不要提前造一个当下用不上的机制。**
"""

import functools
from pathlib import Path

from app.core.exceptions import ConfigError

PROMPT_DIR = Path(__file__).resolve().parent
SUFFIX = ".md"


def available_prompts() -> list[str]:
    """本目录现有的 prompt 名（不含后缀）—— 报错信息里要给得出候选。"""
    return sorted(path.stem for path in PROMPT_DIR.glob(f"*{SUFFIX}"))


@functools.cache
def load_prompt(name: str) -> str:
    """读出 `app/graph/prompts/<name>.md`，去掉首尾空白。

    **带缓存**：每个节点每次执行都要调它，而文件内容在进程生命周期里不会变。
    代价是改完 prompt 要重启进程（或 `load_prompt.cache_clear()`）才生效 ——
    这个取舍是刻意的：换成每次读盘，就得处理「同一次请求里前后两次读到两个版本」
    的可能，那种不一致比「改了没生效」难查得多。

    :raises ConfigError: 文件不存在。**这是打包问题，不降级。** 没有 prompt
        就没法问 LLM，没有「继续跑下去」这个选项 —— 而它在第一次调用时必然
        暴露，所以响亮地失败不会拖到演示中途。
    """
    path = PROMPT_DIR / f"{name}{SUFFIX}"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(
            f"Prompt 文件不存在：{path.name}。本目录现有：{'、'.join(available_prompts())}"
        ) from exc
    return text.strip()


__all__ = ["PROMPT_DIR", "available_prompts", "load_prompt"]
