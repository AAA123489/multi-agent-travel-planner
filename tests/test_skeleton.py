"""骨架冒烟测试（P1.1）。

验证两件容易悄悄坏掉的事：
  1. pyproject.toml 里的 `pythonpath = ["."]` 真的生效 —— 否则 `import app` 全线失败
  2. 所有 `__init__.py` 都在位 —— 否则 import 一个子包才炸，问题暴露得很晚

这不是凑数的测试：它同时是「项目结构与配置一致」的唯一守卫。
"""

import importlib

import app

# 与 docs/方案设计.md §13 的目录结构保持一致
SUBPACKAGES = [
    "app.api",
    "app.core",
    "app.graph",
    "app.graph.nodes",
    "app.graph.validators",
    "app.services",
    "app.tools",
    "app.tools.backends",
]


def test_app_importable():
    """`import app` 能成功 —— 证明 pytest 的 pythonpath 配置正确。"""
    assert app.__version__


def test_subpackages_importable():
    """每个子包都能被导入 —— 证明 __init__.py 没有漏建。"""
    for name in SUBPACKAGES:
        importlib.import_module(name)
