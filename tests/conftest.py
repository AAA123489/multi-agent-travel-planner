"""pytest 全局配置与 fixture。

按 docs/方案设计.md §13，后续阶段会在这里放：
    - 内存 checkpointer（替代 SQLite，避免测试污染 data/app.db）
    - Mock LLM（P4 起，让节点测试不消耗 token）
    - 已构造好的 TravelState fixture（P1.2 起）

P1 阶段这些还没有，先留占位。
"""
