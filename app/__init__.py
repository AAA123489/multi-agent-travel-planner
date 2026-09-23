"""多 Agent 智能旅行规划系统。

分层见 docs/方案设计.md §2.1：
    api/        HTTP 与 SSE 边界（唯一接触请求/响应的地方）
    graph/      LangGraph 状态图：状态、路由、节点、校验、prompt
    tools/      工具层：上层只依赖抽象接口，不感知数据来源
    services/   跨层服务：checkpointer、会话、评估
    core/       基础设施：配置、日志、异常、LLM 工厂
"""

__version__ = "0.1.0"
