"""应用异常体系（P1.5）。

设计目标只有一个：**让 API 层能区分「用户的错」和「服务器的错」**，
从而给出不同的 HTTP 状态码和前端行为（见方案设计 §11 错误码表）。

没有这个体系，代码里会散落裸的 `raise ValueError`，API 层没法分辨
「用户输入非法」和「服务端炸了」，只能一律返回 500 —— 用户看到「服务器错误」，
但其实只是他少填了一个字段。

约定：
  - `code` 是**稳定的字符串标识**，给前端做分支判断用。改它等于破坏兼容
  - `message` 是给人看的中文描述
  - `http_status` 是 API 层的默认状态码
"""
from typing import Any


class AppError(Exception):
    """所有应用级异常的基类。

    只有**预期内**的错误才继承它。真正的程序 bug（IndexError、KeyError）
    不该包装成 AppError —— 那会把 bug 伪装成「已知错误」，把问题掩盖掉。
    """

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str = "", *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        # 内部细节（路径、上游原文），只进日志，不进响应体
        self.detail = detail

    def to_sse_payload(self, node: str | None = None) -> dict[str, Any]:
        """转成 §10 的 SSE `error` 事件载荷 `{code, message, node}`。

        故意不带 detail，避免把内部细节泄露给前端。
        """
        return {"code": self.code, "message": self.message, "node": node}


class ConfigError(AppError):
    """配置错误。启动期抛出 → 进程直接起不来（fail-fast）。

    §12 要求生产不留后门，例如 AUTH_REQUIRED=true 而 token 太短就 raise。
    """
    code = "config_error"


class NotFoundError(AppError):
    """资源不存在 → 404。典型场景：thread_id 查不到，前端应重新创建会话。"""
    code = "not_found"
    http_status = 404


class SessionModeError(AppError):
    """会话模式不匹配 → 409。

    典型场景：图正停在 interrupt 等 resume，客户端却发了新的 message。
    前端收到后应重新拉取状态并对齐 UI（§11）。
    """
    code = "session_mode_error"
    http_status = 409


class RateLimitedError(AppError):
    """触发限流 → 429（默认 20 次/分钟）。"""

    code = "rate_limited"
    http_status = 429


class LLMError(AppError):
    """LLM 调用失败（超时 / 限流 / 鉴权 / 返回不可解析）。

    注意这是**调用层**的异常。工具层的失败不走这里 —— 工具按硬红线 #4
    永不抛异常，一律返回 `ToolResult(ok=False)`。
    """
    code = "llm_error"
