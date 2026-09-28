"""配置管理（P1.6）—— 对应 docs/方案设计.md §12。

三个必须在这一步定下的工程约定（开发流程 P1）：

1. **启动即校验（fail-fast）**：缺 LLM_API_KEY、POI 文件不存在、SQLite 目录
   不可写 → 进程直接起不来。故障在部署那一刻暴露，而不是等第一个用户提问时
   才以一句 AuthenticationError 的形式炸在用户面前。
2. **双模型槽位**：按节点角色分档。需求收集是纯结构化抽取（简单任务）用便宜
   模型；行程生成和反思审核决定质量用强模型。几乎零实现成本的一档省钱配置。
3. **DEMO_MODE**：置 true 时所有外部数据源强制短路成 mock，只留 LLM 一条
   真实依赖。演示防翻车 + 评估可复现。

实现说明：**没有用 pydantic-settings** —— venv 里没装，也不值得为此多一个
依赖。改用 python-dotenv 把 .env 读进 os.environ，再交给 Pydantic BaseModel
做类型转换与校验（Pydantic 的 lax 模式会自动把 "3" 转成 3、"true" 转成 True）。
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import ConfigError

# 本文件位于 app/core/config.py → 往上是 app/core → app → 仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]

PlanToolMode = Literal["deterministic", "agent"]
LogFormat = Literal["text", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# 数据源**按域拆**（§6.4）。注意这不是一个共用的 Literal：高德只出现在
# poi / distance，酒店只认 aigohotel —— 各域的取值集合本来就不同，
# 合成一个 Literal 等于把「谁支持什么」这件事从类型里抹掉。
Domain = Literal["poi", "distance", "hotel", "intercity"]
POIBackendName = Literal["mock", "amap"]
DistanceBackendName = Literal["mock", "amap"]
HotelBackendName = Literal["mock", "aigohotel"]
IntercityBackendName = Literal["mock", "variflight"]

# 域 → (后端字段名, 该域切到非 mock 时需要的密钥字段名)
#
# **放模块级而不放类属性**：Pydantic 会把 BaseModel 上的裸类属性当字段处理，
# 想当常量得标 ClassVar —— 模块级常量没这个坑，也不会混进 model_fields。
DOMAIN_SPEC: dict[Domain, tuple[str, str]] = {
    "poi": ("poi_backend", "amap_api_key"),
    "distance": ("distance_backend", "amap_api_key"),
    "hotel": ("hotel_backend", "aigohotel_api_key"),
    "intercity": ("intercity_backend", "variflight_api_key"),
}


class Settings(BaseModel):
    """全部环境变量（§12）。

    **字段的 alias 就是环境变量名** —— 用它做唯一映射，而不是再维护一张
    「字段名 ↔ 变量名」对照表（两张表迟早会对不上）。
    """

    model_config = ConfigDict(populate_by_name=True)

    # ---------------- LLM ----------------
    llm_base_url: str = Field(default="https://api.deepseek.com/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-v4-flash", alias="LLM_MODEL")
    llm_model_cheap: str = Field(default="deepseek-v4-flash", alias="LLM_MODEL_CHEAP")
    llm_judge_model: str = Field(default="deepseek-v4-flash", alias="LLM_JUDGE_MODEL")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    llm_timeout: int = Field(default=60, gt=0, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, ge=0, alias="LLM_MAX_RETRIES")

    # ---------------- 循环闸门（§5.2）----------------
    max_review_retry: int = Field(default=3, ge=0, alias="MAX_REVIEW_RETRY")
    max_user_revision: int = Field(default=5, ge=0, alias="MAX_USER_REVISION")
    max_ask_rounds: int = Field(default=3, ge=0, alias="MAX_ASK_ROUNDS")

    # ---------------- 工具与数据源 ----------------
    plan_tool_mode: PlanToolMode = Field(default="deterministic", alias="PLAN_TOOL_MODE")

    # 数据源按数据域分别选（§6.4）。单个全局 TOOL_BACKEND 表达不了
    # 「POI 用高德、酒店用 AIGOHOTEL」这种混用，故拆成四个独立开关。
    poi_backend: POIBackendName = Field(default="mock", alias="POI_BACKEND")
    distance_backend: DistanceBackendName = Field(default="mock", alias="DISTANCE_BACKEND")
    hotel_backend: HotelBackendName = Field(default="mock", alias="HOTEL_BACKEND")
    intercity_backend: IntercityBackendName = Field(default="mock", alias="INTERCITY_BACKEND")

    # 密钥只在该域被选中时才需要；后端与密钥不配套 → 启动直接失败（validate_startup）
    amap_api_key: str = Field(default="", alias="AMAP_API_KEY")
    aigohotel_api_key: str = Field(default="", alias="AIGOHOTEL_API_KEY")
    variflight_api_key: str = Field(default="", alias="VARIFLIGHT_API_KEY")

    tool_timeout: int = Field(default=5, gt=0, alias="TOOL_TIMEOUT")
    poi_data_path: str = Field(default="data/poi_clean.json", alias="POI_DATA_PATH")
    sqlite_path: str = Field(default="data/app.db", alias="SQLITE_PATH")
    demo_mode: bool = Field(default=False, alias="DEMO_MODE")

    # ---------------- 服务 ----------------
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="PORT")
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")
    session_ttl_days: int = Field(default=30, gt=0, alias="SESSION_TTL_DAYS")
    auth_required: bool = Field(default=False, alias="AUTH_REQUIRED")
    auth_token: str = Field(default="", alias="AUTH_TOKEN")
    log_level: LogLevel = Field(default="INFO", alias="LOG_LEVEL")
    log_format: LogFormat = Field(default="text", alias="LOG_FORMAT")
    auto_approve: bool = Field(default=False, alias="AUTO_APPROVE")

    # ==================== 构造 ====================

    @classmethod
    def from_env(cls) -> "Settings":
        """从 os.environ 构造。**空字符串按「未设置」处理。**

        这一点必须做：`.env.example` 里 `PORT=`、`AUTH_TOKEN=` 这类留空的行，
        原样丢给 Pydantic 会让 int / bool 转换直接报错。
        「留空 = 用默认值」也符合看模板的人的直觉。
        """
        raw: dict[str, str] = {}
        for name, field in cls.model_fields.items():
            env_key = field.alias or name
            value = os.environ.get(env_key)
            if value is not None and value.strip():
                raw[name] = value.strip()

        try:
            return cls(**raw)
        except ValidationError as exc:
            # 把 Pydantic 的多行报错压成一行中文 —— 启动失败信息要能直接看懂。
            # 名字用**环境变量名**而不是字段名：用户手里拿的是 .env，
            # 报 `poi_backend='aigohotel'` 他还得自己翻译一次才找得到那一行。
            items = []
            for err in exc.errors():
                loc = err["loc"]
                field_name = str(loc[0]) if loc else "?"
                field = cls.model_fields.get(field_name)
                name = (field.alias or field_name) if field is not None else field_name
                items.append(f"{name}={err['input']!r}（{err['msg']}）")
            raise ConfigError("环境变量格式非法：" + "；".join(items)) from exc

    # ==================== 派生值 ====================

    @property
    def poi_file(self) -> Path:
        """POI 数据文件的绝对路径（相对路径按仓库根解析）。"""
        path = Path(self.poi_data_path)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def sqlite_file(self) -> Path:
        """SQLite 会话库的绝对路径（相对路径按仓库根解析）。"""
        path = Path(self.sqlite_path)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def cors_origin_list(self) -> list[str]:
        """逗号分隔 → 列表。留空 = 仅同源，返回空列表。"""
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def effective_backend(self, domain: Domain) -> str:
        """取某个数据域**实际生效**的后端名。

        **registry 必须调它，不许直接读 `self.xxx_backend` 字段。** DEMO_MODE 的
        短路逻辑只此一处 —— 让每个调用点自己判一次 `demo_mode`，迟早漏一处，
        而漏掉的后果正是这个开关要防的事：演示时真的去打外部 API。

        返回 `mock` / `amap` / `aigohotel` / `variflight` 之一。类型标 str 而非
        按域的 Literal：调用方（registry）本就按字符串分发，在这层收窄没收益。
        """
        if self.demo_mode:
            return "mock"
        return getattr(self, DOMAIN_SPEC[domain][0])

    # ==================== 校验 ====================

    def validate_startup(self) -> None:
        """启动期硬校验（跨字段规则）。任何一条不过 → 抛 ConfigError。

        **只管配置自身的自洽性，不碰文件系统**：需要文件真实存在的检查放在
        `validate_paths()`。拆开是因为 `data/poi_clean.json` 要到 P2 才生成，
        在 P1 阶段调它必然失败 —— 那不代表配置写错了。
        """
        if not self.llm_api_key:
            raise ConfigError("LLM_API_KEY 未配置。请 `cp .env.example .env` 后填入密钥。")

        # 生产不留后门（§12）：开着鉴权却配了个短密钥，等于没鉴权
        if self.auth_required and len(self.auth_token) < 16:
            raise ConfigError(
                f"AUTH_REQUIRED=true 时 AUTH_TOKEN 长度必须 >= 16，当前 {len(self.auth_token)}"
            )

        # 后端与密钥必须配套，否则要等到第一次取数才炸在用户面前。
        # 表驱动遍历而不是四条 if：加数据源只改 DOMAIN_SPEC 一行，不会漏掉新分支。
        # 先整体跳过 DEMO_MODE —— 它把每个域强制短路成 mock，此时一律不需要密钥。
        if not self.demo_mode:
            for backend_field, key_field in DOMAIN_SPEC.values():
                backend = getattr(self, backend_field)
                if backend == "mock":
                    continue
                if not getattr(self, key_field):
                    # 报错要报**环境变量名**而不是字段名 —— 用户看的是 .env，
                    # 报 `poi_backend` 他还得自己做一次名字翻译
                    backend_env = type(self).model_fields[backend_field].alias
                    key_env = type(self).model_fields[key_field].alias
                    raise ConfigError(f"{backend_env}={backend} 需要 {key_env}，当前为空")

    def validate_paths(self) -> None:
        """数据文件与路径检查。**P5 的 lifespan 里与 validate_startup() 一起调用。**

        单独拆出来的原因：`data/poi_clean.json` 要到 P2 才生成，
        P1 阶段调它必然失败 —— 那不代表配置写错了。
        """
        if not self.poi_file.is_file():
            raise ConfigError(
                f"POI 数据文件不存在：{self.poi_file}"
                "（P2 阶段由 scripts/preprocess_poi.py 生成）"
            )

        db_dir = self.sqlite_file.parent
        db_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(db_dir, os.W_OK):
            raise ConfigError(f"SQLite 目录不可写：{db_dir}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """配置单例（§12）。

    **故意不在这里调用校验** —— 校验是启动期的动作，由 P5 的 lifespan 显式
    调 `validate_startup()` + `validate_paths()`。拆开是为了让 P1 阶段能在
    没有 API key 的环境里测试配置读取本身。

    改了配置想立刻生效，测试里记得 `get_settings.cache_clear()`。
    """
    load_dotenv(REPO_ROOT / ".env")   # 默认 override=False：真实环境变量优先于 .env
    return Settings.from_env()
