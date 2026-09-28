"""配置与异常体系测试（P1.5 / P1.6 / P1.7）—— 对应 docs/方案设计.md §12。

**这个文件是 P1 那笔欠账的偿还。** 在它存在之前，`app/core/config.py` 与
`app/core/exceptions.py` 是零覆盖的：把 `exceptions.py` 里 `detail` 的默认值
`= None` 删掉（那正是修过的洞，会让 `config.py` 全部 5 处 fail-fast 退化成
`TypeError`），`ruff` 与 `pytest` 依然全绿。**绿不等于过。**

两类测试分开看：

**纯构造层** —— `Settings.from_env()` 的类型转换、空值语义，不碰文件系统。
**启动校验层** —— `validate_startup()` / `validate_paths()` 的 fail-fast 与
「报错要报环境变量名」这条约定。
"""

import pytest

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigError

# 从模型自身推导出全部环境变量名 —— 不手抄一份清单，免得字段加了这里忘了加，
# 于是新字段悄悄从真实环境里漏进来，测试变得不确定
_ALL_ENV_KEYS = [field.alias or name for name, field in Settings.model_fields.items()]


@pytest.fixture
def make_settings(monkeypatch):
    """清干净相关环境变量后按需注入，返回一个 `Settings` 构造器。

    **必须清**：不清的话测试会读到开发者本机或 CI 的环境变量，出现
    「我这跑得过、你那跑不过」。清单从 `Settings.model_fields` 推导，不手抄。

    `.env` 文件不受影响 —— `load_dotenv` 只在 `get_settings()` 里调，
    而这个 fixture 构造的是 `Settings.from_env()`。
    """
    for key in _ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    def _build(**env_vars: str) -> Settings:
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        return Settings.from_env()

    return _build


# ===========================================================================
# 异常体系 —— 那个洞的回归测试
# ===========================================================================


def test_app_error_constructible_without_detail():
    """**回归测试。** `detail` 曾经是必填 keyword-only（漏写 `= None`），

    后果不是「少个可选参数」这么轻：`config.py` 里 5 处 `raise ConfigError("...")`
    全部退化成 `TypeError` —— 本该说「LLM_API_KEY 未配置」，实际吐一句没人看得懂的
    `missing 1 required keyword-only argument`。

    这条测试存在的唯一意义就是让那个洞不可能再回来。**删掉 `= None` 它立刻变红。**
    """
    err = ConfigError("密钥没配")
    assert err.message == "密钥没配"
    assert err.detail is None


def test_app_error_detail_is_optional_keyword():
    """`detail` 是 keyword-only 且可传 —— 传了要能拿到。"""
    err = ConfigError("配置错误", detail=".env 第 12 行")
    assert err.detail == ".env 第 12 行"


def test_to_sse_payload_hides_detail():
    """`detail` 只进日志，**绝不进响应体**。

    它可能带本机绝对路径、上游返回原文 —— 漏给前端就是信息泄露（硬红线 #6 的同一条思路）。
    所以这里既断言载荷的内容，也断言 `detail` 这个 key 根本不存在。
    """
    err = ConfigError("配置错误", detail=r"C:\Users\someone\.env")
    payload = err.to_sse_payload(node="requirement_collect")

    assert payload == {
        "code": "config_error",
        "message": "配置错误",
        "node": "requirement_collect",
    }
    assert "detail" not in payload


def test_error_codes_and_status_are_stable():
    """`code` 是给前端做分支判断的**稳定标识** —— 改它等于破坏兼容。

    这条测试是那份契约的书面凭证：谁改谁红。
    """
    from app.core.exceptions import (
        LLMError,
        NotFoundError,
        RateLimitedError,
        SessionModeError,
    )

    assert ConfigError("x").code == "config_error"
    assert NotFoundError("x").http_status == 404
    assert SessionModeError("x").http_status == 409
    assert RateLimitedError("x").http_status == 429
    assert LLMError("x").http_status == 500


def test_validate_startup_raises_config_error_not_type_error(make_settings):
    """**把异常体系与配置校验连起来验** —— 那个洞正是发生在这个接缝上。

    缺密钥时必须抛 `ConfigError`（API 层据此返回 4xx/5xx 并给出中文原因），
    而不是 `TypeError`。这是本节最值钱的一条断言：它同时覆盖了两个模块。
    """
    settings = make_settings()  # 什么都没配
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        settings.validate_startup()


# ===========================================================================
# from_env —— 空值语义与类型转换
# ===========================================================================


def test_empty_string_means_unset(make_settings):
    """空字符串按「未设置」处理，回落到默认值。

    这是 `.env.example` 里 `PORT=`、`AUTH_TOKEN=` 这类留空行的依据 ——
    原样丢给 Pydantic 会让 int / bool 转换直接报错，而「留空 = 用默认值」
    才是看模板的人的直觉。
    """
    settings = make_settings(LLM_API_KEY="sk-test", PORT="", AUTH_TOKEN="", DEMO_MODE="")
    assert settings.port == 8000
    assert settings.auth_token == ""
    assert settings.demo_mode is False


def test_lax_type_conversion(make_settings):
    """Pydantic 的 lax 模式负责 `"3"` → `3`、`"true"` → `True`。

    没有这一层，`os.environ` 里全是字符串这个事实会漏进每个使用点。
    """
    settings = make_settings(
        LLM_API_KEY="sk-test",
        MAX_REVIEW_RETRY="7",
        LLM_TEMPERATURE="0.9",
        DEMO_MODE="true",
    )
    assert settings.max_review_retry == 7
    assert settings.llm_temperature == 0.9
    assert settings.demo_mode is True


def test_bad_value_reports_env_var_name(make_settings):
    """报错要报**环境变量名**，不是字段名。

    用户手里拿的是 `.env`。报 `poi_backend='aigohotel'` 他还得自己做一次名字翻译
    才找得到那一行 —— 而 Pydantic 默认给的就是字段名，所以这里有一条映射。
    """
    with pytest.raises(ConfigError, match="POI_BACKEND"):
        make_settings(LLM_API_KEY="sk-test", POI_BACKEND="aigohotel")


def test_bad_values_are_reported_together(make_settings):
    """一次报全，不要改一个试一次。"""
    with pytest.raises(ConfigError) as exc_info:
        make_settings(LLM_API_KEY="sk-test", PORT="99999", DEMO_MODE="maybe")

    message = str(exc_info.value)
    assert "PORT" in message
    assert "DEMO_MODE" in message


def test_cross_domain_backend_value_is_rejected(make_settings):
    """按域 `Literal` 拦住「串域取值」。

    高德只出现在 poi / distance，酒店只认 aigohotel。合成一个共用 Literal
    就等于把「谁支持什么」从类型里抹掉，于是 `HOTEL_BACKEND=amap` 能配成功，
    一直到 P2 写 registry 时才发现没有这个实现。
    """
    with pytest.raises(ConfigError, match="HOTEL_BACKEND"):
        make_settings(LLM_API_KEY="sk-test", HOTEL_BACKEND="amap")


# ===========================================================================
# validate_startup —— 按域密钥校验（遍历式）
# ===========================================================================


def test_all_mock_needs_no_keys(make_settings):
    """默认全 mock → 不需要任何数据源密钥。"""
    make_settings(LLM_API_KEY="sk-test").validate_startup()


@pytest.mark.parametrize(
    ("backend_env", "expected_missing"),
    [
        ("AMAP_API_KEY", "POI_BACKEND"),
        ("AIGOHOTEL_API_KEY", "HOTEL_BACKEND"),
        ("VARIFLIGHT_API_KEY", "INTERCITY_BACKEND"),
    ],
)
def test_non_mock_backend_requires_its_key(make_settings, backend_env, expected_missing):
    """切到非 mock 却没配对应密钥 → 启动就失败。

    否则要等到**第一个用户提问**才炸，而且炸在上游 API 的超时里，看不出是配置问题。
    """
    domain_env = {
        "AMAP_API_KEY": {"POI_BACKEND": "amap"},
        "AIGOHOTEL_API_KEY": {"HOTEL_BACKEND": "aigohotel"},
        "VARIFLIGHT_API_KEY": {"INTERCITY_BACKEND": "variflight"},
    }[backend_env]

    settings = make_settings(LLM_API_KEY="sk-test", **domain_env)
    with pytest.raises(ConfigError, match=expected_missing):
        settings.validate_startup()


def test_backend_with_key_passes(make_settings):
    """后端与密钥配套 → 放行。"""
    make_settings(LLM_API_KEY="sk-test", POI_BACKEND="amap", AMAP_API_KEY="amap-key").validate_startup()


def test_mixed_backends_are_independent(make_settings):
    """混用：POI 用本地库、距离用高德 —— 这是 §6.4 拆域的全部理由。"""
    settings = make_settings(
        LLM_API_KEY="sk-test",
        POI_BACKEND="mock",
        DISTANCE_BACKEND="amap",
        AMAP_API_KEY="amap-key",
    )
    settings.validate_startup()
    assert settings.effective_backend("poi") == "mock"
    assert settings.effective_backend("distance") == "amap"


def test_auth_required_needs_long_token(make_settings):
    """生产不留后门：开着鉴权却配个短密钥，等于没鉴权。"""
    with pytest.raises(ConfigError, match="AUTH_TOKEN"):
        make_settings(LLM_API_KEY="sk-test", AUTH_REQUIRED="true", AUTH_TOKEN="short").validate_startup()


def test_auth_required_with_long_token_passes(make_settings):
    make_settings(
        LLM_API_KEY="sk-test", AUTH_REQUIRED="true", AUTH_TOKEN="x" * 16
    ).validate_startup()


# ===========================================================================
# effective_backend —— DEMO_MODE 短路
# ===========================================================================


def test_effective_backend_defaults_to_mock(make_settings):
    settings = make_settings(LLM_API_KEY="sk-test")
    assert [settings.effective_backend(d) for d in ("poi", "distance", "hotel", "intercity")] == [
        "mock"
    ] * 4


def test_demo_mode_short_circuits_every_domain(make_settings):
    """`DEMO_MODE=true` 时**四个域全被强制短路成 mock**，哪怕配的是 amap。

    这条是那个开关的全部意义：演示时不会因为配额用完当场翻车，评估时无网络抖动。
    短路逻辑只此一处（`effective_backend`）—— registry 必须调它而不是直接读字段。
    """
    settings = make_settings(
        LLM_API_KEY="sk-test",
        DEMO_MODE="true",
        POI_BACKEND="amap",
        DISTANCE_BACKEND="amap",
        HOTEL_BACKEND="aigohotel",
    )
    assert [settings.effective_backend(d) for d in ("poi", "distance", "hotel", "intercity")] == [
        "mock"
    ] * 4


def test_demo_mode_also_skips_key_validation(make_settings):
    """DEMO_MODE 下即使配了外部后端也不要求密钥 —— 因为它根本不会去打。"""
    make_settings(LLM_API_KEY="sk-test", DEMO_MODE="true", POI_BACKEND="amap").validate_startup()


# ===========================================================================
# 派生值
# ===========================================================================


def test_cors_origin_list_parsing(make_settings):
    settings = make_settings(LLM_API_KEY="sk-test", CORS_ORIGINS="http://a.com, http://b.com,")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]


def test_cors_origin_list_empty_means_same_origin_only(make_settings):
    assert make_settings(LLM_API_KEY="sk-test", CORS_ORIGINS="").cors_origin_list == []


def test_relative_data_paths_resolve_against_repo_root(make_settings):
    """相对路径按仓库根解析 —— 不依赖「从哪个目录启动进程」。

    这一条很实际：uvicorn 从 `app/` 下启动和从仓库根启动，相对路径会指向不同地方，
    症状是「本地能跑、部署说找不到文件」。
    """
    from app.core.config import REPO_ROOT

    settings = make_settings(LLM_API_KEY="sk-test", POI_DATA_PATH="data/poi_clean.json")
    assert settings.poi_file == REPO_ROOT / "data" / "poi_clean.json"


def test_absolute_data_paths_are_kept(make_settings, tmp_path):
    settings = make_settings(LLM_API_KEY="sk-test", POI_DATA_PATH=str(tmp_path / "poi.json"))
    assert settings.poi_file == tmp_path / "poi.json"


# ===========================================================================
# validate_paths —— 文件系统检查（与 validate_startup 分开的那一半）
# ===========================================================================


def test_validate_paths_rejects_missing_poi_file(make_settings):
    """POI 文件不存在 → 启动失败。

    `.env.example` 里留了 `POI_DATA_PATH` 的默认值，但文件要到 P2 才由
    `scripts/preprocess_poi.py` 生成 —— 这正是它必须与 `validate_startup()`
    分开的原因：P1 阶段调它会必然失败，那不代表配置写错了。
    """
    settings = make_settings(
        LLM_API_KEY="sk-test", POI_DATA_PATH="data/definitely-not-here.json"
    )
    with pytest.raises(ConfigError, match="POI 数据文件不存在"):
        settings.validate_paths()


def test_validate_paths_creates_sqlite_dir(make_settings, tmp_path):
    """SQLite 目录不存在时会被**创建**。

    注意这确实是个副作用（名字叫 validate 的函数在造目录）—— 这条测试把它
    变成显式行为而不是意外。P5 的 lifespan 会调它，届时若觉得不妥再挪出去。
    """
    poi = tmp_path / "poi.json"
    poi.write_text("[]", encoding="utf-8")
    db_dir = tmp_path / "nested" / "db"

    settings = make_settings(
        LLM_API_KEY="sk-test",
        POI_DATA_PATH=str(poi),
        SQLITE_PATH=str(db_dir / "app.db"),
    )
    settings.validate_paths()

    assert db_dir.is_dir()


# ===========================================================================
# get_settings 单例
# ===========================================================================


def test_get_settings_is_cached():
    """`@lru_cache` 单例 —— 同一个对象，不是每次新建。"""
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_get_settings_needs_cache_clear_to_pick_up_changes(monkeypatch):
    """改了环境变量**不会**自动生效，必须 `cache_clear()`。

    这条测试的价值在于把「为什么测试里要写 cache_clear」变成可执行的事实 ——
    不写它会得到「monkeypatch 好像没生效」的错觉，然后去怀疑被测代码。
    """
    get_settings.cache_clear()
    try:
        monkeypatch.setenv("LLM_MODEL", "model-a")
        assert get_settings().llm_model == "model-a"

        monkeypatch.setenv("LLM_MODEL", "model-b")
        assert get_settings().llm_model == "model-a"  # 缓存未清，仍是旧值

        get_settings.cache_clear()
        assert get_settings().llm_model == "model-b"  # 清了才生效
    finally:
        get_settings.cache_clear()


def test_get_settings_does_not_validate():
    """`get_settings()` **故意不调校验** —— 它只读值，不判对错。

    校验是启动期动作，由 P5 的 lifespan 显式调 `validate_startup()`。拆开是为了
    让没有 API key 的环境（例如 CI 跑 P1 测试）也能读配置本身。

    ⚠ **这条测试刻意不断言任何具体值，这是有代价换来的教训。**

    `get_settings()` 内部会 `load_dotenv()` 读仓库根的 `.env`，所以它的结果和
    开发者本机环境绑定。原先这里写的是 `assert get_settings().llm_api_key == ""`，
    在有 `.env` 的机器上必然失败 —— 而 pytest 的断言自省会**把密钥原文打印到终端**：

        AssertionError: assert 'sk-4276...' == ''

    所以规矩是：**测试永不把密钥写进断言**，因为失败路径就是一条泄露路径。
    这里只断言与 `.env` 无关的性质。
    """
    settings = get_settings()  # 不抛异常 = 没有校验
    assert isinstance(settings, Settings)

    # 反证：校验一旦被调用，缺 key 必然抛 —— 说明上面那行确实没校验
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings().validate_startup()
