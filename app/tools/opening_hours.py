"""开放时间查询（P2）—— 门面 `TravelTools.opening_hours` 的实现（§6.3）。

批量取，返回 `dict[poi_id, OpeningHours]`。

**关键约定：没有开放时间的条目「不出现在 dict 里」，同时往 warnings 记一条。**
不是「出现在 dict 里但值是 None」—— `OpeningHours | None` 会让每个调用点都要
判两次（key 在不在、值是不是 None），而这两件事表达的是同一件事：**不知道**。

「不知道」有两种来源，这里都不阻塞（§9.1 第 4 步：不阻塞，只 warn）：
  - id 查不到（数据里没这条）
  - 这条 POI 的 `opening_hours` 是 None（数据源就没给）

> ⚠ 这里与 §6.3 表格里「缺失返回 `None` 并加 warn」的措辞不一致。那句话写在
> P1.4 把返回类型钉成 `dict[str, OpeningHours]`（值非可选）之前。现在的做法以
> **类型**为准：要么在 dict 里且非 None，要么不在。§6.3 已同步更正。
"""

from app.tools.base import OpeningHours, POIBackend, ToolResult


def split_known(
    wanted: list[str], found: dict[str, OpeningHours]
) -> tuple[dict[str, OpeningHours], list[str]]:
    """把「要的」与「拿到的」对齐，产出 (已知的, 未知的 id 列表)。

    保留**请求顺序**并把重复 id 去重：调用方给的顺序就是它想展示的顺序，
    在这里重排会让「第 3 天上午的景点」在结果里乱序。
    """
    known: dict[str, OpeningHours] = {}
    missing: list[str] = []
    for poi_id in wanted:
        if poi_id in known or poi_id in missing:
            continue  # 去重：同一个 id 报两次「查不到」是噪音
        hours = found.get(poi_id)
        if hours is None:
            missing.append(poi_id)
        else:
            known[poi_id] = hours
    return known, missing


class OpeningHoursTool:
    """门面用的工具对象。转发给 backend，把「查不到」翻译成 warning。"""

    name = "opening_hours"
    description = "批量查询指定场所的开放时间（开门、关门、闭馆日、是否全天开放）。"

    def __init__(self, backend: POIBackend) -> None:
        self._backend = backend

    def __call__(self, poi_ids: list[str]) -> ToolResult[dict[str, OpeningHours]]:
        """注意 `ok=False` 只用于**整批失败**（数据源挂了）。

        单个 id 查不到是**部分成功**，不是失败 —— 一批 5 个景点里 1 个没数据，
        不该让另外 4 个的开放时间也拿不到。所以它是 warning + `ok=True`。
        """
        result = self._backend.get_opening_hours(poi_ids)
        if not result.ok or result.data is None:
            return result

        known, missing = split_known(poi_ids, result.data)
        warnings = list(result.warnings)
        if missing:
            warnings.append(f"{len(missing)} 个场所无开放时间数据，按「未知」处理：" + "、".join(missing))

        return ToolResult.success(
            known,
            source=result.source,
            warnings=warnings,
            elapsed_ms=result.elapsed_ms,
        )
