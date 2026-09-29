"""POI 内存索引（P2）—— 三个域共用的一份数据。

读 `data/poi_clean.json`，逐条过 `POI` 校验，按 `id` 与 `city` 建两个索引，
启动时一次性载入（§6.4：< 5 万条约 20 MB）。

**加载器必须容错 —— 这是硬红线 #4 在数据层的延伸。**
四种坏法一律不许抛异常：

| 坏法 | 处理 |
|---|---|
| 文件不存在 | `load_error` 置 `not_found`，索引空 |
| JSON 语法坏 | `load_error` 置 `upstream_error`，索引空 |
| 顶层不是数组 | 同上 |
| **某一行字段越界**（如 `rating=9.9`）| **跳过该行 + 记 warning**，其余照常可用 |

为什么这层要这么小心：工具一旦抛异常，「工具永不抛异常」这条纪律就得在**每个
调用点**重复检查一遍，异常处理路径从一条变成 N 条。所以坏数据在**进门那一步**
就被降级成 warning —— 它在返回值里，看得见、可断言、不会绕过。

最后一行尤其重要：一条脏数据不该让整个城市的 POI 全查不出来。§9.1 已经要求
预处理丢弃坏行，但 `poi_clean.json` 是**手可编辑的文件**（P2 就是手写的），
把「文件永远干净」当成前提，等于把容错责任推给未来某个编辑它的人。
"""

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.tools.base import POI, ErrorCode


@dataclass(frozen=True)
class POIStore:
    """一份只读的 POI 索引。

    **只读是刻意的**：`frozen=True` + 两个 Mapping 就够了，工具层没有任何
    写数据的理由。要加数据源就加一个 backend，不要改这里。
    """

    by_id: Mapping[str, POI]
    by_city: Mapping[str, list[POI]]
    warnings: tuple[str, ...] = ()
    load_error: tuple[str, ErrorCode] | None = None

    # ---------------------------------------------------------------- 构造

    @classmethod
    def load(cls, path: Path) -> "POIStore":
        """从 JSON 文件载入。**任何情况下都返回一个可用的 store**，不抛异常。"""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls.empty(f"POI 数据文件不存在：{path}", "not_found")
        except OSError as exc:  # 权限、路径是目录、盘符掉了……
            return cls.empty(f"POI 数据文件读取失败：{path}（{exc}）", "upstream_error")

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            return cls.empty(f"POI 数据文件不是合法 JSON：{path}（{exc}）", "upstream_error")

        if not isinstance(raw, list):
            return cls.empty(
                f"POI 数据顶层必须是数组，实际是 {type(raw).__name__}：{path}", "upstream_error"
            )

        return cls._from_rows(raw, path)

    @classmethod
    def _from_rows(cls, raw: list, path: Path) -> "POIStore":
        """逐条校验。**坏行跳过并记 warning，不中断整批加载。**"""
        warnings: list[str] = []
        by_id: dict[str, POI] = {}
        by_city: dict[str, list[POI]] = defaultdict(list)

        for index, row in enumerate(raw):
            try:
                poi = POI.model_validate(row)
            except ValidationError as exc:
                # 只报第一个字段错 —— 一行里错十个字段时，报满一屏没人看
                first = exc.errors()[0]
                field = str(first["loc"][0]) if first["loc"] else "?"
                warnings.append(f"第 {index} 条跳过（{field}：{first['msg']}）")
                continue

            if poi.id in by_id:
                # 重复 id 会让 by_id 与 by_city 指向两条不同的记录，是静默漂移。
                # 保留先出现的那条（与 §9.1 第 5 步「保留字段完整度最高的」不同：
                # 那是预处理阶段的语义，加载器没有「完整度」这个概念，只能取确定行为）
                warnings.append(f"第 {index} 条 id 重复（{poi.id}），保留先出现的那条")
                continue

            by_id[poi.id] = poi
            by_city[poi.city].append(poi)

        if not by_id and raw:
            # 有数据但一条都没活下来 —— 这是配置问题（比如文件对错了），
            # 比「空文件」严重，必须让上层知道，不能表现成「这个城市没有景点」
            return cls.empty(
                f"POI 数据 {len(raw)} 条全部校验失败，无可用记录：{path}", "upstream_error"
            )

        return cls(
            by_id=by_id,
            by_city=dict(by_city),
            warnings=tuple(warnings),
        )

    @classmethod
    def empty(cls, error: str, code: ErrorCode) -> "POIStore":
        """一个查不到任何东西、且知道自己为什么查不到的 store。"""
        return cls(by_id={}, by_city={}, load_error=(error, code))

    # ---------------------------------------------------------------- 读

    @property
    def ok(self) -> bool:
        return self.load_error is None

    @property
    def size(self) -> int:
        return len(self.by_id)

    def get(self, poi_id: str) -> POI | None:
        """按 id 取。**不做模糊匹配** —— id 是内部标识，模糊匹配会把丢数据伪装成查到了。"""
        return self.by_id.get(poi_id)

    def in_city(self, city: str) -> list[POI]:
        """按城市取全部 POI（含酒店与交通枢纽）。

        城市名走**精确匹配**。不在这里做「成都 / 成都市」的模糊：归一化是预处理
        阶段（§9.1 第 2 步）的职责，在查询期再做一遍就等于有两处城市名规则，
        迟早不一致。查不到返回空列表 —— 那是「这个城市没有数据」，是正常结论。
        """
        return list(self.by_city.get(city, ()))

    def cities(self) -> list[str]:
        """已载入的城市列表 —— 城市名写错时，报错信息里要给得出候选。"""
        return sorted(self.by_city)
