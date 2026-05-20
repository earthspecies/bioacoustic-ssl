from typing import List, Literal, Any

import polars as pl
from esp_data.backends import DataBackend, PolarsBackend
from esp_data.transforms import LabelFromFeature, register_transform, Filter
from pydantic import BaseModel


class MultiLabelFromFeatureConfig(BaseModel):
    type: Literal["mulitlabel_from_feature"]
    feature: str
    output_feature: str = "label"
    override: bool = False
    is_multilabel: bool = True


class FilterFixConfig(BaseModel):
    type: Literal["filter_fix"]
    mode: Literal["include", "exclude"] = "include"
    property: str
    values: list[str | int | float]


class CastConfig(BaseModel):
    type: Literal["cast"]
    property: str
    cast_class: Any
    extract_all_regex: str = None


class Cast:
    def __init__(self, property: str, cast_class: pl.DataType, extract_all_regex: str = None) -> None:
        self.property = property
        self.cast_class = cast_class
        self.extract_all_regex = extract_all_regex

    @classmethod
    def from_config(cls, cfg: CastConfig) -> "Cast":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        assert isinstance(backend, PolarsBackend)

        df: pl.DataFrame = backend.unwrap

        if self.extract_all_regex is None:
            col = df[self.property].cast(self.cast_class)
        else:
            col = df[self.property].str.extract_all(rf"{self.extract_all_regex}").cast(self.cast_class)

        return backend.add_column(self.property, col), {}


class MultiLabelFromFeature(LabelFromFeature):
    """LabelFromFeature variant for list-type feature columns.

    Each row's feature value is a list of labels. The label_map maps
    individual label values (not lists) to integer indices.
    """

    def __init__(
        self,
        *,
        feature: str,
        label_map: dict[Any, int] | None = None,
        output_feature: str = "label",
        override: bool = False,
        is_multilabel: bool = True
    ) -> None:
        super().__init__(feature=feature, label_map=label_map, output_feature=output_feature, override=override)
        self.is_multilabel = is_multilabel

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        if not self.is_multilabel:
            return super().__call__(backend=backend)

        if self.output_feature in backend.columns and not self.override:
            raise AssertionError(
                "Feature already exists in DataFrame. Set `override=True` to replace it."
            )

        df = backend.unwrap

        assert isinstance(backend, PolarsBackend)

        if self.label_map is None:
            uniques = df[self.feature].explode().drop_nulls().unique().sort().to_list()
            label_map = {lbl: idx for idx, lbl in enumerate(uniques)}
        else:
            label_map = self.label_map

        mapped = df[self.feature].map_elements(
            lambda lst: [label_map.get(v) for v in lst] if lst is not None else None,
            return_dtype=pl.List(pl.Int64),
        ).alias(self.output_feature)

        metadata = {
            "label_feature": self.feature,
            "label_map": label_map,
            "num_classes": len(label_map),
        }

        return backend.add_column(self.output_feature, mapped), metadata



    type: Literal["filter_fix"]
    mode: Literal["include", "exclude"] = "include"
    property: str
    values: list[int | str]


register_transform(MultiLabelFromFeatureConfig, MultiLabelFromFeature)
register_transform(FilterFixConfig, Filter)
register_transform(CastConfig, Cast)
