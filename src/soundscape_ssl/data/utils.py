from collections import Counter
from typing import List, Literal, Any

import polars as pl
from alp_data.backends import DataBackend, PolarsBackend
from alp_data.transforms import LabelFromFeature, register_transform, Filter, LongTailUpsample
from pydantic import BaseModel


class MultiLabelFromFeatureConfig(BaseModel):
    type: Literal["mulitlabel_from_feature"]
    feature: str
    output_feature: str = "label"
    override: bool = False
    is_multilabel: bool = True
    class_ids: list[int] | None = None


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


class LongTailUpsampleTargetConfig(BaseModel):
    type: Literal["long_tail_upsample_target"]
    property: str
    target_count: int = None
    sufficient_threshold: int
    max_repeats: int
    seed: int


class LongTailUpsampleTarget(LongTailUpsample):
    """Upsample *and* downsample all categories to a fixed target count.

    Extends :class:`LongTailUpsample` with an optional ``target_count``
    parameter.  When *no* target is given this class behaves identically to
    its parent — under-represented categories are lifted towards
    ``sufficient_threshold`` while well-represented categories are left
    untouched.

    When ``target_count`` is set every category is resampled to that count:

    - Categories *above* ``target_count`` are downsampled (without
      replacement) to ``target_count``.
    - Categories *below* ``target_count`` are upsampled (with replacement)
      to ``min(target_count, count * max_repeats)``, so very rare categories
      are boosted as much as possible without repeating any single example
      more than ``max_repeats`` times.

    Parameters
    ----------
    property : str
        The name of the property (column) to balance on.
    sufficient_threshold : int
        Forwarded to the parent; governs the no-target strategy.
    max_repeats : int
        Maximum number of times any individual example may appear in the
        output.  Limits how aggressively very rare categories are upsampled.
    seed : int
        Random seed for reproducibility.  Defaults to 42.
    target_count : int | None
        Desired per-category sample count.  When ``None`` the parent strategy
        is used unchanged.
    """

    def __init__(
        self,
        property: str,
        sufficient_threshold: int,
        max_repeats: int,
        seed: int = 42,
        target_count: int | None = None,
    ) -> None:
        super().__init__(
            property=property,
            sufficient_threshold=sufficient_threshold,
            max_repeats=max_repeats,
            seed=seed,
        )
        self.target_count = target_count

    @classmethod
    def from_config(cls, cfg: LongTailUpsampleTargetConfig) -> "LongTailUpsampleTarget":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        if self.target_count is None:
            return super().__call__(backend)

        if self.property not in backend.columns:
            raise KeyError(f"Property '{self.property}' not found in the DataFrame columns.")

        category_counts = backend.histogram(self.property)

        if not category_counts:
            return backend, {"histogram_before": {}, "histogram_after": {}}

        target_counts: dict[str, int] = {}
        for value, count in category_counts.items():
            if count == 0:
                target_counts[value] = 0
            elif count >= self.target_count:
                # Downsample over-represented categories to the target.
                target_counts[value] = self.target_count
            else:
                # Upsample under-represented categories towards the target,
                # bounded by how many times each example may be repeated.
                target_counts[value] = min(self.target_count, count * self.max_repeats)

        sampled_backend = backend.upsample_by_column(
            column=self.property,
            target_counts=target_counts,
            seed=self.seed,
        )

        histogram_after = sampled_backend.histogram(self.property)

        return sampled_backend, {
            "histogram_before": category_counts,
            "histogram_after": histogram_after,
        }


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

    When ``class_ids`` is supplied (and no explicit ``label_map`` is given) the
    label map is built deterministically from that fixed class universe rather
    than inferred from the data observed in this split. This keeps the index
    assignment identical across splits, which is required when prototypes built
    on the train split are evaluated against the test split: a class with no
    samples in one split still keeps its reserved index instead of shifting all
    later classes. ``class_ids`` need not be sorted or unique.
    """

    def __init__(
        self,
        *,
        feature: str,
        label_map: dict[Any, int] | None = None,
        output_feature: str = "label",
        override: bool = False,
        is_multilabel: bool = True,
        class_ids: list[int] | None = None,
    ) -> None:
        if label_map is None and class_ids is not None:
            label_map = {v: i for i, v in enumerate(sorted(set(class_ids)))}
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


def compute_sample_weights(
    dataset: Any,
    label_column: str = "label",
    alpha: float = 1.0,
) -> list[float]:
    """Per-sample weights for a ``WeightedRandomSampler``.

    Weights follow a softened inverse-frequency class distribution
    (``weight = count ** -alpha``): ``alpha=0`` yields uniform weights
    (equivalent to plain shuffling), ``alpha=1`` fully balances the classes.

    Train labels are single-element lists; for multi-label safety a sample's
    weight is the max over its class weights. Samples with no/empty label get
    weight 0.

    Parameters
    ----------
    dataset :
        A map-style ``alp_data`` dataset exposing its Polars frame via
        ``dataset._data.unwrap``.
    label_column :
        Name of the mapped multi-hot label column (``List[Int64]``).
    alpha :
        Softening exponent in ``[0, 1]``.
    """
    df = dataset._data.unwrap  # polars DataFrame
    labels = df[label_column].to_list()  # list[list[int] | None]

    counts: Counter = Counter(c for lst in labels if lst for c in lst)
    class_w = {c: n ** (-alpha) for c, n in counts.items()}

    return [
        max((class_w[c] for c in lst), default=0.0) if lst else 0.0
        for lst in labels
    ]


register_transform(MultiLabelFromFeatureConfig, MultiLabelFromFeature)
register_transform(FilterFixConfig, Filter)
register_transform(CastConfig, Cast)
register_transform(LongTailUpsampleTargetConfig, LongTailUpsampleTarget)
