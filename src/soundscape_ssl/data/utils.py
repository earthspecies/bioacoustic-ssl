from pathlib import Path
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


class AppendLabelWhereConfig(BaseModel):
    """Config for :class:`AppendLabelWhere`."""

    type: Literal["append_label_where"]
    property: str
    key_property: str
    key: str
    value: int


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


class AppendLabelWhere:
    """Append ``value`` to the list column ``property`` where ``key`` is present.

    Repairs BirdSet test splits in which the upstream GBIF linkage failed: the
    species appears in ``ebird_code_multilabel`` but is missing from
    ``gbifID_multispecies``, so its test positives are silently dropped and the
    class cannot be scored at all. Affects exactly one species in each of NES
    (``runwre1``), PER (``scbwoo5``) and SNE (``pasfly``); every other task links
    cleanly.

    Runs *after* the ``Cast`` that turns ``property`` into ``List(Int64)`` and
    before ``MultiLabelFromFeature``. ``key`` is matched against the raw
    ``key_property`` string as a quoted token (``"pasfly"``), so it cannot match a
    longer code that merely contains it.
    """

    def __init__(self, property: str, key_property: str, key: str, value: int) -> None:
        self.property = property
        self.key_property = key_property
        self.key = key
        self.value = value

    @classmethod
    def from_config(cls, cfg: AppendLabelWhereConfig) -> "AppendLabelWhere":
        return cls(**cfg.model_dump(exclude=("type",)))

    def __call__(self, backend: DataBackend) -> tuple[DataBackend, dict]:
        assert isinstance(backend, PolarsBackend)

        df: pl.DataFrame = backend.unwrap
        hit = pl.col(self.key_property).str.contains(f'"{self.key}"', literal=True)
        patched = (
            pl.when(hit)
            .then(
                pl.concat_list(
                    pl.col(self.property).fill_null([]),
                    pl.lit(self.value, dtype=pl.Int64),
                ).list.unique()
            )
            .otherwise(pl.col(self.property))
            .alias(self.property)
        )
        out = df.with_columns(patched)
        n_patched = int(df.select(hit.fill_null(False).sum()).item())

        return backend.add_column(self.property, out[self.property]), {
            "rows_patched": n_patched,
            "appended_value": self.value,
        }


def class_ids_from_parquet(
    path: str,
    column: str = "gbifID",
) -> list[int]:
    """The class universe of a frozen label space, as a list of class ids.

    Used from config as a nested ``_target_`` wherever a class list is needed::

        class_ids:
          _target_: soundscape_ssl.data.class_ids_from_parquet
          path: metadata/xc_v0.1.0_all_classes.parquet

    Written for the full Xeno-Canto head, whose 10 799 ids would otherwise be
    inlined in the dataset config and then copied four times into every W&B run
    config. Pointing at the file instead keeps the label space in exactly one
    place — the same file the model card and BirdSet logit masking read — so the
    head's output indices cannot drift from what maps them back to species.

    The order returned here does not matter: :class:`MultiLabelFromFeature`
    builds its map as ``sorted(set(class_ids))``, so a class's index is its rank
    in ascending id order regardless.

    Parameters
    ----------
    path :
        Parquet file with one row per class. Relative paths resolve against the
        repository root, so a config value is independent of the run's cwd
        (Hydra changes it).
    column :
        Column holding the class id.
    """
    file = Path(path)
    if not file.is_absolute():
        file = Path(__file__).resolve().parents[3] / path
    return pl.read_parquet(file, columns=[column])[column].to_list()


def species_names(
    class_ids: list[int] | None = None,
    path: str = "metadata/gbif_names.parquet",
) -> dict[int, tuple[str | None, str | None]]:
    """Class id to ``(scientific_name, common_name)``, from the frozen name table.

    The one place a gbifID is turned into something readable, so a taxon cannot
    be called two different things in two places. ``common_name`` is ``None``
    where neither Xeno-Canto nor GBIF has an English name (436 insects and
    amphibians), and it is not unique — 38 names are shared by two ids — so
    display it, never key on it.

    Parameters
    ----------
    class_ids :
        Ids to look up, or ``None`` for the whole table. Ids the table does not
        have are absent from the result rather than an error.
    path :
        The table written by ``scripts/build_gbif_names.py``. Relative paths
        resolve against the repository root, as in
        :func:`class_ids_from_parquet`.
    """
    file = Path(path)
    if not file.is_absolute():
        file = Path(__file__).resolve().parents[3] / path
    table = pl.read_parquet(file, columns=["gbifID", "scientific_name", "common_name"])
    if class_ids is not None:
        table = table.filter(pl.col("gbifID").is_in(class_ids))
    return {row[0]: (row[1], row[2]) for row in table.iter_rows()}


def logit_mask(xc_classes: str, class_ids: list[int]) -> tuple["torch.Tensor", int]:
    """Head-output index for each of a task's label indices.

    The head's output index for a class is the rank of its gbifID in ascending
    order — :class:`MultiLabelFromFeature` builds its map as
    ``sorted(set(class_ids))`` — so the head's label space is reconstructed from
    the frozen parquet the training config read, and never from the checkpoint.

    Returns the index vector (``-1`` where the task species has no head output
    at all: 3 BirdSet species have no Xeno-Canto recording, 1 in PER and 2 in
    UHH) and the head's total output width.
    """
    import torch

    head_ids = sorted(set(class_ids_from_parquet(xc_classes)))
    head_index = {cid: i for i, cid in enumerate(head_ids)}
    mask = torch.tensor([head_index.get(cid, -1) for cid in class_ids], dtype=torch.long)
    return mask, len(head_ids)


def apply_logit_mask(logits: "torch.Tensor", mask: "torch.Tensor", fill: float) -> "torch.Tensor":
    """Select a task's columns out of a full-label-space logit vector.

    Column selection, not score modification: a class's logit is untouched, so
    every per-class ranking metric (average precision, AUROC) is identical to
    what the unmasked head would give for that class. What masking changes is
    which classes the macro average runs over — the task's, rather than the full
    head's — and any metric that ranks classes against each other, i.e. top-k
    accuracy. Under the block-diagonal mixer this is exact in a stronger sense
    too: a class's logit reads only its own prototypes, so dropping columns
    cannot change the ones that remain.

    Task species with no head output are filled with a constant, which puts them
    at their chance floor (AUROC 0.5, AP ~ the class's positive rate) rather than
    dropping them, so the class set matches the per-task probes'.
    """
    preds = logits[:, mask.clamp(min=0)]
    if (mask < 0).any():
        preds[:, mask < 0] = fill
    return preds


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
register_transform(AppendLabelWhereConfig, AppendLabelWhere)
