"""Freeze the full Xeno-Canto label space used by the released classifier.

Writes one artefact from the XC v0.2.0 ``train`` split metadata:

``metadata/xc_v0.2.0_classes.parquet``
    One row per class: ``label_index``, ``gbifID``, ``canonical_name``,
    ``family``, ``genus``, ``n_train_recordings``. ``label_index`` is the head's
    output index, and it is *derived*, not chosen: ``MultiLabelFromFeature``
    builds its map as ``sorted(set(class_ids))``, so index == rank of the gbifID
    in ascending numeric order. Anything that maps logits back to species (the
    model card, BirdSet logit masking) reads this file rather than re-deriving
    the order, and so does the dataset config, via
    ``soundscape_ssl.data.class_ids_from_parquet``.

The class universe is the ``train`` split, not ``all``: the head can only learn
a class it has training audio for, and the ``validation`` split contributes 11
species that ``train`` does not have (they are dropped by the include-filter in
the dataset config rather than given dead output units).

Run on the cluster, where the split CSV is reachable:

    uv run python scripts/build_xc_label_space.py
"""

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

from pathlib import Path

import polars as pl

import soundscape_ssl.data.datasets  # noqa: F401  — registers the anonymous-GCS filesystem
from alp_data.datasets import XenoCanto

VERSION = "0.2.0"
SPLIT = "train"
REPO = Path(__file__).resolve().parent.parent
PARQUET_OUT = REPO / "metadata" / f"xc_v{VERSION}_classes.parquet"


def main() -> None:
    csv = XenoCanto.VERSIONS[VERSION]["split_paths"][SPLIT]
    print(f"reading {csv}")
    # infer_schema_length=0 -> everything as Utf8: gbifID is written as "2490719.0"
    # in the CSV, so it has to go through float before int.
    df = pl.read_csv(csv, infer_schema_length=0)
    df = df.with_columns(pl.col("gbifID").cast(pl.Float64).cast(pl.Int64))

    classes = (
        df.group_by("gbifID")
        .agg(
            pl.col("canonical_name").first(),
            pl.col("family").first(),
            pl.col("genus").first(),
            pl.len().alias("n_train_recordings"),
        )
        .sort("gbifID")                       # == MultiLabelFromFeature's ordering
        .with_row_index("label_index")
        .select(["label_index", "gbifID", "canonical_name", "family", "genus",
                 "n_train_recordings"])
    )

    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    classes.write_parquet(PARQUET_OUT)

    print(f"{classes.height} classes over {df.height} recordings")
    print(f"  -> {PARQUET_OUT.relative_to(REPO)}")
    n = classes["n_train_recordings"]
    print(f"  recordings/class: min {n.min()} median {int(n.median())} max {n.max()} "
          f"| singletons {int((n == 1).sum())}")


if __name__ == "__main__":
    main()
