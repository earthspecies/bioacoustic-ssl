"""Freeze the Xeno-Canto label space used by the released classifier.

Writes one artefact per (version, split) pair, e.g. from XC v0.1.0 ``all``:

``metadata/xc_v0.1.0_all_classes.parquet``
    One row per class: ``label_index``, ``gbifID``, ``canonical_name``,
    ``common_name``, ``family``, ``genus``, ``n_train_recordings``. ``label_index`` is the head's
    output index, and it is *derived*, not chosen: ``MultiLabelFromFeature``
    builds its map as ``sorted(set(class_ids))``, so index == rank of the gbifID
    in ascending numeric order. Anything that maps logits back to species (the
    model card, BirdSet logit masking) reads this file rather than re-deriving
    the order, and so does the dataset config, via
    ``bioacoustic_ssl.data.class_ids_from_parquet``.

**The universe must be built from the split the head actually trains on**, and
the file name says which one so the two cannot drift apart unnoticed. They did
drift: the head's dataset config moved to v0.1.0 ``all`` while the label space
was still the older v0.2.0 ``train`` snapshot, which put 8 of the corpus's
gbifIDs outside the label space and left 946 output units with no training
audio. Neither failed loudly. ``MultiLabelFromFeature`` maps an unknown id to
``None``, and ``MultiHotEncoder`` used to accept that — ``vec[None] = 1.0`` is a
whole-tensor assignment in torch, so those rows silently became "every class is
positive" rather than an error. The encoder now rejects it, and this file name
now records the split, but the first defence is running this script against the
split in the config.

A class with no training audio is a class the head cannot learn, so building
from ``train`` while training on ``all`` is also wrong: it discards the 9
species that only ``validation`` has. Build from what you train on.

``common_name`` is joined in from ``metadata/gbif_names.parquet``, so run
``scripts/build_gbif_names.py`` first when the class universe changes. The join
is left: a class XC has no vernacular name for keeps a null, and the label space
is never narrowed by a missing name.

Run where the split CSVs are reachable (they are public GCS objects, read
anonymously via the ``bioacoustic_ssl.data.datasets`` filesystem patch):

    uv run python scripts/build_xc_label_space.py
    uv run python scripts/build_xc_label_space.py --version 0.2.0 --split train

Note that ``metadata/xc_v0.2.0_classes.parquet`` — the 11 737-class universe the
first full-XC head was trained against — predates the split in the file name.
It is kept because the runs that used it are on record; nothing new should read
it.
"""

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import argparse
from pathlib import Path

import polars as pl

import bioacoustic_ssl.data.datasets  # noqa: F401  — registers the anonymous-GCS filesystem
from alp_data.datasets import XenoCanto

REPO = Path(__file__).resolve().parent.parent
NAMES = REPO / "metadata" / "gbif_names.parquet"


def build(version: str, split: str) -> None:
    """Write the class table for one (version, split) pair.

    Args:
        version: Xeno-Canto metadata version, a key of ``XenoCanto.VERSIONS``.
        split: Split whose recordings define the class universe. Must be the
            split the head trains on.
    """
    out = REPO / "metadata" / f"xc_v{version}_{split}_classes.parquet"
    csv = XenoCanto.VERSIONS[version]["split_paths"][split]
    print(f"reading {csv}")
    # infer_schema_length=0 -> everything as Utf8: gbifID is written as "2490719.0"
    # in the CSV, so it has to go through float before int.
    df = pl.read_csv(csv, infer_schema_length=0)
    df = df.with_columns(pl.col("gbifID").cast(pl.Float64).cast(pl.Int64))

    names = pl.read_parquet(NAMES, columns=["gbifID", "common_name"])

    classes = (
        df.group_by("gbifID")
        .agg(
            pl.col("canonical_name").first(),
            pl.col("family").first(),
            pl.col("genus").first(),
            pl.len().alias("n_train_recordings"),
        )
        .join(names, on="gbifID", how="left")
        .sort("gbifID")                       # == MultiLabelFromFeature's ordering
        .with_row_index("label_index")
        .select(["label_index", "gbifID", "canonical_name", "common_name", "family",
                 "genus", "n_train_recordings"])
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    classes.write_parquet(out)

    print(f"{classes.height} classes over {df.height} recordings")
    print(f"  common name for {int(classes['common_name'].is_not_null().sum())} of them")
    print(f"  -> {out.relative_to(REPO)}")
    n = classes["n_train_recordings"]
    print(f"  recordings/class: min {n.min()} median {int(n.median())} max {n.max()} "
          f"| singletons {int((n == 1).sum())}")
    taxa = df.group_by("class").agg(pl.col("gbifID").n_unique().alias("classes"), pl.len())
    for row in taxa.sort("len", descending=True).iter_rows(named=True):
        print(f"  {row['class']}: {row['classes']} classes, {row['len']} recordings")


def main() -> None:
    """Parse arguments and build the requested class table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0", choices=sorted(XenoCanto.VERSIONS))
    parser.add_argument("--split", default="all", help="split the head trains on")
    args = parser.parse_args()
    build(args.version, args.split)


if __name__ == "__main__":
    main()
