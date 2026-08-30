"""Freeze the ``gbifID`` → species-name table every label space translates through.

Writes ``metadata/gbif_names.parquet``: one row per gbifID, with the scientific
name, the English common name, and the taxonomy above them. Anything that has to
show a label to a human — the notebooks, per-class metric keys, the released
model's ``xc_classes.parquet`` — reads this file rather than re-deriving names,
so the same gbifID cannot be called two different things in two places.

The ids are the union of

* the Xeno-Canto class universe of one (version, split) pair, and
* every ``class_ids`` list in ``configs/data/datasets/train/``, i.e. the BirdSet
  and BEANS task label spaces,

so a task is translatable without rebuilding anything when its config changes.

Names come from two sources, in order:

Xeno-Canto split CSV
    Already carries ``canonical_name`` and ``vernacularName`` per recording, so
    the whole XC universe resolves without leaving GCS. Where a taxon's
    recordings disagree on the vernacular name (173 of them do, mostly casing),
    the most frequent spelling wins, ties broken alphabetically — the pick has
    to be deterministic or the table churns between builds.

GBIF species API
    For the ids XC does not have (3 BirdSet species have no Xeno-Canto
    recording) and for the 436 XC taxa with no vernacular name at all — all
    insects and amphibians; every bird and mammal is named in XC.

``common_name`` stays null when neither source has one. It is *not* unique: 38
English names are shared by two gbifIDs, which is why the released model's
``id2label`` is the scientific name and this column rides alongside it.

    uv run python scripts/build_gbif_names.py
    uv run python scripts/build_gbif_names.py --version 0.2.0 --split all
"""

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import argparse
from collections import Counter
from pathlib import Path

import polars as pl
import requests
import tenacity
import yaml
from tqdm import tqdm

import bioacoustic_ssl.data.datasets  # noqa: F401  — registers the anonymous-GCS filesystem
from alp_data.datasets import XenoCanto

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "metadata" / "gbif_names.parquet"
GBIF = "https://api.gbif.org/v1/species"
SESSION = requests.Session()   # ~900 calls; a fresh TLS handshake each doubles the runtime
COLUMNS = ["gbifID", "scientific_name", "common_name", "class", "order", "family", "genus"]


def task_class_ids() -> set[int]:
    """Every class id named by a downstream task config.

    Returns:
        The union of the ``class_ids`` lists under
        ``configs/data/datasets/train/``.
    """
    ids: set[int] = set()
    for path in (REPO / "configs" / "data" / "datasets" / "train").rglob("*.yaml"):
        for split in yaml.safe_load(path.read_text()).values():
            if not isinstance(split, dict):
                continue
            for transformation in split.get("transformations", []):
                # `class_ids` is sometimes a nested `_target_` (the XC universe,
                # read from its own parquet) rather than a list of ids.
                class_ids = transformation.get("class_ids")
                if isinstance(class_ids, list):
                    ids.update(class_ids)
    return ids


def from_xeno_canto(version: str, split: str) -> pl.DataFrame:
    """One row per gbifID, named from the Xeno-Canto split CSV.

    Args:
        version: Xeno-Canto metadata version, a key of ``XenoCanto.VERSIONS``.
        split: Split whose recordings define the class universe.

    Returns:
        A frame with :data:`COLUMNS`, ``common_name`` null where XC has none.
    """
    csv = XenoCanto.VERSIONS[version]["split_paths"][split]
    print(f"reading {csv}")
    # infer_schema_length=0 -> everything as Utf8: gbifID is written as "2490719.0"
    # in the CSV, so it has to go through float before int.
    recordings = pl.read_csv(
        csv,
        infer_schema_length=0,
        columns=["gbifID", "canonical_name", "vernacularName", "class", "order", "family", "genus"],
    ).with_columns(pl.col("gbifID").cast(pl.Float64).cast(pl.Int64))

    return (
        recordings.group_by("gbifID")
        .agg(
            pl.col("canonical_name").first().alias("scientific_name"),
            pl.col("vernacularName").drop_nulls().alias("common_names"),
            pl.col("class").first(),
            pl.col("order").first(),
            pl.col("family").first(),
            pl.col("genus").first(),
        )
        .with_columns(
            pl.col("common_names")
            .map_elements(_most_common, return_dtype=pl.String)
            .alias("common_name")
        )
        .select(COLUMNS)
    )


def _most_common(names: list[str]) -> str | None:
    """The most frequent spelling, ties broken alphabetically.

    Args:
        names: Every vernacular name recorded for one taxon.

    Returns:
        The winning spelling, or ``None`` if there is nothing to pick from.
    """
    if not len(names):
        return None
    counts = Counter(names)
    return min(counts, key=lambda name: (-counts[name], name))


@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    wait=tenacity.wait_exponential(min=1, max=20),
    reraise=True,
)
def _gbif(path: str, **params: object) -> dict:
    """One GBIF species-API call, retried on transient failure.

    Args:
        path: Path under the species endpoint, e.g. ``"2490164"``.
        **params: Query parameters.

    Returns:
        The decoded JSON body.
    """
    response = SESSION.get(f"{GBIF}/{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def from_gbif(gbif_id: int) -> dict[str, object]:
    """One taxon, looked up in GBIF.

    Args:
        gbif_id: The GBIF taxon key, which is what the corpora call ``gbifID``.

    Returns:
        A :data:`COLUMNS` row. ``common_name`` is the most frequent English
        vernacular name GBIF lists, or ``None`` if it lists none.
    """
    taxon = _gbif(str(gbif_id))
    vernacular = _gbif(f"{gbif_id}/vernacularNames", limit=200)
    english = [
        entry["vernacularName"]
        for entry in vernacular.get("results", [])
        if entry.get("language") == "eng" and entry.get("vernacularName")
    ]
    return {
        "gbifID": gbif_id,
        "scientific_name": taxon.get("species") or taxon.get("canonicalName"),
        "common_name": _most_common(english),
        "class": taxon.get("class"),
        "order": taxon.get("order"),
        "family": taxon.get("family"),
        "genus": taxon.get("genus"),
    }


def build(version: str, split: str) -> None:
    """Write the name table for the XC universe plus every task label space.

    Args:
        version: Xeno-Canto metadata version, a key of ``XenoCanto.VERSIONS``.
        split: Split whose recordings define the XC class universe.
    """
    names = from_xeno_canto(version, split)
    missing_ids = sorted(task_class_ids() - set(names["gbifID"].to_list()))
    unnamed_ids = names.filter(pl.col("common_name").is_null())["gbifID"].to_list()
    print(f"{names.height} taxa from Xeno-Canto | {len(missing_ids)} task ids it does not have "
          f"| {len(unnamed_ids)} with no common name")

    rows = [from_gbif(i) for i in tqdm(missing_ids + unnamed_ids, desc="gbif", unit="taxon")]
    looked_up = pl.DataFrame(rows, schema=names.schema) if rows else names.clear()

    # A taxon Xeno-Canto knows takes only its common name from GBIF. Its
    # scientific name and taxonomy stay XC's, because those are what the label
    # spaces are built from — letting a GBIF revision rename a class here would
    # put the name table and the head's classes quietly out of step.
    table = (
        names.join(
            looked_up.select("gbifID", pl.col("common_name").alias("gbif_common_name")),
            on="gbifID",
            how="left",
        )
        .with_columns(pl.col("common_name").fill_null(pl.col("gbif_common_name")))
        .vstack(looked_up.filter(pl.col("gbifID").is_in(missing_ids)).select(names.columns)
                .with_columns(pl.lit(None, dtype=pl.String).alias("gbif_common_name")))
        .sort("gbifID")
        .select(COLUMNS)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(OUT)

    named = table["common_name"].is_not_null().sum()
    print(f"{table.height} taxa -> {OUT.relative_to(REPO)}")
    print(f"  common name for {named} ({table.height - named} without: "
          f"{table.filter(pl.col('common_name').is_null())['class'].value_counts().to_dicts()})")


def main() -> None:
    """Parse arguments and build the name table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0", choices=sorted(XenoCanto.VERSIONS))
    parser.add_argument("--split", default="all", help="split defining the XC class universe")
    args = parser.parse_args()
    build(args.version, args.split)


if __name__ == "__main__":
    main()
