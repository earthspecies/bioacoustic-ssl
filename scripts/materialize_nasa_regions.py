"""Materialize NASA Earthdata *detection regions* to local shards.

Supersedes the fixed 5 s slices of ``scripts/materialize_nasa_events.py``. That
store holds one 5 s window per ``>=0.7`` detection (216 h over 69,369 granules),
so a 400k-step run at 25 % PAM weight re-draws each identical window ~330x —
against ~19x for XenoCanto, whose crop offset is redrawn every epoch. This script
instead stores a *contiguous region* around the detections of a granule, and the
loader draws a fresh random 5 s crop inside it at every access
(``random_crop_seconds``), which is what removes the repetition asymmetry.

Region rule (all thresholds are CLI flags)
-----------------------------------------
1. **Seed granules** — keep only granules holding at least one detection at
   ``--seed-threshold`` (0.7).  This preserves the curated set exactly: the same
   69,369 granules the current ``*_EVENTS_LOCAL`` store was built from, so a
   pretraining run against this store differs from that one *only* in windowing.
2. **Regions** — inside those granules take every detection at ``--threshold``
   (0.4) and merge the overlapping/adjacent ones.  Merging is what keeps a dense
   dawn chorus from entering the index ten times over (see "Sampling" below).
3. **Lone detections** — a region built from a single detection is only 5 s and
   would admit no crop jitter at all, so it is widened by ``--lone-margin``
   (±2.5 s), shifted inwards where it would leave the granule.  Widening can
   close a 5 s gap to a neighbour, so regions are merged **again** afterwards.

At the defaults this yields ~136.5k regions / ~932 h / ~215 GB int16 @32 kHz —
81 % of each curated minute, dropping only what no detector fired on at all.
Run with ``--dry-run`` first: it prints the plan (regions, hours, GB, per-split
breakdown) from the local detection parquets without downloading anything.

Sampling
--------
One index row per region, so training draws **uniformly over regions**: every
detection event gets equal weight regardless of how many seconds surround it.
That is the point of step 2 — a length-weighted draw would restore exactly the
oversampling of connected areas that merging removes.  The per-detection offsets
and confidences are carried per row (``det_starts`` / ``det_confs``), so a
confidence- or peak-biased crop rule can be added later without re-materializing.

Layout & compatibility
----------------------
Writes the same flat layout the loader already reads — a ``shard_*.bin`` blob of
concatenated raw PCM beside a ``shard_*.parquet`` index whose rows carry
``num_samples`` / ``audio_offset`` / ``audio_dtype`` — so
:class:`~soundscape_ssl.data.datasets.NASAEarthAccess` needs no new read path,
only the ``BIOSCAPE_REGIONS`` / ``S2L_REGIONS`` split names.  Rows are variable
length here (10–60 s) where the event store was a fixed 5 s.

Sharding is **by granule**, not by region, so a granule holding several regions
is downloaded exactly once (whole-granule reads beat HTTP-range reads on these
~5.76 MB latency-bound files).

    # plan only, no network
    uv run python scripts/materialize_nasa_regions.py --dry-run

    # one shard of an array job
    uv run python scripts/materialize_nasa_regions.py \
        --split BIOSCAPE --output curated/nasa_audio/BIOSCAPE_REGIONS/shard_0.parquet \
        --num-shards 40 --shard-index 0
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import argparse
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("materialize_nasa_regions")

SAMPLE_RATE = 32_000
GRANULE_SECONDS = 60.0  # every granule in both collections is exactly 60 s
_CURATED_DIR = Path(__file__).resolve().parents[1] / "curated" / "nasa"
_SPLITS = ("BIOSCAPE", "S2L")

#: Detection columns needed to build regions and to label them.
_DET_COLS = [
    "granule_ur", "data_link", "site",
    "start_seconds", "end_seconds", "confidence", "top_species",
]


def load_detections(split: str) -> pd.DataFrame:
    """Return every curated detection of a collection (all confidences >=0.4)."""
    shards = sorted((_CURATED_DIR / split).glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No detection shards under {_CURATED_DIR / split}. Run scripts/curate_nasa.py first."
        )
    return pd.concat(
        [pd.read_parquet(s, columns=_DET_COLS) for s in shards], ignore_index=True
    )


def _merge(rows: list[dict]) -> list[dict]:
    """Merge overlapping/touching regions of one granule, unioning their members.

    ``rows`` must be sorted by ``start``.  Two regions are merged when the next
    one starts at or before the current one's end, so detections that lie side by
    side collapse into a single region (and a single index row).
    """
    out: list[dict] = []
    for r in rows:
        if out and r["start"] <= out[-1]["end"]:
            prev = out[-1]
            prev["end"] = max(prev["end"], r["end"])
            prev["members"].extend(r["members"])
        else:
            out.append(dict(r))
    return out


def build_regions(
    det: pd.DataFrame,
    seed_threshold: float,
    threshold: float,
    lone_margin: float,
    duration: float = GRANULE_SECONDS,
) -> pd.DataFrame:
    """Return one row per merged detection region.

    Implements the three-step rule in the module docstring: seed granules by
    ``seed_threshold``, merge all detections above ``threshold`` inside them,
    then widen single-detection regions by ``lone_margin`` and merge once more.
    """
    seed = det.loc[det.confidence >= seed_threshold, "granule_ur"].unique()
    sub = det[det.granule_ur.isin(set(seed)) & (det.confidence >= threshold)]
    sub = sub.sort_values(["granule_ur", "start_seconds"], kind="stable")

    regions: list[dict] = []
    for granule, g in sub.groupby("granule_ur", sort=False):
        starts = g.start_seconds.to_numpy(float)
        ends = g.end_seconds.to_numpy(float)
        confs = g.confidence.to_numpy(float)
        species = g.top_species.to_numpy(object)
        merged = _merge([
            {"start": s, "end": e, "members": [i]}
            for i, (s, e) in enumerate(zip(starts, ends))
        ])

        # Widen lone detections so every region admits a jittered crop. Clipping
        # at a granule edge would leave <2*lone_margin of slack, so shift the
        # window inwards instead and keep the full width.
        want = 2 * lone_margin
        for r in merged:
            if len(r["members"]) > 1:
                continue
            r["start"] -= lone_margin
            r["end"] += lone_margin
            if r["start"] < 0.0:
                r["end"] = min(duration, r["end"] - r["start"])
                r["start"] = 0.0
            if r["end"] > duration:
                r["start"] = max(0.0, r["start"] - (r["end"] - duration))
                r["end"] = duration
            assert r["end"] - r["start"] >= min(duration, 5.0 + want) - 1e-9

        # Widening can close a 5 s gap between two lone detections.
        for r in _merge(sorted(merged, key=lambda x: x["start"])):
            m = np.asarray(r["members"], dtype=int)
            best = m[confs[m].argmax()]
            regions.append({
                "granule_ur": granule,
                "data_link": g.data_link.iloc[0],
                "site": g.site.iloc[0],
                "start_seconds": r["start"],
                "end_seconds": r["end"],
                "top_species": species[best],
                "confidence": float(confs[best]),
                "n_detections": len(m),
                # offsets relative to the region start, so a later crop rule can
                # bias towards the detections without re-reading the parquets
                "det_starts": np.round(starts[m] - r["start"], 3).astype("float32"),
                "det_confs": confs[m].astype("float32"),
            })
    return pd.DataFrame(regions)


def plan_summary(regions: pd.DataFrame, dtype: str) -> str:
    """One-line size/repetition summary of a region set."""
    width = 2 if dtype == "int16" else 4
    length = (regions.end_seconds - regions.start_seconds).to_numpy()
    hours = length.sum() / 3600
    gb = length.sum() * SAMPLE_RATE * width / 1e9
    lone = (regions.n_detections == 1).sum()
    return (
        f"{len(regions):,} regions over {regions.granule_ur.nunique():,} granules | "
        f"{hours:,.0f} h | {gb:,.0f} GB {dtype} | region len med {np.median(length):.0f}s "
        f"mean {length.mean():.1f}s | {lone:,} from a lone detection"
    )


def _fetch_granule(fs, data_link: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """Download one granule whole and return it mono at ``sample_rate``.

    The whole 60 s file is read in a single request rather than range-reading each
    region: these granules are ~5.76 MB and latency-bound, so one request beats
    several partial ones (measured ~0.6x for partial reads), and a granule often
    holds more than one region anyway.  Resampling happens once for the whole
    granule so region slices carry no boundary artefacts.
    """
    with fs.open(data_link) as fh:
        audio, sr = sf.read(fh, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    if sr != sample_rate:
        audio = librosa.resample(
            y=audio, orig_sr=sr, target_sr=sample_rate, scale=True, res_type="soxr_hq"
        )
    return audio, sample_rate


def _encode(audio: np.ndarray, dtype: str) -> bytes:
    """Return the waveform as raw little-endian bytes (matches the event store)."""
    if dtype == "int16":
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return audio.astype("<f4").tobytes()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", choices=[*_SPLITS, "both"], default="both")
    p.add_argument("--output", type=Path, help="Destination parquet path (not needed for --dry-run).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the region plan from the local parquets and exit (no network).")
    p.add_argument("--seed-threshold", type=float, default=0.7,
                   help="A granule is kept only if it holds a detection this confident.")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="Detections at or above this confidence define the regions.")
    p.add_argument("--lone-margin", type=float, default=2.5,
                   help="Seconds added each side of a region built from one detection.")
    p.add_argument("--dtype", choices=["int16", "float32"], default="int16")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Granules are split across this many shards (one parquet each).")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--prefetch", type=int, default=32, help="Concurrent granule downloads.")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--limit-granules", type=int, default=-1,
                   help="Smoke test: stop after this many granules of the shard.")
    p.add_argument("--flush-every", type=int, default=200,
                   help="Rewrite the index parquet every N granules to checkpoint progress.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = list(_SPLITS) if args.split == "both" else [args.split]

    plans: dict[str, pd.DataFrame] = {}
    for split in splits:
        regions = build_regions(
            load_detections(split), args.seed_threshold, args.threshold, args.lone_margin
        )
        regions["split"] = split
        plans[split] = regions
        log.info("%s: %s", split, plan_summary(regions, args.dtype))
    if len(plans) > 1:
        log.info("TOTAL: %s", plan_summary(pd.concat(plans.values(), ignore_index=True), args.dtype))

    if args.dry_run:
        return
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is given")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bin_path = args.output.with_suffix(".bin")
    # Import here so --dry-run needs neither earthaccess nor credentials.
    import earthaccess
    from soundscape_ssl.data.datasets.nasa_earthaccess import _ensure_login

    _ensure_login()
    fs = earthaccess.get_fsspec_https_session()

    rows: list[dict] = []
    byte_off = 0
    bin_f = open(bin_path, "wb")

    def flush() -> None:
        pd.DataFrame(rows).to_parquet(args.output, index=False)
        bin_f.flush()

    def fetch(item):
        """(granule, its regions) -> (regions, audio); None if it keeps failing."""
        granule, g = item
        for attempt in range(args.max_retries + 1):
            try:
                return g, _fetch_granule(fs, g.data_link.iloc[0], SAMPLE_RATE)[0]
            except Exception as exc:
                if attempt == args.max_retries:
                    log.warning("skipping %s after %d retries: %s", granule, args.max_retries, exc)
                    return None
                time.sleep(min(30.0, 2.0**attempt) + random.uniform(0, 1))

    n_granules = n_regions = 0
    for split, regions in plans.items():
        groups = [(k, v) for k, v in regions.groupby("granule_ur", sort=True)]
        groups = groups[args.shard_index :: args.num_shards] if args.num_shards > 1 else groups
        if args.limit_granules != -1:
            groups = groups[: args.limit_granules]
        log.info("materializing %s: %d granules / %d regions (shard %d/%d), prefetch=%d",
                 split, len(groups), sum(len(g) for _, g in groups),
                 args.shard_index, args.num_shards, args.prefetch)

        with ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as pool:
            for out in pool.map(fetch, groups):
                if out is None:
                    continue
                g, audio = out
                for row in g.to_dict(orient="records"):
                    a = int(round(row["start_seconds"] * SAMPLE_RATE))
                    b = min(int(round(row["end_seconds"] * SAMPLE_RATE)), audio.shape[0])
                    clip = audio[a:b]
                    if clip.size == 0:
                        log.warning("empty region %s [%.1f-%.1f]s; skipping",
                                    row["granule_ur"], row["start_seconds"], row["end_seconds"])
                        continue
                    raw = _encode(clip, args.dtype)
                    bin_f.write(raw)
                    row["sample_rate"] = SAMPLE_RATE
                    row["num_samples"] = int(clip.shape[0])
                    row["audio_offset"] = byte_off
                    row["audio_dtype"] = args.dtype
                    byte_off += len(raw)
                    rows.append(row)
                    n_regions += 1
                n_granules += 1
                if n_granules % args.flush_every == 0:
                    flush()
                    log.info("  ... %d granules / %d regions (%.1f GB)",
                             n_granules, n_regions, byte_off / 1e9)

    flush()
    bin_f.close()
    log.info("done: %d granules / %d regions -> %s + %s (%.1f GB, %s)",
             n_granules, n_regions, args.output, bin_path, byte_off / 1e9, args.dtype)


if __name__ == "__main__":
    main()
