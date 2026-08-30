"""Materialize the NASA Earthdata 5 s event slices to local parquet shards.

The ``BIOSCAPE_EVENTS`` / ``S2L_EVENTS`` splits of
:class:`~bioacoustic_ssl.data.datasets.NASAEarthAccess` fetch each 5 s detection
over the network at run time (~3.9 s/sample, HTTP-range read from Earthdata),
which dominates data-loading wall time during pretraining. This script downloads
and decodes every event once — resampled to 32 kHz mono — and writes the audio
plus its metadata to parquet, so training can read it locally in milliseconds.

Audio is written to a flat ``<output>.bin`` blob (concatenated raw PCM) beside
the metadata parquet; each row records its byte ``audio_offset`` and
``audio_dtype`` into that blob. Stored as ``int16`` PCM by default (lossless vs.
the 16-bit source, ~320 KB/event), or ``float32`` with ``--dtype float32``. The
loader memory-maps the ``.bin`` so DataLoader workers share one copy via the OS
page cache instead of each holding a private decompressed copy.

Sharding
--------
Records are sharded by stride (``records[shard_index::num_shards]``) so a slurm
array splits the events across tasks, one parquet per shard. Downloading
requires an Earthdata login (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` env
vars, or ``~/.netrc`` via ``scripts/earthdata_login.py``).

    uv run python scripts/materialize_nasa_events.py \
        --split BIOSCAPE_EVENTS --output curated/nasa_audio/bioscape/shard_0.parquet \
        --num-shards 20 --shard-index 0
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import argparse
import itertools
import logging
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bioacoustic_ssl.data.datasets import NASAEarthAccess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("materialize_nasa")

SAMPLE_RATE = 32_000
_SPLITS = ("BIOSCAPE_EVENTS", "S2L_EVENTS")

#: Metadata columns copied from each event record into the output parquet.
_META_KEYS = (
    "granule_ur", "data_link", "site",
    "start_seconds", "end_seconds", "top_species", "confidence",
)


def _iter_decoded(records, process_fn, prefetch: int, max_retries: int):
    """Yield ``process_fn(record)`` results, retrying transient failures.

    Keeps ``prefetch`` downloads in flight over a thread pool (fetches are
    network-latency-bound, so concurrency is the main speedup). A record that
    keeps failing is logged and skipped, not yielded.
    """
    def safe(rec):
        for attempt in range(max_retries + 1):
            try:
                return process_fn(rec)
            except Exception as exc:  # network / decode errors are transient here
                if attempt == max_retries:
                    log.warning("skipping %s after %d retries: %s",
                                rec.get("granule_ur"), max_retries, exc)
                    return None
                time.sleep(min(30.0, 2.0**attempt) + random.uniform(0, 1))

    if prefetch <= 0:
        for rec in records:
            out = safe(rec)
            if out is not None:
                yield out
        return

    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        pending: deque = deque()
        for rec in itertools.islice(records, prefetch):
            pending.append(pool.submit(safe, rec))
        for rec in records:
            pending.append(pool.submit(safe, rec))
            out = pending.popleft().result()
            if out is not None:
                yield out
        while pending:
            out = pending.popleft().result()
            if out is not None:
                yield out


def _encode_audio(audio: np.ndarray, dtype: str) -> bytes:
    """Return the mono waveform as raw little-endian bytes in ``dtype``."""
    if dtype == "int16":
        pcm = np.clip(audio, -1.0, 1.0) * 32767.0
        return pcm.astype("<i2").tobytes()
    return audio.astype("<f4").tobytes()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", choices=[*_SPLITS, "both"], default="both")
    p.add_argument("--output", type=Path, required=True, help="Destination parquet path.")
    p.add_argument("--dtype", choices=["int16", "float32"], default="int16",
                   help="Stored audio dtype (int16 halves the size, lossless vs. source).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split each split's records into this many shards.")
    p.add_argument("--shard-index", type=int, default=0, help="Which shard this task handles.")
    p.add_argument("--prefetch", type=int, default=32, help="Concurrent audio downloads.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Retries per event on transient failures before skipping.")
    p.add_argument("--flush-every", type=int, default=500,
                   help="Rewrite the parquet every N events to checkpoint progress.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bin_path = args.output.with_suffix(".bin")
    splits = list(_SPLITS) if args.split == "both" else [args.split]

    rows: list[dict] = []
    bin_f = open(bin_path, "wb")
    byte_off = 0

    def flush() -> None:
        pd.DataFrame(rows).to_parquet(args.output, index=False)
        bin_f.flush()

    kept = 0
    for split in splits:
        ds = NASAEarthAccess(split=split, sample_rate=SAMPLE_RATE, decode_audio=True)
        ds._filesystem()  # pre-warm the authenticated session before threads start
        records = (
            ds._records[args.shard_index :: args.num_shards]
            if args.num_shards > 1 else ds._records
        )
        log.info("materializing %s: %d events (shard %d/%d), prefetch=%d",
                 split, len(records), args.shard_index, args.num_shards, args.prefetch)

        for proc in _iter_decoded(iter(records), ds._process, args.prefetch, args.max_retries):
            audio = np.asarray(proc["audio"], dtype=np.float32)
            ab = _encode_audio(audio, args.dtype)
            bin_f.write(ab)
            row = {k: proc.get(k) for k in _META_KEYS}
            row["split"] = split
            row["sample_rate"] = SAMPLE_RATE
            row["num_samples"] = int(audio.shape[0])
            row["audio_offset"] = byte_off
            row["audio_dtype"] = args.dtype
            byte_off += len(ab)
            rows.append(row)
            kept += 1
            if kept % args.flush_every == 0:
                flush()
                log.info("  ... %d events written", kept)

    flush()
    bin_f.close()
    log.info("done: %d events -> %s + %s (%s)", kept, args.output, bin_path, args.dtype)


if __name__ == "__main__":
    main()
