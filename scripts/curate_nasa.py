"""Curate the NASA Earthdata acoustic collections with AudioProtoPNet.

Streams the per-recording WAVs of the ``BIOSCAPE`` and ``S2L`` splits of
:class:`~bioacoustic_ssl.data.datasets.NASAEarthAccess`, windows each recording
into fixed-length (default 5 s) segments, runs the
``DBD-research-group/AudioProtoPNet-20-BirdSet-XCL`` classifier, and writes the
segments whose top species probability clears a threshold (default 0.4) to a
parquet — keeping the ``confidence`` and predicted ``top_species`` per segment
(plus the granule it came from and the in-recording offset).

Dependencies
------------
The model's remote code is incompatible with the project's pinned
``transformers>=5.6.1``; run with the model-compatible set pinned inline so the
project environment is untouched::

    uv run --with "transformers==4.44.2" --with "huggingface_hub<1.0" \
        scripts/curate_nasa.py --split both --output curated_nasa.parquet

Downloading the audio requires an Earthdata login (``EARTHDATA_USERNAME`` /
``EARTHDATA_PASSWORD`` env vars, or ``~/.netrc`` via
``scripts/earthdata_login.py``).
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import argparse
import itertools
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification

from bioacoustic_ssl.data.datasets import NASAEarthAccess

MODEL_ID = "DBD-research-group/AudioProtoPNet-20-BirdSet-XCL"
SAMPLE_RATE = 32_000

_SPLITS = ("BIOSCAPE", "S2L")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(device: str):
    """Load the AudioProtoPNet model and its feature extractor."""
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model.eval().to(device)
    id2label = model.config.id2label
    return model, feature_extractor, id2label


@torch.no_grad()
def classify(model, feature_extractor, clips: list[np.ndarray], device: str) -> torch.Tensor:
    """Return per-species sigmoid probabilities for a batch of clips.

    Parameters
    ----------
    clips : list[np.ndarray]
        Mono float32 waveforms at ``SAMPLE_RATE``. The feature extractor
        pads/truncates each to its fixed 5 s window internally.

    Returns
    -------
    torch.Tensor
        ``(len(clips), num_species)`` probabilities on CPU.
    """
    feats = feature_extractor(clips).to(device)
    outputs = model(feats)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return torch.sigmoid(logits).float().cpu()


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def iter_windows(audio: np.ndarray, win: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start_sample, window)`` for non-overlapping full windows.

    A trailing fragment shorter than ``win`` is dropped.
    """
    for start in range(0, len(audio) - win + 1, win):
        yield start, audio[start : start + win]


# ---------------------------------------------------------------------------
# Download / retry
# ---------------------------------------------------------------------------


def _process_with_retry(process_fn: Callable, item: Any, label: str, max_retries: int):
    """Run ``process_fn(item)``, retrying transient network failures.

    Downloads occasionally fail with a transport error (e.g. a truncated
    body — ``RemoteProtocolError`` — or a read timeout / 5xx) when the data
    server drops a connection mid-transfer. Those are retried with backoff;
    a persistent failure, or any non-network error (e.g. a decode failure on
    corrupt bytes), returns ``None`` so the caller skips the item instead of
    crashing the whole job.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return process_fn(item)
        except httpx.HTTPError as e:
            print(f"[retry {attempt}/{max_retries}] {label}: "
                  f"{type(e).__name__}: {str(e)[:90]}", flush=True)
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
        except Exception as e:  # noqa: BLE001 — non-network failure; skip this item
            print(f"[skip] {label}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            return None
    print(f"[skip] {label}: failed after {max_retries} attempts", flush=True)
    return None


def _iter_processed(metadata_iter: Iterator, process_fn: Callable,
                    label_fn: Callable[[Any], str], prefetch: int, max_retries: int):
    """Yield successfully processed items, retrying/skipping failures.

    ``prefetch > 0`` keeps that many downloads in flight (preserving the
    loaders' concurrency); ``0`` processes sequentially. Items that fail
    persistently are dropped, not yielded.
    """
    def safe(item: Any):
        return _process_with_retry(process_fn, item, label_fn(item), max_retries)

    if prefetch <= 0:
        for item in metadata_iter:
            result = safe(item)
            if result is not None:
                yield result
        return

    with ThreadPoolExecutor(max_workers=prefetch) as executor:
        pending: deque = deque()
        for item in itertools.islice(metadata_iter, prefetch):
            pending.append(executor.submit(safe, item))
        for item in metadata_iter:
            pending.append(executor.submit(safe, item))
            result = pending.popleft().result()
            if result is not None:
                yield result
        while pending:
            result = pending.popleft().result()
            if result is not None:
                yield result


# ---------------------------------------------------------------------------
# Curation loop
# ---------------------------------------------------------------------------


def make_row(base: dict[str, Any], start_sample: int, win: int, probs_row: torch.Tensor,
             id2label: dict[int, str], top_k: int) -> dict[str, Any]:
    """Build one output parquet row from a window and its probabilities."""
    topk_scores, topk_idx = torch.topk(probs_row, k=top_k)
    topk_idx = topk_idx.tolist()
    topk_scores = topk_scores.tolist()

    row = dict(base)
    row["start_seconds"] = start_sample / SAMPLE_RATE
    row["end_seconds"] = (start_sample + win) / SAMPLE_RATE
    row["confidence"] = topk_scores[0]
    row["top_index"] = topk_idx[0]
    row["top_species"] = id2label[topk_idx[0]]
    row["topk_species"] = [id2label[i] for i in topk_idx]
    row["topk_scores"] = topk_scores
    return row


def curate(recordings, model, feature_extractor, id2label, *, threshold: float,
           segment_seconds: float, batch_size: int, top_k: int, device: str,
           output: Path, flush_every: int) -> None:
    win = int(round(segment_seconds * SAMPLE_RATE))
    rows: list[dict[str, Any]] = []
    kept = 0
    seen = 0

    def flush() -> None:
        if rows:
            pl.DataFrame(rows).write_parquet(output)

    pbar = tqdm(recordings, desc="recordings", unit="rec")
    for rec_idx, (audio, base) in enumerate(pbar, start=1):
        windows = list(iter_windows(audio, win))
        if not windows:
            continue
        for b in range(0, len(windows), batch_size):
            batch = windows[b : b + batch_size]
            clips = [w for _, w in batch]
            probs = classify(model, feature_extractor, clips, device)
            seen += len(batch)
            for (start_sample, _), probs_row in zip(batch, probs):
                if float(probs_row.max()) >= threshold:
                    rows.append(make_row(base, start_sample, win, probs_row, id2label, top_k))
                    kept += 1
        pbar.set_postfix(kept=kept, seen=seen)
        if rec_idx % flush_every == 0:
            flush()

    flush()
    print(f"Done. Kept {kept}/{seen} segments (threshold={threshold}) -> {output}")


def iter_nasa(splits, shard_index: int, num_shards: int, prefetch: int, max_retries: int):
    """Yield ``(audio, base_meta)`` recordings for each requested NASA split.

    Records are sharded by stride (``records[shard_index::num_shards]``) so a
    slurm array can split the ~10^6 granules across tasks. Downloads run with
    ``prefetch`` concurrency and transient failures are retried/skipped (see
    :func:`_iter_processed`).
    """
    for split in splits:
        ds = NASAEarthAccess(split=split, sample_rate=SAMPLE_RATE, decode_audio=True)
        ds._filesystem()  # pre-warm the authenticated session before threads start
        records = ds._records[shard_index::num_shards] if num_shards > 1 else ds._records

        def label(rec: dict, s: str = split) -> str:
            return f"nasa {s} {rec.get('granule_ur')}"

        for proc in _iter_processed(iter(records), ds._process, label, prefetch, max_retries):
            base = {
                "source": "nasa",
                "split": split,
                "granule_ur": proc.get("granule_ur"),
                "data_link": proc.get("data_link"),
                "site": proc.get("site"),
                "temporal_start": proc.get("temporal_start"),
            }
            yield proc["audio"], base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", choices=[*_SPLITS, "both"], default="both")
    p.add_argument("--output", type=Path, required=True, help="Destination parquet path.")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="Keep a segment if its top species probability >= this value.")
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=1,
                   help="Number of top species to record per kept segment.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split each split's records into this many shards.")
    p.add_argument("--shard-index", type=int, default=0, help="Which shard this task handles.")
    p.add_argument("--prefetch", type=int, default=8,
                   help="Concurrent audio downloads.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Retries per recording on transient download failures before skipping.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flush-every", type=int, default=50,
                   help="Rewrite the parquet every N recordings to checkpoint progress.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    splits = list(_SPLITS) if args.split == "both" else [args.split]
    model, feature_extractor, id2label = load_model(args.device)

    recordings = iter_nasa(
        splits, args.shard_index, args.num_shards, args.prefetch, args.max_retries
    )
    curate(
        recordings, model, feature_extractor, id2label,
        threshold=args.threshold,
        segment_seconds=args.segment_seconds,
        batch_size=args.batch_size,
        top_k=args.top_k,
        device=args.device,
        output=args.output,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
