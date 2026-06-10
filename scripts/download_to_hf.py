#!/usr/bin/env python3
"""Download audio detection clips and push to HuggingFace Hub as Parquet shards.

Each shard is built via Dataset.from_generator and uploaded directly from an
in-memory buffer (io.BytesIO) — no Parquet file ever touches disk.  The only
temporary disk usage is the Arrow cache that from_generator writes inside a
TemporaryDirectory (~400 MB per shard), which is automatically deleted after
each upload.

Usage
-----
    # Process A2O detections
    python scripts/download_to_hf.py --source a2o

    # Process Arbimon detections
    python scripts/download_to_hf.py --source arbimon --n-workers 16

    # Dry-run (skip HF upload, useful for testing)
    python scripts/download_to_hf.py --source a2o --chunk-size 20 --skip-upload

Both sources can run concurrently in separate terminals; they write to
different paths inside the same HF repo.

Authentication
--------------
A2O:
  Set A2O_AUTH_TOKEN  (or A2O_EMAIL + A2O_PASSWORD for auto-refresh).

Arbimon:
  Set AUTH0_CLIENT_ID + AUTH0_CLIENT_SECRET (machine auth),
  or run once interactively to cache credentials at ~/.rfcx_credentials.

HuggingFace Hub:
  Run  huggingface-cli login  once before uploading.

Resuming
--------
Progress is saved to .hf_upload_state/{source}/state.json after every shard.
Re-run the same command to continue from where it left off.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import logging
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Generator

import httpx
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from datasets import Audio, Dataset, Features, Value
from esp_data.io import audio_stereo_to_mono
from huggingface_hub import HfApi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project src is on sys.path when run as a script
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from soundscape_ssl.data.datasets.a2o_site import A2ODetections  # noqa: E402
from soundscape_ssl.data.datasets.arbimon import ArbimonDetections  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REPO_ID = "mwirth7/soundscape-pretrain"
DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_N_WORKERS = 8
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_DETECTIONS_DIR = "detections"

# arbimon_j6acnaodcb6t has an epoch-format issue incompatible with ArbimonDetections
SKIP_FILES: frozenset[str] = frozenset({'arbimon_j6acnaodcb6t_detections.csv',
 'a2o_site_5_detections.csv',
 'a2o_site_209_detections.csv',
 'arbimon_lqvkzhdi6q5n_detections.csv',
 'a2o_site_137_detections.csv',
 'a2o_site_197_detections.csv',
 'arbimon_s08rizdwr4j3_detections.csv',
 'arbimon_oprrj21xmfhc_detections.csv'})

# A2O media slice endpoint base URL
_A2O_API_BASE = "https://api.acousticobservatory.org"

# Clips longer than this after download indicate the full-recording fallback
# fired unexpectedly (e.g. slice endpoint returned garbage).  Such clips are
# rejected rather than encoded as oversized shards.
_MAX_CLIP_SAMPLES = 60 * 32_000  # 60 s @ 32 kHz

A2O_META_COLS: list[str] = [
    "recording_id",
    "site",
    "start_seconds",
    "end_seconds",
    "segment_mean_energy",
    "spectral_flatness",
    "spectral_sharpness",
    "low_freq_ratio",
    "temporal_cv",
    "high_freq_cv",
]

ARBIMON_META_COLS: list[str] = [
    "project_id",
    "stream_id",
    "start_seconds",
    "end_seconds",
    "segment_mean_energy",
    "spectral_flatness",
    "spectral_sharpness",
    "low_freq_ratio",
    "temporal_cv",
    "high_freq_cv",
]

A2O_FEATURES = Features(
    {
        "audio": Audio(sampling_rate=32_000),
        "source": Value("string"),
        "recording_id": Value("int64"),
        "site": Value("int32"),
        "start_seconds": Value("float64"),
        "end_seconds": Value("float64"),
        "segment_mean_energy": Value("float32"),
        "spectral_flatness": Value("float32"),
        "spectral_sharpness": Value("float32"),
        "low_freq_ratio": Value("float32"),
        "temporal_cv": Value("float32"),
        "high_freq_cv": Value("float32"),
    }
)

ARBIMON_FEATURES = Features(
    {
        "audio": Audio(sampling_rate=32_000),
        "source": Value("string"),
        "project_id": Value("string"),
        "stream_id": Value("string"),
        "start_seconds": Value("float64"),  # large absolute-offset values
        "end_seconds": Value("float64"),
        "segment_mean_energy": Value("float32"),
        "spectral_flatness": Value("float32"),
        "spectral_sharpness": Value("float32"),
        "low_freq_ratio": Value("float32"),
        "temporal_cv": Value("float32"),
        "high_freq_cv": Value("float32"),
    }
)

_README = """\
---
license: other
task_categories:
  - audio-classification
tags:
  - audio
  - soundscape
  - bioacoustics
configs:
  - config_name: default
    data_files:
      - split: a2o
        path: data/a2o/chunk-*.parquet
      - split: arbimon
        path: data/arbimon/chunk-*.parquet
---

# Soundscape Pretrain Dataset

Audio clips from the Australian Acoustic Observatory (A2O) and Arbimon / RFCx platforms,
pre-processed for self-supervised pre-training of soundscape models.

## Splits

| Split | Source | Approx. clips |
|-------|--------|---------------|
| `a2o` | Australian Acoustic Observatory | ~960 k |
| `arbimon` | Arbimon / RFCx | ~1.58 M |

## Usage

```python
from datasets import load_dataset

# Streaming (recommended for training)
ds = load_dataset("earthspecies/soundscape-pretrain", split="a2o", streaming=True)
for item in ds:
    audio = item["audio"]  # {"array": np.ndarray float32 mono, "sampling_rate": 32000}
    energy = item["segment_mean_energy"]
    break

# Full download
ds = load_dataset("earthspecies/soundscape-pretrain", split="a2o")
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio` | Audio(32 kHz) | Mono FLAC, 2–8 s at 32 kHz |
| `source` | string | `"a2o"` or `"arbimon"` |
| `start_seconds` / `end_seconds` | float64 | Clip offsets (s) within the recording |
| `segment_mean_energy` | float32 | Mean log energy of the segment |
| `spectral_flatness` | float32 | Wiener entropy / spectral flatness |
| `spectral_sharpness` | float32 | Spectral sharpness (high-frequency tilt) |
| `low_freq_ratio` | float32 | Fraction of energy in low frequencies |
| `temporal_cv` | float32 | Coefficient of variation of temporal energy |
| `high_freq_cv` | float32 | Coefficient of variation of high-freq energy |
| `recording_id` | int64 | **A2O only**: A2O recording ID |
| `site` | int32 | **A2O only**: A2O site ID |
| `project_id` | string | **Arbimon only**: Arbimon project ID |
| `stream_id` | string | **Arbimon only**: Arbimon stream ID |
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress verbose third-party loggers
for _noisy in ("httpx", "httpcore", "datasets", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Per-row download helper
# ---------------------------------------------------------------------------


def _encode_flac(audio: np.ndarray, sr: int) -> bytes:
    """Encode a float32 mono array to 16-bit FLAC bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _a2o_process_one(
    row: dict[str, Any],
    ds_instance: A2ODetections,
    meta_cols: list[str],
    sample_rate: int,
) -> dict[str, Any] | None:
    """Download one A2O clip via the media-slice endpoint only.

    ``A2ODetections._process_detection`` has a full-recording fallback: when
    the ``media.flac`` slice endpoint returns a non-audio response it downloads
    the entire (potentially multi-hour) recording and slices it in memory.
    With 32 concurrent threads this causes OOM.

    This function calls the slice endpoint directly and returns ``None`` on any
    failure — no fallback, no multi-GB downloads.
    """
    rec_id = int(row["recording_id"])
    start_s = round(float(row["start_seconds"]), 2)
    end_s = round(float(row["end_seconds"]), 2)

    try:
        resp = ds_instance._get(
            f"{_A2O_API_BASE}/audio_recordings/{rec_id}/media.flac",
            params={"start_offset": start_s, "end_offset": end_s},
            follow_redirects=True,
        )
        ct = resp.headers.get("content-type", "")
        if len(resp.content) < 64 or not ct.startswith(("audio/", "application/octet-stream")):
            log.debug(
                "A2O rec %s [%.2f–%.2f]: invalid slice response (%s, %d B)",
                rec_id, start_s, end_s, ct, len(resp.content),
            )
            return None

        audio, sr = torchaudio.load(io.BytesIO(resp.content))
        if audio.shape[-1] == 0:
            return None

        audio = audio.numpy().astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        # Guard: reject unexpectedly long clips (indicates a bad response slipped through)
        if len(audio) > _MAX_CLIP_SAMPLES:
            log.debug("A2O rec %s: clip too long (%d samples), skipping", rec_id, len(audio))
            return None

        if sr != sample_rate:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=sample_rate, res_type="kaiser_best")
            sr = sample_rate

    except httpx.HTTPStatusError as exc:
        log.debug("A2O rec %s HTTP %s", rec_id, exc.response.status_code)
        return None
    except Exception as exc:
        log.debug("A2O rec %s failed (%s): %s", rec_id, type(exc).__name__, exc)
        return None

    flac = _encode_flac(audio, sr)
    record: dict[str, Any] = {"audio": {"bytes": flac, "path": None}, "source": "a2o"}
    for col in meta_cols:
        record[col] = row.get(col)
    return record


def _process_one(
    row: dict[str, Any],
    ds_instance: A2ODetections | ArbimonDetections,
    source: str,
    meta_cols: list[str],
    sample_rate: int,
) -> dict[str, Any] | None:
    """Download and encode one audio clip.

    Dispatches to a source-specific helper:
    - A2O uses a slice-only path (no full-recording fallback) to bound memory use.
    - Arbimon uses ``_process_detection`` directly (its segments are ≤ 1 min, safe).

    Returns ``None`` on any failure so the shard can continue without that clip.
    """
    if source == "a2o":
        return _a2o_process_one(row, ds_instance, meta_cols, sample_rate)  # type: ignore[arg-type]

    # Arbimon: each segment is a 1-minute FLAC (~3–5 MB), safe to load in full.
    try:
        item = ds_instance._process_detection(row)
        if item is None:
            return None
        flac = _encode_flac(item["audio"], sample_rate)
        record: dict[str, Any] = {
            "audio": {"bytes": flac, "path": None},
            "source": source,
        }
        for col in meta_cols:
            record[col] = row.get(col)
        return record
    except Exception as exc:
        log.debug("Arbimon row skipped (%s): %s", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Shard generator
# ---------------------------------------------------------------------------


def shard_generator(
    chunk_rows: list[dict[str, Any]],
    source: str,
    meta_cols: list[str],
    sample_rate: int,
    n_workers: int,
    chunk_id: int,
    total_chunks: int,
) -> Generator[dict[str, Any], None, None]:
    """Generator for one Parquet shard.

    Creates its own dataset instance (one auth handshake per shard) and
    uses a ``ThreadPoolExecutor`` with a sliding-window deque to keep
    ``n_workers`` downloads in-flight at all times.

    Intended to be passed directly to ``Dataset.from_generator``.

    Parameters
    ----------
    chunk_rows:
        Detection rows for this shard (list of dicts from the CSV).
    source:
        ``"a2o"`` or ``"arbimon"``.
    meta_cols:
        CSV column names to forward to each output record.
    sample_rate:
        Target sample rate in Hz (audio is resampled on download).
    n_workers:
        Number of parallel download threads.
    chunk_id:
        Zero-based shard index (used for the progress-bar label).
    total_chunks:
        Total number of shards (used for the progress-bar label).
    """
    # Auth is cheap: reads cached token / env var.  Creating a new instance
    # per shard avoids passing the unpicklable httpx.Client in gen_kwargs.
    sample = pd.DataFrame([chunk_rows[0]])
    if source == "a2o":
        ds = A2ODetections(detections=sample, sample_rate=sample_rate)
    else:
        ds = ArbimonDetections(detections=sample, sample_rate=sample_rate)

    buffer_size = n_workers * 2
    pbar = tqdm(
        total=len(chunk_rows),
        desc=f"Shard {chunk_id + 1:>4}/{total_chunks}",
        unit="clip",
        leave=False,
        dynamic_ncols=True,
        disable=True
    )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending: deque = deque()
        rows_iter = iter(chunk_rows)

        def _submit(row: dict[str, Any]):
            return executor.submit(_process_one, row, ds, source, meta_cols, sample_rate)

        # Pre-fill sliding window
        for row in itertools.islice(rows_iter, buffer_size):
            pending.append(_submit(row))

        # Slide: submit one new future, yield one completed result
        for row in rows_iter:
            pending.append(_submit(row))
            result = pending.popleft().result()
            pbar.update(1)
            if result is not None:
                yield result

        # Drain remaining futures
        while pending:
            result = pending.popleft().result()
            pbar.update(1)
            if result is not None:
                yield result

    pbar.close()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _load_all_csvs(detections_dir: Path, source: str) -> pd.DataFrame:
    """Load and concatenate all detection CSVs for *source* into one DataFrame."""
    if source == "a2o":
        return pd.read_csv(detections_dir / "a2o_filtered.csv")
    else:
        return pd.read_csv(detections_dir / "arbimon_filtered.csv")


# ---------------------------------------------------------------------------
# State / checkpoint helpers
# ---------------------------------------------------------------------------


def _load_state(state_dir: Path) -> int:
    """Return ``next_row`` from *state_dir*/state.json, or 0 if not found."""
    state_file = state_dir / "state.json"
    if state_file.exists():
        data = json.loads(state_file.read_text())
        return int(data.get("next_row", 0))
    return 0


def _save_state(state_dir: Path, next_row: int) -> None:
    """Persist ``next_row`` to *state_dir*/state.json."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text(json.dumps({"next_row": next_row}, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["a2o", "arbimon"],
        help="Dataset source to process.",
    )
    parser.add_argument(
        "--detections-dir",
        default=DEFAULT_DETECTIONS_DIR,
        help="Directory containing detection CSV files.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="HuggingFace Hub repository ID.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of detection rows per Parquet shard.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help="Number of parallel audio-download threads per shard.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target audio sample rate in Hz.",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "Directory for state.json checkpoint file. "
            "Defaults to .hf_upload_state/{source}/."
        ),
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help=(
            "Build each shard but do not upload to HF Hub. "
            "Useful for smoke-testing without network writes."
        ),
    )
    args = parser.parse_args()

    source: str = args.source
    detections_dir = Path(args.detections_dir)
    repo_id: str = args.repo_id
    chunk_size: int = args.chunk_size
    n_workers: int = args.n_workers
    sample_rate: int = args.sample_rate
    state_dir = (
        Path(args.state_dir) if args.state_dir else Path(f".hf_upload_state/{source}")
    )
    features = A2O_FEATURES if source == "a2o" else ARBIMON_FEATURES
    meta_cols = A2O_META_COLS if source == "a2o" else ARBIMON_META_COLS

    log.info("=" * 60)
    log.info("download_to_hf.py  source=%s  repo=%s", source, repo_id)
    log.info("chunk_size=%d  n_workers=%d  sample_rate=%d Hz", chunk_size, n_workers, sample_rate)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load all detection CSVs
    # ------------------------------------------------------------------
    log.info("Loading detection CSVs from %s ...", detections_dir)
    all_df = _load_all_csvs(detections_dir, source)
    total_rows = len(all_df)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    # ------------------------------------------------------------------
    # 2. Load checkpoint
    # ------------------------------------------------------------------
    next_row = _load_state(state_dir)
    start_chunk = next_row // chunk_size
    remaining_chunks = total_chunks - start_chunk

    if next_row > 0:
        log.info(
            "Resuming: skipping %d completed rows (%d shards done, %d remaining).",
            next_row, start_chunk, remaining_chunks,
        )
    else:
        log.info("Starting fresh: %d rows → %d shards.", total_rows, total_chunks)

    # ------------------------------------------------------------------
    # 3. Set up HuggingFace API
    # ------------------------------------------------------------------
    api = HfApi()
    if not args.skip_upload:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        log.info("HF repo ready: https://huggingface.co/datasets/%s", repo_id)

    # ------------------------------------------------------------------
    # 4. Process shards
    # ------------------------------------------------------------------
    run_start = time.time()
    global_ok = 0
    global_skipped = 0

    outer_pbar = tqdm(
        total=total_chunks,
        initial=start_chunk,
        desc=f"Total shards ({source})",
        unit="shard",
        dynamic_ncols=True,
    )

    for chunk_start in range(next_row, total_rows, chunk_size):
        chunk_id = chunk_start // chunk_size
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_rows = all_df.iloc[chunk_start:chunk_end].to_dict("records")

        shard_start_t = time.time()

        # Use a fresh TemporaryDirectory as cache_dir so the Arrow cache
        # (~400 MB) is automatically deleted when the context exits.
        with tempfile.TemporaryDirectory(prefix=f"hf_shard_{chunk_id}_") as tmp_cache:
            shard_ds = Dataset.from_generator(
                shard_generator,
                gen_kwargs={
                    "chunk_rows": chunk_rows,
                    "source": source,
                    "meta_cols": meta_cols,
                    "sample_rate": sample_rate,
                    "n_workers": n_workers,
                    "chunk_id": chunk_id,
                    "total_chunks": total_chunks,
                },
                features=features,
                cache_dir=tmp_cache,
            )

            n_ok = len(shard_ds)
            n_skipped = len(chunk_rows) - n_ok
            global_ok += n_ok
            global_skipped += n_skipped

            if n_ok == 0:
                log.warning("Shard %d: all %d rows failed — skipping upload.", chunk_id, len(chunk_rows))
            elif not args.skip_upload:
                # Write Parquet to an in-memory buffer (no file on disk)
                buf = io.BytesIO()
                shard_ds.to_parquet(buf)
                buf.seek(0)
                parquet_mb = buf.getbuffer().nbytes / 1e6

                api.upload_file(
                    path_or_fileobj=buf,
                    path_in_repo=f"data/{source}/chunk-{chunk_id:06d}.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Add {source} shard {chunk_id:06d} ({n_ok} clips)",
                )

                elapsed = time.time() - shard_start_t
                log.info(
                    "Shard %d/%d: %d clips | %d skipped | %.1f MB | %.1fs",
                    chunk_id + 1, total_chunks, n_ok, n_skipped, parquet_mb, elapsed,
                )
            else:
                log.info(
                    "Shard %d/%d: %d clips | %d skipped [upload skipped]",
                    chunk_id + 1, total_chunks, n_ok, n_skipped,
                )
        # TemporaryDirectory exits here → Arrow cache auto-deleted

        # Save checkpoint after each successful shard
        _save_state(state_dir, chunk_end)
        outer_pbar.update(1)

    outer_pbar.close()

    # ------------------------------------------------------------------
    # 5. Upload README.md (once, on first run)
    # ------------------------------------------------------------------
    if not args.skip_upload:
        readme_exists = api.file_exists(
            path_in_repo="README.md", repo_id=repo_id, repo_type="dataset"
        )
        if not readme_exists:
            api.upload_file(
                path_or_fileobj=_README.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add dataset card",
            )
            log.info("README.md uploaded.")
        else:
            log.info("README.md already present, skipping.")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - run_start
    log.info("=" * 60)
    log.info(
        "Done!  %d clips uploaded  |  %d skipped  |  %.1f min total",
        global_ok, global_skipped, total_elapsed / 60,
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
