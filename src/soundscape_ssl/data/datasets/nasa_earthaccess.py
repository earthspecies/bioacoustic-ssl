"""NASA Earthdata acoustic collections (via the ``earthaccess`` package).

Two NASA Earthdata passive-acoustic collections are exposed, each as a named
**split**:

- ``BIOSCAPE`` — BioSCape, Cape Floristic region
  (short_name ``Acoustic_Data_Cape_Floristic_2372``).
- ``S2L`` — Soundscapes to Landscapes, Sonoma County, CA
  (short_name ``Acoustic_Data_SonomaCounty_CA_2341``).

Each collection additionally has a curated **event split** (``BIOSCAPE_EVENTS``,
``S2L_EVENTS``): one row per 5 s detection produced by ``scripts/curate_nasa.py``
(with ``top_species`` / ``confidence`` and the in-recording offset).  For these
splits :meth:`~NASAEarthAccess._process` partial-reads only the event's 5 s slice
from its WAV (via an HTTP range request) so a recording shared by several events
is not downloaded in full.

Each collection's granules are mostly **individual ~5 MB WAV recordings** (one
granule per file, e.g. ``s2lam081_230605_2023-06-05_12-30.WAV``); the first few
granules are a site-photos ``.zip`` and a sites ``.csv``.

:meth:`~NASAEarthAccess._load` reads one flat record per granule from a
precomputed metadata CSV per split (under ``metadata/``).  Non-audio granules
(site-photo ``.zip``, sites ``.csv``) are always dropped.  With
``decode_audio=True`` (the default), :meth:`~NASAEarthAccess._process` then
downloads and decodes the full WAV for each granule (resampling to
``sample_rate`` if set, exactly as alp_data does).  Downloading the protected
files *does* require an Earthdata login — see :func:`_ensure_login` and
``scripts/earthdata_login.py``.  Set ``decode_audio=False`` for a metadata-only
view (no login, no download).

Login
-----
Set ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` and run
``scripts/earthdata_login.py`` once on the machine to persist credentials to
``~/.netrc`` (``earthaccess.login(strategy="all", persist=True)``); thereafter
no env vars are needed.  :func:`_ensure_login` performs this lazily and is
reserved for the future audio path — it is **not** called for metadata search.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Iterator
import time, random

import earthaccess
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import audio_stereo_to_mono

logger = logging.getLogger(__name__)

#: Directory holding the precomputed per-collection metadata CSVs (repo root).
_METADATA_DIR = Path(__file__).resolve().parents[4] / "metadata"

#: Directory holding the curated 5 s detection parquets (one subdir per split).
_CURATED_DIR = Path(__file__).resolve().parents[4] / "curated" / "nasa"

#: Directory holding the *materialized* event audio: one subdir per event split,
#: each a set of ``shard_*.parquet`` (metadata) beside ``shard_*.bin`` (the
#: concatenated raw PCM) written by ``scripts/materialize_nasa_events.py`` (the
#: 5 s slice pre-downloaded and resampled to 32 kHz). The audio blob is
#: memory-mapped at read time so DataLoader workers share one copy.
_MATERIALIZED_DIR = Path(__file__).resolve().parents[4] / "curated" / "nasa_audio"

#: Split name -> source path.  Recording splits point at a precomputed metadata
#: parquet (one row per granule); event splits point at a directory of curated
#: detection parquet shards (one row per 5 s event); ``*_LOCAL`` splits point at
#: a directory of materialized-audio shards read locally (no network/login).
_SPLIT_PATHS = {
    "BIOSCAPE": str(_METADATA_DIR / "nasa_bioscape.parquet"),
    "S2L": str(_METADATA_DIR / "nasa_s2l.parquet"),
    "BIOSCAPE_EVENTS": str(_CURATED_DIR / "bioscape07.parquet"),
    "S2L_EVENTS": str(_CURATED_DIR / "s2l07.parquet"),
    "BIOSCAPE_EVENTS_LOCAL": str(_MATERIALIZED_DIR / "BIOSCAPE_EVENTS"),
    "S2L_EVENTS_LOCAL": str(_MATERIALIZED_DIR / "S2L_EVENTS"),
    "BIOSCAPE_RANDOM": str(_MATERIALIZED_DIR / "BIOSCAPE_RANDOM"),
    "S2L_RANDOM": str(_MATERIALIZED_DIR / "S2L_RANDOM"),
    "BIOSCAPE_PEAK": str(_MATERIALIZED_DIR / "BIOSCAPE_PEAK"),
    "S2L_PEAK": str(_MATERIALIZED_DIR / "S2L_PEAK"),
}

#: Splits whose records are curated 5 s events (audio is partial-read per event)
#: rather than whole-granule recordings.
_EVENT_SPLITS = {"BIOSCAPE_EVENTS", "S2L_EVENTS"}

#: Splits whose audio was pre-downloaded + resampled to local parquet shards by
#: scripts/materialize_nasa_events.py (``*_EVENTS_LOCAL``), the §2.1 random arm
#: scripts/curate_nasa_random.py (``*_RANDOM``), or the §2.1 energy-peak arm
#: scripts/curate_nasa_peak.py (``*_PEAK``).  Read locally: no network, no login.
_MATERIALIZED_SPLITS = {
    "BIOSCAPE_EVENTS_LOCAL", "S2L_EVENTS_LOCAL",
    "BIOSCAPE_RANDOM", "S2L_RANDOM",
    "BIOSCAPE_PEAK", "S2L_PEAK",
}

#: fsspec read-block size for event slices.  Must be > 0 so the HTTP file stays
#: seekable (block_size=0 yields a streaming, non-seekable file that libsndfile
#: cannot open); kept small so a slice fetches only the blocks it overlaps via
#: HTTP range requests rather than the whole 5.76 MB file.
_HTTP_BLOCK_SIZE = 1 << 18  # 256 KiB

#: Login is per-process: each spawn DataLoader worker starts with
#: ``earthaccess.__store__ is None`` and must call ``login()`` itself. These
#: tunables spread that first login across workers and retry transient failures
#: (the EDL token endpoint 503s when many workers authenticate at once).
_LOGIN_JITTER_S = 5.0      # first-login jitter, spread across spawn workers
_LOGIN_MAX_RETRIES = 5
_READ_MAX_RETRIES = 4      # transient CloudFront 503s on the data endpoint
_BACKOFF_BASE_S = 1.0
_LOGIN_LOCK = threading.Lock()
_LOGGED_IN = False


def _ensure_login() -> None:
    """Authenticate the current process with NASA Earthdata (idempotent).

    ``earthaccess.__store__`` is a module global that stays ``None`` until
    ``login()`` succeeds in *this* process; under the spawn start method each
    DataLoader worker is a fresh process that must log in on its own.

    Prefers the fully-offline ``EARTHDATA_TOKEN`` path (``strategy="environment"``,
    no network) and falls back to ``~/.netrc`` (one network token request).
    Never uses ``"interactive"`` (it would hang/raise in a headless worker).
    Adds startup jitter + exponential backoff so many workers don't stampede the
    EDL endpoint (503).  Credentials should be set up once via
    ``scripts/earthdata_login.py`` (persists ``~/.netrc``) or by adding
    ``EARTHDATA_TOKEN`` to ``.env``.
    """
    global _LOGGED_IN
    if _LOGGED_IN and earthaccess.__store__ is not None:
        return
    with _LOGIN_LOCK:
        if _LOGGED_IN and earthaccess.__store__ is not None:
            return
        time.sleep(random.uniform(0, _LOGIN_JITTER_S))
        last_err: Exception | None = None
        for attempt in range(_LOGIN_MAX_RETRIES):
            for strategy in ("environment", "netrc"):
                try:
                    earthaccess.login(strategy=strategy, persist=False)
                    if earthaccess.__store__ is not None:
                        _LOGGED_IN = True
                        return
                except Exception as e:  # LoginStrategyUnavailable / 503 / etc.
                    last_err = e
            time.sleep(_BACKOFF_BASE_S * 2**attempt + random.uniform(0, 1))
        raise RuntimeError(
            f"NASA Earthdata login failed after {_LOGIN_MAX_RETRIES} attempts. "
            "Set EARTHDATA_TOKEN in .env or run scripts/earthdata_login.py."
        ) from last_err

@register_dataset
class NASAEarthAccess(Dataset):
    """NASA Earthdata acoustic collections served as per-granule metadata records.

    Loader over the granules of a NASA Earthdata acoustic collection, read from
    a precomputed metadata CSV per split (see :data:`_SPLIT_PATHS`).  Selecting a
    split yields **one item per granule**, each a flat metadata dict::

        {
            "granule_ur":     str,          # granule unique resource name
            "data_link":      str,          # protected https download URL
            "size_mb":        float,        # granule size in MB
            "temporal_start": str | None,   # ISO UTC start (RangeDateTime)
            "temporal_end":   str | None,   # ISO UTC end
            "site":           str,          # recorder id parsed from filename
        }

    With ``decode_audio=True`` (default) the items additionally carry the decoded
    ``"audio"`` (float32 mono) and its ``"sample_rate"``.

    Parameters
    ----------
    split : str, default="BIOSCAPE"
        Split to load (key of :data:`_SPLIT_PATHS`).
    output_take_and_give : dict[str, str], optional
        Optional mapping of ``original_key -> new_key`` that filters and renames
        output fields before returning each item.
    sample_rate : int, optional
        Target sample rate in Hz.  When set and different from a recording's
        native rate, audio is resampled on the fly with ``librosa`` (only
        relevant when ``decode_audio=True``).
    n_records : int, default=-1
        Number of records to keep from the metadata CSV (applied after dropping
        non-audio granules).  ``-1`` keeps all.
    decode_audio : bool, default=True
        When ``True``, each item's full WAV is downloaded and decoded (requires
        an Earthdata login).  When ``False``, items are metadata-only (no login,
        no download).  Non-audio granules are dropped in either case.
    random_crop_seconds : float, optional
        Recording splits only.  When set, a single random ``random_crop_seconds``
        window is HTTP-range partial-read from each granule per access (a fresh
        offset every ``__getitem__``), instead of downloading + decoding the full
        WAV.  This mirrors the XenoCanto "random segment per file per epoch"
        sampling; the crop offset is drawn here (not in ``TimeShift``) because the
        authenticated session lives on this class.  ``None`` keeps the full read.
    backend : BackendType, optional
        Accepted for interface compatibility.
    streaming : bool, optional
        When ``True``, ``__len__`` / ``__getitem__`` are disabled and items are
        produced by iteration only.  Defaults to ``False`` (map-style).

    Examples
    --------
    >>> from soundscape_ssl.data.datasets import NASAEarthAccess
    >>> ds = NASAEarthAccess(split="S2L", n_records=5)
    >>> item = ds[0]
    >>> item["granule_ur"], item["site"]  # doctest: +SKIP
    ('s2llg007_..._10-20.wav', 's2llg007')
    """

    info = DatasetInfo(
        name="nasa-earthaccess",
        owner="moritz",
        split_paths=dict(_SPLIT_PATHS),
        version="0.1.0",
        description=(
            "NASA Earthdata passive-acoustic collections (via earthaccess). One "
            "record per granule (mostly individual WAV recordings), loaded from a "
            "precomputed metadata CSV per split: 'BIOSCAPE' (BioSCape, Cape "
            "Floristic) and 'S2L' (Soundscapes to Landscapes, Sonoma County, CA). "
            "With decode_audio=True the WAV is downloaded and decoded on access."
        ),
        sources=["NASA Earthdata"],
        license="CC0",
    )

    def __init__(
        self,
        split: str = "BIOSCAPE",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        n_records: int = -1,
        decode_audio: bool = True,
        random_crop_seconds: float | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self.n_records = n_records
        self.decode_audio = decode_audio
        self.random_crop_seconds = random_crop_seconds
        self._records: list[dict[str, Any]] = []
        self._fs = None  # lazily-created authenticated fsspec filesystem
        self._shard_paths: list[str] = []  # materialized-split metadata parquets
        self._shard_bin_paths: list[str] = []  # sibling flat audio blobs (.bin)
        self._shard_mmaps = None  # lazily np.memmap'd per worker (see _shard_mmap)
        self._load()
        if self.decode_audio and self.split not in _MATERIALIZED_SPLITS:
            self._filesystem()

    def __getstate__(self) -> dict[str, Any]:
        """Drop the live fsspec session before pickling to DataLoader workers.

        Under the spawn start method each worker unpickles its own copy of this
        dataset; a shared session's connection pool does not survive that, so we
        null ``_fs`` and let :meth:`_filesystem` rebuild it lazily inside each
        worker (mirrors the lazy ``_shard_mmaps`` pattern).
        """
        state = self.__dict__.copy()
        state["_fs"] = None
        return state

    def _filesystem(self):
        """Return a cached authenticated fsspec filesystem.

        Logs in first (``get_fsspec_https_session`` dereferences
        ``earthaccess.__store__``, which is ``None`` until ``login()`` runs in
        this process) and caches the session on ``self._fs``.
        """
        if self._fs is None:
            _ensure_login()
            self._fs = earthaccess.get_fsspec_https_session()
        return self._fs

    @staticmethod
    def _read_event_slice(
        fs, data_link: str, start_seconds: float, end_seconds: float
    ) -> tuple[np.ndarray, int]:
        """Read only the ``[start_seconds, end_seconds)`` slice of a WAV.

        Opens the remote file with a small block size so it stays seekable and
        libsndfile fetches (via HTTP range requests) only the blocks the slice
        overlaps, not the whole recording.  Falls back to a full read + local
        slice if the server does not support seekable range reads.
        """
        try:
            with fs.open(data_link, block_size=_HTTP_BLOCK_SIZE) as fh:
                with sf.SoundFile(fh) as f:
                    sr = f.samplerate
                    f.seek(int(round(start_seconds * sr)))
                    n_frames = int(round((end_seconds - start_seconds) * sr))
                    audio = f.read(n_frames, dtype="float32")
            return audio, sr
        except (RuntimeError, ValueError, sf.LibsndfileError):
            # Server doesn't honor seekable range reads: read the whole file
            # once and slice locally.
            with fs.open(data_link) as fh:
                audio, sr = sf.read(fh, dtype="float32")
            start = int(round(start_seconds * sr))
            n_frames = int(round((end_seconds - start_seconds) * sr))
            return audio[start : start + n_frames], sr

    @staticmethod
    def _read_random_slice(
        fs, data_link: str, crop_seconds: float
    ) -> tuple[np.ndarray, int]:
        """Read a single random ``crop_seconds`` window from a WAV.

        The offset is drawn uniformly at random, fresh each call (this is where
        the XC-style "random segment per file per epoch" randomness lives).

        The whole granule is downloaded in one streamed request and sliced
        locally, rather than issuing HTTP-range reads for just the window.  These
        granules are small (~5.76 MB) and remote reads are latency-bound, so a
        single request beats the several small range requests a partial read
        would need (measured ~0.6x slower).  Granules shorter than the window are
        returned in full (padded downstream by ``WavePadding``).
        """
        with fs.open(data_link) as fh:
            audio, sr = sf.read(fh, dtype="float32")
        n_out = int(round(crop_seconds * sr))
        if audio.shape[0] <= n_out:
            return audio, sr
        start = random.randint(0, audio.shape[0] - n_out)
        return audio[start : start + n_out], sr

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        if self.split in _EVENT_SPLITS or self.split in _MATERIALIZED_SPLITS:
            cols = [
                "granule_ur",
                "data_link",
                "site",
                "start_seconds",
                "end_seconds",
                "top_species",
                "confidence",
            ]
        else:
            cols = [
                "granule_ur",
                "data_link",
                "size_mb",
                "temporal_start",
                "temporal_end",
                "site",
            ]
        if self.decode_audio:
            cols = ["audio", "sample_rate"] + cols
        return cols

    def _load(self) -> None:
        """Load records from the split's source path.

        Recording splits read a precomputed metadata parquet (one row per
        granule); non-audio granules (site-photo ``.zip``, sites ``.csv``) are
        always dropped.  Event splits read a directory of curated detection
        parquet shards (one row per 5 s event), which already contain only
        audio events, so no filtering is applied.

        Raises
        ------
        LookupError
            If ``split`` is not one of the available splits.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. "
                f"Expected one of {list(self.info.split_paths.keys())}"
            )

        path = Path(self.info.split_paths[self.split])
        if self.split in _MATERIALIZED_SPLITS:
            self._load_materialized(path)
            return
        df = pd.read_parquet(path)
        if self.n_records != -1:
            df = df.head(self.n_records)
        if self.random_crop_seconds is not None and self.split not in _EVENT_SPLITS:
            # Random-crop recording mode only needs the download URL; drop the
            # other columns so the per-worker record list (~1.7M rows, copied
            # into every spawn worker) stays small.
            keep = [c for c in ("data_link", "granule_ur") if c in df.columns]
            df = df[keep]
        self._records = df.to_dict(orient="records")

    def _load_materialized(self, path: Path) -> None:
        """Load *metadata-only* records for a materialized split.

        Each shard is a metadata parquet (no audio column) beside a flat
        ``.bin`` blob of the concatenated raw PCM.  Every row carries its byte
        ``audio_offset`` / ``audio_dtype`` into that blob (plus its ``_shard`` /
        ``_row`` position), so the large audio is never held on the Python heap
        or pickled to workers.  The blob is memory-mapped lazily per worker in
        :meth:`_shard_mmap`, so all workers share one copy via the OS page
        cache.  Shards may still be in production, so glob what is present.
        """
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No materialized shards found for split {self.split} under {path}. "
                "Run scripts/materialize_nasa_events.py first."
            )
        self._shard_paths = [str(p) for p in shards]
        self._shard_bin_paths = [str(p.with_suffix(".bin")) for p in shards]
        records: list[dict[str, Any]] = []
        for shard_idx, shard in enumerate(shards):
            meta = pd.read_parquet(shard)
            if "audio_offset" not in meta.columns:
                raise RuntimeError(
                    f"Shard {shard} is in the old audio-in-parquet format. Run "
                    "scripts/convert_nasa_shards.py to migrate it to the flat "
                    ".bin layout (no re-download needed)."
                )
            if not Path(self._shard_bin_paths[shard_idx]).exists():
                raise FileNotFoundError(
                    f"Missing audio blob {self._shard_bin_paths[shard_idx]} for {shard}."
                )
            for row_idx, row in enumerate(meta.to_dict(orient="records")):
                row["_shard"] = shard_idx
                row["_row"] = row_idx
                records.append(row)
        if self.n_records != -1:
            records = records[: self.n_records]
        self._records = records

    def _shard_mmap(self, shard_idx: int) -> np.memmap:
        """Return a shard's flat audio blob as a read-only ``np.memmap``.

        Opened **inside the worker** (never pickled) and memory-mapped as raw
        bytes, so every worker shares the same file-backed pages via the OS page
        cache instead of each holding a private decompressed copy of the audio.
        Follows the lazy pattern of :meth:`_filesystem` so it stays out of the
        pickled state.
        """
        if self._shard_mmaps is None:
            self._shard_mmaps = {}
        mm = self._shard_mmaps.get(shard_idx)
        if mm is None:
            mm = np.memmap(self._shard_bin_paths[shard_idx], dtype=np.uint8, mode="r")
            self._shard_mmaps[shard_idx] = mm
        return mm

    def _decode_remote(self, record: dict[str, Any]) -> tuple[np.ndarray, int]:
        """Read a remote granule's audio, retrying transient errors.

        Dispatches to the right read for the split (event slice / random crop /
        full read).  Transient server or connection errors (e.g. CloudFront 503
        under many-worker load, or a session expired on a multi-day run) are
        retried with exponential backoff, rebuilding the authenticated session
        between attempts.  Raises the last error if all attempts fail.
        """
        last_err: Exception | None = None
        for attempt in range(_READ_MAX_RETRIES):
            try:
                fs = self._filesystem()
                if self.split in _EVENT_SPLITS:
                    # Curated 5 s event: read only its slice so files shared by
                    # several events aren't downloaded in full.
                    return self._read_event_slice(
                        fs, record["data_link"],
                        record["start_seconds"], record["end_seconds"],
                    )
                if self.random_crop_seconds is not None:
                    # Random 5 s window per file per epoch (XC-style).
                    return self._read_random_slice(
                        fs, record["data_link"], self.random_crop_seconds,
                    )
                with fs.open(record["data_link"]) as fh:
                    return sf.read(fh)
            except Exception as e:
                last_err = e
                self._fs = None  # rebuild: session may be expired/rate-limited
                time.sleep(_BACKOFF_BASE_S * 2**attempt + random.uniform(0, 1))
        raise last_err  # type: ignore[misc]

    def _process(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Return the record for a single granule, decoding audio if enabled.

        With ``decode_audio=True`` the full WAV is downloaded over an
        authenticated session and decoded to a float32 mono array (resampled to
        ``sample_rate`` if set), mirroring alp_data's decode path.  The audio is
        added under ``"audio"`` / ``"sample_rate"``.
        """
        row = dict(record)

        if self.decode_audio:
            if self.split in _MATERIALIZED_SPLITS:
                # Pre-downloaded + resampled audio: slice the one event's bytes
                # from the memory-mapped shard blob and decode locally (no
                # network, no login, no per-worker copy of the audio).
                mm = self._shard_mmap(record["_shard"])
                n = int(record["num_samples"])
                off = int(record["audio_offset"])
                if record["audio_dtype"] == "int16":
                    raw = mm[off : off + n * 2]
                    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                else:
                    raw = mm[off : off + n * 4]
                    audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
                sr = int(record["sample_rate"])
            else:
                try:
                    audio, sr = self._decode_remote(record)
                except Exception as e:
                    # Transient 503 storm or dead link: skip this record (the
                    # mix skips None) rather than kill the worker.
                    logger.warning(
                        "Giving up on %s (split=%s) after retries: %s",
                        record.get("data_link", record.get("granule_ur", "<unknown>")),
                        self.split, e,
                    )
                    return None
            audio = audio.astype(np.float32)
            audio = audio_stereo_to_mono(audio, mono_method="average")
            # A degenerate/zero-length event decodes to no samples; skip it
            # (return None) rather than emit an empty clip that crashes
            # amplitude-based transforms downstream. Mirrors NOAA._process.
            if audio.size == 0:
                logger.warning(
                    "Empty audio for %s (split=%s); skipping record.",
                    record.get("data_link", record.get("granule_ur", "<unknown>")),
                    self.split,
                )
                return None
            if self.sample_rate is not None and sr != self.sample_rate:
                audio = librosa.resample(
                    y=audio,
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                    scale=True,
                    res_type="soxr_hq",
                )
                sr = self.sample_rate
            row["audio"] = audio
            row["sample_rate"] = sr

        if self.output_take_and_give:
            return {
                new_key: row[orig_key]
                for orig_key, new_key in self.output_take_and_give.items()
            }
        return row

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NASAEarthAccess", dict[str, Any]]:
        """Create a dataset instance from a :class:`~alp_data.DatasetConfig`.

        ``n_records`` and ``decode_audio`` are read from the config's extra
        fields when present, otherwise the constructor defaults are used.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            n_records=cfg.get("n_records", -1),
            decode_audio=cfg.get("decode_audio", True),
            random_crop_seconds=cfg.get("random_crop_seconds", None),
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __len__(self) -> int:
        """Return the number of granules in the split.

        Raises
        ------
        NotImplementedError
            In streaming mode, where length is unavailable.
        """
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. "
                "Iterate over the dataset instead."
            )
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the metadata record at position ``idx``."""
        if self._streaming:
            raise NotImplementedError(
                "Indexed access is not available in streaming mode. Iterate instead."
            )
        return self._process(self._records[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all granules, yielding one metadata record each."""
        for record in self._records:
            yield self._process(record)

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version}), split={self.split}, "
            f"{len(self._records)} granule(s)\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.available_splits)}"
        )
