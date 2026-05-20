"""NOAA Passive Bioacoustics bucket-wide dataset loader."""

from __future__ import annotations

import itertools
import os
import re
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import gcsfs
import librosa
import numpy as np
import pandas as pd
from esp_data import Dataset, DatasetConfig, DatasetInfo
from esp_data.backends import BackendType, get_backend
from esp_data.io import audio_stereo_to_mono, read_audio

__all__ = ["NOAABucket", "NOAABucketDetections", "NOAASanctSound"]

_GCS_ROOT = "gs://noaa-passive-bioacoustic"
_BUCKET = "noaa-passive-bioacoustic"
_AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")

_SANCTSOUND_NON_ANIMAL = frozenset({"explosions", "ships", "sonar"})
_SANCTSOUND_RESOLUTION_SUFFIXES = ("_1d", "_1h", "_manual")
_SANCTSOUND_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6}Z)\.flac$", re.IGNORECASE)

# Top-level dataset prefixes present in the bucket.
# Some (e.g. nps) contain only unsupported formats (.aif) and will yield
# no items; they are still listed as valid splits for completeness.
_KNOWN_DATASETS = [
    "adeon",
    "aeon",
    "afsc",
    "boem",
    "coastal_studies_institute",
    "cornell",
    "fram",
    "ioos",
    "jasco",
    "listen",
    "mbari",
    "navy",
    "nefsc",
    "nps",
    "nrs",
    "onms",
    "pifsc",
    "sanctsound",
    "sefsc",
    "soundcoop",
    "swfsc",
]

_SPLIT_PATHS: dict[str, str] = {name: f"{_GCS_ROOT}/{name}/" for name in _KNOWN_DATASETS}
_SPLIT_PATHS["all"] = f"{_GCS_ROOT}/"


class NOAABucket(Dataset):
    """NOAA Passive Bioacoustics full-bucket dataset loader.

    Description
    -----------
    Provides access to all audio collections hosted in the
    ``gs://noaa-passive-bioacoustic`` GCS bucket by scanning each
    top-level sub-directory for supported audio files (.wav, .flac,
    .ogg, .mp3).  Every top-level directory is available as a named
    split, plus an ``"all"`` split that spans the entire bucket.

    Because individual collections can contain tens of thousands of
    files (and ``"all"`` over 200 k), the class defaults to
    ``streaming=True``.  In streaming mode audio paths are discovered
    lazily during iteration; in non-streaming mode they are enumerated
    upfront during initialisation.

    Not all collections carry annotation files.  This class returns
    the minimal set of fields that is universally available: the raw
    audio array, its sample rate, and the GCS path of the source file.

    Notes
    -----
    The ``nps`` collection contains only ``.aif`` files, which are not
    in the supported set; iterating over that split will yield no items.

    References
    ----------
    https://registry.opendata.aws/noaa-passive-bioacoustic/

    Examples
    --------
    Stream audio from the PIFSC collection:

    >>> from data.noaa_bucket import NOAABucket
    >>> ds = NOAABucket(split="pifsc")
    >>> for item in ds:
    ...     print(item["audio_path"], item["audio"].shape)
    ...     break

    Load the FRAM collection eagerly (upfront enumeration):

    >>> ds = NOAABucket(split="fram", streaming=False)
    >>> print(len(ds))

    """

    info = DatasetInfo(
        name="noaa_bucket",
        owner="moritz",
        split_paths=_SPLIT_PATHS,
        version="0.1.0",
        description=(
            "Full-bucket loader for all NOAA Passive Bioacoustics audio collections "
            "hosted at gs://noaa-passive-bioacoustic."
        ),
        sources=["NOAA"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(
        self,
        split: str = "pifsc",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        backend: BackendType = "polars",
        streaming: bool = True,
    ) -> None:
        """Initialise the NOAABucket dataset.

        Parameters
        ----------
        split : str, default="pifsc"
            Name of the collection to load.  Must be one of the keys in
            `info.split_paths` (a top-level bucket directory, or ``"all"``).
        output_take_and_give : dict[str, str], optional
            Optional mapping of ``original_key -> new_key`` that filters and
            renames output fields before returning each item.
        sample_rate : int, optional
            Target sample rate in Hz.  When set, audio is resampled
            on-the-fly using ``librosa``.
        backend : BackendType, optional
            DataFrame backend, by default ``"polars"``.  Passed to the base
            class; not used by this implementation directly.
        streaming : bool, optional
            When ``True`` (default) audio paths are discovered lazily during
            iteration.  When ``False`` all paths are enumerated during
            ``__init__``, enabling ``__len__`` and indexed ``__getitem__``
            access.

        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self._data_paths: list[str] | None = None
        self._prefix: str = ""
        self._fs = gcsfs.GCSFileSystem(token="anon")
        self._load()

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        return ["audio", "sample_rate", "audio_path"]

    @property
    def available_splits(self) -> list[str]:
        """Return all available split names."""
        return list(self.info.split_paths.keys())

    def _gcs_prefix(self, gcs_url: str) -> str:
        """Strip the ``gs://`` scheme to get a bare bucket path.

        Returns
        -------
        str
            Bucket-relative path with no leading scheme and no trailing slash.
        """
        return gcs_url.removeprefix("gs://").rstrip("/")

    def _iter_audio_paths(self, gcs_url: str) -> Iterator[str]:
        """Yield GCS URLs of audio files under `gcs_url`.

        Parameters
        ----------
        gcs_url : str
            A ``gs://`` URL pointing to the root directory to scan.

        Yields
        ------
        str
            Full ``gs://`` URL for each discovered audio file whose
            extension is in ``(".wav", ".flac", ".ogg", ".mp3")``.
        """
        prefix = self._gcs_prefix(gcs_url)
        try:
            for dirpath, _subdirs, filenames in self._fs.walk(prefix):
                for fname in filenames:
                    if fname.lower().endswith(_AUDIO_EXTENSIONS):
                        yield f"gs://{dirpath.rstrip('/')}/{fname}"
        except FileNotFoundError:
            return

    def _load(self) -> None:
        """Load (or prepare) audio file paths for the current split.

        In streaming mode the GCS prefix is stored for lazy traversal.
        In non-streaming mode all audio paths are enumerated upfront.

        Raises
        ------
        LookupError
            If `self.split` is not found in `info.split_paths`.
        """
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split!r}. "
                f"Expected one of {list(self.info.split_paths.keys())}."
            )
        self._prefix = self.info.split_paths[self.split]
        if not self._streaming:
            self._data_paths = list(self._iter_audio_paths(self._prefix))

    def _process(self, audio_path: str) -> dict[str, Any]:
        """Load and process a single audio file.

        Parameters
        ----------
        audio_path : str
            Full ``gs://`` URL of the audio file.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"audio"`` (np.ndarray, float32),
            ``"sample_rate"`` (int), and ``"audio_path"`` (str).
            If `output_take_and_give` was set, only the remapped keys
            are returned.
        """
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

        row: dict[str, Any] = {
            "audio": audio,
            "sample_rate": sr,
            "audio_path": audio_path,
        }

        if self.output_take_and_give:
            return {new_key: row[orig_key] for orig_key, new_key in self.output_take_and_give.items()}

        return row

    def __len__(self) -> int:
        """Return the number of audio files in the split.

        Returns
        -------
        int
            Number of audio files enumerated during initialisation.

        Raises
        ------
        NotImplementedError
            If the dataset was initialised in streaming mode.
        RuntimeError
            If no split has been loaded.
        """
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. "
                "Iterate over the dataset instead."
            )
        if self._data_paths is None:
            raise RuntimeError("No split has been loaded yet.")
        return len(self._data_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the item at position `idx`.

        Parameters
        ----------
        idx : int
            Zero-based index into the list of audio paths.

        Returns
        -------
        dict[str, Any]
            Processed audio item (see `_process`).

        Raises
        ------
        NotImplementedError
            If the dataset was initialised in streaming mode.
        RuntimeError
            If no split has been loaded.
        """
        if self._streaming:
            raise NotImplementedError(
                "Indexed access is not available in streaming mode."
            )
        if self._data_paths is None:
            raise RuntimeError("No split has been loaded yet.")
        return self._process(self._data_paths[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all audio items in the split.

        In streaming mode paths are discovered on-the-fly; in
        non-streaming mode the pre-built path list is used.

        Yields
        ------
        dict[str, Any]
            Processed audio item (see `_process`).

        Raises
        ------
        RuntimeError
            If the dataset was initialised in non-streaming mode but no
            split has been loaded.
        """
        if self._streaming:
            for path in self._iter_audio_paths(self._prefix):
                yield self._process(path)
        else:
            if self._data_paths is None:
                raise RuntimeError("No split has been loaded yet.")
            for path in self._data_paths:
                yield self._process(path)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NOAABucket", dict[str, Any]]:
        """Instantiate from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object produced by the data-mixing pipeline.

        Returns
        -------
        tuple[NOAABucket, dict[str, Any]]
            The dataset instance and a (possibly empty) transformation
            metadata dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version}), split={self.split}"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )


# ---------------------------------------------------------------------------
# Detection-based loader
# ---------------------------------------------------------------------------


class _DataFrame(Protocol):
    """Minimal structural type for DataFrame inputs (pandas or polars)."""

    @property
    def columns(self) -> Sequence[str]: ...


class _AudioCache:
    """Thread-safe LRU cache mapping GCS paths to (audio, sample_rate) tuples.

    Parameters
    ----------
    maxsize : int
        Maximum number of files to keep in memory.  When the cache is full
        the least-recently-used entry is evicted.  Use ``0`` to disable
        caching entirely.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[np.ndarray, int] | None:
        """Return the cached value for `key`, or ``None`` if not present.

        Returns
        -------
        tuple[np.ndarray, int] or None
            Cached ``(audio, sample_rate)`` pair, or ``None`` on a cache miss.
        """
        if self._maxsize == 0:
            return None
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: tuple[np.ndarray, int]) -> None:
        """Insert `key`/`value` into the cache, evicting LRU entry if full."""
        if self._maxsize == 0:
            return
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value


class NOAABucketDetections(Dataset):
    """Load audio clips from NOAA GCS files based on a detection list.

    Description
    -----------
    Each row of the ``detections`` DataFrame identifies a time window within
    a GCS audio file expressed as **offsets in seconds from the file start**.
    Because the GCS bucket offers no server-side partial-download API, the
    full file is fetched and then trimmed in memory.

    To avoid redundant downloads when multiple detections share the same
    source file, an LRU file cache is maintained.  With the default
    ``file_cache_size=1`` a single file is kept in memory at a time — sorting
    the detections DataFrame by ``audio_path`` before passing it in ensures
    each file is downloaded exactly once.  Increase ``file_cache_size`` for
    random-access patterns or when files are small enough to hold several
    simultaneously.

    No authentication is required; the bucket is publicly accessible.

    Notes
    -----
    Each unique file may be several hours long.  Sorting detections by
    ``audio_path`` before iterating minimises redundant GCS downloads.

    Examples
    --------
    >>> import pandas as pd
    >>> from data.noaa_bucket import NOAABucketDetections
    >>> detections = pd.DataFrame({
    ...     "audio_path":    ["gs://noaa-passive-bioacoustic/pifsc/file.wav"],
    ...     "start_seconds": [0.0],
    ...     "end_seconds":   [10.0],
    ... })
    >>> ds = NOAABucketDetections(detections, prefetch_factor=4)
    >>> for item in ds:
    ...     print(item["audio_path"], item["audio"].shape)
    """

    info = DatasetInfo(
        name="noaa_bucket_detections",
        owner="moritz",
        split_paths={},
        version="0.1.0",
        description=(
            "NOAA Passive Bioacoustics detection-based audio loader. "
            "Fetches and trims audio clips for each row in a detections DataFrame."
        ),
        sources=["NOAA"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(
        self,
        detections: _DataFrame,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        backend: BackendType = "polars",
        prefetch_factor: int = 0,
        file_cache_size: int = 1,
    ) -> None:
        """Initialise the NOAABucketDetections dataset.

        Parameters
        ----------
        detections : pandas.DataFrame or polars.DataFrame
            Detection records.  Must contain the columns ``audio_path``,
            ``start_seconds``, and ``end_seconds``.  ``audio_path`` must
            be a full ``gs://`` URL.  ``start_seconds`` and ``end_seconds``
            are offsets in seconds from the start of the file (float or int).
        output_take_and_give : dict[str, str], optional
            Optional mapping of ``original_key -> new_key`` that filters
            and renames output fields before returning each item.
        sample_rate : int, optional
            Target sample rate in Hz.  When set, audio is resampled
            on-the-fly using ``librosa`` after trimming.
        backend : BackendType, optional
            DataFrame backend, by default ``"polars"``.
        prefetch_factor : int, optional
            Number of detections to process concurrently in background
            threads.  ``0`` (default) processes detections sequentially.
        file_cache_size : int, optional
            Number of full audio files to keep in the in-memory LRU cache.
            Defaults to ``1``.  Set to ``0`` to disable caching.  Increase
            when multiple detections span the same file and you can afford
            the memory cost of holding several files at once.

        """
        super().__init__(output_take_and_give, backend=backend, streaming=True)
        self._detections = self._to_records(detections)
        self.sample_rate = sample_rate
        self._prefetch_factor = prefetch_factor
        self._fs = gcsfs.GCSFileSystem(token="anon")
        self._cache = _AudioCache(maxsize=file_cache_size)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        return ["audio", "sample_rate", "audio_path", "start_seconds", "end_seconds"]

    @property
    def available_splits(self) -> list[str]:
        """Return available splits."""
        return ["all"]

    def _load(self) -> None:
        """No-op; NOAABucketDetections is streaming-only."""

    def __len__(self) -> int:
        """Not supported; NOAABucketDetections is streaming-only.

        Raises
        ------
        NotImplementedError
            Always; use ``for item in dataset`` to iterate.
        """
        raise NotImplementedError(
            "NOAABucketDetections only supports streaming iteration; "
            "use `for item in dataset`."
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Not supported; NOAABucketDetections is streaming-only.

        Raises
        ------
        NotImplementedError
            Always; use ``for item in dataset`` to iterate.
        """
        raise NotImplementedError(
            "NOAABucketDetections only supports streaming iteration; "
            "use `for item in dataset`."
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all detections, yielding one audio clip each.

        When `prefetch_factor` is greater than zero, up to that many
        detections are processed concurrently in background threads.

        Yields
        ------
        dict[str, Any]
            Processed audio item (see `_process_detection`).
        """
        if self._prefetch_factor <= 0:
            for det in self._detections:
                yield self._process_detection(det)
            return

        dets = iter(self._detections)
        with ThreadPoolExecutor(max_workers=self._prefetch_factor) as executor:
            pending: deque = deque()
            for det in itertools.islice(dets, self._prefetch_factor):
                pending.append(executor.submit(self._process_detection, det))
            for det in dets:
                pending.append(executor.submit(self._process_detection, det))
                yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["NOAABucketDetections", dict[str, Any]]:
        """Instantiate from a `DatasetConfig` by loading detections from a file.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Must include a ``detections_path`` extra field pointing to a CSV
            or Parquet file with columns ``audio_path``, ``start_seconds``,
            and ``end_seconds``.

        Returns
        -------
        tuple[NOAABucketDetections, dict[str, Any]]
            The dataset instance and a (possibly empty) transformation
            metadata dict.

        Raises
        ------
        ValueError
            If ``detections_path`` is missing from the config or the file
            extension is not ``.csv`` or ``.parquet`` / ``.pq``.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        raw_path = cfg.pop("detections_path", None)
        if raw_path is None:
            raise ValueError("DatasetConfig must include a 'detections_path' field.")
        detections_path = Path(raw_path)
        suffix = detections_path.suffix.lower()
        backend = cfg.get("backend", "polars")
        backend_cls = get_backend(backend)

        if suffix == ".csv":
            detections = backend_cls.from_csv(str(detections_path)).unwrap
        elif suffix in (".parquet", ".pq"):
            detections = backend_cls.from_parquet(str(detections_path)).unwrap
        else:
            raise ValueError(
                f"Unsupported detections file format: {suffix!r}. Use .csv or .parquet."
            )

        ds = cls(
            detections=detections,
            output_take_and_give=cfg.get("output_take_and_give"),
            sample_rate=cfg.get("sample_rate"),
            backend=backend,
            prefetch_factor=cfg.get("prefetch_factor", 0),
            file_cache_size=cfg.get("file_cache_size", 1),
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_records(df: _DataFrame) -> list[dict[str, Any]]:
        """Convert a pandas or polars DataFrame to a list of detection dicts.

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            Input detections table.

        Returns
        -------
        list[dict[str, Any]]
            One dict per row containing only the three required keys.

        Raises
        ------
        ValueError
            If any required column is absent from ``df``.
        """
        required = {"audio_path", "start_seconds", "end_seconds"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Detections DataFrame is missing required columns: {sorted(missing)}"
            )
        if hasattr(df, "to_dicts"):  # polars
            return df.select(sorted(required)).to_dicts()
        return df[sorted(required)].to_dict(orient="records")  # pandas

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Load a full audio file from GCS, returning mono float32 audio.

        Results are cached in the LRU file cache to avoid redundant
        downloads when multiple detections share the same source file.

        Parameters
        ----------
        audio_path : str
            Full ``gs://`` URL of the audio file.

        Returns
        -------
        tuple[np.ndarray, int]
            Float32 mono audio array and its native sample rate.
        """
        cached = self._cache.get(audio_path)
        if cached is not None:
            return cached

        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        self._cache.put(audio_path, (audio, sr))
        return audio, sr

    def _process_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        """Load, trim, and optionally resample audio for a single detection.

        The full source file is downloaded once and cached; subsequent
        detections from the same file are served from the in-memory cache.

        Parameters
        ----------
        detection : dict[str, Any]
            A detection record with ``audio_path``, ``start_seconds``,
            and ``end_seconds``.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"audio"`` (np.ndarray, float32 mono),
            ``"sample_rate"`` (int), ``"audio_path"`` (str),
            ``"start_seconds"`` (float), and ``"end_seconds"`` (float).
            If `output_take_and_give` was set, only the remapped keys are
            returned.
        """
        audio_path = detection["audio_path"]
        start_s = detection["start_seconds"]
        end_s = detection["end_seconds"]

        full_audio, sr = self._load_audio(audio_path)

        start_sample = max(0, int(start_s * sr))
        end_sample = min(len(full_audio), int(end_s * sr))
        audio = full_audio[start_sample:end_sample].copy()

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

        row: dict[str, Any] = {
            "audio": audio,
            "sample_rate": sr,
            "audio_path": audio_path,
            "start_seconds": start_s,
            "end_seconds": end_s,
        }

        if self.output_take_and_give:
            return {
                new_key: row[orig_key]
                for orig_key, new_key in self.output_take_and_give.items()
            }
        return row

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version}), "
            f"{len(self._detections)} detection(s)\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}"
        )


class NOAASanctSound(Dataset):
    """NOAA SanctSound passive bioacoustics dataset with animal annotations.

    Description
    -----------
    Loads audio files from the SanctSound collection in the
    ``gs://noaa-passive-bioacoustic`` bucket alongside time-stamped
    animal detection annotations.  Annotations from all animal-only
    detection CSV files are combined into a single table and joined to
    each audio file by timestamp overlap.

    Non-animal detection types (``explosions``, ``ships``, ``sonar``) are
    excluded automatically.  For each audio file the returned item contains
    the audio array and, when at least one annotation falls within the
    recording window, an ``"annotations"`` DataFrame with columns
    ``ISOStartTime``, ``Presence``, ``species``, ``site``, and ``deployment``.

    Notes
    -----
    Initialisation performs two GCS listings: one to discover annotation
    CSVs and one to enumerate audio files.  This may take a minute when
    loading all sites.

    Examples
    --------
    >>> from data.noaa_bucket import NOAASanctSound
    >>> ds = NOAASanctSound(site="ci01")
    >>> for item in ds:
    ...     if "annotations" in item:
    ...         print(item["annotations"][["species", "Presence"]].head())
    ...         break

    """

    info = DatasetInfo(
        name="noaa_sanctsound",
        owner="moritz",
        split_paths={},
        version="0.1.0",
        description=(
            "SanctSound collection from the NOAA Passive Bioacoustics Archive "
            "with animal-only detection annotations joined by recording timestamp."
        ),
        sources=["NOAA"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(
        self,
        site: str | None = None,
        sample_rate: int | None = None,
        output_take_and_give: dict[str, str] | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialise the NOAASanctSound dataset.

        Parameters
        ----------
        site : str, optional
            Restrict loading to a single recording site (e.g. ``"ci01"``).
            When ``None`` all available sites are loaded.
        sample_rate : int, optional
            Target sample rate in Hz.  When set, audio is resampled on-the-fly
            using ``librosa``.
        output_take_and_give : dict[str, str], optional
            Optional mapping of ``original_key -> new_key`` that filters and
            renames output fields before returning each item.
        backend : BackendType, optional
            DataFrame backend, by default ``"polars"``.  Passed to the base
            class; not directly used by this implementation.
        streaming : bool, optional
            Not supported; kept for API consistency. Raises
            ``NotImplementedError`` if ``True``.

        Raises
        ------
        NotImplementedError
            If ``streaming=True`` is requested.
        """
        if streaming:
            raise NotImplementedError("NOAASanctSound does not support streaming mode.")
        super().__init__(output_take_and_give, backend=backend, streaming=False)
        self.site = site.lower() if site is not None else None
        self.sample_rate = sample_rate
        self._fs = gcsfs.GCSFileSystem(token="anon")
        self._annotations: pd.DataFrame = self._load_annotations()
        self._data_paths: list[str] = self._enumerate_audio_paths()

    @property
    def columns(self) -> list[str]:
        """Return output column names."""
        return ["audio", "sample_rate", "audio_path", "annotations"]

    @property
    def available_splits(self) -> list[str]:
        """Return available splits (not applicable; returns empty list)."""
        return []

    def _load(self) -> None:
        """No-op; loading is handled in `__init__`."""

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NOAASanctSound", dict[str, Any]]:
        """Instantiate from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object produced by the data-mixing pipeline.

        Returns
        -------
        tuple[NOAASanctSound, dict[str, Any]]
            The dataset instance and a (possibly empty) transformation
            metadata dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            site=cfg.get("split"),
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def _load_annotations(self) -> pd.DataFrame:
        """Discover and load all animal-only detection CSVs for SanctSound.

        Returns
        -------
        pd.DataFrame
            Combined annotation table with columns ``ISOStartTime``
            (datetime64[ns, UTC]), ``Presence``, ``species``, ``site``,
            ``deployment``.
        """
        detections_root = f"{_BUCKET}/sanctsound/products/detections"
        frames: list[pd.DataFrame] = []

        try:
            site_dirs = self._fs.ls(detections_root)
        except FileNotFoundError:
            return pd.DataFrame(columns=["ISOStartTime", "Presence", "species", "site", "deployment"])

        for site_dir in site_dirs:
            site_name = os.path.basename(site_dir)
            if self.site is not None and site_name != self.site:
                continue

            try:
                det_dirs = self._fs.ls(site_dir)
            except FileNotFoundError:
                continue

            for det_dir in det_dirs:
                dir_name = os.path.basename(det_dir)
                parts = dir_name.split("_")
                # Expected: sanctsound_{site}_{deployment}_{species}[_suffix...]
                if len(parts) < 4:
                    continue

                raw_species = "_".join(parts[3:])
                species = raw_species
                for suffix in _SANCTSOUND_RESOLUTION_SUFFIXES:
                    if species.endswith(suffix):
                        species = species[: -len(suffix)]
                        break

                if species in _SANCTSOUND_NON_ANIMAL:
                    continue

                deployment = parts[2]

                try:
                    csv_paths = self._fs.ls(f"{det_dir}/data")
                except FileNotFoundError:
                    continue

                for csv_path in csv_paths:
                    if not csv_path.lower().endswith(".csv"):
                        continue
                    try:
                        with self._fs.open(csv_path) as fh:
                            df = pd.read_csv(fh)
                    except Exception:
                        continue

                    df["species"] = species
                    df["site"] = site_name
                    df["deployment"] = deployment
                    df["ISOStartTime"] = pd.to_datetime(df["ISOStartTime"], utc=True)
                    frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["ISOStartTime", "Presence", "species", "site", "deployment"])

        return pd.concat(frames, ignore_index=True)

    def _enumerate_audio_paths(self) -> list[str]:
        """List all FLAC audio files in the SanctSound audio directory.

        Only files under ``*/audio/`` subdirectories are included, skipping
        ancillary and metadata directories.

        Returns
        -------
        list[str]
            Sorted ``gs://`` URLs of audio files.
        """
        audio_root = f"{_BUCKET}/sanctsound/audio"
        paths: list[str] = []

        try:
            site_dirs = self._fs.ls(audio_root)
        except FileNotFoundError:
            return []

        for site_dir in site_dirs:
            site_name = os.path.basename(site_dir)
            if self.site is not None and site_name != self.site:
                continue

            try:
                deployment_dirs = self._fs.ls(site_dir)
            except FileNotFoundError:
                continue

            for dep_dir in deployment_dirs:
                audio_subdir = f"{dep_dir}/audio"
                try:
                    files = self._fs.ls(audio_subdir)
                except FileNotFoundError:
                    continue

                for f in files:
                    if f.lower().endswith(".flac"):
                        paths.append(f"gs://{f}")

        return sorted(paths)

    @staticmethod
    def _parse_file_timestamp(audio_path: str) -> datetime | None:
        """Extract the UTC recording start timestamp from a SanctSound filename.

        Parameters
        ----------
        audio_path : str
            Full ``gs://`` URL or bare path to the audio file.

        Returns
        -------
        datetime or None
            Timezone-aware UTC datetime, or ``None`` if the filename does not
            match the expected ``{YYYYMMDD}T{HHMMSS}Z.flac`` pattern.
        """
        basename = os.path.basename(audio_path)
        match = _SANCTSOUND_TIMESTAMP_RE.search(basename)
        if match is None:
            return None
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

    def _process(self, audio_path: str) -> dict[str, Any]:
        """Load a single audio file and attach overlapping annotations.

        Parameters
        ----------
        audio_path : str
            Full ``gs://`` URL of the audio file.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"audio"`` (np.ndarray, float32 mono),
            ``"sample_rate"`` (int), ``"audio_path"`` (str), and optionally
            ``"annotations"`` (pd.DataFrame) when detections overlap the
            recording window.
        """
        audio, sr = read_audio(audio_path)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

        row: dict[str, Any] = {"audio": audio, "sample_rate": sr, "audio_path": audio_path}

        file_start = self._parse_file_timestamp(audio_path)
        if file_start is not None and len(self._annotations) > 0:
            file_end = file_start + timedelta(seconds=len(audio) / sr)
            mask = (self._annotations["ISOStartTime"] >= file_start) & (
                self._annotations["ISOStartTime"] < file_end
            )
            matching = self._annotations.loc[mask]
            if len(matching) > 0:
                row["annotations"] = matching.reset_index(drop=True)

        if self.output_take_and_give:
            return {new_key: row[orig_key] for orig_key, new_key in self.output_take_and_give.items()}

        return row

    def __len__(self) -> int:
        """Return the number of audio files in the dataset.

        Returns
        -------
        int
            Total number of audio files discovered for the selected site(s).
        """
        return len(self._data_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the processed item at position `idx`.

        Parameters
        ----------
        idx : int
            Index into the sorted audio file list.

        Returns
        -------
        dict[str, Any]
            Processed audio item (see `_process`).
        """
        return self._process(self._data_paths[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all audio files, yielding one item each.

        Yields
        ------
        dict[str, Any]
            Processed audio item (see `_process`).
        """
        for path in self._data_paths:
            yield self._process(path)

    def __str__(self) -> str:
        site_str = self.site or "all"
        return (
            f"{self.info.name} (v{self.info.version}), site={site_str}\n"
            f"{len(self._data_paths)} audio file(s), "
            f"{len(self._annotations)} annotation row(s)\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}"
        )
