"""HuggingFace-backed soundscape pre-training dataset.

Unlike :mod:`soundscape_ssl.data.datasets.xeno_canto`, whose source data lives
on GCS and is described by ``alp_data`` split CSVs, this dataset streams its
audio from a HuggingFace Hub dataset repository
(`mwirth7/soundscape-pretrain <https://huggingface.co/datasets/mwirth7/soundscape-pretrain>`_).
That repo exposes two splits — ``a2o`` (Australian Acoustic Observatory) and
``arbimon`` (Arbimon / RFCx) — each a set of Parquet shards whose ``audio``
column holds FLAC-encoded bytes (see :mod:`scripts.download_to_hf`).

The class still derives from :class:`alp_data.Dataset` so it plugs into the
same registry, config system, and :class:`~soundscape_ssl.data.MixedStreamingDataset`
loader as every other ESP dataset.
"""

from typing import Any, Dict, Iterator

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType

DEFAULT_REPO_ID = "mwirth7/soundscape-pretrain"


@register_dataset
class SoundscapePretrain(Dataset):
    """Soundscape pre-training clips served from the HuggingFace Hub.

    Mirrors :class:`~soundscape_ssl.data.datasets.XenoCantoRaw`: each sample
    carries the *raw compressed* audio rather than a decoded waveform, so the
    crop window can be chosen before decoding by a bytes-aware
    :class:`~soundscape_ssl.data.transforms.TimeShift`.  Each sample contains::

        {
            "audio_bytes":  bytes,        # raw FLAC bytes (no decode)
            "audio_format": "FLAC",
            "sample_rate":  int | None,   # *target* SR (from constructor)
            ...metadata...               # all other columns from the parquet
        }

    The ``"audio"`` key produced by HuggingFace's :class:`~datasets.Audio`
    feature is **never** present — the column is cast to ``decode=False`` on
    load so no audio is ever decoded inside the ``datasets`` layer.

    Splits
    ------
    - ``a2o``: Australian Acoustic Observatory clips (~960 k).
    - ``arbimon``: Arbimon / RFCx clips (~1.58 M).

    To pre-train on both at once, instantiate one dataset per split and combine
    them with :class:`~soundscape_ssl.data.MixedStreamingDataset` (or an
    ``alp_data`` ``ConcatConfig``) rather than expecting a single combined split.

    Parameters
    ----------
    split : str, default="a2o"
        Which source to load. One of ``"a2o"`` or ``"arbimon"``. Each is a
        separate HF config on the Hub (loaded as ``load_dataset(repo_id, split)``),
        so this maps to the config name rather than a Hub split.
    repo_id : str, default="mwirth7/soundscape-pretrain"
        HuggingFace Hub dataset repository ID.
    sample_rate : int, optional
        *Target* sample rate forwarded to each sample as ``"sample_rate"``.
        No resampling happens here — the bytes-aware ``TimeShift`` resamples
        on decode.  The shards are already encoded at 32 kHz.
    output_take_and_give : dict[str, str], optional
        Maps original column names to renamed output keys, exactly as in
        :class:`~alp_data.datasets.XenoCanto`.  The audio payload
        (``audio_bytes`` / ``audio_format`` / ``sample_rate``) is always
        carried regardless of this mapping.
    cache_dir : str, optional
        Local cache directory for the downloaded Parquet shards.  Forwarded to
        :func:`datasets.load_dataset`.  ``None`` uses the default HF cache.
    backend : BackendType, optional
        Accepted for interface compatibility with other ESP datasets; unused
        because the data is served by the ``datasets`` library, not a polars /
        pandas backend.
    streaming : bool, optional
        When ``True``, the underlying ``datasets.IterableDataset`` is used:
        iteration works but random access (``__getitem__``) and ``__len__`` do
        not.  Defaults to ``False`` (map-style), as required by
        :class:`~soundscape_ssl.data.MixedStreamingDataset`.

    Examples
    --------
    >>> from soundscape_ssl.data.datasets import SoundscapePretrain
    >>> ds = SoundscapePretrain(split="a2o", sample_rate=32000)
    >>> sample = ds[0]
    >>> isinstance(sample["audio_bytes"], bytes)
    True
    >>> "audio" not in sample
    True
    """

    info = DatasetInfo(
        name="soundscape-pretrain",
        owner="moritz",
        split_paths={
            "a2o": "hf://datasets/{repo_id}@a2o:train",
            "arbimon": "hf://datasets/{repo_id}@arbimon:train",
        },
        version="0.1.0",
        description="Soundscape clips from the Australian Acoustic Observatory (A2O) "
        "and Arbimon / RFCx platforms, pre-processed for self-supervised "
        "pre-training. Served as FLAC-encoded bytes from the HuggingFace Hub "
        f"repository '{DEFAULT_REPO_ID}'. Two splits: 'a2o' (~960k clips) and "
        "'arbimon' (~1.58M clips), each a 2-8s mono clip at 32 kHz.",
        sources=[f"https://huggingface.co/datasets/{DEFAULT_REPO_ID}"],
        license="various",
    )

    # Audio is always stored as FLAC bytes by scripts/download_to_hf.py.
    _audio_format = "FLAC"
    _audio_column = "audio"

    # Each source is published as its own HF config (named after the split),
    # and every config has a single "train" split on the Hub.
    _hf_split = "train"

    def __init__(
        self,
        split: str = "a2o",
        output_take_and_give: dict[str, str] = None,
        sample_rate: int | None = None,
        cache_dir: str | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
        *,
        repo_id: str = DEFAULT_REPO_ID,
    ) -> None:
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.repo_id = repo_id
        self.split = split
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self._data = None
        self._load()

    @property
    def columns(self) -> list[str]:
        """Return the columns of the underlying HuggingFace dataset."""
        return list(self._data.column_names)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    def _load(self) -> None:
        """Load one source from the HuggingFace Hub.

        Each source (``a2o`` / ``arbimon``) is published as its own HF config,
        so it is loaded with ``load_dataset(repo_id, name=split)`` and the
        Hub-side ``train`` split.  Using a per-source config lets each carry its
        own feature schema — the two sources have *different* metadata columns
        (``recording_id`` / ``site`` vs. ``project_id`` / ``stream_id``).

        The ``audio`` column is cast to ``decode=False`` so that :meth:`_process`
        receives the raw FLAC bytes instead of a decoded waveform.

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

        # Imported here so the module imports cheaply when only metadata is read.
        from datasets import Audio, load_dataset

        ds = load_dataset(
            self.repo_id,
            self.split,
            split=self._hf_split,
            streaming=self._streaming,
            cache_dir=self.cache_dir,
        )
        # Keep the audio compressed; we hand raw bytes to a bytes-aware TimeShift.
        self._data = ds.cast_column(self._audio_column, Audio(decode=False))

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["SoundscapePretrain", dict[str, Any]]:
        """Create a dataset instance from a :class:`~alp_data.DatasetConfig`.

        ``repo_id`` and ``cache_dir`` are read from the config's extra fields
        when present (``DatasetConfig`` allows extra keys), otherwise the
        constructor defaults are used.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration describing split, sample rate, transformations, etc.

        Returns
        -------
        tuple[SoundscapePretrain, dict[str, Any]]
            The dataset instance and any transformation metadata (empty dict
            when no transformations are configured).
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})

        ds = cls(
            split=cfg["split"],
            repo_id=cfg.get("repo_id", DEFAULT_REPO_ID),
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            cache_dir=cfg.get("cache_dir"),
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )

        if dataset_config.transformations:
            transform_metadata = ds.apply_transformations(dataset_config.transformations)
            return ds, transform_metadata

        return ds, {}

    def __len__(self) -> int:
        """Return the number of samples in the loaded split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        NotImplementedError
            In streaming mode, where length is unavailable.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return self._data.num_rows

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Attach raw audio bytes to a row without decoding.

        Parameters
        ----------
        row : dict[str, Any]
            A single row from the HuggingFace dataset. Its ``audio`` value is a
            ``{"bytes": ..., "path": ...}`` mapping (because the column was cast
            to ``decode=False``).

        Returns
        -------
        dict[str, Any]
            The row with ``"audio_bytes"``, ``"audio_format"`` and
            ``"sample_rate"`` (target SR) added and ``"audio"`` removed.
        """
        audio = row.get(self._audio_column) or {}
        audio_bytes: bytes = audio.get("bytes")

        row = dict(row)  # shallow copy – do not mutate the cached row
        row["audio_bytes"] = audio_bytes
        row["audio_format"] = self._audio_format
        row["sample_rate"] = self.sample_rate  # target SR; may be None
        row.pop(self._audio_column, None)  # never expose the raw HF audio dict

        if self.output_take_and_give:
            item: dict[str, Any] = {
                new_key: row[orig_key]
                for orig_key, new_key in self.output_take_and_give.items()
            }
            # Always carry the audio payload and target sample rate.
            item["audio_bytes"] = audio_bytes
            item["audio_format"] = self._audio_format
            item["sample_rate"] = self.sample_rate
        else:
            item = row

        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get and process a single sample by index (map-style mode only)."""
        if self._streaming:
            raise NotImplementedError(
                "Random access is not available in streaming mode. Iterate instead."
            )
        return self._process(self._data[idx])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over processed samples (works in both modes)."""
        for row in self._data:
            yield self._process(row)

    def __str__(self) -> str:
        """Return a human-readable description of the dataset."""
        return (
            f"{self.info.name} (v{self.info.version})\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Repo: {self.repo_id}  |  Split: {self.split}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
