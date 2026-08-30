"""XenoCanto dataset variants for lazy / bytes-based audio loading."""

from typing import Any

from alp_data import DatasetInfo, register_dataset
from alp_data.datasets import XenoCanto
from alp_data.io import DATA_HOME, anypath, filesystem_from_path


@register_dataset
class XenoCantoRaw(XenoCanto):
    """XenoCanto dataset that returns raw compressed bytes instead of decoded audio.

    Extends :class:`~alp_data.datasets.XenoCanto` by skipping the decode step
    in :meth:`_process`.  Each sample contains::

        {
            "audio_bytes":  bytes,        # raw compressed audio (no decode)
            "audio_format": str,          # e.g. "FLAC", "MP3", "OGG"
            "sample_rate":  int | None,   # *target* SR (from constructor)
            ...metadata...               # all other metadata columns
        }

    The ``"audio"`` key produced by the parent class is **never** present.

    The primary use-case is to pair this dataset with a bytes-aware
    :class:`~bioacoustic_ssl.data.transforms.TimeShift`, which decides the crop
    window *before* decoding so that only the required frames are ever decoded.
    For long recordings where only a short clip is needed this can yield a
    substantial speed-up.

    .. note::
        This class downloads the **entire** compressed file from GCS before
        handing control to ``TimeShift``.  For truly lazy remote I/O — where
        only the bytes corresponding to the selected crop window are ever
        downloaded — use :class:`XenoCantoLazy` instead.

    Parameters
    ----------
    *args :
        Forwarded verbatim to :class:`~alp_data.datasets.XenoCanto`.
    **kwargs :
        Forwarded verbatim to :class:`~alp_data.datasets.XenoCanto`.

    Examples
    --------
    >>> from bioacoustic_ssl.data.datasets import XenoCantoRaw
    >>> ds = XenoCantoRaw(split="train", sample_rate=32000)
    >>> sample = ds[0]
    >>> isinstance(sample["audio_bytes"], bytes)
    True
    >>> "audio" not in sample
    True
    """

    info = DatasetInfo(
        name="xeno-canto-raw",
        owner="david; gagan",
        split_paths={
            "train": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/train_20260203_v2.csv",
            "validation": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/val_20260203_v2.csv",
            "all": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/all_20260203_v2.csv",
            "train_unseen": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/train_unseen_20260203_v2.csv",
            "validation_unseen": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/val_unseen_20260203_v2.csv",
            "all_unseen": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/all_unseen_20260203_v2.csv",
        },
        version="0.1.0",
        description="Xeno-canto audio dataset with taxonomic metadata. "
        "Available at original (variable) sample rates and 32kHz (pre-resampled). "
        "Pre-resampled audio uses librosa's kaiser_best resampling method. "
        "Xeno-canto dump as of Oct 2025. "
        "Train/val split is 90%/10% with random seed 42.",
        sources=["Xeno-canto"],
        license="CC BY-NC-SA 4.0, CC BY-NC 4.0, CC BY-SA, CC0",
    )

    def _resolve_audio_path(self, row: dict[str, Any]):
        """Return the file path for a given metadata row.

        Mirrors the path-selection logic from the parent class (preferring
        pre-resampled files when available) without reading any audio data.

        Parameters
        ----------
        row : dict[str, Any]
            A metadata row from the loaded split.

        Returns
        -------
        AnyPathT
            Path object (local :class:`pathlib.Path` or cloud path) pointing
            to the audio file.
        """
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                if self.sample_rate == 16000:
                    return self._data_root_16k / row[path_column]
                return self._data_root_32k / row[path_column]

        # Fall back to original variable-rate files.
        rel_path = row[self._originals_path_column]
        if not rel_path.startswith("audio/"):
            return anypath(self.data_root) / "audio" / rel_path
        return anypath(self.data_root) / rel_path

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Load raw bytes without decoding.

        Parameters
        ----------
        row : dict[str, Any]
            A single metadata row from the dataset.

        Returns
        -------
        dict[str, Any]
            The row with ``"audio_bytes"``, ``"audio_format"``, and
            ``"sample_rate"`` (target SR) added.  The ``"audio"`` key is
            never present.
        """
        audio_path = self._resolve_audio_path(row)

        fs = filesystem_from_path(audio_path)
        with fs.open(str(audio_path), "rb") as fh:
            audio_bytes: bytes = fh.read()

        audio_format: str = anypath(audio_path).suffix.lstrip(".").upper()

        row = dict(row)  # shallow copy – do not mutate original
        row["audio_bytes"] = audio_bytes
        row["audio_format"] = audio_format
        row["sample_rate"] = self.sample_rate  # target SR; may be None
        row.pop("audio", None)  # never set by this subclass

        if self.output_take_and_give:
            item: dict[str, Any] = {
                new_key: row[orig_key]
                for orig_key, new_key in self.output_take_and_give.items()
            }
            # Always carry the audio payload and target sample rate.
            item["audio_bytes"] = audio_bytes
            item["audio_format"] = audio_format
            item["sample_rate"] = self.sample_rate
        else:
            item = row

        return item


@register_dataset
class XenoCantoLazy(XenoCanto):
    """XenoCanto dataset that defers all I/O to the transform pipeline.

    Extends :class:`~alp_data.datasets.XenoCanto` so that :meth:`_process`
    returns the **path** to the audio file rather than its content.  Each
    sample contains::

        {
            "audio_path":   str,          # GCS / local path to the audio file
            "audio_format": str,          # e.g. "FLAC", "MP3", "OGG"
            "sample_rate":  int | None,   # *target* SR (from constructor)
            ...metadata...               # all other metadata columns
        }

    Neither ``"audio"`` nor ``"audio_bytes"`` is ever present.

    **Why lazy?**

    When paired with the path-aware mode of
    :class:`~bioacoustic_ssl.data.transforms.TimeShift`, the transform opens
    the remote file directly and passes the file-like object to soundfile.
    Because ``gcsfs`` translates ``seek()`` calls into HTTP ``Range`` headers,
    *only the bytes that soundfile actually reads are ever downloaded* —
    typically the file header (≈ 256 KB) plus the selected window (≈ 256 KB),
    regardless of the total file size.

    For a 5-minute pre-resampled FLAC recording (~9.5 MB) with a 5-second
    target clip (~160 KB), this yields a **~19× reduction in GCS bandwidth**
    compared to downloading the full file.

    .. note::
        This optimisation is most effective for FLAC and WAV (O(1) random
        access via seek table / fixed header).  OGG Vorbis requires several
        extra Range requests to bisect-search for the start page; MP3 has no
        random-access support in libsndfile and degrades to a full download.
        The pre-resampled XenoCanto files (32 kHz / 16 kHz) are FLAC, so the
        primary use-case is fully covered.

    Parameters
    ----------
    *args :
        Forwarded verbatim to :class:`~alp_data.datasets.XenoCanto`.
    **kwargs :
        Forwarded verbatim to :class:`~alp_data.datasets.XenoCanto`.

    Examples
    --------
    >>> from bioacoustic_ssl.data.datasets import XenoCantoLazy
    >>> ds = XenoCantoLazy(split="train", sample_rate=32000)
    >>> sample = ds[0]
    >>> isinstance(sample["audio_path"], str)
    True
    >>> "audio" not in sample and "audio_bytes" not in sample
    True
    """

    info = DatasetInfo(
        name="xeno-canto-lazy",
        owner="david; gagan",
        split_paths={
            "train": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/train_20260203_v2.csv",
            "validation": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/val_20260203_v2.csv",
            "all": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/all_20260203_v2.csv",
            "train_unseen": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/train_unseen_20260203_v2.csv",
            "validation_unseen": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/val_unseen_20260203_v2.csv",
            "all_unseen": f"{DATA_HOME}/xeno-canto/v0.1.0/raw/all_unseen_20260203_v2.csv",
        },
        version="0.1.0",
        description="Xeno-canto audio dataset with taxonomic metadata. "
        "Available at original (variable) sample rates and 32kHz (pre-resampled). "
        "Pre-resampled audio uses librosa's kaiser_best resampling method. "
        "Xeno-canto dump as of Oct 2025. "
        "Train/val split is 90%/10% with random seed 42.",
        sources=["Xeno-canto"],
        license="CC BY-NC-SA 4.0, CC BY-NC 4.0, CC BY-SA, CC0",
    )

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return the audio path without performing any I/O.

        Parameters
        ----------
        row : dict[str, Any]
            A single metadata row from the dataset.

        Returns
        -------
        dict[str, Any]
            The row with ``"audio_path"``, ``"audio_format"``, and
            ``"sample_rate"`` (target SR) added.  No audio data is loaded.
        """
        # Reuse the path-resolution logic from XenoCantoRaw (mixin approach).
        audio_path = XenoCantoRaw._resolve_audio_path(self, row)
        audio_path_str: str = str(audio_path)
        audio_format: str = anypath(audio_path).suffix.lstrip(".").upper()

        row = dict(row)  # shallow copy – do not mutate original
        row["audio_path"] = audio_path_str
        row["audio_format"] = audio_format
        row["sample_rate"] = self.sample_rate  # target SR; may be None
        row.pop("audio", None)

        if self.output_take_and_give:
            item: dict[str, Any] = {
                new_key: row[orig_key]
                for orig_key, new_key in self.output_take_and_give.items()
            }
            item["audio_path"] = audio_path_str
            item["audio_format"] = audio_format
            item["sample_rate"] = self.sample_rate
        else:
            item = row

        return item
