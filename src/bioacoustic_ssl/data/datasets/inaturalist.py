"""iNaturalist dataset variant for bytes-based audio loading."""

from typing import Any

from alp_data import DatasetInfo, register_dataset
from alp_data.datasets import INaturalist
from alp_data.io import DATA_HOME, anypath, filesystem_from_path


@register_dataset
class INaturalistRaw(INaturalist):
    """iNaturalist dataset that returns raw compressed bytes instead of decoded audio.

    Extends :class:`~alp_data.datasets.INaturalist` by skipping the decode step
    in :meth:`_process`.  Each sample contains::

        {
            "audio_bytes":  bytes,        # raw compressed audio (no decode)
            "audio_format": str,          # e.g. "FLAC", "WAV", "MP3"
            "sample_rate":  int | None,   # *target* SR (from constructor)
            ...metadata...               # all other metadata columns
        }

    The ``"audio"`` key produced by the parent class is **never** present.

    Like :class:`~bioacoustic_ssl.data.datasets.XenoCantoRaw`, the primary
    use-case is to pair this dataset with a bytes-aware
    :class:`~bioacoustic_ssl.data.transforms.TimeShift`, which decides the crop
    window *before* decoding so that only the required frames are ever decoded.

    .. note::
        This class downloads the **entire** compressed file before handing
        control to ``TimeShift``.

    Parameters
    ----------
    *args :
        Forwarded verbatim to :class:`~alp_data.datasets.INaturalist`.
    **kwargs :
        Forwarded verbatim to :class:`~alp_data.datasets.INaturalist`.

    Examples
    --------
    >>> from bioacoustic_ssl.data.datasets import INaturalistRaw
    >>> ds = INaturalistRaw(split="train", sample_rate=32000)
    >>> sample = ds[0]
    >>> isinstance(sample["audio_bytes"], bytes)
    True
    >>> "audio" not in sample
    True
    """

    info = DatasetInfo(
        name="inaturalist-raw",
        owner="gagan; david",
        split_paths={
            "train": f"{DATA_HOME}/inaturalist/v0.1.0/raw/train_20260201_v3.csv",
            "train_unseen": f"{DATA_HOME}/inaturalist/v0.1.0/raw/train_unseen_20260201_v3.csv",
            "val": f"{DATA_HOME}/inaturalist/v0.1.0/raw/val_20260201_v3.csv",
            "val_unseen": f"{DATA_HOME}/inaturalist/v0.1.0/raw/val_unseen_20260201_v3.csv",
            "all": f"{DATA_HOME}/inaturalist/v0.1.0/raw/all_20260201_v3.csv",
            "all_unseen": f"{DATA_HOME}/inaturalist/v0.1.0/raw/all_unseen_20260201_v3.csv",
        },
        version="0.1.0",
        description="iNaturalist audio dataset returning raw compressed audio bytes (no decode).",
        sources=["iNaturalist"],
        license="CC BY-NC 4.0, CC BY 4.0, CC0 1.0",
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
                return anypath(self.data_root) / row[path_column]

        # Fall back to original variable-rate files.
        return anypath(self.data_root) / row[self._originals_path_column]

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
