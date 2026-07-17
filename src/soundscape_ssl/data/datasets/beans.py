"""BEANS dataset variant for bytes-based (deferred-decode) audio loading."""

from typing import Any

from alp_data import DatasetInfo, register_dataset
from alp_data.datasets import Beans
from alp_data.io import anypath, filesystem_from_path


@register_dataset
class BeansRaw(Beans):
    """BEANS dataset that returns raw compressed bytes instead of decoded audio.

    Extends :class:`~alp_data.datasets.Beans` by skipping the decode step in
    :meth:`_process`.  Each sample contains::

        {
            "audio_bytes":  bytes,        # raw compressed audio (no decode)
            "audio_format": str,          # e.g. "FLAC", "WAV", "MP3"
            "sample_rate":  int | None,   # *target* SR (from constructor)
            ...metadata...               # all other metadata columns
        }

    The ``"audio"`` key produced by the parent class is **never** present.

    Pair this with a bytes-aware
    :class:`~soundscape_ssl.data.transforms.TimeShift`, which decides the crop
    window *before* decoding so that only the required frames are ever decoded.
    """

    info = DatasetInfo(
        name="beans-raw",
        owner="gagan",
        split_paths=Beans.info.split_paths,
        version=Beans.info.version,
        description=Beans.info.description,
        sources=Beans.info.sources,
        license=Beans.info.license,
    )

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
            ``"sample_rate"`` (target SR) added.  The ``"audio"`` key is never
            present.
        """
        audio_path = anypath(self.data_root) / row["local_path"]

        fs = filesystem_from_path(audio_path)
        with fs.open(str(audio_path), "rb") as fh:
            audio_bytes: bytes = fh.read()

        row["audio_bytes"] = audio_bytes
        row["audio_format"] = anypath(audio_path).suffix.lstrip(".").upper()
        row["sample_rate"] = self.sample_rate  # target SR; may be None

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
        else:
            item = row

        return item
