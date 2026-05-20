"""NOAA dataset collection"""

from typing import Any, Iterator

import librosa
import numpy as np
from esp_data import Dataset, DatasetConfig, DatasetInfo
from esp_data.backends import BackendType
from esp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

_GCS_ROOT = "gs://noaa-passive-bioacoustic"


# @register_dataset
class NOAA(Dataset):
    """NOAA Passive Bioacoustics Dataset.

    Description
    -----------
    NOAA is a large scale colleciton project of passive acoustic recordings from various marine environments,
    with a focus on marine mammals. The dataset includes annotated soundscape recordings and extracted clips
    across multiple subsets, each representing different geographic locations and recording conditions.
    Currently, only the PIFSC-10 subset is available Pacific Islands Fisheries Science Center (PIFSC).
    Metadata are processed on the fly and other datasets have a different format and may not work immediatly.
    The native sample rate is 10kHz.


    References
    ----------
    https://storage.cloud.google.com/noaa-passive-bioacoustic/pifsc/README.md

    Examples
    --------
    >>> from esp_data.datasets import NOAA
    >>> dataset = NOAA(split="PIFSC-10")

    """

    info = DatasetInfo(
        name="noaa",
        owner="moritz",
        split_paths={
            # other datasets may have a different format and not work with the current setup
            "PIFSC-10": f"{_GCS_ROOT}/pifsc/products/detections/annotations.csv",  # pipan
        },
        version="0.1.0",
        description=("[MISSING]"),
        sources=["NOAA"],
        license="CC-BY-4.0, CC0",
    )

    _sample_rates: list[int] = [10_000]

    _originals_path_column: dict[str, str] = {"PIFSC-10": "flac_compressed_xwav_object"}

    _data_path_replaces: dict[str, tuple[str, str]] = {"PIFSC-10": ("/pipan/", "/pipan_10/")}

    def __init__(
        self,
        split: str = "PIFSC-10",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """Initialize the BirdSet dataset.

        Parameters
        ----------
        split : str, default="PIFSC-10"
            Split to load (key in info.split_paths).
        output_take_and_give : dict[str, str], optional
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int, optional
            If set, audio is resampled to this rate. 10kHz is native.
        data_root : str | AnyPathT, optional
            Root directory prepended to relative audio paths.  Defaults to the
            GCS path for this dataset version.
        backend : BackendType, optional
            Backend engine ("pandas" or "polars"), by default "polars".
        streaming : bool, optional
            Whether to use streaming mode, by default False.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self._load()
        self.sample_rate = sample_rate

        if data_root is None:
            self.data_root = anypath(f"{_GCS_ROOT}/")
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        """Return the columns of the dataset."""
        return list(self._data.columns)

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return sample rates supported by this dataset.

        Pre-resampled audio is loaded directly when a matching column exists
        in the data; otherwise the original audio is resampled on-the-fly.

        Returns
        -------
        list[int]
            Sorted sample rates (Hz) declared in ``_sample_rate_paths``.
        """
        return self._sample_rates

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(location, streaming=self._streaming)
        self._data_paths = self._data.get_unique("flac_compressed_xwav_object")

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NOAA", dict[str, Any]]:
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __len__(self) -> int:
        if self._data is None:
            raise RuntimeError("No split has been loaded yet.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. Iterate over the dataset instead."
            )
        return len(self._data)

    def _process(self, audio_path: str) -> dict[str, Any]:
        row: dict[str, Any] = {}
        audio_path_col = self._originals_path_column[self.split]
        annotations = self._data.filter_isin(column=audio_path_col, values=[audio_path]).unwrap
        if len(annotations) > 0:
            row["annotations"] = annotations

        audio_path = audio_path.replace(*self._data_path_replaces[self.split])
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

        row["audio"] = audio
        row["sample_rate"] = sr
        row[self._originals_path_column[self.split]] = audio_path

        if self.output_take_and_give:
            item = {}
            for key, value in self.output_take_and_give.items():
                item[value] = row[key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._data_paths[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for row in self._data_paths:
            yield self._process(row)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version}), split={self.split}"
        return (
            f"{base}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
