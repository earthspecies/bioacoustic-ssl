"""NOAA PIFSC detection-event dataset (thin alias over :class:`NOAA`).

The PIFSC (Pacific Islands Fisheries Science Center) subset of the NOAA
Passive Bioacoustics archive ships a single annotation file at::

    gs://noaa-passive-bioacoustic/pifsc/products/detections/annotations.csv

Every row is one labelled **detection event** inside a long HARP ``xwav``
(FLAC) recording.  The actual loading — in-file offset computation, lazy
windowed GCS reads, resampling — lives in :class:`~soundscape_ssl.data.datasets.noaa.NOAA`,
where PIFSC is registered as the ``PIFSC-10`` split (see that module for the
subchunk time-bookkeeping details).

This subclass is a convenience wrapper that defaults to the PIFSC split; it is
exactly equivalent to ``NOAA(split="PIFSC-10", ...)``.
"""

from esp_data import DatasetInfo, register_dataset

from .noaa import _GCS_ROOT, NOAA


@register_dataset
class PIFSC(NOAA):
    """NOAA PIFSC detection events served as fixed-length audio clips.

    Convenience wrapper over :class:`~soundscape_ssl.data.datasets.noaa.NOAA`
    that fixes the split to PIFSC.  See :class:`NOAA` for the full parameter
    and return-value documentation; the only difference is that ``split``
    defaults to (and is restricted to) the PIFSC subset.

    Examples
    --------
    >>> from soundscape_ssl.data.datasets import PIFSC
    >>> ds = PIFSC(sample_rate=10_000)
    >>> item = ds[0]
    >>> item["audio"].shape, item["sample_rate"], item["label"]  # doctest: +SKIP
    ((50000,), 10000, 'Other')
    """

    info = DatasetInfo(
        name="pifsc",
        owner="moritz",
        split_paths={
            "PIFSC-10": f"{_GCS_ROOT}/pifsc/products/detections/annotations.csv",
        },
        version="0.1.0",
        description=(
            "NOAA PIFSC (Pacific Islands Fisheries Science Center) passive "
            "acoustic detection events. One labelled clip per annotation row, "
            "centred on the event within long HARP xwav (FLAC) recordings. "
            "Native sample rate 10 kHz. Alias of NOAA(split='PIFSC-10')."
        ),
        sources=["NOAA"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(self, split: str = "PIFSC-10", *args, **kwargs) -> None:
        super().__init__(*args, split=split, **kwargs)
