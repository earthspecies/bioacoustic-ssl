"""NOAA SanctSound animal-detection dataset (thin alias over :class:`NOAA`).

The SanctSound subset of the NOAA Passive Bioacoustics archive ships detection
products as many small per-deployment, per-category CSVs under::

    gs://noaa-passive-bioacoustic/sanctsound/products/detections/<site>/
        sanctsound_<site>_<dep>_<category>/data/<file>.csv

Each row of an animal, time-localised detector is one **detection event**
timestamped in absolute UTC.  The actual loading — globbing the detection
files, filtering to animal/localised events, resolving each event's UTC time
against its deployment's 96 kHz audio listing, windowed GCS reads, resampling —
lives in :class:`~bioacoustic_ssl.data.datasets.noaa.NOAA`, where SanctSound is
registered as the ``SANCTSOUND`` split (see that module for the UTC
time-bookkeeping and the animal/localised filtering details).

This subclass is a convenience wrapper that defaults to the SanctSound split;
it is exactly equivalent to ``NOAA(split="SANCTSOUND", ...)``.
"""

from alp_data import DatasetInfo, register_dataset

from .noaa import _SANCTSOUND_ROOT, NOAA


@register_dataset
class SanctSound(NOAA):
    """NOAA SanctSound animal detection events served as fixed-length clips.

    Convenience wrapper over :class:`~bioacoustic_ssl.data.datasets.noaa.NOAA`
    that fixes the split to SanctSound.  See :class:`NOAA` for the full
    parameter and return-value documentation; the only difference is that
    ``split`` defaults to (and is restricted to) the SanctSound subset.

    Only animal, time-localised detections are loaded — non-animal categories
    (ships, explosions, sonar, …) and aggregated daily/hourly presence products
    are excluded (see the ``noaa`` module for details).  Each item's ``label``
    is the detection category (e.g. ``"bluewhale"``, ``"killerwhale"``), with
    the originating ``site``, ``deployment`` and ``category`` passed through.

    Examples
    --------
    >>> from bioacoustic_ssl.data.datasets import SanctSound
    >>> ds = SanctSound(sample_rate=32_000)
    >>> item = ds[0]
    >>> item["audio"].shape, item["sample_rate"], item["label"]  # doctest: +SKIP
    ((160000,), 32000, 'bluewhale')
    """

    info = DatasetInfo(
        name="sanctsound",
        owner="moritz",
        split_paths={
            "SANCTSOUND": f"{_SANCTSOUND_ROOT}/products/detections",
        },
        version="0.1.0",
        description=(
            "NOAA SanctSound passive acoustic animal detection events. One "
            "labelled clip per localised animal detection, centred on the event "
            "within the deployment's 96 kHz FLAC recordings (resolved by UTC "
            "timestamp). Non-animal and aggregated-presence products are "
            "excluded. Alias of NOAA(split='SANCTSOUND')."
        ),
        sources=["NOAA"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(
        self,
        split: str = "SANCTSOUND",
        max_event_seconds: float | None = 6,
        min_event_gap_seconds: float | None = 300,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, split=split, max_event_seconds=max_event_seconds, min_event_gap_seconds=min_event_gap_seconds, **kwargs)
