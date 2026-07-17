"""NOAA Passive Bioacoustics detection-event datasets.

The NOAA Passive Bioacoustics archive (``gs://noaa-passive-bioacoustic``) hosts
many sub-collections, several of which ship detection-annotation files: each
row of an annotation CSV is one labelled **detection event** — a time region
inside a long, multi-hour recording (typically a HARP ``xwav`` re-encoded as
FLAC).

:class:`NOAA` is a generic loader over those annotation files.  It yields *one
clip per event* — a fixed-duration window centred on the event — read lazily
from GCS.  Each annotated sub-collection is exposed as a named **split**.  A
split is fully described by a :class:`_NOAASplit` spec that knows:

* where its annotation CSV lives,
* which column holds the audio-object path and how to rewrite it to the actual
  (sample-rate-specific) audio path,
* how to turn each annotation row into an in-file ``(start, end)`` offset in
  seconds, and
* the native sample rate.

Adding another annotated NOAA collection is therefore a matter of writing one
``_NOAASplit`` entry (and, if its time bookkeeping differs, one small offset
function) — no new class required.

Event time-bookkeeping (PIFSC)
------------------------------
PIFSC recordings are stored as a concatenation of fixed-length HARP ``xwav``
*subchunks*.  In the audio (sample) domain the subchunks are contiguous, so an
event's offset from the start of the FLAC file is::

    offset_seconds = subchunk_index * SUBCHUNK_SECONDS + rel_subchunk_seconds

with ``SUBCHUNK_SECONDS = 75``.  (The per-subchunk *UTC* timestamps are **not**
contiguous — PIFSC deployments are duty-cycled — so only the audio domain is
gap-free, which is what matters for reading the FLAC.)  This mirrors the offset
computation in ``notebooks/testing.ipynb``.

Event time-bookkeeping (SanctSound)
-----------------------------------
SanctSound (``gs://noaa-passive-bioacoustic/sanctsound``) is organised very
differently from PIFSC.  Detections live in many small per-deployment,
per-category CSVs under ``products/detections/<site>/sanctsound_<site>_<dep>_<category>/data/``,
and each row is timestamped in **absolute UTC** (``ISOStartTime`` and, for
true-span detectors, ``ISOEndTime``) rather than as an in-file offset.  The
deployment's audio is a chronological run of ~2 h FLAC files named
``SanctSound_<SITE>_<DEP>_<serial>_<YYYYMMDDTHHMMSSZ>.flac``.  Mapping an event
to a clip therefore means resolving its UTC start against the deployment's
audio directory listing: the containing file is the last one whose embedded
start time precedes the event, and the in-file offset is ``event_utc -
file_start_utc``.  Events falling in a recording gap (offset beyond the gap to
the next file) or in a deployment with no audio are dropped at load time.

Only **animal**, **time-localised** detectors are loaded: categories such as
``ships``/``explosions``/``sonar``/``mfa``/``unknownimpulse`` are excluded, as
are aggregated daily/hourly/minute presence products (``*_1d``, ``*_1h``,
``*_1hr``, ``*_1min``, ``googleai_*``, or any file carrying ``Effort`` /
``Count`` / proportion columns) — a daily presence flag cannot localise a clip.
What remains is true-span events (e.g. ``bocaccio``, ``plainfinmidshipman``,
``killerwhale``, ``atlanticcod``, ``pinnipeds``) and fine-grained point
detections (``bluewhale``); negative rows (``Presence == 0``) are dropped.
Some true-span detectors (``killerwhale``, ``plainfinmidshipman``,
``pinnipeds``) mark multi-hour — occasionally multi-day — *presence intervals*
rather than single calls; events whose span exceeds ``NOAA.max_event_seconds``
(default 60 s) are dropped at load time, so every retained event is a localised
detection.

Audio loading
-------------
Audio is read lazily with :func:`alp_data.io.read_audio_by_time`, which issues
HTTP ``Range`` requests against GCS and therefore downloads only the FLAC
header plus the requested window — never the full multi-hour file.
"""

import bisect
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

import gcsfs
import librosa
import numpy as np
import pandas as pd
from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import audio_stereo_to_mono
from alp_data.io.read_utils import get_audio_info, read_audio_by_time

logger = logging.getLogger(__name__)

_GCS_ROOT = "gs://noaa-passive-bioacoustic"

#: Length (seconds) of one HARP ``xwav`` subchunk in the audio domain.
SUBCHUNK_SECONDS = 75.0

#: Root of the SanctSound collection within the bucket.
_SANCTSOUND_ROOT = f"{_GCS_ROOT}/sanctsound"

#: SanctSound detection categories that are **not** animal sounds.  Excluded.
_SANCTSOUND_NON_ANIMAL = frozenset(
    {"ships", "explosions", "sonar", "mfa", "unknownimpulse"}
)

#: Category-name suffixes / markers identifying aggregated presence products
#: (daily/hourly/minute bins, GoogleAI proportions) — not time-localised events.
_SANCTSOUND_AGG_SUFFIXES = ("_1d", "_1h", "_1hr", "_1min")

#: Columns whose presence marks an aggregated/effort product rather than a
#: per-event detection file.  Any of these ⇒ the file is skipped.
_SANCTSOUND_AGG_COLUMNS = frozenset({"Effort", "Count"})

#: Number of concurrent GCS reads used to load the many small SanctSound CSVs.
_SANCTSOUND_READ_WORKERS = 16

#: Type of a per-split offset function: ``df -> (start_seconds, end_seconds)``,
#: each a pandas Series of in-file offsets (seconds) aligned to ``df``.
OffsetFn = Callable[[pd.DataFrame], tuple["pd.Series", "pd.Series"]]


@dataclass(frozen=True)
class _NOAASplit:
    """Specification of one annotated NOAA sub-collection.

    A split is loaded one of two ways.  Most splits (e.g. PIFSC) read a single
    annotation CSV whose rows carry both an audio-object path (``path_column``)
    and the data to compute in-file offsets (``offsets``).  Splits whose
    annotations are spread across many files and/or timestamped in absolute UTC
    (e.g. SanctSound) instead supply a ``loader`` that builds the whole event
    table — including a resolved ``audio_path`` and in-file
    ``event_start_seconds`` / ``event_end_seconds`` — and a ``resolver`` that
    returns each event's ``(audio_path, start_seconds, end_seconds)``.

    Parameters
    ----------
    annotations : str
        ``gs://`` path to the split's annotation CSV (loader-based splits pass a
        representative root path here for display in ``split_paths``).
    sample_rate : int
        Native sample rate (Hz) of this split's audio.
    path_column : str, optional
        Column holding the audio-object path for each annotation row.  Required
        for CSV-based splits; ``None`` for ``loader``-based splits.
    path_replace : tuple[str, str], optional
        ``(old, new)`` substring replacement applied to ``path_column`` to
        turn the annotation path into the actual audio path (e.g. selecting a
        sample-rate-specific product directory).
    offsets : OffsetFn, optional
        Function mapping the loaded annotation DataFrame to a pair of pandas
        Series ``(event_start_seconds, event_end_seconds)`` — the in-file
        (audio-domain) offsets of each event.  Required for CSV-based splits.
    label_column : str, optional
        Column to surface under the normalised ``"label"`` output key.  The
        original column is also passed through unchanged.
    path_repair : Callable[[str], str], optional
        Applied to the raw annotation path *before* ``path_replace`` to correct
        known data-quality issues in the annotation file.  Must be a no-op for
        unaffected paths.  ``None`` (default) means no repair.
    variant_resolver : Callable, optional
        ``(dataset, event, repaired_raw_path) -> (path, start_seconds,
        end_seconds) | None``.  When ``prefer_highres`` is set, this is given a
        chance to redirect the read to a higher-resolution product, returning
        the alternate audio path together with the clip window **in that file's
        own timebase** (which generally differs from the 10 kHz timebase — the
        products are segmented differently).  Returns ``None`` to fall back to
        ``path_replace`` (the canonical product).  ``None`` (default) disables
        variant selection for the split.
    loader : Callable[[NOAA], pd.DataFrame], optional
        Builds the full event table for the split, replacing the default
        single-CSV read.  Its DataFrame must include ``event_start_seconds`` /
        ``event_end_seconds`` (in-file, audio-domain offsets) plus whatever
        columns the ``resolver`` needs.  ``None`` (default) uses the CSV loader.
    resolver : Callable[[NOAA, dict], tuple[str, float, float]], optional
        Maps a fully-loaded event to ``(audio_path, start_seconds,
        end_seconds)``, replacing the ``path_column`` / ``path_replace`` /
        ``variant_resolver`` resolution path.  Pairs with ``loader``.
    """

    annotations: str
    sample_rate: int
    path_column: str | None = None
    path_replace: tuple[str, str] | None = None
    offsets: OffsetFn | None = None
    label_column: str | None = None
    path_repair: Callable[[str], str] | None = None
    variant_resolver: Callable[["NOAA", dict[str, Any], str],
                               tuple[str, float, float] | None] | None = None
    loader: Callable[["NOAA"], pd.DataFrame] | None = None
    resolver: Callable[["NOAA", dict[str, Any]],
                       tuple[str, float, float]] | None = None


def _pifsc_offsets(df: pd.DataFrame) -> tuple["pd.Series", "pd.Series"]:
    """Compute in-file event offsets for the PIFSC subchunk layout.

    Subchunks are contiguous in the audio domain, so the offset is
    ``subchunk_index * SUBCHUNK_SECONDS`` plus the within-subchunk seconds.
    """
    base = df["subchunk_index"] * SUBCHUNK_SECONDS
    return base + df["begin_rel_subchunk"], base + df["end_rel_subchunk"]


def _pifsc_repair_path(path: str) -> str:
    """Fix truncated PIFSC deployment directories (``_0/audio/``).

    Some annotation rows carry a placeholder deployment directory
    ``pipan_<site>_0`` instead of the real one; the true deployment number is
    the 3rd ``_``-token of the filename (e.g. ``Hawaii_K_14_...`` -> ``14``).
    Valid paths never contain ``_0/audio/`` (deployments start at ``_01``), so
    this returns ``path`` unchanged when not affected.
    """
    if "_0/audio/" not in path:
        return path
    filename = path.rsplit("/", 1)[-1]
    parts = filename.split("_")
    if len(parts) < 3:
        return path
    m = re.match(r"\d+", parts[2])
    if not m:
        return path
    return path.replace("_0/audio/", f"_{int(m.group(0)):02d}/audio/")


def _pifsc_200k_path(path: str) -> str:
    """Construct the 200 kHz (``pipan_200``) path for a PIFSC annotation path.

    The 200 kHz product uses a differently-constructed layout than the 10 kHz
    one, all derivable from the filename ``<Site>_<Chan>_<dep>_...``::

        pipan_200/<site>_<chan>/pipan_<site>_<chan>_<dep:02d>_200/audio/<stem>.x.flac

    where ``<stem>`` is the filename with its decimation token (``.d20`` /
    ``.df20`` / ``.df32``, dotted or underscored) stripped.  This locates the
    200 kHz file holding the recording's **first** subchunk; later subchunks
    live in subsequent files of the same deployment (see
    :func:`_pifsc_resolve_200k`).
    """
    fn = path.rsplit("/", 1)[-1]
    parts = fn.split("_")
    site, chan = parts[0].lower(), parts[1].lower()
    dep = int(re.match(r"\d+", parts[2]).group(0))
    stem = re.sub(r"[._]d?f?\d+\.x\.flac$|\.x\.flac$", "", fn)
    return (
        f"{_GCS_ROOT}/pifsc/audio/pipan_200/"
        f"{site}_{chan}/pipan_{site}_{chan}_{dep:02d}_200/audio/{stem}.x.flac"
    )


def _pifsc_filename_utc(name: str) -> datetime | None:
    """Parse the UTC start time encoded in a PIFSC filename (``..._YYMMDD_HHMMSS``)."""
    m = re.search(r"_(\d{6}_\d{6})", name)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def _pifsc_resolve_200k(
    ds: "NOAA", event: dict[str, Any], raw_path: str
) -> tuple[str, float, float] | None:
    """Map a PIFSC event to its 200 kHz file and clip window.

    The 10 kHz product concatenates a deployment's raw recordings into long
    files indexed by a global ``subchunk_index``; the 200 kHz product keeps each
    raw recording as a separate file (~30 subchunks of 75 s).  This walks the
    deployment's 200 kHz files in chronological order from the one holding the
    recording's first subchunk (:func:`_pifsc_200k_path`), accumulating each
    file's subchunk count (``duration / 75``) until it reaches the event's
    global ``subchunk_index`` — yielding the containing file and the *local*
    subchunk index within it.

    The mapping is then cross-checked against the event's ``begin_utc``: the
    chosen file's filename timestamp must bracket the subchunk's start time
    (``begin_utc - begin_rel_subchunk``).  Any mismatch — a missing file, a
    non-standard segment, or a window that overruns the file — returns ``None``
    so the caller falls back to the 10 kHz product.  Returns
    ``(path, start_seconds, end_seconds)`` in the 200 kHz file's timebase.
    """
    try:
        g = int(event["subchunk_index"])
        begin_rel = float(event["begin_rel_subchunk"])
        end_rel = float(event["end_rel_subchunk"])
    except (KeyError, TypeError, ValueError):
        return None

    start_path = _pifsc_200k_path(raw_path)
    dep_dir, _, start_name = start_path.rpartition("/")
    listing = ds._dir_listing(dep_dir)
    i0 = listing["index"].get(start_name)
    if i0 is None:
        return None

    # Walk forward accumulating subchunk counts to locate the containing file.
    names = listing["names"]
    acc = 0
    j = i0
    while True:
        if j >= len(names):
            return None
        file_path = f"{dep_dir}/{names[j]}"
        n_sub = round(ds._duration(file_path) / SUBCHUNK_SECONDS)
        if acc + n_sub > g:
            local = g - acc
            break
        acc += n_sub
        j += 1

    # Cross-check against begin_utc: the file's timestamp must bracket the
    # subchunk start (rejects missing files / non-standard segmentation).
    try:
        subchunk_utc = datetime.fromisoformat(str(event["begin_utc"])) - timedelta(
            seconds=begin_rel
        )
    except ValueError:
        return None
    lo = _pifsc_filename_utc(names[j])
    if lo is None or subchunk_utc < lo - timedelta(seconds=2):
        return None
    if j + 1 < len(names):
        hi = _pifsc_filename_utc(names[j + 1])
        if hi is not None and subchunk_utc >= hi:
            return None

    # Clip window in the 200 kHz file's own (local) timebase.
    event_start = local * SUBCHUNK_SECONDS + begin_rel
    event_end = local * SUBCHUNK_SECONDS + end_rel
    if ds.clip_duration is None:
        start_s, end_s = event_start, event_end
    else:
        middle = (event_start + event_end) / 2.0
        half = ds.clip_duration / 2.0
        start_s = max(0.0, middle - half)
        end_s = start_s + ds.clip_duration

    if end_s > ds._duration(file_path):  # window overruns the file -> fall back
        return None
    return file_path, start_s, end_s


def _sanctsound_filename_utc(name: str) -> datetime | None:
    """Parse the UTC start time encoded in a SanctSound audio filename.

    Two filename conventions occur in the bucket, both UTC::

        SanctSound_<SITE>_<DEP>_<serial>_20181031T220000Z.flac  (YYYYMMDDTHHMMSS)
        SanctSound_<SITE>_<DEP>_<serial>_190325210000.flac      (YYMMDDHHMMSS)
    """
    m = re.search(r"_(\d{8}T\d{6})Z", name)
    if m is not None:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    m = re.search(r"_(\d{12})\.flac$", name)
    if m is not None:
        return datetime.strptime(m.group(1), "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return None


def _sanctsound_is_aggregated(category: str) -> bool:
    """Return ``True`` for aggregated presence categories (not localised events)."""
    return "googleai" in category or category.endswith(_SANCTSOUND_AGG_SUFFIXES)


def _sanctsound_audio_dir(site: str, dep: str) -> str:
    """Return the ``gs://`` audio directory for a SanctSound site/deployment."""
    return f"{_SANCTSOUND_ROOT}/audio/{site}/sanctsound_{site}_{dep}/audio"


def _sanctsound_read_csv(
    fs: "gcsfs.GCSFileSystem", csv_path: str
) -> tuple[str, str, str, pd.DataFrame] | None:
    """Read one detection CSV, keeping only animal, time-localised event rows.

    Returns ``(site, deployment, category, dataframe)`` or ``None`` when the
    file is not an animal per-event detection file.  Localisation is decided by
    schema: a ``ISOEndTime`` column marks true spans; otherwise the file must be
    a clean ``{ISOStartTime, Presence}`` point-detection file with sub-day
    timestamps.  Aggregated products (daily/hourly bins, effort/count columns)
    and ``Presence == 0`` rows are dropped.
    """
    folder = csv_path.split("/")[-3].split("_")
    if len(folder) < 4:
        return None
    site, dep, category = folder[1], folder[2], "_".join(folder[3:])
    if category in _SANCTSOUND_NON_ANIMAL or _sanctsound_is_aggregated(category):
        return None

    with fs.open(csv_path) as fh:
        df = pd.read_csv(fh, encoding="utf-8-sig")
    cols = set(df.columns)
    if _SANCTSOUND_AGG_COLUMNS & cols or "ISOStartTime" not in cols:
        return None

    df["event_start_utc"] = pd.to_datetime(df["ISOStartTime"], utc=True, errors="coerce")
    has_end = "ISOEndTime" in cols
    if has_end:
        df["event_end_utc"] = pd.to_datetime(df["ISOEndTime"], utc=True, errors="coerce")
    else:
        # Point detections: require a clean {ISOStartTime, Presence} schema and
        # sub-day resolution (reject date-only / midnight-binned daily files).
        if not cols <= {"ISOStartTime", "Presence"}:
            return None
        starts = df["event_start_utc"].dropna()
        if starts.empty or (starts == starts.dt.normalize()).all():
            return None
        df["event_end_utc"] = df["event_start_utc"]

    df = df[df["event_start_utc"].notna()]
    if "Presence" in cols:
        df = df[df["Presence"] == 1]
    if df.empty:
        return None
    return site, dep, category, df


def _sanctsound_load(ds: "NOAA") -> pd.DataFrame:
    """Build the SanctSound event table: animal, time-localised detections.

    Globs every per-deployment detection CSV, keeps the animal/localised rows
    (see :func:`_sanctsound_read_csv`), then resolves each event's absolute UTC
    time against its deployment's audio directory listing into a concrete
    ``audio_path`` and in-file ``event_start_seconds`` / ``event_end_seconds``.
    Events with no audio, or falling in a recording gap, are dropped.  The CSV
    reads (many small files) are issued concurrently; the per-deployment
    directory listings are cached and reused.
    """
    fs = ds._fs
    csv_paths = fs.glob(
        f"{_GCS_ROOT.split('://', 1)[1]}/sanctsound/products/detections/*/*/data/*.csv"
    )
    with ThreadPoolExecutor(max_workers=_SANCTSOUND_READ_WORKERS) as pool:
        loaded = [r for r in pool.map(lambda p: _sanctsound_read_csv(fs, p), csv_paths) if r]

    records: list[dict[str, Any]] = []
    for site, dep, category, df in loaded:
        listing = ds._dir_listing(_sanctsound_audio_dir(site, dep), _sanctsound_filename_utc)
        starts, names = listing["starts"], listing["names"]
        if not starts:
            continue
        audio_dir = _sanctsound_audio_dir(site, dep)
        label = re.sub(r"_manual$", "", category)
        # Columns to surface besides the derived audio/offset fields.
        extra_cols = [c for c in df.columns if c not in {"event_start_utc", "event_end_utc"}]
        for row in df.itertuples(index=False):
            r = row._asdict()
            t0, t1 = r["event_start_utc"].to_pydatetime(), r["event_end_utc"].to_pydatetime()
            i = bisect.bisect_right(starts, t0) - 1
            if i < 0:
                continue
            rel = (t0 - starts[i]).total_seconds()
            if i + 1 < len(starts) and rel >= (starts[i + 1] - starts[i]).total_seconds():
                continue  # event falls in a recording gap
            rec = {c: r[c] for c in extra_cols}
            rec.update(
                site=site,
                deployment=dep,
                category=category,
                label=label,
                audio_path=f"{audio_dir}/{names[i]}",
                event_start_seconds=rel,
                event_end_seconds=rel + (t1 - t0).total_seconds(),
            )
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _sanctsound_resolve(
    ds: "NOAA", event: dict[str, Any]
) -> tuple[str, float, float]:
    """Return the pre-resolved ``(audio_path, start_seconds, end_seconds)``.

    SanctSound events are fully resolved at load time by :func:`_sanctsound_load`
    (audio path + in-file offsets), and the clip window is centred by the shared
    :meth:`NOAA._load` logic, so resolution here is a simple lookup.
    """
    return (
        str(event["audio_path"]),
        float(event["start_seconds"]),
        float(event["end_seconds"]),
    )


#: Registry of annotated NOAA splits.  Add a new entry to support another
#: annotated sub-collection from the bucket.
_SPLITS: dict[str, _NOAASplit] = {
    "PIFSC-10": _NOAASplit(
        annotations=f"{_GCS_ROOT}/pifsc/products/detections/annotations.csv",
        path_column="flac_compressed_xwav_object",
        path_replace=("/pipan/", "/pipan_10/"),
        offsets=_pifsc_offsets,
        sample_rate=10_000,
        label_column="label",
        path_repair=_pifsc_repair_path,
        variant_resolver=_pifsc_resolve_200k,
    ),
    "SANCTSOUND": _NOAASplit(
        annotations=f"{_SANCTSOUND_ROOT}/products/detections",
        sample_rate=96_000,
        label_column="label",
        loader=_sanctsound_load,
        resolver=_sanctsound_resolve,
    ),
}


@register_dataset
class NOAA(Dataset):
    """NOAA Passive Bioacoustics detection events served as audio clips.

    Description
    -----------
    A generic loader over the annotation files of the NOAA Passive
    Bioacoustics archive.  Each annotated sub-collection is a **split** (see
    :data:`_SPLITS`); selecting one yields **one item per detection event**.
    For each event the in-file offset is computed by the split's offset
    function, and a clip of ``clip_duration`` seconds centred on the event
    midpoint is read directly from GCS.

    Currently available splits:

    - ``PIFSC-10``: Pacific Islands Fisheries Science Center, 10 kHz product.
    - ``SANCTSOUND``: SanctSound project, animal detections across all sites
      and deployments, 96 kHz audio (one clip per localised animal detection).

    Each item is a dict containing the standard audio keys plus every column
    from the annotation row (so split-specific metadata is preserved)::

        {
            "audio":               np.ndarray,  # float32 mono
            "sample_rate":         int,
            "audio_path":          str,         # resolved gs:// audio path
            "start_seconds":       float,       # clip window start in file
            "end_seconds":         float,       # clip window end in file
            "event_start_seconds": float,       # raw event span start in file
            "event_end_seconds":   float,       # raw event span end in file
            "label":               Any,         # if the split sets label_column
            ...                                 # all other annotation columns
        }

    Parameters
    ----------
    split : str, default="PIFSC-10"
        Split to load (key in ``info.split_paths`` / :data:`_SPLITS`).
    output_take_and_give : dict[str, str], optional
        Optional mapping of ``original_key -> new_key`` that filters and
        renames output fields before returning each item.
    sample_rate : int, optional
        Target sample rate in Hz.  When set and different from the split's
        native rate, audio is resampled on the fly with ``librosa``.
    clip_duration : float or None, default=5.0
        Length (seconds) of the clip returned per event, centred on the event
        midpoint.  When ``None``, the exact event span is returned instead.
    max_event_seconds : float or None, default=60.0
        Maximum annotated event span (seconds); events whose span exceeds this
        are dropped at load time.  Multi-hour — occasionally multi-day —
        presence-interval detectors (e.g. SanctSound's ``killerwhale`` /
        ``plainfinmidshipman`` / ``pinnipeds``) mark presence over a window
        rather than a localised call, so a fixed clip cannot reliably contain
        the sound (and centring on the span midpoint would read past the file's
        EOF).  Excluding them keeps every retained event a localised detection.
        ``None`` disables the filter (keep all events, including over-long ones).
    min_event_gap_seconds : float or None, default=60.0
        Minimum spacing (seconds) between kept events within one audio
        recording.  Detectors log calls note-by-note, so consecutive events are
        often a fraction of a second apart (e.g. SanctSound ``bluewhale``: 99%
        within 5 s), producing near-identical overlapping clips that dominate
        and add little for SSL.  Events are de-duplicated greedily per recording
        — keeping an event only if its onset is at least this many seconds after
        the last kept one — which both removes redundancy and rebalances away
        from over-represented detectors.  ``None`` disables de-duplication.
    prefer_highres : bool, default=True
        When ``True`` and the split defines a ``variant_resolver``, each event is
        served from the highest-resolution product available for its recording
        (e.g. PIFSC's 200 kHz product, better for downsampling to a 32 kHz
        target), falling back to the canonical product (10 kHz) otherwise.  The
        10 kHz product concatenates a deployment's raw recordings into long
        files while the 200 kHz product keeps them separate, so the resolver
        maps the event's global ``subchunk_index`` to the right 200 kHz file and
        offset (verified against ``begin_utc``); events it cannot confidently map
        read from 10 kHz.  The chosen source is reflected in each item's
        ``"audio_path"`` and native ``"sample_rate"``.

        .. note::
            With ``prefer_highres=True`` and ``sample_rate=None``, items can
            have **mixed native rates** (200 kHz vs 10 kHz).  Set a target
            ``sample_rate`` (e.g. ``32000``) for uniform output.
    backend : BackendType, optional
        Accepted for interface compatibility; annotations are loaded with
        pandas.
    streaming : bool, optional
        When ``True``, ``__len__`` / ``__getitem__`` are disabled and items
        are produced by iteration only.  Defaults to ``False`` (map-style).

    References
    ----------
    https://storage.cloud.google.com/noaa-passive-bioacoustic/pifsc/README.md

    Examples
    --------
    >>> from soundscape_ssl.data.datasets import NOAA
    >>> ds = NOAA(split="PIFSC-10", sample_rate=10_000)
    >>> item = ds[0]
    >>> item["audio"].shape, item["sample_rate"], item["label"]  # doctest: +SKIP
    ((50000,), 10000, 'Other')
    """

    info = DatasetInfo(
        name="noaa",
        owner="moritz",
        split_paths={name: spec.annotations for name, spec in _SPLITS.items()},
        version="0.1.0",
        description=(
            "NOAA Passive Bioacoustics detection events. One labelled clip per "
            "annotation row, centred on the event within long (HARP xwav / FLAC) "
            "recordings, read lazily from gs://noaa-passive-bioacoustic. Each "
            "annotated sub-collection is a split (e.g. 'PIFSC-10', 10 kHz; "
            "'SANCTSOUND', animal detections at 96 kHz)."
        ),
        sources=["NOAA"],
        license="CC-BY-4.0, CC0",
    )

    def __init__(
        self,
        split: str = "PIFSC-10",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        clip_duration: float | None = 5.0,
        max_event_seconds: float | None = None,
        min_event_gap_seconds: float | None = None,
        prefer_highres: bool = True,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.max_event_seconds = max_event_seconds
        self.min_event_gap_seconds = min_event_gap_seconds
        self.prefer_highres = prefer_highres
        self._spec: _NOAASplit | None = None
        self._annotation_columns: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._fs = gcsfs.GCSFileSystem(token="anon")
        # Caches for variant resolution: GCS directory -> {names, index}, and
        # audio path -> duration (seconds).  Each unique listing / file is read
        # at most once.
        self._dir_cache: dict[str, dict[str, Any]] = {}
        self._dur_cache: dict[str, float] = {}
        self._load()

    def _dir_listing(
        self,
        gcs_dir: str,
        utc_parser: Callable[[str], datetime | None] = _pifsc_filename_utc,
    ) -> dict[str, Any]:
        """List a GCS directory once, returning ``{"names", "index", "starts"}``.

        ``names`` are the audio filenames sorted by their embedded UTC start
        time (parsed by ``utc_parser``); ``starts`` are the corresponding start
        datetimes (same order); ``index`` maps each name to its position.  Files
        whose name does not parse to a UTC time are dropped (they cannot be
        ordered, and ``starts`` must stay free of ``None`` for offset lookups).
        Missing/unlistable directories yield empty results (callers then fall
        back, or skip the event).
        """
        cached = self._dir_cache.get(gcs_dir)
        if cached is not None:
            return cached
        try:
            paths = [p for p in self._fs.ls(gcs_dir) if p.lower().endswith(".flac")]
        except (FileNotFoundError, OSError):
            paths = []
        dated = sorted(
            ((utc_parser(n), n) for n in (p.rsplit("/", 1)[-1] for p in paths)),
            key=lambda ds: ds[0] or datetime.min.replace(tzinfo=timezone.utc),
        )
        dated = [(s, n) for s, n in dated if s is not None]
        names = [n for _, n in dated]
        listing = {
            "names": names,
            "index": {n: i for i, n in enumerate(names)},
            "starts": [s for s, _ in dated],
        }
        self._dir_cache[gcs_dir] = listing
        return listing

    def _duration(self, path: str) -> float:
        """Return the duration (seconds) of an audio file, cached per path."""
        cached = self._dur_cache.get(path)
        if cached is None:
            cached = float(get_audio_info(path)["duration"])
            self._dur_cache[path] = cached
        return cached

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        cols = ["audio", "sample_rate", "audio_path"]
        if self._spec and self._spec.label_column:
            cols.append("label")
        cols += [c for c in self._annotation_columns if c not in cols]
        return cols

    @property
    def available_splits(self) -> list[str]:
        """Return the available splits of the dataset."""
        return list(_SPLITS.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return the native sample rate(s) of the loaded split."""
        return [self._spec.sample_rate] if self._spec else []

    def _load(self) -> None:
        """Load annotations and pre-compute per-event in-file clip windows.

        Raises
        ------
        LookupError
            If ``split`` is not one of the available splits.
        """
        if self.split not in _SPLITS:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(_SPLITS.keys())}"
            )
        self._spec = _SPLITS[self.split]

        if self._spec.loader is not None:
            # Loader-based split (e.g. SanctSound): the loader builds the full
            # event table, already carrying in-file event offsets.
            df = self._spec.loader(self)
            self._annotation_columns = list(df.columns)
        else:
            df = pd.read_csv(self._spec.annotations)
            self._annotation_columns = list(df.columns)
            # In-file (audio-domain) event offsets, per the split's layout.
            event_start, event_end = self._spec.offsets(df)
            df["event_start_seconds"] = event_start
            df["event_end_seconds"] = event_end

        event_start = df["event_start_seconds"]
        event_end = df["event_end_seconds"]

        # Drop over-long events whose annotated span exceeds the cutoff.  Some
        # SanctSound detectors (e.g. killerwhale, plainfinmidshipman, pinnipeds)
        # mark multi-hour — occasionally multi-day — *presence intervals* rather
        # than localised calls; a fixed clip cannot reliably contain the sound,
        # and centring on the span midpoint would land past the file's EOF
        # (empty reads).  Excluding them keeps every retained event a localised
        # detection.
        if self.max_event_seconds is not None:
            keep = (event_end - event_start) <= self.max_event_seconds
            df = df[keep].reset_index(drop=True)
            event_start = df["event_start_seconds"]
            event_end = df["event_end_seconds"]

        # Temporal de-duplication: detectors log calls note-by-note, so within a
        # single recording consecutive events are often a fraction of a second
        # apart (e.g. SanctSound bluewhale: 99% within 5 s), yielding near-
        # identical overlapping clips that dominate and add little for SSL.
        # Greedily keep, per audio recording, only events whose onset is at
        # least ``min_event_gap_seconds`` after the last kept one.
        if self.min_event_gap_seconds is not None and len(df):
            group_col = "audio_path" if "audio_path" in df.columns else self._spec.path_column
            df = df.sort_values([group_col, "event_start_seconds"], kind="stable")
            gap = self.min_event_gap_seconds
            last: dict[Any, float] = {}
            keep_mask: list[bool] = []
            for grp, onset in zip(df[group_col].to_numpy(), df["event_start_seconds"].to_numpy()):
                prev = last.get(grp)
                if prev is None or onset - prev >= gap:
                    keep_mask.append(True)
                    last[grp] = onset
                else:
                    keep_mask.append(False)
            df = df[keep_mask].reset_index(drop=True)
            event_start = df["event_start_seconds"]
            event_end = df["event_end_seconds"]

        if self.clip_duration is None:
            df["start_seconds"] = event_start
            df["end_seconds"] = event_end
        else:
            middle = (event_start + event_end) / 2.0
            half = self.clip_duration / 2.0
            df["start_seconds"] = (middle - half).clip(lower=0.0)
            df["end_seconds"] = df["start_seconds"] + self.clip_duration

        self._events = df.to_dict(orient="records")

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["NOAA", dict[str, Any]]:
        """Create a dataset instance from a :class:`~alp_data.DatasetConfig`.

        ``clip_duration``, ``max_event_seconds``, ``min_event_gap_seconds`` and
        ``prefer_highres`` are read from the config's extra fields when present,
        otherwise the constructor defaults are used.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            sample_rate=cfg["sample_rate"],
            clip_duration=cfg.get("clip_duration", 5.0),
            max_event_seconds=cfg.get("max_event_seconds", 60.0),
            min_event_gap_seconds=cfg.get("min_event_gap_seconds", 60.0),
            prefer_highres=cfg.get("prefer_highres", True),
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def _resolve_audio(self, event: dict[str, Any], raw_path: str) -> tuple[str, float, float]:
        """Resolve the audio path and clip window for an event.

        Defaults to the 10 kHz product (``path_replace``) with the clip window
        pre-computed in :meth:`_load`.  When ``prefer_highres`` is set and the
        split defines a ``variant_resolver``, that resolver may redirect to a
        higher-resolution product, returning its own ``(path, start, end)`` —
        the window is re-expressed in the variant's timebase.  The resolver
        returns ``None`` (→ 10 kHz) whenever it cannot confidently map the event.
        """
        spec = self._spec
        if self.prefer_highres and spec.variant_resolver is not None:
            hit = spec.variant_resolver(self, event, raw_path)
            if hit is not None:
                return hit
        return (
            raw_path.replace(*spec.path_replace),
            float(event["start_seconds"]),
            float(event["end_seconds"]),
        )

    def _process(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Load and process the audio clip for a single detection event.

        Parameters
        ----------
        event : dict[str, Any]
            One annotation record, including the ``start_seconds`` /
            ``end_seconds`` clip window pre-computed in :meth:`_load`.

        Returns
        -------
        dict[str, Any] or None
            All annotation columns plus the standard audio keys (see the class
            docstring).  If ``output_take_and_give`` was set, only the remapped
            keys are returned.  ``None`` if the event's clip window decoded to
            zero samples (e.g. it lands past the audio file's EOF), signalling
            the caller to skip this event.
        """
        spec = self._spec
        if spec.resolver is not None:
            audio_path, start_s, end_s = spec.resolver(self, event)
        else:
            raw_path = str(event[spec.path_column])
            if spec.path_repair is not None:
                raw_path = spec.path_repair(raw_path)
            audio_path, start_s, end_s = self._resolve_audio(event, raw_path)

        audio, sr = read_audio_by_time(audio_path, start_time=start_s, end_time=end_s)
        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        # The clip window is centred on the event and clamped only on the low
        # end (see _load), so an event whose offset lands at/after the file's
        # EOF yields a 0-sample read. Skip these rather than crash downstream
        # (e.g. librosa.resample rejects length-0 input). The caller draws
        # another sample in place of the dropped one.
        if audio.size == 0:
            logger.warning(
                "Empty audio read for %s [%.3f, %.3f]s; skipping event.",
                audio_path,
                start_s,
                end_s,
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

        # Pass every annotation column through, then overlay the audio payload.
        row: dict[str, Any] = dict(event)
        row["audio"] = audio
        row["sample_rate"] = sr
        row["audio_path"] = audio_path
        row["start_seconds"] = start_s
        row["end_seconds"] = end_s
        if spec.label_column and spec.label_column in event:
            row["label"] = event[spec.label_column]

        if self.output_take_and_give:
            return {
                new_key: row[orig_key]
                for orig_key, new_key in self.output_take_and_give.items()
            }
        return row

    def __len__(self) -> int:
        """Return the number of detection events in the split.

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
        return len(self._events)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """Return the processed clip for the event at position ``idx``.

        ``None`` if the event decoded to zero samples (see :meth:`_process`).
        """
        if self._streaming:
            raise NotImplementedError(
                "Indexed access is not available in streaming mode. Iterate instead."
            )
        return self._process(self._events[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all detection events, yielding one clip each.

        Events that decode to zero samples (see :meth:`_process`) are skipped.
        """
        for event in self._events:
            sample = self._process(event)
            if sample is not None:
                yield sample

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version}), split={self.split}, "
            f"{len(self._events)} event(s)\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.available_splits)}"
        )
