"""Australian Acoustic Observatory site-based dataset loader."""

from __future__ import annotations

import io
import itertools
import os
import random
import subprocess
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import httpx
import librosa
import numpy as np
import torchaudio
from alp_data import Dataset, DatasetConfig, DatasetInfo
from alp_data.backends import BackendType, get_backend
from alp_data.io import audio_stereo_to_mono

__all__ = ["A2OSite", "A2ODetections"]

_API_BASE = "https://api.acousticobservatory.org"
_PAGE_SIZE = 25
_AUTH_TOKEN_ENV = "A2O_AUTH_TOKEN"
_EMAIL_ENV = "A2O_EMAIL"
_PASSWORD_ENV = "A2O_PASSWORD"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_region_info(name_or_id: int | str) -> dict[str, Any] | None:
    """
    Look up region information by `name` or `id`.
    All region ids can be obtained by `A2OSite.list_regions()`.

    Parameters
    ----------
        name_or_id: Either a region `name` (str) or `id` (int)

    Returns
    -------
        dict with keys `name`, `id`, `site_ids_with_audio` or None if not found
    """
    if isinstance(name_or_id, int):
        # Lookup by id
        for name, info in A2OSite.regions_with_audio.items():
            if info["id"] == name_or_id:
                return {"name": name, **info}
    else:
        # Lookup by name
        if name_or_id in A2OSite.regions_with_audio:
            info = A2OSite.regions_with_audio[name_or_id]
            return {"name": name_or_id, **info}

    return None


# ---------------------------------------------------------------------------
# Shared types and HTTP / auth mixin
# ---------------------------------------------------------------------------


class _DataFrame(Protocol):
    """Minimal structural type for DataFrame inputs (pandas or polars)."""

    @property
    def columns(self) -> Sequence[str]: ...


class _A2OMixin:
    """Authentication and HTTP helpers shared by A2O dataset classes.

    Subclasses must set ``self._client`` (``httpx.Client``),
    ``self._email`` (``str | None``), and ``self._password``
    (``str | None``) before calling any of these methods.
    """

    _client: httpx.Client
    _email: str | None
    _password: str | None

    def _fetch_token(self) -> str:
        """Sign in with stored credentials and return a fresh auth token.

        Returns
        -------
        str
            New API auth token from ``POST /security``.

        Raises
        ------
        RuntimeError
            If sign-in fails (non-2xx response or missing token in body).
        """
        resp = self._client.post(
            f"{_API_BASE}/security",
            json={"email": self._email, "password": self._password},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"A2O sign-in failed (HTTP {resp.status_code}). "
                "Check A2O_EMAIL / A2O_PASSWORD credentials."
            )
        data = resp.json().get("data", {})
        token = data.get("auth_token")
        if not token:
            raise RuntimeError(
                "A2O sign-in succeeded but response contained no auth_token."
            )
        return token

    def _refresh_auth(self) -> None:
        """Re-fetch the auth token using stored credentials and update the client.

        Raises
        ------
        RuntimeError
            If no credentials are stored (static-token mode) or sign-in fails.
        """
        if self._email is None or self._password is None:
            raise RuntimeError(
                "Auth token expired but no credentials are available for refresh. "
                "Provide A2O_EMAIL and A2O_PASSWORD to enable automatic re-login."
            )
        token = self._fetch_token()
        self._client.headers["Authorization"] = f'Token token="{token}"'

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Authenticated GET with one automatic token refresh on 401.

        Parameters
        ----------
        url : str
            Target URL.
        **kwargs
            Forwarded to ``httpx.Client.get``.

        Returns
        -------
        httpx.Response
            Successful response.
        """
        resp = self._client.get(url, **kwargs)
        if resp.status_code == 401:
            self._refresh_auth()
            resp = self._client.get(url, **kwargs)
        resp.raise_for_status()
        return resp


class A2OSite(_A2OMixin, Dataset):
    """Australian Acoustic Observatory dataset loader by site ID.

    Description
    -----------
    Provides access to audio recordings from a specific site (point) on the
    Australian Acoustic Observatory (A2O) platform via the Acoustic Workbench
    REST API. Recordings are enumerated by paginating
    ``GET /audio_recordings?filter[site_id]=<site_id>`` and audio is fetched
    on demand from ``GET /audio_recordings/<id>/original``.

    Authentication is required to download audio.  Provide an API token via
    the ``auth_token`` parameter or set the ``A2O_AUTH_TOKEN`` environment
    variable before instantiation.

    Because sites can contain thousands of multi-hour recordings the class
    defaults to ``streaming=True``.  In streaming mode recording metadata is
    paginated lazily; in non-streaming mode all metadata is fetched upfront
    during initialisation to enable ``__len__`` and ``__getitem__``.

    Notes
    -----
    Individual recordings are often several hours long.  Expect each
    ``_process`` call to make a large HTTP download.

    References
    ----------
    https://data.acousticobservatory.org
    https://github.com/QutEcoacoustics/baw-server

    Examples
    --------
    Stream audio from site 209:

    >>> from data.a2o_site import A2OSite
    >>> ds = A2OSite(site_id=209)
    >>> for item in ds:
    ...     print(item["recording_id"], item["audio"].shape)
    ...     break

    Load site 209 eagerly (upfront metadata fetch):

    >>> ds = A2OSite(site_id=209, streaming=False)
    >>> print(len(ds))

    """

    regions_with_audio = {
        "Arkaba": {"id": 72, "site_ids_with_audio": [286, 287, 288, 285]},
        "Aroona Station": {"id": 107, "site_ids_with_audio": [426, 428, 427, 429]},
        "Binya": {"id": 84, "site_ids_with_audio": [334, 333, 335, 336]},
        "Blackbraes National Park": {"id": 31, "site_ids_with_audio": [121, 123, 124, 122]},
        "Blacksoil Creek (Bowling Green Bay National Park)": {"id": 29, "site_ids_with_audio": [115]},
        "Bon Bon Station": {"id": 2, "site_ids_with_audio": [5, 6, 8, 7]},
        "Boodjamulla (Lawn Hill) National Park": {"id": 26, "site_ids_with_audio": [101, 103, 102, 104]},
        "Boolcoomatta": {"id": 3, "site_ids_with_audio": [9, 10, 11, 12]},
        "Booroopki (Bank Australia Conservation Reserve)": {"id": 77, "site_ids_with_audio": [306, 307, 305, 308]},
        "Bowra": {"id": 65, "site_ids_with_audio": [258, 260, 259, 257]},
        "Boyagin Nature Reserve": {"id": 55, "site_ids_with_audio": [217, 220, 219, 218]},
        "Brogo": {"id": 6, "site_ids_with_audio": [23, 24, 21, 22]},
        "'Burrima' Macquarie Marshes": {"id": 109, "site_ids_with_audio": [434, 436, 437, 435]},
        "Calperum Mallee": {"id": 56, "site_ids_with_audio": [224, 223, 221, 222]},
        "Cape Barren Island": {"id": 76, "site_ids_with_audio": [301, 302, 303, 304]},
        "Carnarvon Station Reserve": {"id": 92, "site_ids_with_audio": [366, 368, 365, 367]},
        "Charles Darwin Reserve": {"id": 7, "site_ids_with_audio": [25, 28, 26, 27]},
        "Chillagoe": {"id": 42, "site_ids_with_audio": [165, 167, 168, 166]},
        "Cumberland Plain": {"id": 53, "site_ids_with_audio": [209, 210, 212, 211]},
        "Daintree Rainforest Observatory": {"id": 50, "site_ids_with_audio": [200, 197, 199, 198]},
        "Doonan Creek Environmental Reserve": {"id": 22, "site_ids_with_audio": [87, 88, 86, 85]},
        "Duval": {"id": 23, "site_ids_with_audio": [89, 91, 92, 90]},
        "Eungella National Park": {"id": 28, "site_ids_with_audio": [110, 112, 111]},
        "Five Rivers": {"id": 70, "site_ids_with_audio": [277, 278, 279, 280]},
        "Fletcherview Research Station": {"id": 61, "site_ids_with_audio": [241, 244, 243, 242]},
        "Fowlers Gap": {"id": 24, "site_ids_with_audio": [95, 96, 94, 93]},
        "Gingin": {"id": 57, "site_ids_with_audio": [226, 228, 225, 227]},
        "Gluepot Reserve": {"id": 18, "site_ids_with_audio": [69, 70, 71, 72]},
        "Great Cumbung": {"id": 82, "site_ids_with_audio": [326]},
        "Great Western Woodlands": {"id": 58, "site_ids_with_audio": [229, 230, 232, 231]},
        "Hamelin Station": {"id": 10, "site_ids_with_audio": [39, 37, 38, 40]},
        "Kalamurina": {"id": 71, "site_ids_with_audio": [282, 283, 281, 284]},
        "Kangaroo Island": {"id": 103, "site_ids_with_audio": [410, 411, 409, 412]},
        "Katarapko": {"id": 68, "site_ids_with_audio": [270, 269, 271, 272]},
        "Litchfield Savanna": {"id": 52, "site_ids_with_audio": [208, 205, 206]},
        "Little Desert Nature Lodge": {"id": 74, "site_ids_with_audio": [295, 293, 294, 296]},
        "Little Llangothlin Reserve/Warra National Park": {"id": 11, "site_ids_with_audio": [43, 44, 41, 42]},
        "Marshmead (MLC)": {"id": 78, "site_ids_with_audio": [311, 310, 312]},
        "Matuwa Indigenous Protected Area": {"id": 90, "site_ids_with_audio": [357, 358, 359, 360]},
        "Minjerribah": {"id": 86, "site_ids_with_audio": [343, 342, 344, 341]},
        "Mitchell Grass Rangeland": {"id": 59, "site_ids_with_audio": [233, 236, 234, 235]},
        "Monjebup Reserve/SWWA Floristic Region": {"id": 91, "site_ids_with_audio": [362, 364, 361, 363]},
        "Moorrinya National Park": {"id": 32, "site_ids_with_audio": [128]},
        "Mount Barney": {"id": 1, "site_ids_with_audio": [4, 3, 2, 1]},
        "Mourachan": {"id": 106, "site_ids_with_audio": [421, 424, 423, 422]},
        "Naree Station": {"id": 12, "site_ids_with_audio": [45, 47, 46, 48]},
        "Newhaven": {"id": 73, "site_ids_with_audio": [289, 290, 292, 291]},
        "Orpheus Island": {"id": 43, "site_ids_with_audio": [171, 172]},
        "Paluma Range National Park": {"id": 39, "site_ids_with_audio": [153, 155, 154, 156]},
        "Reedy Creek": {"id": 13, "site_ids_with_audio": [49, 50, 51, 52]},
        "Rinyirru (Lakefield) National Park": {"id": 40, "site_ids_with_audio": [158, 159, 157, 160]},
        "Robson Creek": {"id": 51, "site_ids_with_audio": [201, 202, 203, 204]},
        "Scottsdale": {"id": 14, "site_ids_with_audio": [53, 54, 55, 56]},
        "SEQP Samford": {"id": 64, "site_ids_with_audio": [256, 253, 255, 254]},
        "Spyglass": {"id": 44, "site_ids_with_audio": [175, 173, 176, 174]},
        "Staaten River National Park": {"id": 41, "site_ids_with_audio": [164, 161, 162, 163]},
        "Staaten River National Park West": {"id": 88, "site_ids_with_audio": [351, 352, 349, 350]},
        "Steve Irwin Wildlife Reserve": {"id": 105, "site_ids_with_audio": [417, 419, 420, 418]},
        "Sturt National Park": {"id": 79, "site_ids_with_audio": [314, 315, 316, 313]},
        "Tarcutta Hills": {"id": 15, "site_ids_with_audio": [57, 59, 58, 60]},
        "Tjoritja (West MacDonnell National Park) 1 & 2": {"id": 87, "site_ids_with_audio": [348]},
        "Toorale National Park": {"id": 81, "site_ids_with_audio": [322, 323, 324, 321]},
        "Townsville Town Common Conservation Park": {"id": 35, "site_ids_with_audio": [137, 138, 140]},
        "Tumbarumba Wet Eucalypt": {"id": 60, "site_ids_with_audio": [240, 239, 237, 238]},
        "Undara National Park": {"id": 36, "site_ids_with_audio": [144, 142, 141, 143]},
        "Uunguu Indigenous Protected Area (Wunambal Gaambera)": {"id": 16, "site_ids_with_audio": [61, 62, 63, 64]},
        "Victorian Dry Eucalypt: Wombat": {"id": 62, "site_ids_with_audio": [245, 246, 247]},
        "Wambiana Cattle Station": {"id": 45, "site_ids_with_audio": [178, 179, 180, 177]},
        "Warra Tall Eucalypt": {"id": 63, "site_ids_with_audio": [249, 250, 251, 252]},
        "Yourka": {"id": 46, "site_ids_with_audio": [181, 183, 184, 182]},
    }

    info = DatasetInfo(
        name="a2o_site",
        owner="moritz",
        split_paths={},
        version="0.1.0",
        description=(
            "Australian Acoustic Observatory recordings loader. "
            "Streams audio from a given site via the Acoustic Workbench REST API."
        ),
        sources=["Australian Acoustic Observatory"],
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        site_id: int,
        auth_token: str | None = None,
        email: str | None = None,
        password: str | None = None,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        backend: BackendType = "polars",
        streaming: bool = True,
    ) -> None:
        """Initialise the A2OSite dataset.

        Parameters
        ----------
        site_id : int
            A2O site (point) ID.  Visible in the URL of the data portal, e.g.
            ``/projects/1/regions/53/points/209`` → ``site_id=209``.
        auth_token : str, optional
            Acoustic Workbench API token.  Falls back to the
            ``A2O_AUTH_TOKEN`` environment variable when not provided.
            When set, no automatic token refresh is performed.
        email : str, optional
            A2O account e-mail address used for automatic re-login when the
            token expires.  Falls back to the ``A2O_EMAIL`` environment
            variable.  Requires `password` to be set as well.
        password : str, optional
            A2O account password used for automatic re-login.  Falls back to
            the ``A2O_PASSWORD`` environment variable.  Requires `email` to be
            set as well.
        output_take_and_give : dict[str, str], optional
            Optional mapping of ``original_key -> new_key`` that filters and
            renames output fields before returning each item.
        sample_rate : int, optional
            Target sample rate in Hz.  When set, audio is resampled on-the-fly
            using ``librosa``.
        backend : BackendType, optional
            DataFrame backend, by default ``"polars"``.
        streaming : bool, optional
            When ``True`` (default) recording metadata is paginated lazily
            during iteration.  When ``False`` all metadata is fetched during
            ``__init__``, enabling ``__len__`` and indexed ``__getitem__``.

        Raises
        ------
        ValueError
            If neither an auth token nor email/password credentials are
            provided (directly or via ``A2O_AUTH_TOKEN`` / ``A2O_EMAIL`` /
            ``A2O_PASSWORD`` environment variables).

        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.site_id = site_id
        self.sample_rate = sample_rate

        self._email = email or os.environ.get(_EMAIL_ENV)
        self._password = password or os.environ.get(_PASSWORD_ENV)

        # Build the client first so _fetch_token() can use it for the POST.
        self._client = httpx.Client(
            headers={"Authorization": ""},
            timeout=httpx.Timeout(30.0, read=None),
        )

        token = auth_token or os.environ.get(_AUTH_TOKEN_ENV)
        if token:
            self._client.headers["Authorization"] = f'Token token="{token}"'
        elif self._email and self._password:
            token = self._fetch_token()
            self._client.headers["Authorization"] = f'Token token="{token}"'
        else:
            raise ValueError(
                "No credentials provided. Pass `auth_token` or set "
                f"{_AUTH_TOKEN_ENV!r}, or pass `email`+`password` / set "
                f"{_EMAIL_ENV!r} and {_PASSWORD_ENV!r} for automatic token refresh."
            )

        self._recordings: list[dict[str, Any]] | None = None
        self._load()

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        return ["audio", "sample_rate", "recording_id", "recorded_date", "duration"]

    @property
    def available_splits(self) -> list[str]:
        """Return available splits (the single configured site ID as a string)."""
        return [str(self.site_id)]

    def _iter_recordings(self) -> Iterator[dict[str, Any]]:
        """Paginate through recording metadata for the site.

        Yields
        ------
        dict[str, Any]
            Raw recording metadata dict from the Acoustic Workbench API.
        """
        page = 1
        while True:
            resp = self._get(
                f"{_API_BASE}/audio_recordings",
                params={"filter_site_id": self.site_id, "page": page, "items": _PAGE_SIZE},
            )
            body = resp.json()
            for rec in body["data"]:
                yield rec
            max_page = body["meta"]["paging"].get("max_page", 1)
            if page >= max_page:
                break
            page += 1

    def _load(self) -> None:
        """Prepare recording metadata.

        In streaming mode this is a no-op; metadata is fetched lazily.
        In non-streaming mode all pages are fetched upfront.
        """
        if not self._streaming:
            self._recordings = list(self._iter_recordings())

    def _process(self, recording: dict[str, Any]) -> dict[str, Any]:
        """Download and decode a single recording.

        Parameters
        ----------
        recording : dict[str, Any]
            Recording metadata dict as returned by the A2O API, containing at
            least ``"id"``, ``"recorded_date"``, and ``"duration"`` keys.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"audio"`` (np.ndarray, float32),
            ``"sample_rate"`` (int), ``"recording_id"`` (int), ``"site"`` (int), `
            ``"recorded_date"`` (str), ``"duration"`` (float), and ``"creator_id"`` (int).
            If `output_take_and_give` was set, only the remapped keys
            are returned.
        """
        rec_id = recording["id"]
        resp = self._get(
            f"{_API_BASE}/audio_recordings/{rec_id}/original",
            follow_redirects=True,
        )

        audio, sr = torchaudio.load(io.BytesIO(resp.content))
        audio = audio.numpy()

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
            "recording_id": rec_id,
            "site": recording["site_id"],
            "recorded_date": recording["recorded_date"],
            "duration": recording["duration_seconds"],
            "creator_id": recording["creator_id"],
        }

        if self.output_take_and_give:
            return {new_key: row[orig_key] for orig_key, new_key in self.output_take_and_give.items()}

        return row

    def __len__(self) -> int:
        """Return the number of recordings for the site.

        Returns
        -------
        int
            Number of recordings fetched during initialisation.

        Raises
        ------
        NotImplementedError
            If the dataset was initialised in streaming mode.
        RuntimeError
            If no recordings have been loaded.
        """
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode. "
                "Iterate over the dataset instead."
            )
        if self._recordings is None:
            raise RuntimeError("No recordings have been loaded yet.")
        return len(self._recordings)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the item at position `idx`.

        Parameters
        ----------
        idx : int
            Zero-based index into the list of recordings.

        Returns
        -------
        dict[str, Any]
            Processed audio item (see `_process`).

        Raises
        ------
        NotImplementedError
            If the dataset was initialised in streaming mode.
        RuntimeError
            If no recordings have been loaded.
        """
        if self._streaming:
            raise NotImplementedError(
                "Indexed access is not available in streaming mode."
            )
        if self._recordings is None:
            raise RuntimeError("No recordings have been loaded yet.")
        return self._process(self._recordings[idx])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all recordings for the site.

        In streaming mode metadata is paginated lazily; in non-streaming mode
        the pre-fetched metadata list is used.

        Yields
        ------
        dict[str, Any]
            Processed audio item (see `_process`).

        Raises
        ------
        RuntimeError
            If the dataset was initialised in non-streaming mode but no
            recordings have been loaded.
        """
        if self._streaming:
            for rec in self._iter_recordings():
                yield self._process(rec)
        else:
            if self._recordings is None:
                raise RuntimeError("No recordings have been loaded yet.")
            for rec in self._recordings:
                yield self._process(rec)

    @classmethod
    def list_regions(cls) -> list[dict[str, Any]]:
        """Return all regions with their site IDs that have audio recordings.

        Returns
        -------
        list[dict[str, Any]]
            List of region dicts, each containing ``"id"``, ``"name"``,
            and ``"site_ids_with_audio"``.
        """
        return [
            {"name": name, "id": info["id"], "site_ids": info["site_ids_with_audio"]}
            for name, info in cls.regions_with_audio.items()
        ]

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["A2OSite", dict[str, Any]]:
        """Instantiate from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object produced by the data-mixing pipeline.

        Returns
        -------
        tuple[A2OSite, dict[str, Any]]
            The dataset instance and a (possibly empty) transformation
            metadata dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            site_id=cfg["site_id"],
            auth_token=cfg.get("auth_token"),
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
        return (
            f"{self.info.name} (v{self.info.version}), site_id={self.site_id}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}"
        )


# ---------------------------------------------------------------------------
# Detection-based loader
# ---------------------------------------------------------------------------


class A2ODetections(_A2OMixin, Dataset):
    """Load audio clips from A2O recordings based on a detection list.

    Description
    -----------
    Each row of the ``detections`` DataFrame identifies a time window within
    a given recording expressed as **offsets in seconds from the recording
    start**.  For each detection the class fetches the corresponding audio
    slice via the Acoustic Workbench media endpoint
    (``GET /audio_recordings/{id}/media.flac?start_offset=…&end_offset=…``),
    which retrieves only the requested portion server-side — no full
    multi-hour recording is downloaded.

    Auth is shared across all requests via a single HTTP client (token
    refreshed automatically on 401).

    Notes
    -----
    Set ``prefetch_factor`` to run several detections concurrently and
    overlap network I/O.

    Examples
    --------
    >>> import pandas as pd
    >>> from data.a2o_site import A2ODetections
    >>> detections = pd.DataFrame({
    ...     "recording_id":  [12345],
    ...     "start_seconds": [0.0],
    ...     "end_seconds":   [10.0],
    ... })
    >>> ds = A2ODetections(detections, prefetch_factor=4)
    >>> for item in ds:
    ...     print(item["recording_id"], item["audio"].shape)
    """

    info = DatasetInfo(
        name="a2o_detections",
        owner="moritz",
        split_paths={},
        version="0.1.0",
        description=(
            "Australian Acoustic Observatory detection-based audio loader. "
            "Fetches and trims audio clips for each row in a detections DataFrame."
        ),
        sources=["Australian Acoustic Observatory"],
        license="CC-BY-4.0",
    )

    def __init__(
        self,
        detections: _DataFrame | list[_DataFrame | str | Path] | str | Path,
        auth_token: str | None = None,
        email: str | None = None,
        password: str | None = None,
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        backend: BackendType = "polars",
        prefetch_factor: int = 0,
    ) -> None:
        """Initialise the A2ODetections dataset.

        Parameters
        ----------
        detections : DataFrame, str, Path, or list thereof
            Detection records.  Accepts a pandas/polars DataFrame, a path to
            a CSV file (``str`` or ``Path``), or a list mixing any of the
            above.  All sources are concatenated in order.  Each must contain
            the columns ``recording_id``, ``start_seconds``, and
            ``end_seconds``.  ``start_seconds`` and ``end_seconds`` are
            offsets in seconds from the recording start (float or int).
        auth_token : str, optional
            Acoustic Workbench API token.  Falls back to the
            ``A2O_AUTH_TOKEN`` environment variable.  When set, no
            automatic token refresh is performed.
        email : str, optional
            A2O account e-mail address for automatic re-login when the
            token expires.  Falls back to ``A2O_EMAIL``.  Requires
            `password`.
        password : str, optional
            A2O account password for automatic re-login.  Falls back to
            ``A2O_PASSWORD``.  Requires `email`.
        output_take_and_give : dict[str, str], optional
            Optional mapping of ``original_key -> new_key`` that filters
            and renames output fields before returning each item.
        sample_rate : int, optional
            Target sample rate in Hz.  When set, audio is resampled
            on-the-fly using ``librosa``.
        backend : BackendType, optional
            DataFrame backend, by default ``"polars"``.
        prefetch_factor : int, optional
            Number of detections to download concurrently in background
            threads during iteration.  ``0`` (default) processes
            detections sequentially.

        Raises
        ------
        ValueError
            If neither an auth token nor email/password credentials are
            provided, or if the detections DataFrame is missing required
            columns.

        """
        super().__init__(output_take_and_give, backend=backend, streaming=True)
        self._detections = self._to_records(detections)
        self.sample_rate = sample_rate
        self._prefetch_factor = prefetch_factor

        self._email = email or os.environ.get(_EMAIL_ENV)
        self._password = password or os.environ.get(_PASSWORD_ENV)

        self._client = httpx.Client(
            headers={"Authorization": ""},
            timeout=httpx.Timeout(30.0, read=None),
        )

        token = auth_token or os.environ.get(_AUTH_TOKEN_ENV)
        if token:
            self._client.headers["Authorization"] = f'Token token="{token}"'
        elif self._email and self._password:
            token = self._fetch_token()
            self._client.headers["Authorization"] = f'Token token="{token}"'
        else:
            raise ValueError(
                "No credentials provided. Pass `auth_token` or set "
                f"{_AUTH_TOKEN_ENV!r}, or pass `email`+`password` / set "
                f"{_EMAIL_ENV!r} and {_PASSWORD_ENV!r} for automatic token refresh."
            )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # httpx.Client contains thread locks and is not picklable.
        # Save the auth header so the client can be rebuilt in each worker.
        state["_auth_header"] = str(self._client.headers.get("Authorization", ""))
        del state["_client"]
        return state

    def __setstate__(self, state: dict) -> None:
        auth_header = state.pop("_auth_header", "")
        self.__dict__.update(state)
        self._client = httpx.Client(
            headers={"Authorization": auth_header},
            timeout=httpx.Timeout(30.0, read=None),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        return ["audio", "sample_rate", "recording_id", "start_seconds", "end_seconds"]

    @property
    def available_splits(self) -> list[str]:
        """Return available splits."""
        return ["all"]

    def _load(self) -> None:
        """No-op for now."""
        pass

    def __len__(self) -> int:
        return len(self._detections)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        result = self._process_detection(self._detections[idx])
        if result is not None:
            return result
        for _ in range(5):
            alt_idx = random.randrange(len(self._detections))
            result = self._process_detection(self._detections[alt_idx])
            if result is not None:
                return result
        raise RuntimeError(f"Failed to load a valid sample after 5 retries (index {idx})")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all detections, yielding one audio clip each.

        When `prefetch_factor` is greater than zero, up to that many
        detections are downloaded concurrently in background threads.

        Yields
        ------
        dict[str, Any]
            Processed audio item (see `_process_detection`).
        """
        if self._prefetch_factor <= 0:
            for det in self._detections:
                yield self._process_detection(det)
            return

        def _safe_process(det: dict[str, Any]) -> dict[str, Any] | None:
            return self._process_detection(det)

        dets = iter(self._detections)
        with ThreadPoolExecutor(max_workers=self._prefetch_factor) as executor:
            pending: deque = deque()
            for det in itertools.islice(dets, self._prefetch_factor):
                pending.append(executor.submit(_safe_process, det))
            for det in dets:
                pending.append(executor.submit(_safe_process, det))
                result = pending.popleft().result()
                if result is not None:
                    yield result
            while pending:
                result = pending.popleft().result()
                if result is not None:
                    yield result

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig
    ) -> tuple["A2ODetections", dict[str, Any]]:
        """Instantiate from a `DatasetConfig` by loading detections from a file.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Must include a ``detections_path`` extra field pointing to a CSV
            or Parquet file with columns ``recording_id``, ``start_seconds``,
            and ``end_seconds``.

        Returns
        -------
        tuple[A2ODetections, dict[str, Any]]
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
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_records(
        detections: _DataFrame | list[_DataFrame | str | Path] | str | Path,
    ) -> list[dict[str, Any]]:
        """Convert detections to a list of dicts.

        Accepts a pandas/polars DataFrame, a CSV file path (``str`` or
        ``Path``), or a list mixing any of the above.  All sources are
        validated for required columns and concatenated in order.

        Parameters
        ----------
        detections : DataFrame, str, Path, or list thereof
            Input detections.

        Returns
        -------
        list[dict[str, Any]]
            One dict per row containing only the three required keys.

        Raises
        ------
        ValueError
            If any required column is absent from any source.
        """
        import pandas as pd

        required = {"recording_id", "start_seconds", "end_seconds"}

        # Treat as a single item when it's a path or a DataFrame (has .columns).
        # list, omegaconf.ListConfig, and other iterables are treated as collections.
        if isinstance(detections, (str, Path)) or hasattr(detections, "columns"):
            items: list[_DataFrame | str | Path] = [detections]  # type: ignore[list-item]
        else:
            items = list(detections)  # type: ignore[arg-type]

        all_records: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, (str, Path)):
                df = pd.read_csv(item)
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(
                        f"Detections file {str(item)!r} is missing required columns: {sorted(missing)}"
                    )
                all_records.extend(df[sorted(required)].to_dict(orient="records"))
            else:
                missing = required - set(item.columns)
                if missing:
                    raise ValueError(
                        f"Detections DataFrame is missing required columns: {sorted(missing)}"
                    )
                if hasattr(item, "to_dicts"):  # polars
                    all_records.extend(item.select(sorted(required)).to_dicts())
                else:
                    all_records.extend(item[sorted(required)].to_dict(orient="records"))

        return all_records

    @staticmethod
    def _is_valid_audio_response(resp: httpx.Response) -> bool:
        content_type = resp.headers.get("content-type", "")
        return len(resp.content) >= 64 and content_type.startswith(
            ("audio/", "application/octet-stream")
        )

    def _process_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        """Fetch and decode audio for a single detection.

        First attempts a targeted server-side slice via the media endpoint.
        If that response contains no audio bytes, falls back to downloading
        the full recording and slicing the decoded audio array locally.

        Parameters
        ----------
        detection : dict[str, Any]
            A detection record with ``recording_id``, ``start_seconds``,
            and ``end_seconds``.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"audio"`` (np.ndarray, float32 mono),
            ``"sample_rate"`` (int), ``"recording_id"`` (int),
            ``"start_seconds"`` (float), and ``"end_seconds"`` (float).
            If `output_take_and_give` was set, only the remapped keys are
            returned.

        Raises
        ------
        httpx.HTTPStatusError
            If the slice or full-recording endpoint returns a non-422 HTTP error.
        RuntimeError
            If both the slice endpoint and the full-recording fallback fail.
        """
        rec_id = detection["recording_id"]
        start_s = round(detection["start_seconds"], 2)
        end_s = round(detection["end_seconds"], 2)

        audio = None
        sr = None
        try:
            slice_resp = self._get(
                f"{_API_BASE}/audio_recordings/{rec_id}/media.flac",
                params={"start_offset": start_s, "end_offset": end_s},
                follow_redirects=True,
            )
            if self._is_valid_audio_response(slice_resp):
                try:
                    audio, sr = torchaudio.load(io.BytesIO(slice_resp.content))
                    audio = audio.numpy()
                except RuntimeError:
                    pass  # undecodable slice → fall through to full-recording fallback
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 422:
                raise

        if audio is None or audio.shape[-1] == 0:
            full_resp = self._get(
                f"{_API_BASE}/audio_recordings/{rec_id}/original",
                follow_redirects=True,
            )
            # if not self._is_valid_audio_response(full_resp):
                # content_type = full_resp.headers.get("content-type", "")
                # raise RuntimeError(
                #     f"A2O API returned unexpected response for recording {rec_id} "
                #     f"[{start_s:.3f}s - {end_s:.3f}s] (slice and full-recording "
                #     f"fallback both failed): "
                #     f"HTTP {full_resp.status_code}, content-type={content_type!r}, "
                #     f"{len(full_resp.content)} bytes."
                # )

            try:
                audio, sr = torchaudio.load(io.BytesIO(full_resp.content))
            except RuntimeError as exc:
                # warnings.warn(
                #     f"A2O recording {rec_id} [{start_s:.3f}s-{end_s:.3f}s] "
                #     f"could not be decoded and will be skipped: {exc}",
                #     stacklevel=2,
                # )
                return None
            audio = audio.numpy()
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            if end_sample > start_sample and end_sample <= audio.shape[-1]:
                audio = audio[..., start_sample:end_sample]

        audio = audio.astype(np.float32)
        audio = audio_stereo_to_mono(audio, mono_method="average")

        if self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="soxr_hq",
            )
            sr = self.sample_rate

        row: dict[str, Any] = {
            "audio": audio,
            "sample_rate": sr,
            "recording_id": rec_id,
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
