"""Arbimon / RFCx stream-based dataset loader."""

from __future__ import annotations

import io
import itertools
import os
import random
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

import httpx
import librosa
import numpy as np
from esp_data import Dataset, DatasetConfig, DatasetInfo
from esp_data.backends import BackendType, get_backend
from esp_data.io import audio_stereo_to_mono
from esp_data.io.read_utils import _read_audio_from_file

__all__ = ["Arbimon", "ArbimonDetections"]

_API_BASE = "https://api.rfcx.org"
_AUTH_BASE = "https://auth.rfcx.org"
_AUDIENCE = "https://rfcx.org"
_SCOPE = "openid email profile offline_access"
# Public web-app client ID used by the RFCx SDK for device-flow auth.
_DEFAULT_CLIENT_ID = "LS4dJlP8J2iOBr2snzm6N8I5u7FLSUGd"
_DEFAULT_CREDENTIALS_FILE = Path("~/.rfcx_credentials")
_PAGE_SIZE = 1000
_EARLIEST_DATE = "2000-01-01T00:00:00.000Z"


# ---------------------------------------------------------------------------
# Shared types and HTTP / auth mixin
# ---------------------------------------------------------------------------


class _DataFrame(Protocol):
    """Minimal structural type for DataFrame inputs (pandas or polars)."""

    @property
    def columns(self) -> Sequence[str]: ...


class _ArbimonMixin:
    """Shared RFCx authentication and HTTP helpers.

    Subclasses must initialise the following instance attributes before
    calling any instance method:

    * ``_client_id`` — Auth0 client ID
    * ``_client_secret`` — client secret or ``None`` for device flow
    * ``_credentials_file`` — :class:`~pathlib.Path` to the token cache
    * ``_http`` — shared :class:`httpx.Client`
    * ``_token`` — current bearer token
    """

    # Instance attribute declarations so type checkers know they exist.
    _client_id: str
    _client_secret: str | None
    _credentials_file: Path
    _http: httpx.Client
    _token: str

    # ------------------------------------------------------------------
    # Static auth helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_auth_params(
        client_id: str | None,
        client_secret: str | None,
        credentials_file: str | Path,
    ) -> tuple[str, str | None, Path]:
        """Resolve auth parameters from arguments and environment variables.

        Parameters
        ----------
        client_id : str or None
            Explicit client ID, or ``None`` to read from env / use default.
        client_secret : str or None
            Explicit client secret, or ``None`` to read from env.
        credentials_file : str or Path
            Path to the credentials cache file.

        Returns
        -------
        tuple[str, str | None, Path]
            Resolved ``(client_id, client_secret, credentials_file)``.
        """
        resolved_id = client_id or os.environ.get("AUTH0_CLIENT_ID") or _DEFAULT_CLIENT_ID
        resolved_secret = client_secret or os.environ.get("AUTH0_CLIENT_SECRET")
        resolved_file = Path(credentials_file).expanduser()
        return resolved_id, resolved_secret, resolved_file

    @staticmethod
    def _get_token(
        client_id: str,
        client_secret: str | None,
        credentials_file: Path,
        http: httpx.Client,
    ) -> str:
        """Obtain a bearer token via machine auth or device flow.

        Parameters
        ----------
        client_id : str
            Auth0 client ID.
        client_secret : str or None
            Auth0 client secret.  When set, machine auth
            (client-credentials grant) is used; otherwise the device
            flow is attempted.
        credentials_file : Path
            File path for caching/reading refresh tokens.
        http : httpx.Client
            Shared HTTP client.

        Returns
        -------
        str
            A valid bearer token.
        """
        if client_secret:
            return _ArbimonMixin._machine_auth(client_id, client_secret, http)
        if credentials_file.exists():
            token = _ArbimonMixin._load_cached_token(client_id, credentials_file, http)
            if token:
                return token
        access_token, refresh_token = _ArbimonMixin._device_flow(client_id, http)
        _ArbimonMixin._save_token(
            access_token, refresh_token=refresh_token, credentials_file=credentials_file
        )
        return access_token

    @staticmethod
    def _machine_auth(client_id: str, client_secret: str, http: httpx.Client) -> str:
        """Obtain a token via the OAuth 2.0 client-credentials grant.

        Parameters
        ----------
        client_id : str
            Auth0 client ID.
        client_secret : str
            Auth0 client secret.
        http : httpx.Client
            Shared HTTP client.

        Returns
        -------
        str
            Access token.

        Raises
        ------
        RuntimeError
            If the Auth0 server returns a non-200 response.
        """
        resp = http.post(
            f"{_AUTH_BASE}/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "audience": _AUDIENCE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Machine auth failed ({resp.status_code}): {resp.text}")
        return resp.json()["access_token"]

    @staticmethod
    def _device_flow(client_id: str, http: httpx.Client) -> tuple[str, str]:
        """Run the OAuth 2.0 device authorisation flow interactively.

        Prints a verification URL and user code to the terminal, then
        polls until the user approves the request in their browser.

        Parameters
        ----------
        client_id : str
            Auth0 client ID.
        http : httpx.Client
            Shared HTTP client.

        Returns
        -------
        tuple[str, str]
            ``(access_token, refresh_token)`` where ``refresh_token`` is
            an empty string if the server did not return one.

        Raises
        ------
        RuntimeError
            If the device code request fails, polling fails, or the flow
            times out.
        """
        resp = http.post(
            f"{_AUTH_BASE}/oauth/device/code",
            data={"client_id": client_id, "scope": _SCOPE, "audience": _AUDIENCE},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Device code request failed ({resp.status_code}): {resp.text}"
            )
        body = resp.json()
        device_code = body["device_code"]
        interval = body.get("interval", 5)
        print(
            f"\nOpen this URL in your browser to authenticate:\n"
            f"  {body['verification_uri']}\n"
            f"  Code: {body['user_code']}\n"
        )
        deadline = time.time() + body.get("expires_in", 300)
        while time.time() < deadline:
            time.sleep(interval)
            poll = http.post(
                f"{_AUTH_BASE}/oauth/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": client_id,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            data = poll.json()
            if poll.status_code == 200:
                return data["access_token"], data.get("refresh_token", "")
            if data.get("error") not in ("authorization_pending", "slow_down"):
                raise RuntimeError(
                    f"Token polling failed ({poll.status_code}): {poll.text}"
                )
        raise RuntimeError("Device flow timed out - user did not authorise in time.")

    @staticmethod
    def _load_cached_token(
        client_id: str, credentials_file: Path, http: httpx.Client
    ) -> str | None:
        """Load and validate a cached credential file.

        Tries to refresh the token if it is expiring within the next hour.

        The file format (four lines) is::

            version 1
            {access_token}
            {refresh_token}
            {expiry_isoformat}

        Parameters
        ----------
        client_id : str
            Auth0 client ID (needed for token refresh).
        credentials_file : Path
            Path to the credentials cache file.
        http : httpx.Client
            Shared HTTP client.

        Returns
        -------
        str or None
            A valid access token, or ``None`` if the cache is missing,
            malformed, or expired and cannot be refreshed.
        """
        try:
            lines = credentials_file.read_text().splitlines()
            if len(lines) < 4 or not lines[0].startswith("version"):
                return None
            access_token = lines[1].strip()
            refresh_token = lines[2].strip()
            expiry_str = lines[3].strip().rstrip("Z")
            expiry = datetime.fromisoformat(expiry_str).replace(tzinfo=timezone.utc)
            if (expiry - datetime.now(tz=timezone.utc)).total_seconds() > 3600:
                return access_token
            if refresh_token:
                return _ArbimonMixin._refresh_token(
                    client_id, refresh_token, credentials_file, http
                )
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _refresh_token(
        client_id: str,
        refresh_token: str,
        credentials_file: Path,
        http: httpx.Client,
    ) -> str | None:
        """Exchange a refresh token for a new access token.

        Parameters
        ----------
        client_id : str
            Auth0 client ID.
        refresh_token : str
            A valid OAuth 2.0 refresh token.
        credentials_file : Path
            Path used to persist the new token on success.
        http : httpx.Client
            Shared HTTP client.

        Returns
        -------
        str or None
            New access token, or ``None`` if the refresh request fails.
        """
        resp = http.post(
            f"{_AUTH_BASE}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "client_id": client_id,
                "refresh_token": refresh_token,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        _ArbimonMixin._save_token(
            data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_in=data.get("expires_in", 86400),
            credentials_file=credentials_file,
        )
        return data["access_token"]

    @staticmethod
    def _save_token(
        access_token: str,
        refresh_token: str = "",
        expires_in: int = 86400,
        credentials_file: Path = _DEFAULT_CREDENTIALS_FILE,
    ) -> None:
        """Persist credentials to disk.

        Parameters
        ----------
        access_token : str
            OAuth access token.
        refresh_token : str, optional
            OAuth refresh token (empty string if not available).
        expires_in : int, optional
            Token lifetime in seconds, by default 86400.
        credentials_file : Path, optional
            Destination file, by default ``~/.rfcx_credentials``.
        """
        expiry = (datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in)).isoformat()
        credentials_file.expanduser().write_text(
            f"version 1\n{access_token}\n{refresh_token}\n{expiry}\n"
        )

    @staticmethod
    def _parse_rfcx_timestamp(ts: str) -> datetime:
        """Parse an RFCx ISO-8601 timestamp string to a timezone-aware datetime.

        Parameters
        ----------
        ts : str
            Timestamp string such as ``"2023-01-01T00:00:00.000Z"``.

        Returns
        -------
        datetime
            UTC-aware datetime.
        """
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    # ------------------------------------------------------------------
    # Instance HTTP helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _refresh_auth(self) -> None:
        """Refresh ``self._token`` without user interaction.

        For machine auth (``client_secret`` set) a new token is fetched
        via the client-credentials grant.  For device-flow auth the
        refresh token cached in ``~/.rfcx_credentials`` is exchanged for
        a new access token.

        Raises
        ------
        RuntimeError
            If the credentials file is missing or malformed, no refresh
            token was stored, or the refresh request fails.
        """
        if self._client_secret:
            self._token = _ArbimonMixin._machine_auth(
                self._client_id, self._client_secret, self._http
            )
            return

        if not self._credentials_file.exists():
            raise RuntimeError(
                "Credentials file not found; re-authenticate by recreating the dataset."
            )
        lines = self._credentials_file.read_text().splitlines()
        if len(lines) < 3:
            raise RuntimeError(
                "Credentials file is malformed; re-authenticate by recreating the dataset."
            )
        refresh_token = lines[2].strip()
        if not refresh_token:
            raise RuntimeError(
                "No refresh token in credentials file; re-authenticate by recreating "
                "the dataset.  This can happen when the initial device flow did not "
                "grant offline access."
            )
        new_token = _ArbimonMixin._refresh_token(
            self._client_id, refresh_token, self._credentials_file, self._http
        )
        if new_token is None:
            raise RuntimeError(
                "Token refresh failed (refresh token may be expired); "
                "re-authenticate by recreating the dataset."
            )
        self._token = new_token

    def _get_auth(self, url: str, **kwargs: Any) -> httpx.Response:
        """Authenticated GET with automatic token refresh on 401.

        On a 401 response the token is refreshed once via `_refresh_auth`
        and the request is retried.  Any remaining non-2xx response raises
        via ``raise_for_status``.

        Parameters
        ----------
        url : str
            Target URL.
        **kwargs
            Forwarded to ``httpx.Client.get``.

        Returns
        -------
        httpx.Response
            Successful response (2xx).
        """
        resp = self._http.get(url, headers=self._auth_headers(), **kwargs)
        if resp.status_code == 401:
            self._refresh_auth()
            resp = self._http.get(url, headers=self._auth_headers(), **kwargs)
        resp.raise_for_status()
        return resp

    def _get_redirecting(self, url: str, **kwargs: Any) -> httpx.Response:
        """Authenticated GET that expects a redirect, with token refresh on 401.

        Redirects are not followed; the caller reads
        ``response.headers["location"]``.  On a 401 the token is refreshed
        once via `_refresh_auth` and the request is retried.  A
        non-redirect response that is not a 401 raises via
        ``raise_for_status``.

        Parameters
        ----------
        url : str
            Target URL.
        **kwargs
            Forwarded to ``httpx.Client.get`` (``follow_redirects`` is
            always set to ``False``).

        Returns
        -------
        httpx.Response
            Redirect response (3xx).
        """
        resp = self._http.get(
            url, headers=self._auth_headers(), follow_redirects=False, **kwargs
        )
        if resp.status_code == 401:
            self._refresh_auth()
            resp = self._http.get(
                url, headers=self._auth_headers(), follow_redirects=False, **kwargs
            )
        for attempt in range(3):
            if resp.status_code < 500:
                break
            time.sleep(2**attempt)
            resp = self._http.get(
                url, headers=self._auth_headers(), follow_redirects=False, **kwargs
            )
        if not resp.is_redirect:
            resp.raise_for_status()
        return resp


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


class Arbimon(_ArbimonMixin, Dataset):
    """Arbimon / RFCx stream-based dataset loader.

    Description
    -----------
    Provides access to audio recordings from a specific stream on the
    Arbimon / RFCx platform via the RFCx REST API. Recordings are
    enumerated by paginating ``GET /streams/{stream_id}/segments`` for
    the given time window and audio is fetched on demand from
    ``GET /streams/{stream_id}/segments/{start}/file``.

    The ``start`` and ``end`` parameters define the recording window.
    Arbimon streams can span months or years of 1-minute segments;
    bounding the window is required to make iteration finite.

    The dataset is streaming-only: segment metadata is paginated lazily
    and audio is downloaded on demand.  Use ``for item in dataset`` to
    iterate; ``__len__`` and indexed access are not supported.

    Two authentication modes are supported:

    * **Machine auth** - Set ``AUTH0_CLIENT_ID`` and
      ``AUTH0_CLIENT_SECRET`` environment variables (or pass them
      directly).  Uses the OAuth 2.0 client-credentials grant.
    * **Device flow** - Interactive browser-based login printed to the
      terminal.  The resulting token is cached at
      ``~/.rfcx_credentials`` so subsequent runs skip the browser step.

    Use the `list_projects` and `list_streams` classmethods to discover
    available data before creating an instance.

    Notes
    -----
    Arbimon segments are typically 1-minute recordings.  Each
    ``_process`` call makes two HTTP requests (an API redirect and an
    S3 download).  Set ``prefetch_factor`` to overlap these downloads
    with item consumption and improve throughput.

    References
    ----------
    https://arbimon.org
    https://github.com/rfcx/rfcx-sdk-python

    Examples
    --------
    Discover public projects and streams (no stream_id needed):

    >>> from data.arbimon import Arbimon
    >>> projects = Arbimon.list_projects()
    >>> streams = Arbimon.list_streams(project_id=projects[0]["id"])

    Load a dataset with automatically resolved bounds:

    >>> ds = Arbimon(stream_id=streams[0]["id"])
    >>> print(ds.start, ds.end)  # resolved from stream metadata
    >>> for item in ds:
    ...     print(item["start"], item["audio"].shape)

    Restrict to an explicit time window with concurrent prefetching:

    >>> ds = Arbimon(
    ...     stream_id=streams[0]["id"],
    ...     start="2023-01-01T00:00:00Z",
    ...     end="2023-01-01T00:10:00Z",
    ...     prefetch_factor=4,
    ... )
    """

    info = DatasetInfo(
        name="arbimon",
        owner="moritz",
        split_paths={},
        version="0.1.0",
        description=(
            "Arbimon / RFCx stream recordings loader. "
            "Streams audio segments from a given stream via the RFCx REST API."
        ),
        sources=["Arbimon / RFCx"],
        license="various",
    )

    def __init__(
        self,
        stream_id: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        credentials_file: str | Path = "~/.rfcx_credentials",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        backend: BackendType = "polars",
        prefetch_factor: int = 0,
    ) -> None:
        """Initialise the Arbimon dataset.

        Parameters
        ----------
        stream_id : str
            RFCx stream ID.  Use `list_projects` and `list_streams` to
            discover available stream IDs.
        start : str, datetime, or None, optional
            Start of the recording window (ISO-8601 string or datetime).
            When ``None`` (default), resolved via ``GET /streams/{stream_id}``.
            Falls back to ``_EARLIEST_DATE`` (``"2000-01-01T00:00:00.000Z"``)
            if the stream also reports ``None``.  Note that some streams
            contain recordings that predate their reported ``start``; pass
            an explicit value to guarantee coverage of earlier recordings.
        end : str, datetime, or None, optional
            End of the recording window (ISO-8601 string or datetime).
            When ``None`` (default), resolved via ``GET /streams/{stream_id}``.
            Falls back to ``datetime.now(UTC)`` if the stream also reports
            ``None``.
        client_id : str, optional
            Auth0 client ID.  Falls back to the ``AUTH0_CLIENT_ID``
            environment variable, then the public RFCx SDK client ID.
        client_secret : str, optional
            Auth0 client secret.  Falls back to ``AUTH0_CLIENT_SECRET``.
            When set, machine auth is used; otherwise device flow is used.
        credentials_file : str or Path, optional
            Path for caching the access/refresh token between runs.
            Defaults to ``~/.rfcx_credentials``.
        output_take_and_give : dict[str, str], optional
            Optional mapping of ``original_key -> new_key`` that filters
            and renames output fields before returning each item.
        sample_rate : int, optional
            Target sample rate in Hz.  When set, audio is resampled
            on-the-fly using ``librosa``.
        backend : BackendType, optional
            DataFrame backend, by default ``"polars"``.
        prefetch_factor : int, optional
            Number of audio segments to download concurrently in the
            background during iteration.  ``0`` (default) downloads
            segments sequentially.  A value of ``N > 0`` keeps N
            downloads in flight ahead of the current item, overlapping
            network I/O with item consumption.
        """
        super().__init__(output_take_and_give, backend=backend, streaming=True)
        self.stream_id = stream_id
        self.start: str | None = (
            start if start is None or isinstance(start, str) else start.isoformat()
        )
        self.end: str | None = (
            end if end is None or isinstance(end, str) else end.isoformat()
        )
        self.sample_rate = sample_rate
        self._prefetch_factor = prefetch_factor
        self._stream_meta: dict[str, Any] | None = None

        resolved_id, resolved_secret, resolved_file = self._resolve_auth_params(
            client_id, client_secret, credentials_file
        )
        self._client_id = resolved_id
        self._client_secret = resolved_secret
        self._credentials_file = resolved_file
        self._http = httpx.Client(timeout=httpx.Timeout(30.0, read=None))
        self._token = self._get_token(resolved_id, resolved_secret, resolved_file, self._http)

        if self.start is None or self.end is None:
            self._fetch_stream_bounds()

    def _load(self) -> None:
        """No-op; Arbimon is streaming-only."""
        pass

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        return ["audio", "sample_rate", "stream_id", "start", "end"]

    @property
    def available_splits(self) -> list[str]:
        """Return available splits (the configured stream ID)."""
        return [self.stream_id]

    def _process(self, segment: dict[str, Any]) -> dict[str, Any]:
        """Download and decode a single audio segment.

        Parameters
        ----------
        segment : dict[str, Any]
            Segment metadata dict as returned by the RFCx API, containing
            at least a ``"start"`` key with an ISO-8601 timestamp.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"audio"`` (np.ndarray, float32),
            ``"sample_rate"`` (int), ``"stream_id"`` (str), ``"start"``
            (str), and ``"end"`` (str).  If `output_take_and_give` was
            set, only the remapped keys are returned.
        """
        seg_start = segment["start"]
        start_str = seg_start if seg_start.endswith("Z") else seg_start + "Z"
        url = f"{_API_BASE}/streams/{self.stream_id}/segments/{start_str}/file"

        # Step 1: ask the RFCx API for the file - it redirects to a pre-signed S3 URL.
        # Do NOT follow the redirect here; S3 pre-signed URLs carry auth in the query
        # string, and forwarding the Authorization header to S3 causes a 403.
        redirect_resp = self._get_redirecting(url, params={"file_extension": "flac"})

        # Step 2: download from the S3 URL without any custom headers.
        s3_url = redirect_resp.headers["location"]
        s3_resp = self._http.get(s3_url, follow_redirects=True)
        s3_resp.raise_for_status()
        audio_bytes = s3_resp.content

        audio, sr = _read_audio_from_file(io.BytesIO(audio_bytes), format="FLAC")
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
            "stream_id": self.stream_id,
            "start": seg_start,
            "end": segment.get("end", ""),
        }

        if self.output_take_and_give:
            return {new_key: row[orig_key] for orig_key, new_key in self.output_take_and_give.items()}

        return row

    def __len__(self) -> int:
        """Not supported; Arbimon is streaming-only.

        Raises
        ------
        NotImplementedError
            Always; use ``for item in dataset`` to iterate.
        """
        raise NotImplementedError(
            "Arbimon only supports streaming iteration; use `for item in dataset`."
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Not supported; Arbimon is streaming-only.

        Raises
        ------
        NotImplementedError
            Always; use ``for item in dataset`` to iterate.
        """
        raise NotImplementedError(
            "Arbimon only supports streaming iteration; use `for item in dataset`."
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all segments in the configured time window.

        Segment metadata is paginated lazily.  When `prefetch_factor` is
        greater than zero, up to that many audio downloads run concurrently
        in background threads, overlapping network I/O with item consumption.

        Yields
        ------
        dict[str, Any]
            Processed audio item (see `_process`).
        """
        if self._prefetch_factor <= 0:
            for seg in self._iter_segments():
                yield self._process(seg)
            return

        segments = self._iter_segments()
        with ThreadPoolExecutor(max_workers=self._prefetch_factor) as executor:
            pending: deque = deque()
            # Pre-fill the buffer
            for seg in itertools.islice(segments, self._prefetch_factor):
                pending.append(executor.submit(self._process, seg))
            # Slide the window
            for seg in segments:
                pending.append(executor.submit(self._process, seg))
                yield pending.popleft().result()
            # Drain remaining futures
            while pending:
                yield pending.popleft().result()

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[Arbimon, dict[str, Any]]:
        """Instantiate from a `DatasetConfig`.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration object produced by the data-mixing pipeline.

        Returns
        -------
        tuple[Arbimon, dict[str, Any]]
            The dataset instance and a (possibly empty) transformation
            metadata dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            stream_id=cfg["stream_id"],
            start=cfg.get("start"),
            end=cfg.get("end"),
            output_take_and_give=cfg.get("output_take_and_give"),
            sample_rate=cfg.get("sample_rate"),
            backend=cfg.get("backend", "polars"),
            prefetch_factor=cfg.get("prefetch_factor", 0),
        )
        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta
        return ds, {}

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version}), stream_id={self.stream_id}\n"
            f"Window: {self.start} - {self.end}\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}"
        )

    # ------------------------------------------------------------------
    # Discovery classmethods
    # ------------------------------------------------------------------

    @classmethod
    def list_projects(
        cls,
        keyword: str | None = None,
        only_public: bool = True,
        limit: int = 1000,
        client_id: str | None = None,
        client_secret: str | None = None,
        credentials_file: str | Path = "~/.rfcx_credentials",
    ) -> list[dict[str, Any]]:
        """List Arbimon projects.

        This classmethod can be called without creating a dataset instance,
        making it the starting point for discovering available data.

        Parameters
        ----------
        keyword : str, optional
            Filter by project name (partial match).
        only_public : bool, optional
            When ``True`` (default), restrict to public projects.
        limit : int, optional
            Maximum number of results, by default 1000.
        client_id : str, optional
            Auth0 client ID.  Falls back to ``AUTH0_CLIENT_ID`` env var.
        client_secret : str, optional
            Auth0 client secret.  Falls back to ``AUTH0_CLIENT_SECRET``.
        credentials_file : str or Path, optional
            Token cache path, by default ``~/.rfcx_credentials``.

        Returns
        -------
        list[dict[str, Any]]
            List of project dicts with keys ``id``, ``name``,
            ``is_public``.
        """
        resolved_id, resolved_secret, resolved_file = cls._resolve_auth_params(
            client_id, client_secret, credentials_file
        )
        params: dict[str, Any] = {"limit": limit}
        if keyword:
            params["keyword"] = keyword
        if only_public:
            params["only_public"] = "true"
        with httpx.Client(timeout=httpx.Timeout(30.0, read=None)) as http:
            token = cls._get_token(resolved_id, resolved_secret, resolved_file, http)
            resp = http.get(
                f"{_API_BASE}/projects",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    @classmethod
    def list_streams(
        cls,
        project_id: str | None = None,
        keyword: str | None = None,
        only_public: bool = True,
        limit: int = 1000,
        client_id: str | None = None,
        client_secret: str | None = None,
        credentials_file: str | Path = "~/.rfcx_credentials",
    ) -> list[dict[str, Any]]:
        """List Arbimon streams, optionally filtered by project.

        This classmethod can be called without creating a dataset instance,
        making it the starting point for discovering available data.
        Each stream dict contains ``start`` and ``end`` timestamps that
        can be passed directly to `Arbimon.__init__`.

        Parameters
        ----------
        project_id : str, optional
            Restrict to streams belonging to this project.
        keyword : str, optional
            Filter by stream name (partial match).
        only_public : bool, optional
            When ``True`` (default), restrict to public streams.
        limit : int, optional
            Maximum number of results, by default 1000.
        client_id : str, optional
            Auth0 client ID.  Falls back to ``AUTH0_CLIENT_ID`` env var.
        client_secret : str, optional
            Auth0 client secret.  Falls back to ``AUTH0_CLIENT_SECRET``.
        credentials_file : str or Path, optional
            Token cache path, by default ``~/.rfcx_credentials``.

        Returns
        -------
        list[dict[str, Any]]
            List of stream dicts with keys ``id``, ``name``, ``start``,
            ``end``, ``latitude``, ``longitude``, etc.
        """
        resolved_id, resolved_secret, resolved_file = cls._resolve_auth_params(
            client_id, client_secret, credentials_file
        )
        params: dict[str, Any] = {"limit": limit}
        if project_id:
            params["projects[]"] = project_id
        if keyword:
            params["keyword"] = keyword
        if only_public:
            params["only_public"] = "true"
        with httpx.Client(timeout=httpx.Timeout(30.0, read=None)) as http:
            token = cls._get_token(resolved_id, resolved_secret, resolved_file, http)
            resp = http.get(
                f"{_API_BASE}/streams",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_segments(self) -> Iterator[dict[str, Any]]:
        """Paginate through segment metadata for the stream and time window.

        Yields
        ------
        dict[str, Any]
            Raw segment metadata dict from the RFCx API.
        """
        offset = 0
        while True:
            resp = self._get_auth(
                f"{_API_BASE}/streams/{self.stream_id}/segments",
                params={
                    "start": self.start,
                    "end": self.end,
                    "limit": _PAGE_SIZE,
                    "offset": offset,
                },
            )
            page: list[dict[str, Any]] = resp.json()
            if not page:
                break
            yield from page
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

    def _fetch_stream_bounds(self) -> None:
        """Resolve ``self.start`` and/or ``self.end`` from the stream detail endpoint.

        Calls ``GET /streams/{stream_id}`` and stores the full response dict
        in ``self._stream_meta``.  Only the ``None`` bounds are overwritten.

        Fallback rules:

        * ``start``: uses the API value when non-``None``; otherwise
          ``_EARLIEST_DATE`` (``"2000-01-01T00:00:00.000Z"``).
        * ``end``:   uses the API value when non-``None``; otherwise
          ``datetime.now(timezone.utc)`` formatted as ISO-8601 with
          milliseconds.
        """
        resp = self._get_auth(f"{_API_BASE}/streams/{self.stream_id}")
        meta: dict[str, Any] = resp.json()
        self._stream_meta = meta

        if self.start is None:
            self.start = meta.get("start") or _EARLIEST_DATE

        if self.end is None:
            self.end = (
                meta.get("end")
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            )


class ArbimonDetections(_ArbimonMixin, Dataset):
    """Load specific audio clips from Arbimon streams based on a detection list.

    Description
    -----------
    Each row of the ``detections`` DataFrame identifies a time window within
    a given stream expressed as **offsets in seconds from the stream's own
    start time**.  For example, ``start_seconds=0, end_seconds=120`` retrieves
    the first two minutes of the stream.  For each detection the class fetches
    the overlapping 1-minute RFCx segments, concatenates their audio, and
    trims to the exact window, returning a sub-segment clip ready for inference.

    Detections may span multiple streams.  Auth is shared across all requests
    via a single HTTP client and bearer token (refreshed automatically on 401).

    Notes
    -----
    Each detection typically requires one or two segment downloads (one API
    redirect + one S3 download each).  Set ``prefetch_factor`` to run several
    detections concurrently and overlap network I/O.

    Examples
    --------
    >>> import pandas as pd
    >>> from data.arbimon import ArbimonDetections
    >>> detections = pd.DataFrame({
    ...     "project_id": ["abc"],
    ...     "stream_id": ["xyz"],
    ...     "start_seconds": [0],
    ...     "end_seconds":   [120],
    ... })
    >>> ds = ArbimonDetections(detections, prefetch_factor=4)
    >>> for item in ds:
    ...     print(item["start"], item["audio"].shape)
    """

    info = DatasetInfo(
        name="arbimon_detections",
        owner="moritz",
        split_paths={},
        version="0.1.0",
        description=(
            "Arbimon detection-based audio loader. "
            "Fetches and trims audio clips for each row in a detections DataFrame."
        ),
        sources=["Arbimon / RFCx"],
        license="various",
    )

    def __init__(
        self,
        detections: _DataFrame | list[_DataFrame | str | Path] | str | Path,
        client_id: str | None = None,
        client_secret: str | None = None,
        credentials_file: str | Path = "~/.rfcx_credentials",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = None,
        backend: BackendType = "polars",
        prefetch_factor: int = 0,
    ) -> None:
        """Initialise the ArbimonDetections dataset.

        Parameters
        ----------
        detections : DataFrame, str, Path, or list thereof
            Detection records.  Accepts a pandas/polars DataFrame, a path to
            a CSV file (``str`` or ``Path``), or a list mixing any of the
            above.  All sources are concatenated in order.  Each must contain
            the columns ``project_id``, ``stream_id``, ``start_seconds``, and
            ``end_seconds``.  ``start_seconds`` and ``end_seconds`` are offsets
            in seconds from the stream's own start time (float or int).  For
            example, ``start_seconds=0, end_seconds=120`` retrieves the first
            two minutes of the stream.
        client_id : str, optional
            Auth0 client ID.  Falls back to ``AUTH0_CLIENT_ID`` env var,
            then the public RFCx SDK client ID.
        client_secret : str, optional
            Auth0 client secret.  Falls back to ``AUTH0_CLIENT_SECRET``.
            When set, machine auth is used; otherwise device flow is used.
        credentials_file : str or Path, optional
            Path for caching the access/refresh token between runs.
            Defaults to ``~/.rfcx_credentials``.
        output_take_and_give : dict[str, str], optional
            Optional mapping of ``original_key -> new_key`` that filters
            and renames output fields before returning each item.
        sample_rate : int, optional
            Target sample rate in Hz.  When set, audio is resampled
            on-the-fly using ``librosa`` after trimming.
        backend : BackendType, optional
            DataFrame backend, by default ``"polars"``.
        prefetch_factor : int, optional
            Number of detections to download concurrently in the background
            during iteration.  ``0`` (default) processes detections
            sequentially.

        """
        super().__init__(output_take_and_give, backend=backend, streaming=True)
        self._detections = self._to_records(detections)
        self.sample_rate = sample_rate
        self._prefetch_factor = prefetch_factor
        self._stream_start_cache: dict[str, datetime] = {}

        resolved_id, resolved_secret, resolved_file = self._resolve_auth_params(
            client_id, client_secret, credentials_file
        )
        self._client_id = resolved_id
        self._client_secret = resolved_secret
        self._credentials_file = resolved_file
        self._http = httpx.Client(timeout=httpx.Timeout(30.0, read=None))
        self._token = self._get_token(resolved_id, resolved_secret, resolved_file, self._http)

    # ------------------------------------------------------------------
    # Pickle protocol (required for DataLoader num_workers > 0)
    # ------------------------------------------------------------------

    def _ensure_http(self) -> None:
        """Create ``self._http`` if it does not exist.

        Called after unpickling so each DataLoader worker gets its own
        ``httpx.Client``.  The existing ``_token`` is reused; the normal
        401-handling in ``_get_auth`` will refresh it if expired.
        """
        if not hasattr(self, "_http") or self._http is None:
            self._http = httpx.Client(timeout=httpx.Timeout(30.0, read=None))

    def __getstate__(self) -> dict:
        """Return pickle state without the unpicklable HTTP client.

        Returns
        -------
        dict
            Instance ``__dict__`` with ``_http`` removed.
        """
        state = self.__dict__.copy()
        state.pop("_http", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling and recreate the HTTP client."""
        self.__dict__.update(state)
        self._ensure_http()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    @property
    def columns(self) -> list[str]:
        """Return the output column names."""
        return ["audio", "sample_rate", "stream_id", "project_id", "start", "end"]

    @property
    def available_splits(self) -> list[str]:
        """Return available splits."""
        return ["all"]

    def _load(self) -> None:
        """No-op; ArbimonDetections is streaming-only."""
        pass

    def __len__(self) -> int:
        """Return the number of detections.

        Returns
        -------
        int
            Number of detection records.
        """
        return len(self._detections)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Fetch and return the audio clip for detection at ``idx``.

        Parameters
        ----------
        idx : int
            Index into the detections list (negative indices are supported).

        Returns
        -------
        dict[str, Any]
            Processed audio item (see `_process_detection`).
        """
        self._ensure_http()
        return self._process_detection(self._detections[idx])


    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all detections, yielding one trimmed audio clip each.

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
    def from_config(cls, dataset_config: DatasetConfig) -> tuple[ArbimonDetections, dict[str, Any]]:
        """Instantiate from a `DatasetConfig` by loading detections from a file.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Must include a ``detections_path`` extra field pointing to a CSV
            or Parquet file with columns ``project_id``, ``stream_id``,
            ``start_seconds``, and ``end_seconds``.

        Returns
        -------
        tuple[ArbimonDetections, dict[str, Any]]
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
            One dict per row containing only the four required keys.

        Raises
        ------
        ValueError
            If any required column is absent from any source.
        """
        import pandas as pd

        required = {"project_id", "stream_id", "start_seconds", "end_seconds"}

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

    def _get_stream_start(self, stream_id: str) -> datetime:
        """Return the UTC start datetime of a stream, fetching and caching on first call.

        First checks the stream metadata endpoint.  If the stream reports
        no start time, falls back to querying the first available segment
        to find the real recording start.

        Parameters
        ----------
        stream_id : str
            RFCx stream ID.

        Returns
        -------
        datetime
            UTC-aware start datetime of the stream.

        Raises
        ------
        ValueError
            If the stream has no reported start time and contains no segments.
        """
        if stream_id not in self._stream_start_cache:
            resp = self._get_auth(f"{_API_BASE}/streams/{stream_id}")
            start_str = resp.json().get("start")
            if start_str:
                self._stream_start_cache[stream_id] = self._parse_rfcx_timestamp(start_str)
            else:
                # Stream metadata has no start; find the first actual segment.
                now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                seg_resp = self._get_auth(
                    f"{_API_BASE}/streams/{stream_id}/segments",
                    params={"start": _EARLIEST_DATE, "end": now_iso, "limit": 1, "offset": 0},
                )
                segments = seg_resp.json()
                if not segments:
                    raise ValueError(
                        f"Stream {stream_id!r} has no reported start time and no segments."
                    )
                self._stream_start_cache[stream_id] = self._parse_rfcx_timestamp(
                    segments[0]["start"]
                )
        return self._stream_start_cache[stream_id]

    def _process_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        """Fetch, stitch, and trim audio for a single detection.

        Parameters
        ----------
        detection : dict[str, Any]
            A detection record with ``project_id``, ``stream_id``,
            ``start_seconds``, and ``end_seconds``.  ``start_seconds``
            and ``end_seconds`` are offsets in seconds from the stream's
            own start time (e.g. ``start_seconds=0, end_seconds=120``
            retrieves the first two minutes of the stream).

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``"audio"`` (np.ndarray, float32 mono),
            ``"sample_rate"`` (int), ``"stream_id"`` (str),
            ``"project_id"`` (str), ``"start"`` (ISO-8601 str), and
            ``"end"`` (ISO-8601 str).  If `output_take_and_give` was set,
            only the remapped keys are returned.

        Raises
        ------
        ValueError
            If no segments exist in the RFCx API for the detection window.
        """
        stream_id = detection["stream_id"]
        stream_start = self._get_stream_start(stream_id)
        start_dt = stream_start + timedelta(seconds=detection["start_seconds"])
        end_dt = stream_start + timedelta(seconds=detection["end_seconds"])

        # Round the fetch start down to the minute boundary so that the segment
        # containing start_dt is included (RFCx filters by segment start time).
        fetch_start = start_dt.replace(second=0, microsecond=0)
        fetch_start_iso = fetch_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        fetch_end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        resp = self._get_auth(
            f"{_API_BASE}/streams/{stream_id}/segments",
            params={
                "start": fetch_start_iso,
                "end": fetch_end_iso,
                "limit": _PAGE_SIZE,
                "offset": 0,
            },
        )
        segments: list[dict[str, Any]] = resp.json()

        if not segments:
            raise ValueError(
                f"No segments found for stream {stream_id!r} "
                f"in window {fetch_start_iso} - {fetch_end_iso}."
            )

        # Download and concatenate all overlapping segments.
        audio_parts: list[tuple[datetime, np.ndarray]] = []
        sr = 0
        for seg in segments:
            seg_audio, sr = self._download_segment(stream_id, seg["start"])
            audio_parts.append((self._parse_rfcx_timestamp(seg["start"]), seg_audio))

        full_audio = np.concatenate([a for _, a in audio_parts])

        # Trim to the exact detection window relative to the first segment's start.
        first_seg_start = audio_parts[0][0]
        start_offset_s = (start_dt - first_seg_start).total_seconds()
        end_offset_s = (end_dt - first_seg_start).total_seconds()

        start_sample = max(0, int(start_offset_s * sr))
        end_sample = min(len(full_audio), int(end_offset_s * sr))
        audio = full_audio[start_sample:end_sample]

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
            "stream_id": stream_id,
            "project_id": detection["project_id"],
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        }

        if self.output_take_and_give:
            return {
                new_key: row[orig_key]
                for orig_key, new_key in self.output_take_and_give.items()
            }
        return row

    def _download_segment(self, stream_id: str, seg_start: str) -> tuple[np.ndarray, int]:
        """Download and decode one RFCx audio segment.

        Stereo audio is converted to mono.  Resampling is intentionally
        deferred to `_process_detection` so it operates on the final
        trimmed clip rather than on each full segment.

        Parameters
        ----------
        stream_id : str
            RFCx stream ID.
        seg_start : str
            Segment start timestamp as returned by the API (ISO-8601).

        Returns
        -------
        tuple[np.ndarray, int]
            Float32 mono audio array and its native sample rate.
        """
        start_str = seg_start if seg_start.endswith("Z") else seg_start + "Z"
        url = f"{_API_BASE}/streams/{stream_id}/segments/{start_str}/file"
        redirect_resp = self._get_redirecting(url, params={"file_extension": "flac"})
        s3_url = redirect_resp.headers["location"]
        s3_resp = self._http.get(s3_url, follow_redirects=True)
        s3_resp.raise_for_status()
        audio, sr = _read_audio_from_file(io.BytesIO(s3_resp.content), format="FLAC")
        audio = audio.astype(np.float32)
        return audio_stereo_to_mono(audio, mono_method="average"), sr

    def __str__(self) -> str:
        return (
            f"{self.info.name} (v{self.info.version}), "
            f"{len(self._detections)} detection(s)\n"
            f"Description: {self.info.description}\n"
            f"Sources: {', '.join(self.info.sources)}\n"
            f"License: {self.info.license}"
        )
