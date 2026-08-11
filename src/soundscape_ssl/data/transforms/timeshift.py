import io
import random
import tempfile

import numpy as np
import soundfile as sf
import torch
from alp_data.io.read_utils import get_audio_info, read_audio
from soundfile import LibsndfileError

from soundscape_ssl.data.transforms.base import Transform


class TimeShift(Transform):
    """Randomly crops the waveform to a fixed output length.

    Selects a random start offset and extracts ``output_length`` samples from
    the audio tensor. When the audio is shorter than ``output_length`` the
    sample is returned unchanged — pair with a padding transform if strictly
    fixed-length output is required.

    Three operating modes are supported, selected automatically based on the
    keys present in the incoming sample dict:

    **Standard mode** (``"audio"`` key present)
        Crops an already-decoded :class:`torch.Tensor`.  No I/O.

    **Bytes mode** (``"audio_bytes"`` key present, from :class:`~soundscape_ssl.data.datasets.XenoCantoRaw`)
        The crop window is decided *before* decoding.  Only the selected
        frames are decoded from the compressed stream (partial decode).
        The entire compressed file must already be in memory.

    **Lazy / path mode** (``"audio_path"`` key present, from :class:`~soundscape_ssl.data.datasets.XenoCantoLazy`)
        The file is opened directly — local or remote (GCS / S3).  For remote
        files, ``gcsfs`` translates soundfile's ``seek()`` calls into HTTP
        ``Range`` requests, so *only the header bytes plus the selected window
        bytes are ever downloaded*.  This is the most bandwidth-efficient mode
        for long GCS recordings.

    Args:
        output_length: Number of output samples when *sample_rate* is
            ``None``, or duration in **seconds** when *sample_rate* is given.
        sample_rate: If provided, ``output_length`` is interpreted as seconds
            and converted to ``int(sample_rate * output_length)`` samples.
            In bytes / path mode this is also used as the target sample rate
            for on-the-fly resampling when the file's native rate differs.
        p: Probability of applying the crop.
        audio_key: Key in the sample dict for the audio tensor (standard mode)
            or for the decoded output tensor (bytes / path mode).
    """

    def __init__(
        self,
        output_length: float,
        sample_rate: int,
        p: float = 1.0,
        audio_key: str = "audio",
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.output_length = output_length
        self.p = p
        self.audio_key = audio_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            if "audio_path" in batch[i]:
                batch[i] = self._shift_path(batch[i])
            elif "audio_bytes" in batch[i]:
                batch[i] = self._shift_bytes(batch[i])
            else:
                batch[i] = self._shift(batch[i])
        return batch

    # ------------------------------------------------------------------
    # Standard mode – operates on an already-decoded torch.Tensor
    # ------------------------------------------------------------------

    def _shift(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        audio = sample[self.audio_key]
        T = audio.shape[-1]
        output_length = int(self.output_length * self.sample_rate)
        if T <= output_length:
            return sample
        start = int(torch.randint(0, T - output_length, (1,)).item())
        return {**sample, self.audio_key: audio[..., start : start + output_length]}

    # ------------------------------------------------------------------
    # Lazy / path mode – true partial download via HTTP Range requests
    # ------------------------------------------------------------------

    def _shift_path(self, sample: dict) -> dict:
        """Open the remote / local file directly and download only the needed frames.

        For GCS-backed files (``gcsfs``), soundfile's ``seek()`` calls are
        translated into HTTP ``Range`` headers.  Only two network requests are
        issued: one for the file header (≤ 256 KB) and one for the selected
        audio window (≤ 256 KB), regardless of total file size.

        Parameters
        ----------
        sample : dict
            Must contain:

            ``"audio_path"``
                Path string understood by :func:`~alp_data.io.filesystem_from_path`
                (local path, ``gs://…``, ``s3://…``).
            ``"audio_format"`` *(optional)*
                Format hint (e.g. ``"FLAC"``).  Not strictly required for path
                mode — soundfile auto-detects from the file header — but kept
                for symmetry with bytes mode.
            ``"sample_rate"`` *(optional)*
                Target sample rate in Hz.  Falls back to ``self.sample_rate``.

        Returns
        -------
        dict
            Same as *sample* but with ``"audio_path"`` / ``"audio_format"``
            removed and ``"audio"`` (1-D :class:`torch.Tensor`, float32) plus
            ``"sample_rate"`` (int) added.
        """
        audio_path: str = sample["audio_path"]
        info = get_audio_info(audio_path)
        start_secs = random.uniform(0, max(info["duration"] - self.output_length, 0.0))

        # anonymous=True: our GCS buckets are public, so skip the ambient
        # (expiring) credentials on the ffmpeg range-read path — see
        # soundscape_ssl.data.datasets._gcs_anon.
        audio, sr = read_audio(
            audio_path, start_time=start_secs, end_time=start_secs + self.output_length, anonymous=True
        )

        # get_audio_info can overestimate duration for VBR files (e.g. MP3), so
        # the window may land past the true EOF and decode to zero frames. Retry
        # from the start, which yields real audio for any decodable file.
        if audio.size == 0 and start_secs > 0.0:
            audio, sr = read_audio(audio_path, start_time=0.0, end_time=self.output_length, anonymous=True)

        target_sr: int | None = self.sample_rate or info["sr"]

        audio_tensor = self._postprocess(audio, sr, target_sr)

        out = {k: v for k, v in sample.items() if k not in ("audio_path", "audio_format")}
        out[self.audio_key] = audio_tensor
        out["sample_rate"] = target_sr
        return out

    # ------------------------------------------------------------------
    # Bytes mode – decides the crop window before decoding
    # ------------------------------------------------------------------

    def _shift_bytes(self, sample: dict) -> dict:
        """Decode only the selected window from compressed audio bytes.

        Parameters
        ----------
        sample : dict
            Must contain:

            ``"audio_bytes"``
                Raw compressed audio as :class:`bytes`.
            ``"audio_format"`` *(optional)*
                Format hint understood by soundfile, e.g. ``"FLAC"``,
                ``"WAV"``, ``"OGG"``, ``"MP3"``.  Required as a fallback when
                the BytesIO-based decode fails (e.g. for MP3 with libsndfile).
            ``"sample_rate"`` *(optional)*
                Target sample rate in Hz.  Falls back to ``self.sample_rate``.

        Returns
        -------
        dict
            Same as *sample* but with ``"audio_bytes"`` / ``"audio_format"``
            removed and ``"audio"`` (1-D :class:`torch.Tensor`, float32) plus
            ``"sample_rate"`` (int) added.
        """
        if torch.rand(()) >= self.p:
            # Skip crop – still need to decode bytes → tensor
            return self._decode_full_bytes(sample)

        audio_bytes: bytes = sample["audio_bytes"]
        audio_format: str = sample.get("audio_format", "")
        target_sr: int | None = self.sample_rate or sample.get("sample_rate")

        # ---- 1. query file metadata (no audio decoded yet) ----------------
        buf = io.BytesIO(audio_bytes)
        try:
            info = sf.info(buf)
            native_sr: int = info.samplerate
            total_frames: int = info.frames
        except LibsndfileError:
            total_frames, native_sr = self._info_via_tmpfile(audio_bytes, audio_format)

        # ---- 2. crop dimensions in *native* frame space -------------------
        # output_length is a duration in seconds; the window to read in native
        # frame space is that duration times the native sample rate, regardless
        # of the target rate (resampling happens after decode).
        native_output_frames = int(self.output_length * native_sr)

        # ---- 3. random start, decode only the window ----------------------
        if total_frames <= native_output_frames:
            # File shorter than requested – decode everything, padding is
            # handled downstream.
            audio, sr = self._read_bytes(audio_bytes, audio_format, start=0, frames=-1)
        else:
            start_frame = random.randint(0, total_frames - native_output_frames)
            audio, sr = self._read_bytes(
                audio_bytes, audio_format,
                start=start_frame, frames=native_output_frames,
            )

        # ---- 4. post-process: mono → float32 → optional resample ----------
        audio_tensor = self._postprocess(audio, sr, target_sr)
        sr = target_sr if (target_sr and sr != target_sr) else sr

        # ---- 5. rebuild sample dict ---------------------------------------
        out = {k: v for k, v in sample.items() if k not in ("audio_bytes", "audio_format")}
        out[self.audio_key] = audio_tensor
        out["sample_rate"] = sr
        return out

    # ------------------------------------------------------------------
    # Shared post-processing
    # ------------------------------------------------------------------

    def _postprocess(
        self, audio: np.ndarray, native_sr: int, target_sr: int | None
    ) -> torch.Tensor:
        """Convert raw numpy array to a mono float32 tensor, resampling if needed.

        Parameters
        ----------
        audio : np.ndarray
            Audio array from soundfile.  Shape ``(frames,)`` for mono or
            ``(frames, channels)`` for multi-channel.
        native_sr : int
            Sample rate of *audio* as returned by soundfile.
        target_sr : int or None
            Desired output sample rate.  If ``None`` or equal to *native_sr*,
            no resampling is performed.

        Returns
        -------
        torch.Tensor
            1-D float32 tensor.
        """
        audio = audio.astype(np.float32)
        if audio.ndim == 2:
            # soundfile returns (frames, channels); average to mono → (frames,)
            audio = audio.mean(axis=-1)

        if target_sr and native_sr != target_sr:
            import librosa  # deferred import – not always needed
            audio = librosa.resample(
                y=audio, orig_sr=native_sr, target_sr=target_sr,
                scale=True, res_type="soxr_hq",
            )

        return torch.from_numpy(np.ascontiguousarray(audio))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_bytes(
        audio_bytes: bytes,
        audio_format: str,
        start: int = 0,
        frames: int = -1,
    ) -> tuple[np.ndarray, int]:
        """Read (possibly partial) audio from raw bytes.

        Parameters
        ----------
        audio_bytes : bytes
            Raw compressed audio.
        audio_format : str
            File format hint (e.g. ``"FLAC"``), used for the temp-file
            fallback when BytesIO decoding fails.
        start : int
            First frame to read.
        frames : int
            Number of frames to read; ``-1`` reads to end of file.

        Returns
        -------
        tuple[np.ndarray, int]
            ``(audio_array, sample_rate)``
        """
        buf = io.BytesIO(audio_bytes)
        try:
            return sf.read(buf, start=start, frames=frames)
        except LibsndfileError:
            suffix = f".{audio_format.lower()}" if audio_format else ".tmp"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                return sf.read(tmp.name, start=start, frames=frames)

    @staticmethod
    def _info_via_tmpfile(audio_bytes: bytes, audio_format: str) -> tuple[int, int]:
        """Return ``(total_frames, sample_rate)`` via a temporary file.

        Parameters
        ----------
        audio_bytes : bytes
            Raw compressed audio.
        audio_format : str
            File format hint used as the temp-file suffix.

        Returns
        -------
        tuple[int, int]
            ``(total_frames, sample_rate)``
        """
        suffix = f".{audio_format.lower()}" if audio_format else ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            info = sf.info(tmp.name)
        return info.frames, info.samplerate

    def _decode_full_bytes(self, sample: dict) -> dict:
        """Decode all bytes to a tensor without cropping (used when *p* skips the shift).

        Parameters
        ----------
        sample : dict
            Sample with ``"audio_bytes"`` and optionally ``"audio_format"``,
            ``"sample_rate"``.

        Returns
        -------
        dict
            Same as *sample* with ``"audio_bytes"`` / ``"audio_format"``
            removed and ``"audio"`` / ``"sample_rate"`` added.
        """
        audio_bytes: bytes = sample["audio_bytes"]
        audio_format: str = sample.get("audio_format", "")
        target_sr: int | None = self.sample_rate or sample.get("sample_rate")

        audio, sr = self._read_bytes(audio_bytes, audio_format, start=0, frames=-1)
        audio_tensor = self._postprocess(audio, sr, target_sr)
        sr = target_sr if (target_sr and sr != target_sr) else sr

        out = {k: v for k, v in sample.items() if k not in ("audio_bytes", "audio_format")}
        out[self.audio_key] = audio_tensor
        out["sample_rate"] = sr
        return out
