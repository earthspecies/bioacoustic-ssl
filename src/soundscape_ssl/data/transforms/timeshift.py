import torch

from soundscape_ssl.data.transforms.base import Transform


class TimeShift(Transform):
    """Randomly crops the waveform to a fixed output length.

    Selects a random start offset and extracts ``output_length`` samples from
    the audio tensor. When the audio is shorter than ``output_length`` the
    sample is returned unchanged — pair with a padding transform if strictly
    fixed-length output is required.

    Args:
        output_length: Number of output samples.
        p: Probability of applying the crop.
        audio_key: Key in the sample dict containing the audio tensor.
    """

    def __init__(
        self,
        output_length: int,
        sample_rate: int = None,
        p: float = 1.0,
        audio_key: str = "audio",
    ) -> None:
        super().__init__()
        self.output_length = output_length if sample_rate is None else int(sample_rate * output_length)
        self.p = p
        self.audio_key = audio_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            batch[i] = self._shift(batch[i])
        return batch

    def _shift(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        audio = sample[self.audio_key]
        T = audio.shape[-1]
        if T <= self.output_length:
            return sample
        start = int(torch.randint(0, T - self.output_length, (1,)).item())
        return {**sample, self.audio_key: audio[..., start : start + self.output_length]}
