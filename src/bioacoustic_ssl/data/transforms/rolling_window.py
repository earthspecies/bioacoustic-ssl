import torch

from bioacoustic_ssl.data.transforms.base import Transform


class RollingWindow(Transform):
    """Randomly circular-shifts the waveform along the time axis.

    Args:
        p: Probability of applying the shift.
        audio_key: Key in the sample dict containing the audio tensor.
        training_only: If ``True`` (default), this transform is skipped when
            the module is in eval mode (``.eval()``).
    """

    def __init__(
        self,
        p: float = 0.5,
        target_key: str = "audio",
    ) -> None:
        super().__init__()
        self.p = p
        self.target_key = target_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            batch[i] = self._roll(batch[i])
        return batch

    def _roll(self, sample: dict) -> dict:
        if torch.rand(()) >= self.p:
            return sample
        audio = sample[self.target_key]
        shift = int(torch.randint(0, audio.shape[-1], (1,)).item())
        return {**sample, self.target_key: torch.roll(audio, shifts=shift, dims=-1)}
