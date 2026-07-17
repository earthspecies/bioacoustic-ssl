import logging

from soundscape_ssl.data.transforms.base import Transform

logger = logging.getLogger(__name__)


class DropEmptyAudio(Transform):
    """Drop batch items whose audio decoded to zero samples.

    Empty clips crash amplitude-based transforms downstream (e.g.
    ``PeakNormalize``'s ``max`` over an empty tensor). Datasets that decode
    lazily inside the collate pipeline (``XenoCantoLazy`` via ``TimeShift``)
    bypass the mix's ``None``-skip, so a mis-sized / unreadable read can still
    reach here. Drop such items and log the sample's scalar keys so the
    offending source can be traced.
    """

    def __init__(self, target_key: str = "audio") -> None:
        super().__init__()
        self.target_key = target_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        kept = []
        for sample in batch:
            x = sample.get(self.target_key)
            n = x.numel() if hasattr(x, "numel") else (0 if x is None else x.size)
            if n == 0:
                info = {
                    k: v
                    for k, v in sample.items()
                    if k != self.target_key and isinstance(v, (str, int, float, bool))
                }
                logger.warning("Dropping empty-audio sample (%s).", info)
                continue
            kept.append(sample)
        return kept
