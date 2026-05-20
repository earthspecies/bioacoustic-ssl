import torch

from soundscape_ssl.data.transforms.base import Transform


class ToTensor(Transform):

    def __init__(self, keys: str | list[str]):
        super().__init__()
        self.keys = [keys] if isinstance(keys, str) else keys

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            for key in self.keys:
                batch[i][key] = torch.as_tensor(batch[i][key])
        return batch
