import torch

from bioacoustic_ssl.data.transforms.base import Transform


class MultiHotEncoder(Transform):
    """Converts a raw label or list of labels into a multi-hot float tensor.

    Reads ``label_key`` from each sample dict, treats each value as a direct
    class index, and writes a float32 tensor of zeros and ones back to the
    same key.

    Example::

        enc = MultiHotEncoder(num_classes=3, label_key="label")
        sample = enc({"audio": ..., "label": [1, 2]})
        # sample["label"] == tensor([0., 1., 1.])

    Args:
        num_classes: Total number of classes.
        label_key: Key in the sample dict containing the raw label(s).
    """

    def __init__(self, num_classes: int, label_key: str = "label") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_key = label_key

    def __call__(self, batch: list[dict]) -> list[dict]:
        for i in range(len(batch)):
            raw = batch[i][self.label_key]
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for label in raw:
                # `MultiLabelFromFeature` yields None for a class outside its
                # label map, and `vec[None] = 1.0` is a whole-tensor assignment
                # in torch: it marks *every* class positive instead of raising.
                # That turned a label-space/split mismatch into 100 000 steps of
                # silently corrupted targets, so it fails here instead.
                if label is None:
                    raise ValueError(
                        "MultiHotEncoder got a None label, which means the sample's "
                        "class is absent from the label map. Rebuild the label space "
                        "from the split being trained on (scripts/build_xc_label_space.py) "
                        "or filter the sample out."
                    )
                try:
                    vec[label] = 1.0
                except Exception as e:
                    print(label, type(label))
                    raise e

            batch[i][self.label_key] = vec
        return batch
