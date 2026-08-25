"""Regression tests for :class:`MultiHotEncoder`'s handling of unmapped labels.

The label space of the full-Xeno-Canto head is frozen in a parquet, and
``MultiLabelFromFeature`` maps a class outside it to ``None``. Torch reads
``vec[None] = 1.0`` as a whole-tensor assignment, so such a sample used to come
out with every class positive — a mismatch between the label space and the split
being trained on corrupted targets for 100 000 steps without a single warning.
These tests pin the two halves of that: normal labels still encode, and a
``None`` label is an error.
"""

import pytest
import torch

from soundscape_ssl.data.transforms.label_encoder import MultiHotEncoder


def test_encodes_a_list_of_class_indices() -> None:
    """A multi-label sample becomes a multi-hot vector."""
    encoder = MultiHotEncoder(num_classes=4)
    batch = encoder([{"label": [1, 3]}, {"label": 0}])

    assert torch.equal(batch[0]["label"], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(batch[1]["label"], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_rejects_an_unmapped_label() -> None:
    """A ``None`` label raises instead of marking every class positive."""
    encoder = MultiHotEncoder(num_classes=4)

    with pytest.raises(ValueError, match="absent from the label map"):
        encoder([{"label": [1, None]}])
