"""Bit-exactness tests for the released HuggingFace model code.

The release gate (spec ADR 0002) is that features from the published artifact
match features from the training checkpoint loaded through the in-repo model,
bit-exactly. These tests assert exactly that on a tiny geometry with random
weights, which is where the property can actually be *tested*: the same check
against the 1.33 GB training checkpoints is the parity step of
``scripts/export_hf_model.py`` and runs at export time.

Three properties, in the order they matter:

1. The published modules reproduce ``ViTEncoder`` / ``ViTProtoLayerwise``
   bit-exactly, loading the in-repo state dict with ``strict=True``. Strictness
   is the point: it is what makes the conversion a key rename rather than a
   re-implementation that happens to agree.
2. ``AutoModel`` / ``AutoModelForAudioClassification`` round-trip a saved
   artifact without changing a single output value, label map included.
3. The feature extractor reproduces the ``BatchSpectrogram`` front-end the
   encoder was trained on, and — unlike it — gives a clip the same spectrogram
   whatever else shares its batch.

The project env pins ``transformers>=5.6.1``, which needs torch >= 2.7 while the
env has 2.6, so ``transformers.PreTrainedModel`` is unimportable here and every
test below skips. Run them with the override this repo already uses for its other
transformers-version conflicts::

    uv run --with "transformers==4.57.1" pytest tests/unittests/test_hf_xenomae.py
"""

import numpy as np
import pytest
import torch

try:  # transformers 5.x needs torch >= 2.7; this env has 2.6, so it cannot import
    from transformers import PreTrainedModel  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"transformers is unimportable here: {exc}", allow_module_level=True)

from soundscape_ssl.data.transforms.padding import BatchPadding  # noqa: E402
from soundscape_ssl.data.transforms.spectrogram import BatchSpectrogram  # noqa: E402
from hf_model import (  # noqa: E402
    XenoMAEConfig,
    XenoMAEFeatureExtractor,
    XenoMAEForAudioClassification,
    XenoMAEModel,
)
from hf_model.conversion import classifier_state_dict, encoder_state_dict  # noqa: E402
from soundscape_ssl.models import ViTEncoder, ViTProtoLayerwise  # noqa: E402

# Tiny stand-in for the released geometry: a (2, 4) token grid instead of (8, 32),
# 2 blocks instead of 12. Everything that could differ between the two
# implementations — QK-norm, the sincos grid, the CLS-subtraction, the chunked
# prototype similarity — is exercised at this size too.
GEOMETRY = dict(
    img_size=(32, 64),
    patch_size=16,
    in_chans=1,
    embed_dim=32,
    depth=2,
    num_heads=2,
    mlp_ratio=2.0,
    norm_eps=1e-6,
    pos_embed_type="sinusoidal_2d",
    qkv_norm=True,
)
HEAD = dict(num_classes=5, num_prototypes=3, mixer="block_diagonal", proto_chunk=4)


@pytest.fixture
def config() -> XenoMAEConfig:
    """The published config matching :data:`GEOMETRY` and :data:`HEAD`."""
    return XenoMAEConfig(
        num_mel_bins=GEOMETRY["img_size"][0],
        num_frames=GEOMETRY["img_size"][1],
        patch_size=GEOMETRY["patch_size"],
        hidden_size=GEOMETRY["embed_dim"],
        num_hidden_layers=GEOMETRY["depth"],
        num_attention_heads=GEOMETRY["num_heads"],
        mlp_ratio=GEOMETRY["mlp_ratio"],
        qkv_norm=GEOMETRY["qkv_norm"],
        num_prototypes=HEAD["num_prototypes"],
        proto_chunk=HEAD["proto_chunk"],
        id2label={i: f"Species {i}" for i in range(HEAD["num_classes"])},
        label2id={f"Species {i}": i for i in range(HEAD["num_classes"])},
    )


@pytest.fixture
def spectrogram(config: XenoMAEConfig) -> torch.Tensor:
    """A fixed batch of two spectrograms shaped like the front-end's output."""
    torch.manual_seed(0)
    return torch.randn(2, 1, config.num_mel_bins, config.num_frames)


def test_encoder_reproduces_repo_encoder(config: XenoMAEConfig, spectrogram: torch.Tensor) -> None:
    """The published encoder is bit-exact against ``ViTEncoder``."""
    torch.manual_seed(1)
    reference = ViTEncoder(**GEOMETRY).eval()
    published = XenoMAEModel(config).eval()
    published.load_state_dict(encoder_state_dict(reference.state_dict()), strict=True)

    with torch.no_grad():
        expected = reference(spectrogram)
        actual = published(spectrogram)

    assert torch.equal(actual.last_hidden_state, expected)
    assert torch.equal(actual.pooler_output, expected[:, 0])


def test_encoder_hidden_states_are_the_pre_norm_block_outputs(
    config: XenoMAEConfig, spectrogram: torch.Tensor
) -> None:
    """``output_hidden_states`` returns what the layerwise head consumes.

    ``ViTEncoder(return_hidden=True)`` yields the raw block outputs, *before* the
    final shared norm; the published model must expose the same tensors, since
    the classification head's layer fusion reads them.
    """
    torch.manual_seed(1)
    reference = ViTEncoder(**GEOMETRY).eval()
    published = XenoMAEModel(config).eval()
    published.load_state_dict(encoder_state_dict(reference.state_dict()), strict=True)

    with torch.no_grad():
        expected = reference(spectrogram, return_hidden=True)
        actual = published(spectrogram, output_hidden_states=True).hidden_states

    # (embeddings, block_1, ..., block_depth) — HF's convention, so one more.
    assert len(actual) == len(expected) + 1
    for block, (published_h, reference_h) in enumerate(zip(actual[1:], expected)):
        assert torch.equal(published_h, reference_h), f"block {block} differs"


def test_classifier_reproduces_repo_classifier(config: XenoMAEConfig, spectrogram: torch.Tensor) -> None:
    """The published classifier is bit-exact against ``ViTProtoLayerwise``."""
    torch.manual_seed(2)
    reference = ViTProtoLayerwise(**HEAD, layer_norm=True, **GEOMETRY).eval()
    published = XenoMAEForAudioClassification(config).eval()
    published.load_state_dict(classifier_state_dict(reference.state_dict()), strict=True)

    with torch.no_grad():
        expected = reference(spectrogram)
        actual = published(spectrogram).logits

    assert torch.equal(actual, expected)


def test_auto_classes_round_trip(
    tmp_path, config: XenoMAEConfig, spectrogram: torch.Tensor
) -> None:
    """A saved artifact reloads through ``AutoModel`` with identical outputs."""
    from transformers import AutoModel, AutoModelForAudioClassification

    torch.manual_seed(3)
    classifier = XenoMAEForAudioClassification(config).eval()
    classifier.save_pretrained(tmp_path)

    reloaded = AutoModelForAudioClassification.from_pretrained(tmp_path).eval()
    encoder_only = AutoModel.from_pretrained(tmp_path).eval()

    with torch.no_grad():
        assert torch.equal(reloaded(spectrogram).logits, classifier(spectrogram).logits)
        assert torch.equal(
            encoder_only(spectrogram).last_hidden_state,
            classifier.encoder(spectrogram).last_hidden_state,
        )

    assert reloaded.config.id2label == config.id2label
    assert reloaded.config.label2id["Species 3"] == 3


def test_feature_extractor_matches_training_front_end() -> None:
    """The extractor reproduces the ``BatchSpectrogram`` chain the encoder saw.

    The clip is duplicated because that is the case where the two front-ends are
    provably comparable: training clamped ``top_db`` against the maximum of the
    whole batch and the extractor clamps per sample, and for a batch of one
    repeated clip those are the same number. A batch of *different* clips is
    where they part company, which is the point of
    :func:`test_feature_extractor_is_batch_independent`.
    """
    extractor = XenoMAEFeatureExtractor()
    torch.manual_seed(4)
    waveform = torch.randn(int(extractor.sampling_rate * extractor.clip_seconds))
    # PeakNormalize precedes BatchSpectrogram in the training pipeline, and the
    # extractor does it internally, so hand both sides the normalised waveform.
    waveform = waveform / waveform.abs().max()

    batch = {"audio": torch.stack([waveform, waveform])}
    batch = BatchSpectrogram(sample_rate=extractor.sampling_rate)(batch)
    expected = BatchPadding(target_shape=(extractor.num_mel_bins, extractor.num_frames))(batch)["spectrogram"]

    actual = extractor([waveform, waveform], sampling_rate=extractor.sampling_rate)["input_values"]

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_feature_extractor_pads_and_truncates_to_the_trained_window() -> None:
    """Clips shorter or longer than 5 s come out at the trained input shape."""
    extractor = XenoMAEFeatureExtractor()
    torch.manual_seed(5)
    clip_samples = int(extractor.sampling_rate * extractor.clip_seconds)

    short = extractor(torch.randn(clip_samples // 3), sampling_rate=extractor.sampling_rate)["input_values"]
    long = extractor(torch.randn(clip_samples * 2), sampling_rate=extractor.sampling_rate)["input_values"]

    shape = (1, 1, extractor.num_mel_bins, extractor.num_frames)
    assert short.shape == shape
    assert long.shape == shape

    # The batch of a recording collection is ragged in practice, and numpy is
    # what a user reading files with soundfile or librosa will hand over.
    ragged = extractor(
        [np.random.randn(clip_samples // 3), np.random.randn(clip_samples * 2)],
        sampling_rate=extractor.sampling_rate,
    )["input_values"]
    assert ragged.shape == (2, *shape[1:])


def test_feature_extractor_is_batch_independent() -> None:
    """A clip's spectrogram does not depend on what else is in the batch.

    This is the one deliberate divergence from the training front-end, whose
    ``top_db`` clamp is taken against the batch maximum.
    """
    extractor = XenoMAEFeatureExtractor()
    torch.manual_seed(6)
    clip_samples = int(extractor.sampling_rate * extractor.clip_seconds)
    quiet = torch.randn(clip_samples) * 1e-3
    loud = torch.randn(clip_samples)

    batched = extractor([quiet, loud], sampling_rate=extractor.sampling_rate)["input_values"]
    alone = extractor(quiet, sampling_rate=extractor.sampling_rate)["input_values"]

    assert batched.shape[0] == 2
    # Not bit-exact: batching changes the blocking of the mel matmul, which moved
    # a single one of these 65 536 values by 2.4e-7 when this was written. The
    # claim is that nothing *systematic* rides on batch composition.
    assert torch.allclose(batched[:1], alone, rtol=0.0, atol=1e-6)


def test_feature_extractor_rejects_a_wrong_sampling_rate() -> None:
    """A mismatched sampling rate is an error, not a silent resample."""
    extractor = XenoMAEFeatureExtractor()
    with pytest.raises(ValueError, match="sampling rate"):
        extractor(torch.zeros(16_000), sampling_rate=16_000)
