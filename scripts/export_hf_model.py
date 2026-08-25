"""Convert a training checkpoint into a publishable HuggingFace artifact.

Two artifacts, one HuggingFace model repo, one subfolder each (spec ADR 0002):

``base``
    The pretrained encoder — a ViT-B/16 MAE encoder, 86M parameters. The MAE
    decoder in the checkpoint is dropped: the release publishes the encoder,
    which is what ``AutoModel`` returns.

``xc-classifier``
    That same encoder, frozen, under the layerwise prototypical head trained
    over all 10 799 Xeno-Canto taxa. Carries its own encoder copy, so it loads on
    its own, and ships ``xc_classes.parquet`` beside the weights — the map from
    output index to ``gbifID`` that logit masking needs.

Both get ``config.json``, ``model.safetensors``, ``preprocessor_config.json``
(the mel front-end), and a copy of the three published modules so the weights
load without cloning this repository.

**The export is gated.** Before anything is written, the published model is
checked against the in-repo model that produced the checkpoint on a fixed input,
and the check is repeated against the artifact reloaded from disk. Bit-exact or
nothing: a conversion that cannot prove parity is not published. The gate is why
this script exists rather than a notebook cell.

The project env pins ``transformers>=5.6.1``, which needs torch >= 2.7 against
this env's 2.6, so run it under the same override the repo uses elsewhere::

    uv run --with "transformers==4.57.1" python scripts/export_hf_model.py base \
        --ckpt /mnt/home/soundscape_mae/checkpoints/XC_1M.ckpt \
        --out artifacts/soundscape-mae/base

    uv run --with "transformers==4.57.1" python scripts/export_hf_model.py xc-classifier \
        --ckpt /mnt/home/soundscape_mae/checkpoints/xc_head_XC_1M_step_0100000.ckpt \
        --out artifacts/soundscape-mae/xc-classifier
"""

from dotenv import load_dotenv

load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from soundscape_ssl.hf import (
    BirdMAE2FeatureExtractor,
    BirdMAE2ForAudioClassification,
    BirdMAE2Model,
)
from soundscape_ssl.hf.conversion import (
    RELEASE_HEAD,
    build_config,
    classifier_state_dict,
    encoder_state_dict,
    reference_encoder_kwargs,
)
from soundscape_ssl.models import ViTEncoder, ViTProtoLayerwise

# Pointers into the copied modules, so `AutoModel.from_pretrained(..., trust_remote_code=True)`
# resolves without this repository. Written into both config.json and
# preprocessor_config.json.
AUTO_MAP = {
    "AutoConfig": "configuration_birdmae2.BirdMAE2Config",
    "AutoModel": "modeling_birdmae2.BirdMAE2Model",
    "AutoModelForAudioClassification": (
        "modeling_birdmae2.BirdMAE2ForAudioClassification"
    ),
    "AutoFeatureExtractor": "feature_extraction_birdmae2.BirdMAE2FeatureExtractor",
}
PUBLISHED_MODULES = (
    "configuration_birdmae2.py",
    "modeling_birdmae2.py",
    "feature_extraction_birdmae2.py",
)
DEFAULT_CLASSES = Path("metadata/xc_v0.1.0_all_classes.parquet")


def load_model_state(ckpt: Path) -> dict[str, torch.Tensor]:
    """Read the ``model`` sub-dict of a training checkpoint.

    Memory-mapped: the head checkpoints are 2.5 GB of which 1.4 GB is optimizer
    state this never touches.

    Args:
        ckpt: Path to the training checkpoint.

    Returns:
        The model state dict.
    """
    return torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)["model"]


def verify(reference: nn.Module, published: nn.Module, inputs: torch.Tensor, what: str) -> None:
    """Gate the export on the two implementations agreeing bit-exactly.

    Args:
        reference: The in-repo model, holding the training checkpoint's weights.
        published: The published model, holding the converted weights.
        inputs: A fixed input batch.
        what: What is being compared, for the message.

    Raises:
        SystemExit: If the outputs differ at all.
    """
    with torch.no_grad():
        expected = reference.eval()(inputs)
        actual = published.eval()(inputs)
    actual = actual.logits if hasattr(actual, "logits") else actual.last_hidden_state

    if not torch.equal(actual, expected):
        difference = (actual - expected).abs().max().item()
        raise SystemExit(
            f"PARITY FAILED ({what}): max |published - reference| = {difference:.3e}. "
            "Do not publish this artifact."
        )
    print(f"  parity OK ({what}): bit-exact over {expected.numel()} values")


def write_artifact(model: nn.Module, out: Path) -> None:
    """Write the weights, the config, the front-end and the published modules.

    Args:
        model: The published model to save.
        out: Artifact directory, created if missing.
    """
    model.config.auto_map = AUTO_MAP
    model.save_pretrained(out)

    extractor = BirdMAE2FeatureExtractor()
    extractor.auto_map = AUTO_MAP
    extractor.save_pretrained(out)

    source = Path(__file__).resolve().parent.parent / "src" / "soundscape_ssl" / "hf"
    for module in PUBLISHED_MODULES:
        shutil.copy(source / module, out / module)


def export_base(ckpt: Path, out: Path) -> None:
    """Export the pretrained encoder.

    Args:
        ckpt: An MAE pretraining checkpoint.
        out: Artifact directory.
    """
    state = load_model_state(ckpt)
    reference = ViTEncoder(**reference_encoder_kwargs())
    reference.load_state_dict(encoder_state_dict(state), strict=True)

    published = BirdMAE2Model(build_config())
    published.load_state_dict(encoder_state_dict(state), strict=True)

    torch.manual_seed(0)
    inputs = torch.randn(2, 1, published.config.num_mel_bins, published.config.num_frames)

    verify(reference, published, inputs, "encoder")
    write_artifact(published, out)
    verify(reference, BirdMAE2Model.from_pretrained(out), inputs, "encoder, reloaded from disk")

    print(f"{ckpt} -> {out}")
    print(f"  {sum(p.numel() for p in published.parameters()):,} parameters")


def export_classifier(ckpt: Path, out: Path, classes: Path) -> None:
    """Export the full-Xeno-Canto classification head with its label map.

    Args:
        ckpt: A ``ViTProtoLayerwise`` probe checkpoint.
        out: Artifact directory.
        classes: The frozen class parquet the head was trained against.
    """
    labels = pd.read_parquet(classes).sort_values("label_index")
    config = build_config(id2label=dict(zip(labels.label_index, labels.canonical_name, strict=True)))

    state = load_model_state(ckpt)
    checkpoint_labels = state["head.class_bias"].shape[0]
    if checkpoint_labels != config.num_labels:
        raise SystemExit(
            f"{ckpt} has a {checkpoint_labels}-class head, but {classes} defines "
            f"{config.num_labels} classes. The label space and the checkpoint must be the "
            "same one — pass the parquet the head was trained against, or re-export a head "
            "trained on this label space. Publishing the mismatch would mislabel every logit."
        )

    reference = ViTProtoLayerwise(
        num_classes=config.num_labels,
        mixer="block_diagonal",
        layer_norm=True,
        **RELEASE_HEAD,
        **reference_encoder_kwargs(),
    )
    reference.load_state_dict(state, strict=True)

    published = BirdMAE2ForAudioClassification(config)
    published.load_state_dict(classifier_state_dict(state), strict=True)

    torch.manual_seed(0)
    inputs = torch.randn(1, 1, config.num_mel_bins, config.num_frames)

    verify(reference, published, inputs, "classifier logits")
    write_artifact(published, out)
    shutil.copy(classes, out / "xc_classes.parquet")
    verify(
        reference,
        BirdMAE2ForAudioClassification.from_pretrained(out),
        inputs,
        "classifier logits, reloaded from disk",
    )

    print(f"{ckpt} -> {out}")
    print(f"  {sum(p.numel() for p in published.parameters()):,} parameters")
    print(f"  {config.num_labels} labels, output index -> gbifID in xc_classes.parquet")


def main() -> None:
    """Parse arguments and export the requested artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", choices=["base", "xc-classifier"])
    parser.add_argument("--ckpt", type=Path, required=True, help="training checkpoint to convert")
    parser.add_argument("--out", type=Path, required=True, help="artifact directory to write")
    parser.add_argument(
        "--classes",
        type=Path,
        default=DEFAULT_CLASSES,
        help=f"class parquet for the head's label map (default: {DEFAULT_CLASSES})",
    )
    args = parser.parse_args()

    if args.artifact == "base":
        export_base(args.ckpt, args.out)
    else:
        export_classifier(args.ckpt, args.out, args.classes)


if __name__ == "__main__":
    main()
