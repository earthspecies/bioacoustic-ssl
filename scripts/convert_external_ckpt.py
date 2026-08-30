"""Convert published Bird-MAE / AudioMAE ViT-B weights into our checkpoint format.

Both released encoders are architecturally identical to our :class:`ViTEncoder`
(verified bit-exact against the reference implementations), so probing them needs
no wrapper model — only a state dict the eval scripts already know how to load:
nested under ``"model"`` with every key prefixed ``"encoder."``.

``pos_embed`` is deliberately kept: our encoder builds its sincos grid the other
way round, and it is the checkpoint's own tensor overwriting that buffer on load
that makes the forward pass match the published models exactly.

    uv run python scripts/convert_external_ckpt.py birdmae_base
    uv run python scripts/convert_external_ckpt.py audiomae --out checkpoints/amae.ckpt

Then probe as usual, with the matching model + transform configs:

    uv run python scripts/birdset_eval.py \
        module/model=proto/birdmae data/transforms=birdmae \
        trainer.resume_from_checkpoint=checkpoints/birdmae_base.ckpt
"""

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from soundscape_ssl.models import ViTEncoder
from soundscape_ssl.models.components import get_2d_sincos_pos_embed

# name -> (HF repo, encoder img_size as (frames, n_mels), pos_embed grid to
# rebuild as (n_mels // patch, frames // patch), or None to keep the
# checkpoint's own tensor).
#
# AudioMAE ships a 1024-frame (513-token) pos_embed, but its own recipe probes 5 s
# clips at 512 frames and regenerates the sincos grid for that length
# (main_finetune_esc.py, "esc size: (8,32)"; Bird-MAE's vit.py does the same).
# The grid is (8, 32) rather than (32, 8) because both models generate positions
# with the mel axis first while patch tokens are flattened time-first — a fixed
# quirk inherited from the original AudioMAE code that the weights were trained
# with, so it must be reproduced rather than corrected.
MODELS = {
    "birdmae_base": ("DBD-research-group/Bird-MAE-Base", (512, 128), None),
    "audiomae": ("gaunernst/vit_base_patch16_1024_128.audiomae_as2m", (512, 128), (8, 32)),
}

# fc_norm belongs to Bird-MAE's mean-pooling readout (proto probing uses `norm`
# instead); head.* is timm's classifier stub on the AudioMAE export.
DROP_PREFIXES = ("fc_norm.", "head.")


def convert(model: str, out: Path) -> None:
    repo_id, img_size, pos_grid = MODELS[model]
    state = load_file(hf_hub_download(repo_id, "model.safetensors"))

    dropped = sorted(k for k in state if k.startswith(DROP_PREFIXES))
    encoder_state = {k: v for k, v in state.items() if not k.startswith(DROP_PREFIXES)}

    if pos_grid is not None:
        embed_dim = encoder_state["pos_embed"].shape[-1]
        encoder_state["pos_embed"] = get_2d_sincos_pos_embed(
            embed_dim, *pos_grid, cls_token=True
        ).unsqueeze(0)

    # Gate the conversion: strict=True fails loudly if the published weights ever
    # stop matching the geometry the paired model config assumes.
    encoder = ViTEncoder(
        img_size=img_size,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_norm=False,
        pos_embed_type="sinusoidal_2d",
    )
    encoder.load_state_dict(encoder_state, strict=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {f"encoder.{k}": v for k, v in encoder_state.items()}}, out)

    print(f"{repo_id} -> {out}")
    print(f"  img_size={img_size}  tensors={len(encoder_state)}  dropped={dropped}")
    if pos_grid is not None:
        print(f"  pos_embed rebuilt on the {pos_grid} sincos grid "
              f"-> {tuple(encoder_state['pos_embed'].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(MODELS), default="birdmae_base")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output checkpoint path (default: checkpoints/<model>.ckpt)",
    )
    args = parser.parse_args()
    convert(args.model, args.out or Path("checkpoints") / f"{args.model}.ckpt")


if __name__ == "__main__":
    main()
