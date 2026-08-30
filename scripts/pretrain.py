"""MAE pretraining entry point.

Single GPU:
    python scripts/pretrain.py datamodule.a2o_detections_path=/path/to/detections.csv

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 scripts/pretrain.py \
        datamodule.a2o_detections_path=/path/to/detections.csv \
        trainer.devices=4 trainer.strategy=ddp

Override any config key at the command line:
    python scripts/pretrain.py module.model.encoder_depth=6 trainer.max_epochs=100

Resume from checkpoint:
    python scripts/pretrain.py \
        datamodule.a2o_detections_path=/path/to/detections.csv \
        trainer.resume_from_checkpoint=/path/to/epoch_0010.ckpt
"""

from dotenv import load_dotenv
load_dotenv()  # load repo .env (secrets, HF cache, CA bundle) before other imports

import hydra
import torch
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf

from bioacoustic_ssl.training.mae_pretrainer import pretrain


@hydra.main(version_base=None, config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    print("Torch version:", torch.__version__)

    fabric = Fabric(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
    )
    fabric.launch(pretrain, cfg)


if __name__ == "__main__":
    main()
