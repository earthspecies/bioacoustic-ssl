"""Report train-dataset sizes and suggest `max_steps` for each config.

Loads every config under configs/data/datasets/train/, reports the raw size,
number of classes, and class imbalance, then suggests `trainer.max_steps`
under two anchors:

  - raw      : E effective passes over the raw files
  - balanced : E passes giving each class ~P balanced draws (the analog of the
               old LongTailUpsampleTarget target_count)

Usage:
    python scripts/dataset_sizes.py                 # defaults: E=30, P=500, bs=256
    python scripts/dataset_sizes.py 20 500 256      # E P batch_size
"""

import math
import sys
from collections import Counter
from pathlib import Path

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from esp_data import dataset_from_config

CONFIG_DIR = "../configs"
TRAIN_DIR = Path(__file__).resolve().parent.parent / "configs/data/datasets/train"


def main(E: float, P: int, batch_size: int) -> None:
    names = sorted(p.stem for p in TRAIN_DIR.glob("*.yaml"))

    rows = []
    for name in names:
        with initialize(version_base=None, config_path=CONFIG_DIR):
            cfg = compose(
                config_name="train",
                overrides=[f"data/datasets=train/{name}"],
            )
        ds, meta = dataset_from_config(instantiate(cfg.data.datasets["train"]))

        n = len(ds)
        num_classes = meta["mulitlabel_from_feature"]["num_classes"]

        # per-class counts from the mapped label column
        labels = ds._data.unwrap["label"].to_list()
        counts = Counter(c for lst in labels if lst for c in lst)
        cmin, cmax = min(counts.values()), max(counts.values())

        steps_raw = round(E * math.ceil(n / batch_size))
        steps_bal = round(E * P * num_classes / batch_size)

        rows.append((name, n, num_classes, cmin, cmax, cmax / cmin, steps_raw, steps_bal))

    hdr = f"{'dataset':<8}{'size':>8}{'classes':>9}{'min/cls':>9}{'max/cls':>9}{'imbal':>8}{'steps_raw':>11}{'steps_bal':>11}"
    print(f"\nE={E}  P={P}  batch_size={batch_size}\n")
    print(hdr)
    print("-" * len(hdr))
    for name, n, nc, cmin, cmax, imbal, sraw, sbal in rows:
        print(f"{name:<8}{n:>8}{nc:>9}{cmin:>9}{cmax:>9}{imbal:>8.1f}{sraw:>11}{sbal:>11}")
    print("\nsteps_raw = E * ceil(size/bs)        (E passes over raw files)")
    print("steps_bal = E * P * classes / bs     (each class ~P balanced draws per pass) [recommended]\n")


if __name__ == "__main__":
    E = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    main(E, P, bs)
