"""Training entry point: parse config and launch training."""

import argparse
import yaml
from training.trainer import Trainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Handle inheritance
    if "_base_" in cfg:
        base_path = config_path.rsplit("/", 1)[0] + "/" + cfg.pop("_base_")
        base = load_config(base_path)
        for key, val in cfg.items():
            if isinstance(val, dict) and key in base:
                base[key].update(val)
            else:
                base[key] = val
        return base
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train()
