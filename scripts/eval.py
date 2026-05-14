"""Evaluation entry point: load checkpoint → run metrics → output table."""

import argparse
import yaml
import torch

from model import ModelConfig, Transformer
from model.utils import count_parameters, print_model_summary
from eval.perplexity import evaluate_perplexity
from eval.compare import RunMetrics, estimate_kv_cache_per_token, format_comparison_table


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
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


def evaluate_checkpoint(config_path: str, checkpoint_path: str = None) -> RunMetrics:
    """Load a trained model and compute all evaluation metrics."""
    config = load_config(config_path)
    model_cfg = ModelConfig(**config["model"])
    device = config["infra"]["device"]

    # Build model
    model = Transformer(model_cfg).to(device)
    print_model_summary(model, model_cfg)

    # Load checkpoint
    if checkpoint_path is None:
        from training.checkpoint import CheckpointManager
        ckpt_mgr = CheckpointManager(config["infra"]["checkpoint_dir"])
        state = ckpt_mgr.load_latest()
    else:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if state:
        model.load_state_dict(state["model_state_dict"])
        step = state.get("step", 0)
        val_loss = state.get("val_loss", None)
    else:
        print("[Eval] No checkpoint found, evaluating random init.")
        step = 0
        val_loss = None

    # Compute perplexity
    if val_loss is None:
        val_loss_computed = 0.0
        ppl = 0.0
        try:
            ppl = evaluate_perplexity(
                model, config["data"]["data_dir"], config["data"]["seq_len"],
                config["training"]["batch_size"],
                eval_steps=config["training"].get("eval_steps", 50),
                device=device,
            )
            import math
            val_loss_computed = math.log(ppl)
        except Exception as e:
            print(f"[Eval] Perplexity evaluation failed: {e}")
    else:
        ppl = torch.exp(torch.tensor(val_loss)).item()
        val_loss_computed = val_loss

    # Collect metrics
    params = count_parameters(model)
    metrics = RunMetrics(
        variant=model_cfg.attention_type,
        val_loss=val_loss_computed,
        val_ppl=ppl,
        total_params=params.get("total", 0),
        kv_cache_bytes_per_token=estimate_kv_cache_per_token(model_cfg),
        final_step=step,
    )

    print(f"\n{'=' * 50}")
    print(f"Variant: {metrics.variant}")
    print(f"Val PPL:  {metrics.val_ppl:.2f}")
    print(f"Params:   {metrics.total_params / 1e6:.1f}M")
    print(f"KV$/tok:  {metrics.kv_cache_bytes_per_token} bytes")
    print(f"Step:     {metrics.final_step}")
    print(f"{'=' * 50}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional)")
    args = parser.parse_args()

    evaluate_checkpoint(args.config, args.checkpoint)
