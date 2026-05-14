"""
Cross-architecture comparison: metrics collection and report generation.

Loads results from multiple training runs and generates comparison tables,
plots, and statistical summaries.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
import numpy as np

from model.config import ModelConfig
from model.transformer import Transformer
from model.utils import count_parameters
from eval.perplexity import evaluate_perplexity


@dataclass
class RunMetrics:
    """Metrics from a single training run."""
    variant: str
    val_loss: float = 0.0
    val_ppl: float = 0.0
    tokens_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    mfu: float = 0.0
    total_params: int = 0
    kv_cache_bytes_per_token: int = 0
    final_step: int = 0


def estimate_kv_cache_per_token(config: ModelConfig) -> int:
    """Estimate KV cache memory per token per layer (in bytes, BF16).

    This is the key metric that differentiates attention variants.
    """
    bytes_per_elem = 2  # BF16

    if config.attention_type == "mha":
        # Full K + V: 2 * n_heads * head_dim
        return 2 * config.n_heads * config.head_dim * bytes_per_elem

    elif config.attention_type in ("gqa", "swa"):
        # Reduced: 2 * n_kv_heads * head_dim
        return 2 * config.n_kv_heads * config.head_dim * bytes_per_elem

    elif config.attention_type == "mqa":
        # Minimal: 2 * 1 * head_dim
        return 2 * config.head_dim * bytes_per_elem

    elif config.attention_type == "mla":
        # Only latent c_KV is cached: d_latent + rope_head_dim
        return (config.d_latent + config.rope_head_dim) * bytes_per_elem

    elif config.attention_type == "diff":
        # Same as MHA (no cache reduction)
        return 2 * config.n_heads * config.head_dim * bytes_per_elem

    elif config.attention_type == "moh":
        # Same structure as MHA (routing doesn't reduce cache)
        return 2 * config.n_heads * config.head_dim * bytes_per_elem

    elif config.attention_type == "nsa":
        # Only window tokens cached (compression + selection are ephemeral)
        return 2 * config.n_heads * config.head_dim * bytes_per_elem

    return 2 * config.n_heads * config.head_dim * bytes_per_elem


def format_comparison_table(metrics_list: list[RunMetrics]) -> str:
    """Generate a markdown comparison table from run metrics."""
    header = (
        "| Variant | Val PPL ↓ | Tokens/s ↑ | Peak Mem ↓ | MFU ↑ | "
        "KV$/tok ↓ | Params |\n"
        "|---------|----------|-----------|-----------|-------|"
        "----------|--------|\n"
    )
    rows = []
    for m in sorted(metrics_list, key=lambda x: x.val_ppl):
        rows.append(
            f"| {m.variant:8s} | {m.val_ppl:8.1f} | {m.tokens_per_sec:>9,.0f} | "
            f"{m.peak_memory_mb:>7,.0f} MB | {m.mfu:>4.1f}% | "
            f"{m.kv_cache_bytes_per_token:>6d} B | {m.total_params / 1e6:>5.1f}M |"
        )
    return header + "\n".join(rows)


def save_results(metrics_list: list[RunMetrics], output_dir: str) -> None:
    """Save results as JSON for downstream visualization."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    data = [asdict(m) for m in metrics_list]
    with open(path / "comparison_results.json", "w") as f:
        json.dump(data, f, indent=2)

    # Also save markdown table
    table = format_comparison_table(metrics_list)
    with open(path / "comparison_table.md", "w") as f:
        f.write("# Attention Variant Comparison\n\n")
        f.write(table)
        f.write("\n")

    print(f"[Compare] Results saved to {path}")
    print(table)
