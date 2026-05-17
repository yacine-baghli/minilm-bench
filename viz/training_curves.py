"""
Training curve plotting utilities.

Generates publication-quality plots from training logs or W&B exports.
"""

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def smooth(values: list[float], weight: float = 0.9) -> list[float]:
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_loss_curves(
    variant_data: dict[str, list[dict]],
    output_path: str = "results/loss_curves.png",
    smoothing: float = 0.9,
) -> None:
    """Plot training loss curves for multiple variants overlaid.

    Args:
        variant_data: {variant_name: [{step, loss}, ...]}
        output_path: Path to save the plot.
        smoothing: EMA smoothing weight.
    """
    if not HAS_MPL:
        print("[Plot] matplotlib not installed, skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(variant_data)))
    for (name, data), color in zip(variant_data.items(), colors):
        steps = [d["step"] for d in data]
        losses = [d["loss"] for d in data]
        smoothed = smooth(losses, smoothing)
        ax.plot(steps, losses, alpha=0.2, color=color)
        ax.plot(steps, smoothed, label=name, color=color, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss by Attention Variant", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Loss curves saved to {output_path}")


def plot_throughput_comparison(
    variant_data: dict[str, float],
    output_path: str = "results/throughput.png",
) -> None:
    """Bar chart comparing tokens/sec across variants."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    names = list(variant_data.keys())
    values = list(variant_data.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title("Training Throughput by Attention Variant", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Throughput chart saved to {output_path}")


def plot_memory_comparison(
    variant_data: dict[str, float],
    output_path: str = "results/memory.png",
) -> None:
    """Bar chart comparing peak memory across variants."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    names = list(variant_data.keys())
    values = list(variant_data.values())
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("Peak GPU Memory by Attention Variant", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Memory chart saved to {output_path}")
