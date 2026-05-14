# MiniLM-Bench: Google Colab Training Guide
#
# This script trains all 8 attention variants on Colab with:
# - Google Drive persistence (survives disconnections)
# - Automatic checkpoint resume
# - Optimized settings for T4 (free) or A100 (Colab Pro)
# - Sequential training with progress tracking
#
# Usage:
# 1. Upload minilm-bench/ folder to Google Drive
# 2. Open this as a Colab notebook or run the cells sequentially
# 3. Training auto-resumes on reconnection

# ============================================================
# Cell 1: Mount Drive + Setup
# ============================================================

import os
import subprocess

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Project paths — ALL state lives on Drive
DRIVE_PROJECT = "/content/drive/MyDrive/minilm-bench"
DRIVE_DATA = f"{DRIVE_PROJECT}/data/tokenized"
DRIVE_CHECKPOINTS = f"{DRIVE_PROJECT}/checkpoints"
DRIVE_RESULTS = f"{DRIVE_PROJECT}/results"

# Clone or sync project
if not os.path.exists(f"{DRIVE_PROJECT}/model"):
    print("ERROR: Upload minilm-bench/ to Google Drive first!")
    print("Expected path: My Drive/minilm-bench/")
    raise FileNotFoundError(f"{DRIVE_PROJECT}/model not found")

# Work from Drive directly (persistence!)
os.chdir(DRIVE_PROJECT)

# Install dependencies
subprocess.run(["pip", "install", "-q", "torch", "numpy", "tiktoken", "pyyaml", "wandb"], check=True)

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

# ============================================================
# Cell 2: Detect GPU + Set Optimal Config
# ============================================================

gpu_name = torch.cuda.get_device_name(0).lower()
gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

# Auto-configure based on GPU type
if "a100" in gpu_name:
    # A100 (40GB/80GB) — Colab Pro
    CONFIG = {
        "batch_size": 32,
        "grad_accum_steps": 4,
        "d_model": 768,
        "n_layers": 12,
        "max_steps": 20000,
        "eval_interval": 500,
        "checkpoint_interval": 1000,
        "compile": True,
    }
    print("🚀 A100 detected — using full config")
elif "t4" in gpu_name:
    # T4 (16GB) — Colab Free
    CONFIG = {
        "batch_size": 16,
        "grad_accum_steps": 8,
        "d_model": 512,
        "n_layers": 8,
        "max_steps": 10000,
        "eval_interval": 500,
        "checkpoint_interval": 500,
        "compile": True,
    }
    print("🆓 T4 detected — using memory-optimized config")
elif "v100" in gpu_name:
    CONFIG = {
        "batch_size": 24,
        "grad_accum_steps": 4,
        "d_model": 768,
        "n_layers": 12,
        "max_steps": 15000,
        "eval_interval": 500,
        "checkpoint_interval": 1000,
        "compile": False,  # V100 doesn't support torch.compile well
    }
    print("⚡ V100 detected")
else:
    # Conservative fallback
    CONFIG = {
        "batch_size": 8,
        "grad_accum_steps": 16,
        "d_model": 512,
        "n_layers": 8,
        "max_steps": 5000,
        "eval_interval": 250,
        "checkpoint_interval": 250,
        "compile": False,
    }
    print(f"⚠️ Unknown GPU ({gpu_name}) — using conservative config")

effective_batch = CONFIG["batch_size"] * CONFIG["grad_accum_steps"]
print(f"Effective batch size: {effective_batch}")
print(f"Model: d={CONFIG['d_model']}, L={CONFIG['n_layers']}")

# ============================================================
# Cell 3: Download + Tokenize Data (runs once, saved to Drive)
# ============================================================

if os.path.exists(DRIVE_DATA) and len(os.listdir(DRIVE_DATA)) > 0:
    n_shards = len([f for f in os.listdir(DRIVE_DATA) if f.endswith('.bin')])
    print(f"✅ Data already on Drive ({n_shards} shards). Skipping download.")
else:
    print("📥 Downloading and tokenizing FineWeb-Edu...")
    os.makedirs(DRIVE_DATA, exist_ok=True)
    subprocess.run([
        "python", "-m", "data.download",
        "--output_dir", DRIVE_DATA,
        "--max_shards", "10",
    ], check=True)
    print("✅ Data saved to Drive!")

# ============================================================
# Cell 4: Training Function with Auto-Resume
# ============================================================

import yaml
import time
import json
from pathlib import Path

def load_base_config():
    """Load base.yaml config."""
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)

def build_config(variant: str) -> dict:
    """Build training config for a variant, optimized for current GPU."""
    cfg = load_base_config()
    n_heads = CONFIG["d_model"] // 64

    # Model overrides
    cfg["model"]["d_model"] = CONFIG["d_model"]
    cfg["model"]["n_layers"] = CONFIG["n_layers"]
    cfg["model"]["n_heads"] = n_heads
    cfg["model"]["n_kv_heads"] = n_heads
    cfg["model"]["attention_type"] = variant

    # Variant-specific overrides
    if variant == "mqa":
        cfg["model"]["n_kv_heads"] = 1
    elif variant == "gqa":
        cfg["model"]["n_kv_heads"] = max(1, n_heads // 4)
    elif variant == "mla":
        cfg["model"]["d_latent"] = CONFIG["d_model"] // 4
        cfg["model"]["rope_head_dim"] = 64
    elif variant == "moh":
        cfg["model"]["moh_top_k"] = max(2, n_heads // 2)
        cfg["model"]["moh_n_shared"] = 1
    elif variant == "nsa":
        cfg["model"]["nsa_block_size"] = 16
        cfg["model"]["nsa_top_k_blocks"] = 8
        cfg["model"]["nsa_window_size"] = 512

    # Training overrides
    cfg["training"]["batch_size"] = CONFIG["batch_size"]
    cfg["training"]["grad_accum_steps"] = CONFIG["grad_accum_steps"]
    cfg["training"]["max_steps"] = CONFIG["max_steps"]
    cfg["training"]["eval_interval"] = CONFIG["eval_interval"]
    cfg["training"]["checkpoint_interval"] = CONFIG["checkpoint_interval"]

    # Data
    cfg["data"]["data_dir"] = DRIVE_DATA

    # Infrastructure — save to Drive!
    cfg["infra"]["device"] = "cuda"
    cfg["infra"]["dtype"] = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    cfg["infra"]["compile"] = CONFIG["compile"]
    cfg["infra"]["checkpoint_dir"] = f"{DRIVE_CHECKPOINTS}/{variant}"
    cfg["infra"]["log_dir"] = f"{DRIVE_PROJECT}/logs/{variant}"
    cfg["infra"]["wandb_enabled"] = False  # Set True if you have W&B

    return cfg


def check_training_status(variant: str) -> dict:
    """Check if a variant has already been trained (fully or partially)."""
    ckpt_dir = Path(f"{DRIVE_CHECKPOINTS}/{variant}")
    if not ckpt_dir.exists():
        return {"status": "not_started", "step": 0}

    checkpoints = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return {"status": "not_started", "step": 0}

    latest = checkpoints[-1]
    step = int(latest.stem.split("_")[-1])

    if step >= CONFIG["max_steps"]:
        return {"status": "complete", "step": step}
    else:
        return {"status": "in_progress", "step": step}


def train_variant(variant: str):
    """Train a single variant with auto-resume."""
    from training.trainer import Trainer

    status = check_training_status(variant)
    if status["status"] == "complete":
        print(f"✅ {variant.upper()} already complete (step {status['step']}). Skipping.")
        return

    if status["status"] == "in_progress":
        print(f"🔄 {variant.upper()} resuming from step {status['step']}...")
    else:
        print(f"🆕 {variant.upper()} starting fresh...")

    cfg = build_config(variant)
    trainer = Trainer(cfg)
    trainer.train()

    print(f"✅ {variant.upper()} training complete!")


# ============================================================
# Cell 5: Train All Variants Sequentially
# ============================================================
# Run this cell. If Colab disconnects, just re-run — it auto-resumes.

VARIANTS = ["mha", "gqa", "mqa", "swa", "diff", "mla", "moh", "nsa"]

print("=" * 60)
print("TRAINING STATUS")
print("=" * 60)
for v in VARIANTS:
    s = check_training_status(v)
    emoji = {"not_started": "⬜", "in_progress": "🟡", "complete": "🟢"}[s["status"]]
    print(f"  {emoji} {v:>5s}: {s['status']:>12s} (step {s['step']})")
print("=" * 60)

# Train each variant
for variant in VARIANTS:
    try:
        train_variant(variant)
    except Exception as e:
        print(f"❌ {variant.upper()} failed: {e}")
        print("Continuing to next variant...")
        continue

# ============================================================
# Cell 6: Profile All Variants
# ============================================================

print("\n📊 Running throughput profiling...")
os.environ["PYTHONPATH"] = DRIVE_PROJECT
subprocess.run([
    "python", "scripts/profile.py",
    "--device", "cuda",
    "--d_model", str(CONFIG["d_model"]),
    "--n_layers", str(CONFIG["n_layers"]),
    "--batch_size", str(CONFIG["batch_size"]),
    "--seq_len", "1024",
], check=True)

# ============================================================
# Cell 7: Generate Comparison Table
# ============================================================

from eval.compare import RunMetrics, estimate_kv_cache_per_token, format_comparison_table, save_results
from model.config import ModelConfig
from model.transformer import Transformer
from training.checkpoint import CheckpointManager

results = []
for variant in VARIANTS:
    status = check_training_status(variant)
    if status["status"] != "complete":
        print(f"⚠️ {variant} not complete (step {status['step']}), skipping evaluation")
        continue

    cfg = build_config(variant)
    model_cfg = ModelConfig(**cfg["model"])

    # Load checkpoint
    ckpt_mgr = CheckpointManager(cfg["infra"]["checkpoint_dir"])
    state = ckpt_mgr.load_latest()

    metrics = RunMetrics(
        variant=variant,
        val_loss=state.get("val_loss", 0) or 0,
        val_ppl=torch.exp(torch.tensor(state.get("val_loss", 0) or 0)).item(),
        total_params=model_cfg.num_params(),
        kv_cache_bytes_per_token=estimate_kv_cache_per_token(model_cfg),
        final_step=state["step"],
    )
    results.append(metrics)

if results:
    save_results(results, DRIVE_RESULTS)
    print("\n" + format_comparison_table(results))
else:
    print("No completed training runs to compare.")
