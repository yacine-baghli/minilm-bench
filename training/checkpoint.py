"""
Fault-tolerant checkpoint management.

Supports atomic writes, auto-resume, graceful preemption handling,
and periodic + best-loss checkpointing.
"""

import os
import signal
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages model checkpoints with fault tolerance.

    Features:
        - Atomic writes (temp file + rename) to prevent corruption on crash.
        - Auto-resume: seamlessly continues from latest checkpoint.
        - Graceful preemption: catches SIGTERM and saves before exit.
        - Tracks both periodic and best-validation-loss checkpoints.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        max_keep: Maximum number of periodic checkpoints to keep.
    """

    def __init__(self, checkpoint_dir: str, max_keep: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self._should_save_and_exit = False

        # Register SIGTERM handler for graceful preemption
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM: flag that we should save and exit."""
        print("\n[CheckpointManager] SIGTERM received. Will save checkpoint and exit.")
        self._should_save_and_exit = True

    @property
    def should_exit(self) -> bool:
        """Check if graceful exit was requested."""
        return self._should_save_and_exit

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        val_loss: float | None = None,
        is_best: bool = False,
        **extra_state,
    ) -> Path:
        """Save checkpoint atomically.

        Uses temp file + os.replace() to ensure the checkpoint file
        is never in a partially-written state.

        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: LR scheduler to save.
            step: Current training step.
            val_loss: Current validation loss (optional).
            is_best: Whether this is the best validation loss so far.
            **extra_state: Additional state to save (e.g., rng_state).

        Returns:
            Path to saved checkpoint.
        """
        # Unwrap torch.compile wrapper if present
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        state = {
            "step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "torch_rng_state": torch.random.get_rng_state(),
            **extra_state,
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

        # Atomic write: save to temp, then rename
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step:08d}.pt"
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.checkpoint_dir, suffix=".pt.tmp"
        )
        os.close(tmp_fd)

        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        print(f"[Checkpoint] Saved: {checkpoint_path.name} (step={step})")

        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy2(checkpoint_path, best_path)
            print(f"[Checkpoint] New best! val_loss={val_loss:.4f}")

        # Cleanup old checkpoints (keep max_keep most recent)
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_latest(self) -> dict | None:
        """Load the most recent checkpoint, if any.

        Returns:
            Checkpoint state dict, or None if no checkpoint found.
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        if not checkpoints:
            print("[Checkpoint] No checkpoint found. Starting from scratch.")
            return None

        latest = checkpoints[-1]
        print(f"[Checkpoint] Resuming from: {latest.name}")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        return state

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old periodic checkpoints, keeping only max_keep most recent."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        while len(checkpoints) > self.max_keep:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"[Checkpoint] Removed old: {oldest.name}")
