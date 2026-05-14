"""
Main training loop with mixed-precision, gradient accumulation, W&B logging,
validation evaluation, and MoH auxiliary loss support.
"""

import os
import time
import yaml
import torch
import torch.nn as nn
from pathlib import Path

from model import ModelConfig, Transformer
from model.utils import print_model_summary
from training.optimizer import configure_optimizer, build_lr_scheduler
from training.checkpoint import CheckpointManager
from training.profiler import Profiler
from data.dataloader import create_dataloader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer:
    """Pre-training loop with BF16, grad accumulation, profiling, W&B, and checkpointing."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.device = config["infra"]["device"]
        self.dtype = getattr(torch, config["infra"]["dtype"])

        # Set seed
        torch.manual_seed(config["infra"].get("seed", 42))

        # Build model
        model_cfg = ModelConfig(**config["model"])
        self.model_cfg = model_cfg
        self.model = Transformer(model_cfg).to(self.device)
        print_model_summary(self.model, model_cfg)

        # Compile if enabled (skip on CPU)
        if config["infra"].get("compile", False) and self.device == "cuda":
            print("[Trainer] Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Optimizer + scheduler
        opt_cfg = config["optimizer"]
        self.optimizer = configure_optimizer(
            self.model, opt_cfg["lr"], opt_cfg["weight_decay"], tuple(opt_cfg["betas"])
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer, opt_cfg["warmup_steps"], config["training"]["max_steps"]
        )

        # Checkpoint manager
        self.ckpt_mgr = CheckpointManager(config["infra"]["checkpoint_dir"])

        # Profiler
        self.profiler = Profiler(
            model_cfg, config["training"]["batch_size"],
            config["data"]["seq_len"], config["training"]["grad_accum_steps"]
        )

        # Data
        self.train_loader = create_dataloader(
            config["data"]["data_dir"], config["data"]["seq_len"],
            config["training"]["batch_size"], config["data"]["num_workers"], shuffle=True
        )
        self.train_iter = iter(self.train_loader)

        self.step = 0
        self.best_val_loss = float("inf")

        # W&B
        self.use_wandb = (
            HAS_WANDB
            and config["infra"].get("wandb_enabled", False)
            and config["infra"].get("wandb_project")
        )
        if self.use_wandb:
            wandb.init(
                project=config["infra"]["wandb_project"],
                name=f"{model_cfg.attention_type}-{model_cfg.d_model}d-{model_cfg.n_layers}L",
                config=config,
            )
            wandb.watch(self.model, log="gradients", log_freq=100)

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next training batch, cycling through DataLoader."""
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
        return x.to(self.device), y.to(self.device)

    def _collect_moh_aux_loss(self) -> torch.Tensor:
        """Collect auxiliary load-balancing loss from MoH attention layers."""
        aux_loss = torch.tensor(0.0, device=self.device)
        count = 0
        for block in self.model.blocks:
            attn = block.attention
            if hasattr(attn, "aux_loss") and attn.aux_loss is not None:
                aux_loss = aux_loss + attn.aux_loss
                count += 1
        return aux_loss / max(count, 1) if count > 0 else aux_loss

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Run validation evaluation. Returns average loss."""
        self.model.eval()
        tcfg = self.cfg["training"]

        # Create val loader (use same data dir for now, separate split later)
        val_loader = create_dataloader(
            self.cfg["data"]["data_dir"], self.cfg["data"]["seq_len"],
            self.cfg["training"]["batch_size"], num_workers=2, shuffle=False
        )

        total_loss = 0.0
        count = 0
        for i, (x, y) in enumerate(val_loader):
            if i >= tcfg.get("eval_steps", 50):
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda"):
                _, loss = self.model(x, y)
            total_loss += loss.item()
            count += 1

        self.model.train()
        return total_loss / max(count, 1)

    def train(self) -> None:
        """Main training loop."""
        tcfg = self.cfg["training"]

        # Auto-resume from checkpoint
        ckpt = self.ckpt_mgr.load_latest()
        if ckpt:
            model_to_load = self.model
            # Handle torch.compile wrapper
            if hasattr(self.model, "_orig_mod"):
                model_to_load = self.model._orig_mod
            model_to_load.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.step = ckpt["step"]
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"[Trainer] Resumed from step {self.step}")

        self.model.train()
        print(f"\n[Trainer] Starting training from step {self.step}")
        print(f"[Trainer] Attention type: {self.model_cfg.attention_type}")
        print(f"[Trainer] Max steps: {tcfg['max_steps']}")
        print(f"[Trainer] Effective batch: {tcfg['batch_size'] * tcfg['grad_accum_steps']}")

        while self.step < tcfg["max_steps"]:
            self.profiler.step_start()

            # Gradient accumulation
            self.optimizer.zero_grad()
            total_loss = 0.0

            for micro_step in range(tcfg["grad_accum_steps"]):
                x, y = self._get_batch()
                with torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda"):
                    _, loss = self.model(x, y)
                    loss = loss / tcfg["grad_accum_steps"]

                    # Add MoH auxiliary loss if applicable
                    if self.model_cfg.attention_type == "moh":
                        aux = self._collect_moh_aux_loss() / tcfg["grad_accum_steps"]
                        loss = loss + aux

                loss.backward()
                total_loss += loss.item()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["optimizer"]["grad_clip"]
            )

            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            dt = self.profiler.step_end()

            # Logging
            if self.step % tcfg["log_interval"] == 0:
                tps = self.profiler.tokens_per_second(dt)
                lr = self.scheduler.get_last_lr()[0]
                mem = self.profiler.peak_memory_mb()
                log_msg = (
                    f"step={self.step:>6d} | loss={total_loss:.4f} | "
                    f"lr={lr:.2e} | grad_norm={grad_norm:.2f} | "
                    f"tok/s={tps:,.0f} | mem={mem:,.0f}MB | dt={dt:.3f}s"
                )
                print(log_msg)

                if self.use_wandb:
                    wandb.log({
                        "train/loss": total_loss,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "perf/tokens_per_sec": tps,
                        "perf/step_time_s": dt,
                        "perf/peak_memory_mb": mem,
                    }, step=self.step)

            # Validation
            if self.step % tcfg.get("eval_interval", 500) == 0:
                val_loss = self._evaluate()
                ppl = torch.exp(torch.tensor(val_loss)).item()
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                print(f"  [Eval] val_loss={val_loss:.4f} | ppl={ppl:.1f} | best={is_best}")

                if self.use_wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "val/perplexity": ppl,
                        "val/best_loss": self.best_val_loss,
                    }, step=self.step)

                self.ckpt_mgr.save(
                    self.model, self.optimizer, self.scheduler, self.step,
                    val_loss=val_loss, is_best=is_best, best_val_loss=self.best_val_loss,
                )

            # Periodic checkpointing
            elif self.step % tcfg["checkpoint_interval"] == 0:
                self.ckpt_mgr.save(
                    self.model, self.optimizer, self.scheduler, self.step,
                    best_val_loss=self.best_val_loss,
                )

            # Graceful exit
            if self.ckpt_mgr.should_exit:
                self.ckpt_mgr.save(
                    self.model, self.optimizer, self.scheduler, self.step,
                    best_val_loss=self.best_val_loss,
                )
                print("[Trainer] Graceful exit after SIGTERM.")
                break

        print(f"\n[Trainer] Training complete at step {self.step}")
        self.ckpt_mgr.save(
            self.model, self.optimizer, self.scheduler, self.step,
            best_val_loss=self.best_val_loss,
        )

        if self.use_wandb:
            wandb.finish()
