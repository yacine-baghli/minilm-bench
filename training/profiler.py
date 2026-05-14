"""
Throughput and memory profiling utilities.

Measures tokens/sec, Model FLOPs Utilization (MFU), and peak GPU memory.
"""

import time
import torch
from model.config import ModelConfig
from model.utils import estimate_flops_per_step


class Profiler:
    """Training throughput and memory profiler."""

    GPU_PEAK_FLOPS = {"A100": 312e12, "H100": 989e12, "RTX 4090": 165e12, "RTX 3090": 71e12, "T4": 65e12}

    def __init__(self, config: ModelConfig, batch_size: int, seq_len: int, grad_accum_steps: int = 1) -> None:
        self.tokens_per_step = batch_size * seq_len * grad_accum_steps
        self.flops_per_step = estimate_flops_per_step(config, batch_size * grad_accum_steps, seq_len)
        self._step_start: float | None = None
        self._step_times: list[float] = []

    def step_start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def step_end(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._step_start
        self._step_times.append(dt)
        self._step_start = None
        return dt

    def tokens_per_second(self, dt: float) -> float:
        return self.tokens_per_step / dt

    def mfu(self, dt: float, gpu_name: str = "A100") -> float:
        peak = self.GPU_PEAK_FLOPS.get(gpu_name, 312e12)
        return (self.flops_per_step / dt) / peak

    @staticmethod
    def peak_memory_mb() -> float:
        return torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    def summary(self) -> dict[str, float]:
        if not self._step_times:
            return {}
        avg_dt = sum(self._step_times) / len(self._step_times)
        return {"avg_step_time_s": avg_dt, "avg_tokens_per_sec": self.tokens_per_second(avg_dt), "peak_memory_mb": self.peak_memory_mb()}
