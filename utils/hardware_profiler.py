"""
SPECTRA-Sum: Hardware Telemetry & Systems Profiler
Measures peak memory, latency, token throughput, and arithmetic intensity.
"""

import time
import os
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import resource
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class HardwareMetrics:
    wall_time_sec: float
    cpu_utilization_pct: float
    peak_ram_mb: float
    ram_delta_mb: float
    peak_device_mem_mb: float
    tokens_generated: int
    throughput_tokens_per_sec: float
    arithmetic_intensity_estimate: float
    precision_mode: str
    device_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_time_sec": round(self.wall_time_sec, 4),
            "cpu_utilization_pct": round(self.cpu_utilization_pct, 1),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "ram_delta_mb": round(self.ram_delta_mb, 2),
            "peak_device_mem_mb": round(self.peak_device_mem_mb, 2),
            "tokens_generated": self.tokens_generated,
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
            "arithmetic_intensity_estimate": round(self.arithmetic_intensity_estimate, 2),
            "precision_mode": self.precision_mode,
            "device_name": self.device_name,
        }


class HardwareProfiler:
    """
    Context manager and telemetry monitor for hardware-level profiling during model inference.
    """

    def __init__(self, precision_mode: str = "fp32", model_param_count: int = 140_000_000):
        self.precision_mode = precision_mode
        self.model_param_count = model_param_count
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.start_cpu_time: float = 0.0
        self.end_cpu_time: float = 0.0
        self.start_ram_mb: float = 0.0
        self.peak_ram_mb: float = 0.0
        self.tokens_generated: int = 0
        self.device_name: str = self._detect_device()

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
        return "CPU Host"

    def _get_current_ram_mb(self) -> float:
        if self.process is not None:
            return self.process.memory_info().rss / (1024.0 * 1024.0)
        # Fallback to resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS, ru_maxrss is in bytes; on Linux in KB
        import sys
        if sys.platform == 'darwin':
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0

    def _get_device_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "current_allocated_memory"):
                return torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
        return 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self.start_ram_mb = self._get_current_ram_mb()
        self.peak_ram_mb = self.start_ram_mb
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_time = time.perf_counter()
        current_ram = self._get_current_ram_mb()
        self.peak_ram_mb = max(self.peak_ram_mb, current_ram)

    def record_tokens(self, token_count: int):
        self.tokens_generated += max(0, token_count)

    def get_metrics(self) -> HardwareMetrics:
        elapsed = max(self.end_time - self.start_time, 1e-6)
        throughput = self.tokens_generated / elapsed if self.tokens_generated > 0 else 0.0
        ram_delta = max(0.0, self.peak_ram_mb - self.start_ram_mb)
        device_mem = self._get_device_memory_mb()

        # Arithmetic intensity: FLOPs per Byte transferred
        # Transformer forward pass: ~2 * P FLOPs per generated token
        # Memory transfer: ~P * bytes_per_param (4 for fp32, 2 for fp16, 1 for int8)
        bytes_per_param = 1.0 if "int8" in self.precision_mode else (2.0 if "fp16" in self.precision_mode else 4.0)
        estimated_flops = 2.0 * self.model_param_count * max(1, self.tokens_generated)
        estimated_bytes = self.model_param_count * bytes_per_param * max(1, self.tokens_generated)
        intensity = estimated_flops / max(estimated_bytes, 1e-6)

        return HardwareMetrics(
            wall_time_sec=elapsed,
            cpu_utilization_pct=psutil.cpu_percent(),
            peak_ram_mb=self.peak_ram_mb,
            ram_delta_mb=ram_delta,
            peak_device_mem_mb=device_mem,
            tokens_generated=self.tokens_generated,
            throughput_tokens_per_sec=throughput,
            arithmetic_intensity_estimate=intensity,
            precision_mode=self.precision_mode,
            device_name=self.device_name,
        )
