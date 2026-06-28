# Copyright (c) Opendatalab. All rights reserved.
"""Memory profiler for auto-detecting system capabilities at startup.

On Apple Silicon (M1/M2/M3+), CPU and GPU share unified memory,
so total system RAM is the budget for everything.
On Linux/Windows with discrete GPUs, system RAM and VRAM are separate.
"""
import os
import platform
from dataclasses import dataclass, field
from loguru import logger

from mineru.utils.check_sys_env import (
    is_mac_environment,
    is_apple_silicon_cpu,
    is_mac_os_version_supported,
)


@dataclass
class MemoryProfile:
    total_system_memory_gb: float
    available_memory_gb: float
    is_unified_memory: bool
    is_apple_silicon: bool
    chip_description: str
    vram_gb: int = 0
    has_gpu: bool = False
    platform: str = field(default_factory=platform.system)

    @property
    def effective_memory_gb(self) -> float:
        if self.is_unified_memory:
            return self.available_memory_gb
        return float(self.vram_gb) if self.vram_gb > 0 else self.available_memory_gb

    @property
    def is_low_memory(self) -> bool:
        return self.effective_memory_gb <= 16

    @property
    def is_very_low_memory(self) -> bool:
        return self.effective_memory_gb <= 8


def _get_system_memory_gb() -> tuple[float, float]:
    try:
        import psutil
        mem = psutil.virtual_memory()
        total = mem.total / (1024 ** 3)
        available = mem.available / (1024 ** 3)
        return round(total, 1), round(available, 1)
    except ImportError:
        pass

    try:
        import os as _os
        total = _os.sysconf('SC_PAGE_SIZE') * _os.sysconf('SC_PHYS_PAGES')
        total_gb = total / (1024 ** 3)
        return round(total_gb, 1), round(total_gb * 0.6, 1)
    except (ValueError, OSError, AttributeError):
        pass

    return 16.0, 10.0


def _get_mac_chip_description() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        pass

    if is_apple_silicon_cpu():
        try:
            import subprocess
            result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64.FEAT_FCMA"],
            capture_output=True, text=True, timeout=2,
        )
        except Exception:
            pass
        return "Apple Silicon"
    return "Unknown"


def _get_vram_gb() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
    except ImportError:
        pass
    except Exception:
        pass
    return 0


def profile_system_memory() -> MemoryProfile:
    total_mem, available_mem = _get_system_memory_gb()
    is_unified = is_mac_environment() and is_apple_silicon_cpu()
    is_silicon = is_apple_silicon_cpu()
    chip_desc = _get_mac_chip_description() if is_mac_environment() else platform.processor()
    vram = _get_vram_gb() if not is_unified else 0
    has_gpu = vram > 0 or (is_unified and is_mac_os_version_supported())

    profile = MemoryProfile(
        total_system_memory_gb=total_mem,
        available_memory_gb=available_mem,
        is_unified_memory=is_unified,
        is_apple_silicon=is_silicon,
        chip_description=chip_desc,
        vram_gb=vram,
        has_gpu=has_gpu,
    )

    return profile


_cached_profile: MemoryProfile | None = None


def get_memory_profile() -> MemoryProfile:
    global _cached_profile
    if _cached_profile is None:
        _cached_profile = profile_system_memory()
    return _cached_profile
