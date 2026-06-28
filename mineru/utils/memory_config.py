# Copyright (c) Opendatalab. All rights reserved.
"""Auto-configuration resolver for memory-optimized parsing.

Probes system capabilities via memory_profiler and auto-configures:
  - VLM quantization level (int8/none)
  - Processing window size (pages held in memory)
  - PDF render DPI
  - VLM batch size
  - Model eviction budget

All settings can be overridden via environment variables.
"""
import os
from dataclasses import dataclass
from loguru import logger

from mineru.utils.memory_profiler import get_memory_profile, MemoryProfile


@dataclass
class MemoryOptimizationConfig:
    quantization: str
    processing_window_size: int
    pdf_image_dpi: int
    vlm_batch_size: int
    model_eviction_enabled: bool
    model_eviction_budget_gb: float
    gpu_memory_utilization: float
    profile: MemoryProfile

    def log_summary(self):
        p = self.profile
        mem_type = "unified memory" if p.is_unified_memory else "system RAM"
        gpu_info = f", VRAM: {p.vram_gb}GB" if p.vram_gb > 0 else ""
        logger.info(
            f"MinerU Memory Configuration (auto-detected):\n"
            f"  System: {p.total_system_memory_gb}GB {mem_type} ({p.chip_description}){gpu_info}\n"
            f"  Available: {p.available_memory_gb}GB\n"
            f"  VLM quantization: {self.quantization.upper() or 'none (BF16)'}\n"
            f"  Processing window: {self.processing_window_size} pages\n"
            f"  PDF render DPI: {self.pdf_image_dpi}\n"
            f"  VLM batch size: {self.vlm_batch_size}\n"
            f"  Model eviction: {'enabled' if self.model_eviction_enabled else 'disabled'}"
            f" (budget: {self.model_eviction_budget_gb:.1f}GB)\n"
            f"  GPU memory utilization: {self.gpu_memory_utilization:.0%}"
        )


def _resolve_quantization(profile: MemoryProfile) -> str:
    env_val = os.getenv("MINERU_VLM_QUANTIZATION", "").strip().lower()
    if env_val in ("int8", "int4", "none", ""):
        if env_val:
            return env_val if env_val != "none" else ""

    if profile.is_unified_memory:
        if profile.available_memory_gb <= 16:
            return "int8"
        elif profile.available_memory_gb <= 32:
            return "int8"
        return ""
    else:
        vram = profile.vram_gb
        if vram > 0 and vram <= 8:
            return "int8"
        elif vram > 0 and vram <= 16:
            return "int8"
        elif profile.available_memory_gb <= 16:
            return "int8"
        return ""


def _resolve_window_size(profile: MemoryProfile) -> int:
    env_val = os.getenv("MINERU_PROCESSING_WINDOW_SIZE")
    if env_val is not None:
        try:
            return max(1, int(env_val))
        except ValueError:
            logger.warning(f"Invalid MINERU_PROCESSING_WINDOW_SIZE: {env_val}")

    eff = profile.effective_memory_gb
    if eff <= 8:
        return 2
    elif eff <= 16:
        return 4
    elif eff <= 32:
        return 16
    elif eff <= 64:
        return 32
    else:
        return 64


def _resolve_dpi(profile: MemoryProfile) -> int:
    env_val = os.getenv("MINERU_PDF_IMAGE_DPI")
    if env_val is not None:
        try:
            return max(72, int(env_val))
        except ValueError:
            logger.warning(f"Invalid MINERU_PDF_IMAGE_DPI: {env_val}")

    eff = profile.effective_memory_gb
    if eff <= 16:
        return 144
    else:
        return 200


def _resolve_vlm_batch_size(profile: MemoryProfile) -> int:
    env_val = os.getenv("MINERU_VLM_BATCH_SIZE")
    if env_val is not None:
        try:
            return max(1, int(env_val))
        except ValueError:
            logger.warning(f"Invalid MINERU_VLM_BATCH_SIZE: {env_val}")

    eff = profile.effective_memory_gb
    if eff <= 8:
        return 1
    elif eff <= 16:
        return 1
    elif eff <= 32:
        return 4
    else:
        return 8


def _resolve_model_eviction(profile: MemoryProfile) -> tuple[bool, float]:
    env_val = os.getenv("MINERU_MODEL_EVICTION", "").strip().lower()
    if env_val in ("true", "false"):
        enabled = env_val == "true"
    else:
        enabled = profile.effective_memory_gb <= 32

    budget_env = os.getenv("MINERU_MODEL_EVICTION_BUDGET_GB")
    if budget_env is not None:
        try:
            budget = float(budget_env)
            return enabled, budget
        except ValueError:
            pass

    budget = profile.available_memory_gb * 0.4
    return enabled, round(budget, 1)


def _resolve_gpu_memory_utilization(profile: MemoryProfile) -> float:
    env_val = os.getenv("MINERU_GPU_MEMORY_UTILIZATION")
    if env_val is not None:
        try:
            return max(0.1, min(0.95, float(env_val)))
        except ValueError:
            logger.warning(f"Invalid MINERU_GPU_MEMORY_UTILIZATION: {env_val}")

    eff = profile.effective_memory_gb
    if profile.is_unified_memory:
        if eff <= 16:
            return 0.3
        elif eff <= 32:
            return 0.4
        else:
            return 0.5
    else:
        if eff <= 8:
            return 0.7
        else:
            return 0.5


_cached_config: MemoryOptimizationConfig | None = None


def get_memory_optimization_config() -> MemoryOptimizationConfig:
    global _cached_config
    if _cached_config is None:
        profile = get_memory_profile()
        quant = _resolve_quantization(profile)
        window = _resolve_window_size(profile)
        dpi = _resolve_dpi(profile)
        batch = _resolve_vlm_batch_size(profile)
        evict_enabled, evict_budget = _resolve_model_eviction(profile)
        gpu_util = _resolve_gpu_memory_utilization(profile)

        _cached_config = MemoryOptimizationConfig(
            quantization=quant,
            processing_window_size=window,
            pdf_image_dpi=dpi,
            vlm_batch_size=batch,
            model_eviction_enabled=evict_enabled,
            model_eviction_budget_gb=evict_budget,
            gpu_memory_utilization=gpu_util,
            profile=profile,
        )
        _cached_config.log_summary()

    return _cached_config


def should_use_adaptive_backend_selection() -> bool:
    env_val = os.getenv("MINERU_ADAPTIVE_BACKEND", "").strip().lower()
    if env_val in ("true", "false"):
        return env_val == "true"
    try:
        config = get_memory_optimization_config()
        return config.profile.is_low_memory
    except Exception:
        return False


def should_route_to_pipeline(pdf_bytes: bytes, current_backend: str) -> bool:
    if not should_use_adaptive_backend_selection():
        return False
    if current_backend == "pipeline":
        return False
    if not current_backend.startswith("hybrid-"):
        return False
    try:
        from mineru.utils.pdf_classify import classify
        result = classify(pdf_bytes)
        if result == "txt":
            logger.info("Adaptive backend: routing text PDF to pipeline backend (no VLM needed) to save memory")
            return True
    except Exception as e:
        logger.debug(f"Adaptive backend selection skipped: {e}")
    return False
