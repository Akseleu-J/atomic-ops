from __future__ import annotations
import jax
from .configs import KernelConfig, KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE


def is_tpu_available() -> bool:
    try:
        dev = jax.devices("tpu")[0]
        return dev.platform == "tpu"
    except Exception:
        return False


def estimate_memory(batch: int, seq_len: int, heads: int, d_head: int = 128,
                    bt: int = 256, bytes_per_param: int = 4) -> float:
    """
    Грубая оценка HBM (MB) на один TPU chip.
    Полезно для Kaggle, где v5e-8 ~ 8GB HBM на чип.
    """
    n_chunks = seq_len // bt
    # Aqk + Akk + A + w_pseudo + u + kg + qg + gc_last + v_new + h_pre
    per_chunk = (bt * bt * 3) + (bt * d_head * 7) + (d_head * d_head * 2)
    per_head_mb = n_chunks * per_chunk * bytes_per_param / (1024 * 1024)
    total_mb = batch * heads * per_head_mb
    return total_mb


def get_recommended_config(batch: int, seq_len: int, heads: int, d_head: int = 128) -> KernelConfig:
    """Автоподбор пресета по размеру."""
    mem = estimate_memory(batch, seq_len, heads, d_head)
    if mem > 6000:
        return KAGGLE_LARGE
    if mem > 2500:
        return KAGGLE_MEDIUM
    return KAGGLE_SMALL
