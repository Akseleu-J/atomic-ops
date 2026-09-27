"""Domain of correctness + guards.

I-1: g <= 0.   I-2: beta*||k||^2 <= 2.   I-3 (BTL): half_span < 88 where
half_span = max over inner blocks/channels of 0.5*(gc[s0]-gc[s1-1]).
Violating I-3 silently gives 0*inf -> NaN -> 0 (bug T-22), so it is checked
by MAX (not mean) and enforced by clamping g in the layer.
"""
from __future__ import annotations
import math
import jax, jax.numpy as jnp
from .config import BTL_STOP, BTL_WARN, BTL_HARD_LIMIT


def half_span_max(gc_chunks, bs2: int):
    """gc_chunks: (B,H,nc,bt,D) cumsum inside chunk. Returns scalar max half-span."""
    lead = gc_chunks.shape[:-2]
    bt, D = gc_chunks.shape[-2:]
    blk = gc_chunks.reshape(lead + (bt // bs2, bs2, D))
    span = blk[..., 0, :] - blk[..., -1, :]
    return 0.5 * jnp.max(span)


def half_span_from_g(g_hm, bt: int, bs2: int):
    """g_hm: (B,H,L,D) -> max half span."""
    b, h, L, d = g_hm.shape
    gc = jnp.cumsum(g_hm.astype(jnp.float32).reshape(b, h, L // bt, bt, d), axis=-2)
    return half_span_max(gc, bs2)


def bk2_stat(k, b):
    """max/mean of sum_d b*k^2 (I-2)."""
    x = jnp.sum(b * k * k, axis=-1)
    return jnp.max(x), jnp.mean(x)


def clamp_g(g_neg, g_max: float):
    """g_neg <= 0; enforce |g| <= g_max (smooth-enough: hard clamp, keeps grad inside)."""
    return jnp.maximum(g_neg, -g_max)


def init_g_bias(target: float = 0.05):
    """bias so that softplus(bias) = target (slow forgetting at init)."""
    return math.log(math.expm1(target))


def check_preconditions(g_hm, cfg, strict=True):
    hs = float(half_span_from_g(g_hm, cfg.bt, cfg.bs2)) if cfg.diag == "btl" else 0.0
    status = "ok"
    if hs > BTL_STOP:
        status = "STOP"
    elif hs > BTL_WARN:
        status = "warn"
    if strict and status == "STOP":
        raise RuntimeError(f"BTL half_span={hs:.1f} > {BTL_STOP} (limit {BTL_HARD_LIMIT}); reduce bs2 or clamp g")
    return {"half_span_max": hs, "status": status}


def safe_bs2_for(g_max: float, candidates=(128, 64, 32, 16, 8)):
    for c in candidates:
        if g_max * c / 2.0 < BTL_STOP:
            return c
    return candidates[-1]
