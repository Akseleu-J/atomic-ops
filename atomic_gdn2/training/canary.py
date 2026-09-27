"""Runtime canary: compare BTL scores with exact (lean) scores on a sample of chunks."""
import jax.numpy as jnp, numpy as np
from ..fwd import build_scores
from ..domain import half_span_max


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def btl_vs_lean_canary(cfg, q=None, k=None, b=None, g=None, D=64, g_level=None, thresh=1e-3, seed=0):
    """q,k,b,g: (B,H,nc,bt,D) chunks (sample). If omitted, synthetic with constant |g|=g_level."""
    if q is None:
        rng = np.random.default_rng(seed); sh = (1, 1, 1, cfg.bt, D)
        q, k = (jnp.asarray(rng.normal(size=sh).astype(np.float32) * 0.3) for _ in range(2))
        b = jnp.asarray(np.abs(rng.normal(size=sh)).astype(np.float32) * 0.5)
        g = -jnp.full(sh, g_level, jnp.float32)
    gc = jnp.cumsum(g, axis=-2)
    _, Ab = build_scores(q, k, b, gc, 1.0 / np.sqrt(q.shape[-1]), cfg.with_(diag="btl"))
    _, Al = build_scores(q, k, b, gc, 1.0 / np.sqrt(q.shape[-1]), cfg.with_(diag="lean"))
    err = _rel(Ab, Al)
    return dict(rel_err=err, half_span=float(half_span_max(gc, cfg.bs2)), alarm=err > thresh)
