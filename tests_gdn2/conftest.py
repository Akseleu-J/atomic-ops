import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax.numpy as jnp
import pytest

SEEDS = [  # name, seed, g_scale, k_scale, b_scale, in_domain
    ("benign0", 0, 0.05, 1.0, 0.5, True), ("benign1", 1, 0.05, 1.0, 0.5, True),
    ("strong_decay", 2, 3.0, 1.0, 0.5, True), ("near_zero", 3, 0.001, 1.0, 0.5, True),
    ("b_near_one", 5, 0.3, 1.0, 0.95, True), ("large_k_stable", 4, 0.3, 8.0, 0.02, True),
    ("large_kb_OOD", 4, 0.3, 8.0, 4.0, False)]
DOMAIN = [s for s in SEEDS if s[5]]


def mk_inputs(bsz, L, H, D, seed, g=0.05, k=1.0, b=0.5, layout="blhd"):
    rng = np.random.default_rng(seed); sh = (bsz, L, H, D)
    q = rng.normal(size=sh) * 0.1; kk = rng.normal(size=sh) * 0.1 * k; v = rng.normal(size=sh) * 0.1
    w = np.ones(sh); bb = b * rng.uniform(0.5, 1.0, size=sh); gg = -np.abs(rng.normal(size=sh) * g)
    out = tuple(jnp.asarray(x.astype(np.float32)) for x in (q, kk, v, w, bb, gg))
    if layout == "bhld":
        out = tuple(jnp.swapaxes(t, 1, 2) for t in out)
    return out


def rel_l2(a, b, floor=1e-8):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n, d = np.linalg.norm((a - b).ravel()), np.linalg.norm(b.ravel())
    return float(n if d < floor else n / d)   # absolute error if reference underflows


@pytest.fixture
def small_cfg():
    from atomic_gdn2 import make_cfg
    return make_cfg(bt=64, bc=32, score_bs=64, bs2=32, mb=8, max_chunks_per_call=32)
