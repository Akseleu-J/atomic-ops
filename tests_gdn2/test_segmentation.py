"""Long sequences are cut into segments with h carried; result must equal the single-call result."""
import jax, jax.numpy as jnp
from atomic_gdn2 import make_cfg, make_blr_trainable
from conftest import mk_inputs

def test_segmented_equals_full():
    cfg = make_cfg(bt=64, bc=32, score_bs=64, bs2=32, mb=8)
    a = mk_inputs(1, 256, 1, 64, 0, layout="bhld")
    o1, h1 = make_blr_trainable(cfg, .125)(*a)
    o2, h2 = make_blr_trainable(cfg.with_(max_chunks_per_call=1), .125)(*a)
    assert float(jnp.max(jnp.abs(o1 - o2))) < 1e-5 and float(jnp.max(jnp.abs(h1 - h2))) < 1e-5
    f = lambda c: jax.grad(lambda *x: jnp.sum(make_blr_trainable(c, .125)(*x)[0] ** 2), argnums=(0, 5))(*a)
    g1, g2 = f(cfg), f(cfg.with_(max_chunks_per_call=1))
    assert all(float(jnp.max(jnp.abs(x - y))) < 1e-4 for x, y in zip(g1, g2))
