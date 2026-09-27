"""T8: speed/memory regression. Gate: total <= 11.02 ms * 1.05 at bt=128,bs2=32 (bsz=8,H=6,L=4096,D=128)."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp, numpy as np
from atomic_gdn2 import make_cfg, make_blr_trainable

def med(f, *a, n=12, w=3):
    jf = jax.jit(f)
    for _ in range(w): jax.block_until_ready(jf(*a))
    ts = []
    for _ in range(n):
        t = time.perf_counter(); jax.block_until_ready(jf(*a)); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))

def run(L=4096, bsz=8, H=6, D=128, **kw):
    cfg = make_cfg(**kw); fn = make_blr_trainable(cfg, 1 / math.sqrt(D))
    r = np.random.default_rng(0); mk = lambda s: jnp.asarray(r.normal(size=(bsz, H, L, D)).astype(np.float32) * s)
    q, k, v = mk(.1), mk(.1), mk(.1); w = jnp.ones_like(q); b = jnp.abs(mk(.5)); g = -jnp.abs(mk(.05))
    fwd = med(lambda *a: fn(*a), q, k, v, w, b, g)
    loss = lambda *a: (lambda o, h: jnp.sum(o * o) + jnp.sum(h * h))(*fn(*a))
    tot = med(lambda *a: jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(*a), q, k, v, w, b, g)
    print(f"L={L} {kw}: fwd={fwd:.2f} total={tot:.2f} ms  ({tot*1e3/(bsz*L):.3f} us/tok)")
    return tot

if __name__ == "__main__":
    t = run(bs2=32)
    print("PASS" if t <= 11.02 * 1.05 else "FAIL: regression")
    run(L=8192, bs2=32)      # segmentation: must not OOM
