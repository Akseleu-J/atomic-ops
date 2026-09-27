#!/usr/bin/env python3
"""Reproduce the kernel fwd+bwd benchmark. Usage:
    python scripts/reproduce_kernel_bench.py --shape 4,2048,6,128
"""
import argparse, json, math, time
import numpy as np

def parse_shape(s):
    B, L, H, D = map(int, s.split(','))
    return B, L, H, D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', default='4,2048,6,128')
    ap.add_argument('--n-iters', type=int, default=5)
    args = ap.parse_args()
    import jax, jax.numpy as jnp
    from atomic_gdn2 import make_cfg, make_blr_trainable
    B, L, H, D = parse_shape(args.shape)
    cfg = make_cfg(bt=128, bc=64, score_bs=128, bs2=64, mb=8)
    try: cfg = cfg.with_(b3_dot_mode='bf16x3', b4_dot_mode='bf16x3', diag='btl')
    except Exception: pass
    fn = make_blr_trainable(cfg, 1.0 / math.sqrt(D))
    rng = np.random.default_rng(0)
    sh = (B, H, L, D)
    q = jnp.asarray((rng.normal(size=sh)*0.1).astype(np.float32))
    k = jnp.asarray((rng.normal(size=sh)*0.1).astype(np.float32))
    v = jnp.asarray((rng.normal(size=sh)*0.1).astype(np.float32))
    w = jnp.ones(sh, jnp.float32)
    b = jnp.asarray((np.abs(rng.normal(size=sh))*0.5).astype(np.float32))
    g = jnp.asarray((-np.abs(rng.normal(size=sh))*0.05).astype(np.float32))
    h0 = jnp.zeros((B, H, D, D), jnp.float32)
    def loss(*a):
        o, hf = fn(*a, h0)
        return jnp.sum(o.astype(jnp.float32)**2) + jnp.sum(hf.astype(jnp.float32)**2)
    gf = jax.jit(jax.grad(loss, argnums=(0,1,2,3,4,5)))
    args_t = (q, k, v, w, b, g)
    for _ in range(2): jax.block_until_ready(gf(*args_t))
    ts = []
    for _ in range(args.n_iters):
        t0 = time.perf_counter()
        jax.block_until_ready(gf(*args_t))
        ts.append(time.perf_counter() - t0)
    out = {'shape': [B, L, H, D], 'jax': jax.__version__,
           'backend': jax.default_backend(),
           'ms_fwd_bwd_median': float(np.median(ts))*1e3}
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
