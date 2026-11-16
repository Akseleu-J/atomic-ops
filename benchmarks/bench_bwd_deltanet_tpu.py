"""Backward benchmark: atomic_gdn2 fused mega-backward vs deltanet-tpu
value_and_grad backward, same shapes, same TPU.

Run on a TPU host with both packages installed:
    pip install -e . "deltanet-tpu[tpu]"
    python benchmarks/bench_bwd_deltanet_tpu.py

Both implementations compute the full GDN-2 chunkwise backward for the same
inputs (q, k, v, w, b, g in their native layouts); timings are wall-clock
medians over timed iterations after warmup, synchronized with
block_until_ready. This script is the evidence artifact for the proposed
contribution of the fused single-launch backward (bwd/mega.py) upstream.
"""
import statistics
import time

import jax
import jax.numpy as jnp

SHAPES = [  # (B, L, Hk, Hv, K, V) — deltanet-tpu public shapes first
    (1, 2048, 12, 24, 128, 128),
    (2, 1024, 12, 24, 128, 128),
    (1, 4096, 12, 24, 128, 128),
]
BT = 128  # atomic_gdn2 chunk size; deltanet-tpu chunk_size=128 default


def _time(fn, warmup=2, iters=8):
    for _ in range(warmup):
        fn().block_until_ready()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_atomic(B, L, Hk, Hv, K, V):
    from atomic_gdn2.config import make_cfg
    from atomic_gdn2.pipeline import make_blr_trainable
    cfg = make_cfg(bt=BT)
    key = jax.random.PRNGKey(0)
    # atomic_gdn2 is head-major (B, H, L, D); Hk == Hv == H for the compare
    mk = lambda: jax.random.normal(key, (B, Hk, L, K), dtype=jnp.float32)
    q, k, v, w, b = mk(), mk(), mk(), mk(), mk()
    g = -jax.random.uniform(key, (B, Hk, L, K), minval=0.0, maxval=1.0)
    call = make_blr_trainable(cfg, 1.0)
    def loss():
        o, hf = call(q, k, v, w, b, g)
        return jnp.sum(o) + jnp.sum(hf)
    return _time(lambda: jax.value_and_grad(loss)())


def bench_deltanet_tpu(B, L, Hk, Hv, K, V):
    from deltanet_tpu import chunk_gdn2
    key = jax.random.PRNGKey(0)
    mk = lambda *shape: jax.random.normal(key, shape, dtype=jnp.float32)
    q, k = mk(B, L, Hk, K), mk(B, L, Hk, K)
    v, w = mk(B, L, Hv, V), jax.nn.sigmoid(mk(B, L, Hv, V))
    b = jax.nn.sigmoid(mk(B, L, Hk, K))
    g = -jax.random.uniform(key, (B, L, Hk, K), minval=0.0, maxval=1.0)
    def loss():
        o, _ = chunk_gdn2(q, k, v, g, b, w, chunk_size=BT, output_final_state=True)
        return jnp.sum(o)
    return _time(lambda: jax.value_and_grad(loss)())


def main():
    print(f"{'B,L,Hk,Hv,K,V':>28} {'atomic_gdn2 (ms)':>18} {'deltanet-tpu (ms)':>18}")
    for shape in SHAPES:
        row = [f"{shape}"]
        for bench in (bench_atomic, bench_deltanet_tpu):
            try:
                ms = bench(*shape) * 1e3
                row.append(f"{ms:18.1f}")
            except Exception as e:
                row.append(f"{'FAIL: ' + repr(e)[:40]:>18}")
        print(f"{row[0]:>28} {row[1]:>18} {row[2]:>18}", flush=True)


if __name__ == "__main__":
    main()
