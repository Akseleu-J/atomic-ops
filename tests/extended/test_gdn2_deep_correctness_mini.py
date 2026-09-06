from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from atomic_ops.configs import KernelConfig
from atomic_ops.gdn2_fwd import build_chunk_scores_pallas
from atomic_ops.gdn2_bwd import intra_backward_pallas, reverse_cumsum_bwd

_HIGHEST = jax.lax.Precision.HIGHEST
_FAILURES = []


def _check(name, rel_err, tol, extra=""):
    status = "PASS" if rel_err <= tol else "FAIL"
    print(f"[{status}] {name}: rel_err={rel_err:.3e}  (tol={tol:.1e}) {extra}")
    if rel_err > tol:
        _FAILURES.append(name)
    return rel_err <= tol


def _rel_err(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    num = jnp.max(jnp.abs(a - b))
    den = jnp.maximum(jnp.max(jnp.abs(b)), 1e-8)
    return float(num / den)


def _make_inputs(key, bsz, n_chunks, bt, H, D, decay_scale):
    L = n_chunks * bt
    k1, k2, k3, k4 = jax.random.split(key, 4)
    shape = (bsz, L, H, D)

    q = jax.random.normal(k1, shape)
    k = jax.random.normal(k2, shape)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    b = jax.random.uniform(k3, shape, minval=0.2, maxval=1.0)

    if decay_scale <= 0.0:
        g = jnp.zeros(shape, dtype=jnp.float32)
    else:
        g = -jnp.abs(jax.random.normal(k4, shape)) * decay_scale

    return q, k, b, g.astype(jnp.float32)


def test_b4_intra_backward_mini(cfg):
    print("\n--- D4a-mini: B4 (intra_backward_pallas), interpret=True ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 1
    key = jax.random.PRNGKey(85)
    q, k, b, g = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"], decay_scale=0.1)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config, interpret=True)

    rkey = jax.random.PRNGKey(86)
    r1, r2 = jax.random.split(rkey)
    dAqk = jax.random.normal(r1, Aqk.shape) * 0.1 * jnp.tril(jnp.ones((bt, bt)))[None, None, None]
    dAkk = jax.random.normal(r2, Akk.shape) * 0.1 * jnp.tril(jnp.ones((bt, bt)), k=-1)[None, None, None]

    dq_p, dk_p, db_p, _dgc_p = intra_backward_pallas(
        dAqk, dAkk, q, k, b, g, scale=1.0, config=config, interpret=True
    )

    def scores_fwd(q_, k_, b_, g_):
        gc = jnp.cumsum(g_.astype(jnp.float32), axis=1)
        gc_bhcd = jnp.moveaxis(gc, 2, 1)
        q_bhcd = jnp.moveaxis(q_, 2, 1).astype(jnp.float32)
        k_bhcd = jnp.moveaxis(k_, 2, 1).astype(jnp.float32)
        b_bhcd = jnp.moveaxis(b_, 2, 1).astype(jnp.float32)
        decay_diff = gc_bhcd[:, :, :, None, :] - gc_bhcd[:, :, None, :, :]
        edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))
        causal = jnp.tril(jnp.ones((bt, bt)))
        strict = jnp.tril(jnp.ones((bt, bt)), k=-1)
        Aqk_ = jnp.einsum("bhid,bhijd,bhjd->bhij", q_bhcd, edecay, k_bhcd, precision=_HIGHEST) * causal
        bk_bhcd = b_bhcd * k_bhcd
        Akk_ = jnp.einsum("bhid,bhijd,bhjd->bhij", bk_bhcd, edecay, k_bhcd, precision=_HIGHEST) * strict
        return Aqk_, Akk_

    dAqk_bh = dAqk[:, :, 0]
    dAkk_bh = dAkk[:, :, 0]
    _, vjp_fn = jax.vjp(scores_fwd, q, k, b, g)
    dq_m, dk_m, db_m, _dg_m = vjp_fn((dAqk_bh, dAkk_bh))

    dq_p_cmp = jnp.moveaxis(dq_p[:, :, 0], 1, 2)
    dk_p_cmp = jnp.moveaxis(dk_p[:, :, 0], 1, 2)
    db_p_cmp = jnp.moveaxis(db_p[:, :, 0], 1, 2)
    _check("mini.b4.dq", _rel_err(dq_p_cmp, dq_m), 2e-2)
    _check("mini.b4.dk", _rel_err(dk_p_cmp, dk_m), 2e-2)
    _check("mini.b4.db", _rel_err(db_p_cmp, db_m), 2e-2)


def test_b5_reverse_cumsum_mini(cfg):
    print("\n--- D4b-mini: B5 (reverse_cumsum_bwd) ---")
    bt = cfg["bt"]
    key = jax.random.PRNGKey(87)
    dgc = jax.random.normal(key, (1, 1, 1, bt, cfg["D"])) * 0.5

    dg_p = reverse_cumsum_bwd(dgc, chunk_size=bt)
    dg_manual = jnp.flip(jnp.cumsum(jnp.flip(dgc, axis=-2), axis=-2), axis=-2)

    _check("mini.b5.reverse_cumsum", _rel_err(dg_p, dg_manual), 1e-3)


RUN_CONFIG = dict(
    bsz=1,
    H=1,
    D=128,
    bt=32,
)


def main(cfg=RUN_CONFIG):
    print("Running CPU-fast mini smoke suite (atomic_ops, interpret=True where needed).")
    print("This does NOT replace the full TPU suite -- see TESTING_STRATEGY.md.")
    test_b4_intra_backward_mini(cfg)
    test_b5_reverse_cumsum_mini(cfg)

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s):")
        for name in _FAILURES:
            print(f"  - {name}")
        print("=" * 70)
        sys.exit(1)
    else:
        print("OK: all mini checks passed (CPU).")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
