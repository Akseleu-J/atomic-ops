from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import pytest

from atomic_ops.configs import KernelConfig, KAGGLE_SMALL, KAGGLE_MEDIUM, DEFAULT_CONFIG
from atomic_ops.reference import (
    gdn2_token_serial_reference,
    gdn2_chunked_wy_reference,
    _wy_inverse,
)
from atomic_ops.gdn2_fwd import build_chunk_scores_pallas
from atomic_ops.gdn2_bwd import intra_backward_pallas

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


def _make_inputs(key, bsz, n_chunks, bt, H, D, decay_scale, h0_nonzero=False):
    L = n_chunks * bt
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    shape = (bsz, L, H, D)

    q = jax.random.normal(k1, shape)
    k = jax.random.normal(k2, shape)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(k3, shape) * 0.5
    w = jax.random.uniform(k4, shape, minval=0.2, maxval=1.0)
    b = jax.random.uniform(jax.random.fold_in(k4, 1), shape, minval=0.2, maxval=1.0)

    if decay_scale <= 0.0:
        g = jnp.zeros(shape, dtype=jnp.float32)
    else:
        g = -jnp.abs(jax.random.normal(k5, shape)) * decay_scale

    h0 = None
    if h0_nonzero:
        h0 = jax.random.normal(jax.random.fold_in(key, 99), (bsz, H, D, D)) * 0.1

    return q, k, v, w, b, g, h0


RUN_CONFIG = dict(bsz=2, H=2, D=128, bt=256, n_seeds=5)


# ==========================================================================
# Section 0: config / shape sanity (Layer 0)
# ==========================================================================
def test_config_invariants(cfg):
    print("\n--- Section 0: KernelConfig invariants ---")
    for name, kc in (("SMALL", KAGGLE_SMALL), ("MEDIUM", KAGGLE_MEDIUM)):
        ok = (kc.bt % kc.bc == 0) and (kc.bt == 2 * kc.bc) and (kc.bc % kc.mb == 0)
        _check(f"config.{name}.invariants", 0.0 if ok else 1.0, 0.5)
    ok_default = DEFAULT_CONFIG == KAGGLE_MEDIUM
    _check("config.default_is_medium", 0.0 if ok_default else 1.0, 0.5)


# ==========================================================================
# Section 1: reference smoke test (Layer 1) -- shapes + finiteness only
# ==========================================================================
def test_reference_smoke(cfg):
    print("\n--- Section 1: reference smoke test ---")
    bt = cfg["bt"]
    key = jax.random.PRNGKey(1)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], 2, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True)
    o, h_final = gdn2_chunked_wy_reference(q, k, v, g, b, w, scale=1.0, chunk_size=bt, h0=h0)
    shape_ok = (o.shape == q.shape) and (h_final.shape == (cfg["bsz"], cfg["H"], cfg["D"], cfg["D"]))
    finite_ok = bool(jnp.all(jnp.isfinite(o))) and bool(jnp.all(jnp.isfinite(h_final)))
    _check("reference.shapes", 0.0 if shape_ok else 1.0, 0.5)
    _check("reference.finite", 0.0 if finite_ok else 1.0, 0.5)


# ==========================================================================
# Section 2: chunked-WY vs token-serial (Layer 2/3) -- the core math check.
# ==========================================================================
def test_chunked_wy_vs_token_serial(cfg):
    print("\n--- Section 2: chunked-WY vs token-serial (multi-seed) ---")
    bt = cfg["bt"]
    n_chunks = 2
    for seed in range(cfg["n_seeds"]):
        key = jax.random.PRNGKey(1000 + seed)
        q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                             decay_scale=0.15, h0_nonzero=True)

        o_wy, h_wy = gdn2_chunked_wy_reference(q, k, v, g, b, w, scale=1.0, chunk_size=bt, h0=h0)
        o_ts, h_ts = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)

        _check(f"wy_vs_serial[seed={seed}].o", _rel_err(o_wy, o_ts), 2e-2)
        _check(f"wy_vs_serial[seed={seed}].h_final", _rel_err(h_wy, h_ts), 2e-2)


def test_chunked_wy_vs_token_serial_zero_decay(cfg):
    """decay_scale=0 -> g=0 everywhere, i.e. plain (undecayed) DeltaNet.
    Degenerate case worth isolating: exp(g)=1 identically, so any bug
    that only manifests through the decay/WY interaction (as opposed to
    the pure delta-rule update) would NOT show up here -- useful for
    bisecting where a Section 2 failure actually lives."""
    print("\n--- Section 2b: chunked-WY vs token-serial (zero decay) ---")
    bt = cfg["bt"]
    key = jax.random.PRNGKey(2024)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], 2, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.0, h0_nonzero=True)
    o_wy, h_wy = gdn2_chunked_wy_reference(q, k, v, g, b, w, scale=1.0, chunk_size=bt, h0=h0)
    o_ts, h_ts = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
    _check("wy_vs_serial.zero_decay.o", _rel_err(o_wy, o_ts), 1e-2)
    _check("wy_vs_serial.zero_decay.h_final", _rel_err(h_wy, h_ts), 1e-2)


def test_chunked_wy_single_chunk_equals_full_seq(cfg):
    """Sanity invariant: with n_chunks=1, the chunked recurrence has no
    inter-chunk carry to get wrong -- any residual mismatch vs
    token-serial here isolates to the intra-chunk WY-inverse math
    itself, not the inter-chunk scan (Kernel D)."""
    print("\n--- Section 2c: single-chunk (isolates intra-chunk WY math) ---")
    bt = cfg["bt"]
    key = jax.random.PRNGKey(55)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], 1, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True)
    o_wy, h_wy = gdn2_chunked_wy_reference(q, k, v, g, b, w, scale=1.0, chunk_size=bt, h0=h0)
    o_ts, h_ts = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
    _check("wy_vs_serial.single_chunk.o", _rel_err(o_wy, o_ts), 2e-2)
    _check("wy_vs_serial.single_chunk.h_final", _rel_err(h_wy, h_ts), 2e-2)


# ==========================================================================
# Section 3: wy_eps damping -- reference-level check (no Pallas needed).
# ==========================================================================
def test_wy_inverse_damping_solves_damped_system(cfg):
    print("\n--- Section 3: _wy_inverse damping solves the damped system ---")
    key = jax.random.PRNGKey(77)
    C = 32
    Akk_raw = jax.random.normal(key, (2, 3, C, C)) * 0.3
    Akk = Akk_raw * jnp.tril(jnp.ones((C, C)), k=-1)[None, None]

    for eps in (0.0, 1e-3, 1e-2):
        A = _wy_inverse(Akk, eps=eps)
        eye = jnp.eye(C)[None, None]
        residual = eye + (1.0 - eps) * jnp.einsum("...ij,...jk->...ik", Akk, A, precision=_HIGHEST) - eye
        # (I + (1-eps)*Akk) @ A should equal I  =>  Akk_damped @ A + A - I ~ 0
        lhs = jnp.einsum("...ij,...jk->...ik", (1.0 - eps) * Akk, A, precision=_HIGHEST) + A
        err = _rel_err(lhs, eye)
        _check(f"wy_inverse.damped_system_solved[eps={eps}]", err, 5e-3)


# ==========================================================================
# Section 4: Kernel A (build_chunk_scores_pallas, interpret=True) --
# ==========================================================================
def _manual_scores(q_c, k_c, b_c, g_c, scale):
    C = q_c.shape[1]
    gc = jnp.cumsum(g_c.astype(jnp.float32), axis=1)
    gc_bhcd = jnp.moveaxis(gc, 2, 1)
    q_bhcd = jnp.moveaxis(q_c, 2, 1).astype(jnp.float32)
    k_bhcd = jnp.moveaxis(k_c, 2, 1).astype(jnp.float32)
    b_bhcd = jnp.moveaxis(b_c, 2, 1).astype(jnp.float32)
    decay_diff = gc_bhcd[:, :, :, None, :] - gc_bhcd[:, :, None, :, :]
    edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))
    causal = jnp.tril(jnp.ones((C, C)))
    strict = jnp.tril(jnp.ones((C, C)), k=-1)
    Aqk = scale * jnp.einsum("bhid,bhijd,bhjd->bhij", q_bhcd, edecay, k_bhcd, precision=_HIGHEST) * causal
    bk_bhcd = b_bhcd * k_bhcd
    Akk = jnp.einsum("bhid,bhijd,bhjd->bhij", bk_bhcd, edecay, k_bhcd, precision=_HIGHEST) * strict
    return Aqk, Akk


def test_kernel_a_scores_cpu_interpret(cfg):
    print("\n--- Section 4: Kernel A (build_chunk_scores_pallas, interpret=True) ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    key = jax.random.PRNGKey(321)
    q, k, v, w, b, g, _ = _make_inputs(key, cfg["bsz"], 1, bt, cfg["H"], cfg["D"], decay_scale=0.1)

    Aqk_p, Akk_p = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config, interpret=True)
    Aqk_m, Akk_m = _manual_scores(q, k, b, g, scale=1.0)

    _check("kernel_a.Aqk_cpu_interpret", _rel_err(Aqk_p, Aqk_m), 1e-3)
    _check("kernel_a.Akk_cpu_interpret", _rel_err(Akk_p, Akk_m), 1e-3)

    upper_akk = jnp.triu(Akk_p[0, 0, 0], k=0)
    upper_aqk_strict = jnp.triu(Aqk_p[0, 0, 0], k=1)
    _check("kernel_a.Akk_strictly_lower_triangular", float(jnp.max(jnp.abs(upper_akk))), 1e-6)
    _check("kernel_a.Aqk_causal_incl_diag", float(jnp.max(jnp.abs(upper_aqk_strict))), 1e-6)


# ==========================================================================
# Section 5: Kernel B4 (intra_backward_pallas, interpret=True) -- the
# ==========================================================================
def test_kernel_b4_intra_backward_cpu_interpret(cfg):
    print("\n--- Section 5: Kernel B4 (intra_backward_pallas, interpret=True) ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    key = jax.random.PRNGKey(85)
    q, k, v, w, b, g, _ = _make_inputs(key, cfg["bsz"], 1, bt, cfg["H"], cfg["D"], decay_scale=0.1)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config, interpret=True)

    rkey = jax.random.PRNGKey(86)
    r1, r2 = jax.random.split(rkey)
    dAqk = jax.random.normal(r1, Aqk.shape) * 0.1 * jnp.tril(jnp.ones((bt, bt)))[None, None, None]
    dAkk = jax.random.normal(r2, Akk.shape) * 0.1 * jnp.tril(jnp.ones((bt, bt)), k=-1)[None, None, None]

    dq_p, dk_p, db_p, dgc_p = intra_backward_pallas(
        dAqk, dAkk, q, k, b, g, scale=1.0, config=config, interpret=True
    )

    def scores_fwd(q_, k_, b_, g_):
        return _manual_scores(q_, k_, b_, g_, scale=1.0)

    dAqk_bh = dAqk[:, :, 0]
    dAkk_bh = dAkk[:, :, 0]
    _, vjp_fn = jax.vjp(scores_fwd, q, k, b, g)
    dq_m, dk_m, db_m, dg_m = vjp_fn((dAqk_bh, dAkk_bh))

    dq_p_cmp = jnp.moveaxis(dq_p[:, :, 0], 1, 2)  # (bsz,H,bt,D) -> (bsz,bt,H,D)
    dk_p_cmp = jnp.moveaxis(dk_p[:, :, 0], 1, 2)
    db_p_cmp = jnp.moveaxis(db_p[:, :, 0], 1, 2)

    _check("kernel_b4.dq_cpu_interpret", _rel_err(dq_p_cmp, dq_m), 2e-2)
    _check("kernel_b4.dk_cpu_interpret", _rel_err(dk_p_cmp, dk_m), 2e-2)
    _check("kernel_b4.db_cpu_interpret", _rel_err(db_p_cmp, db_m), 2e-2)


# ==========================================================================
# Section 6: bf16 input coverage at the reference level (no Pallas) --
# ==========================================================================
def test_reference_bf16_inputs_finite(cfg):
    print("\n--- Section 6: bf16 inputs stay finite through chunked-WY reference ---")
    bt = cfg["bt"]
    key = jax.random.PRNGKey(111)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], 2, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True)
    q, k, v, w, b = (t.astype(jnp.bfloat16) for t in (q, k, v, w, b))
    o, h_final = gdn2_chunked_wy_reference(q, k, v, g, b, w, scale=1.0, chunk_size=bt, h0=h0)
    finite_ok = bool(jnp.all(jnp.isfinite(o))) and bool(jnp.all(jnp.isfinite(h_final)))
    _check("reference.bf16.finite", 0.0 if finite_ok else 1.0, 0.5)


# ==========================================================================
# pytest entry points
# ==========================================================================
@pytest.fixture
def cfg():
    return RUN_CONFIG


def test_all_sections(cfg):
    _FAILURES.clear()
    test_config_invariants(cfg)
    test_reference_smoke(cfg)
    test_chunked_wy_vs_token_serial(cfg)
    test_chunked_wy_vs_token_serial_zero_decay(cfg)
    test_chunked_wy_single_chunk_equals_full_seq(cfg)
    test_wy_inverse_damping_solves_damped_system(cfg)
    test_kernel_a_scores_cpu_interpret(cfg)
    test_kernel_b4_intra_backward_cpu_interpret(cfg)
    test_reference_bf16_inputs_finite(cfg)
    assert not _FAILURES, f"{len(_FAILURES)} failure(s): {_FAILURES}"


# ==========================================================================
# Script entrypoint (python test_gdn2_full_math_correctness.py)
# ==========================================================================
def main(cfg=RUN_CONFIG):
    _FAILURES.clear()
    test_config_invariants(cfg)
    test_reference_smoke(cfg)
    test_chunked_wy_vs_token_serial(cfg)
    test_chunked_wy_vs_token_serial_zero_decay(cfg)
    test_chunked_wy_single_chunk_equals_full_seq(cfg)
    test_wy_inverse_damping_solves_damped_system(cfg)
    test_kernel_a_scores_cpu_interpret(cfg)
    test_kernel_b4_intra_backward_cpu_interpret(cfg)
    test_reference_bf16_inputs_finite(cfg)

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"РЕЗУЛЬТАТ: {len(_FAILURES)} провал(ов):")
        for name in _FAILURES:
            print(f"  - {name}")
        print("=" * 78)
        sys.exit(1)
    else:
        print("РЕЗУЛЬТАТ: ВСЕ проверки прошли (CPU, без TPU).")
        print("=" * 78)
        sys.exit(0)


if __name__ == "__main__":
    main()
