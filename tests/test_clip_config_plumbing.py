from __future__ import annotations
import dataclasses as dc
import jax
import jax.numpy as jnp
import pytest

from atomic_ops.configs import KAGGLE_LARGE, KernelConfig
from atomic_ops.gdn2_fwd import (
    build_chunk_scores_pallas,
    wy_solve_pallas,
    recompute_wy_pallas,
    gdn2_inter_chunk_combine,
    gdn2_inter_chunk_combine_with_state,
)
from atomic_ops.gdn2_bwd import (
    dav_backward_pallas,
    wy_dqkg_backward_pallas,
    intra_backward_pallas,
    gdn2_dhu_backward,
    reverse_cumsum_bwd,
)

CFG_TIGHT = dc.replace(KAGGLE_LARGE, clip=5e3)
CFG_LOOSE = dc.replace(KAGGLE_LARGE, clip=1e4)
BT = CFG_TIGHT.bt
BSZ, H, D = 1, 1, 128
TOL = 1.0  # fp roundoff slack on top of the clip bound


def _bounds_check(name, arr_tight, arr_loose, cfg_tight=CFG_TIGHT, cfg_loose=CFG_LOOSE, expect_diverge=True):
    mt = float(jnp.max(jnp.abs(arr_tight)))
    ml = float(jnp.max(jnp.abs(arr_loose)))
    assert mt <= cfg_tight.clip + TOL, (
        f"[{name}] tight config clip={cfg_tight.clip} NOT honored: max|x|={mt:.3e}"
    )
    assert ml <= cfg_loose.clip + TOL, (
        f"[{name}] loose config clip={cfg_loose.clip} NOT honored: max|x|={ml:.3e}"
    )
    if expect_diverge:
        assert abs(mt - ml) > TOL, (
            f"[{name}] tight/loose outputs identical (mt={mt:.3e}, ml={ml:.3e}) "
            f"despite raw values exceeding both clips -- config.clip is being "
            f"ignored at this stage (hardcoded default clip is being used instead)."
        )
    return mt, ml


def _huge_inputs(key, scale=30.0, bval=50.0, zero_decay=True):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    shape = (BSZ, BT, H, D)
    q = jax.random.normal(k1, shape) * scale
    k = jax.random.normal(k2, shape) * scale
    v = jax.random.normal(k3, shape) * scale
    w = jax.random.uniform(k4, shape, minval=0.5, maxval=1.5) * scale
    b = jnp.ones(shape) * bval
    g = jnp.zeros(shape) if zero_decay else -jnp.abs(jax.random.normal(k5, shape)) * 0.01
    return q, k, v, w, b, g


# ---------------- Kernel A ----------------
def test_kernel_a_aqk_akk_clip():
    q, k, v, w, b, g = _huge_inputs(jax.random.PRNGKey(1))
    _, akk_t = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=CFG_TIGHT)
    aqk_t, _ = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=CFG_TIGHT)
    aqk_l, akk_l = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=CFG_LOOSE)
    _bounds_check("kernel_A/Aqk", aqk_t, aqk_l)
    _bounds_check("kernel_A/Akk", akk_t, akk_l)


# ---------------- Kernel B (WY solve) ----------------
def test_kernel_b_wy_solve_clip():
    # Feed an already-huge Akk directly (bypassing Kernel A) to stress
    # _block_solve / _kernel_b_body's internal sanitize calls specifically.
    key = jax.random.PRNGKey(2)
    n_chunks = 1
    raw = jax.random.normal(key, (BSZ, H, n_chunks, BT, BT)) * 1e5
    idx = jnp.arange(BT)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    akk_huge = raw * strict[None, None, None]

    a_t = wy_solve_pallas(akk_huge, config=CFG_TIGHT)
    a_l = wy_solve_pallas(akk_huge, config=CFG_LOOSE)
    _bounds_check("kernel_B/A", a_t, a_l)


# ---------------- Kernel C ----------------
def test_kernel_c_recompute_wy_clip():
    key = jax.random.PRNGKey(3)
    q, k, v, w, b, g = _huge_inputs(key, scale=3000.0, bval=50.0)
    n_chunks = 1
    # Use an extreme A (post-WY-solve) to blow up w_pseudo/u as well.
    a_key = jax.random.PRNGKey(4)
    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    A = jax.random.normal(a_key, (BSZ, H, n_chunks, BT, BT)) * 1e4 * causal[None, None, None]

    def r2c(t):
        return t.reshape(BSZ, n_chunks, BT, H, D).transpose(0, 3, 1, 2, 4)

    w_pseudo_t, u_t, kg_t, qg_t, _ = recompute_wy_pallas(
        r2c(q).reshape(BSZ, BT, H, D), r2c(k).reshape(BSZ, BT, H, D),
        r2c(v).reshape(BSZ, BT, H, D), r2c(w).reshape(BSZ, BT, H, D),
        r2c(b).reshape(BSZ, BT, H, D), r2c(g).reshape(BSZ, BT, H, D),
        A, config=CFG_TIGHT,
    )
    w_pseudo_l, u_l, kg_l, qg_l, _ = recompute_wy_pallas(
        r2c(q).reshape(BSZ, BT, H, D), r2c(k).reshape(BSZ, BT, H, D),
        r2c(v).reshape(BSZ, BT, H, D), r2c(w).reshape(BSZ, BT, H, D),
        r2c(b).reshape(BSZ, BT, H, D), r2c(g).reshape(BSZ, BT, H, D),
        A, config=CFG_LOOSE,
    )
    _bounds_check("kernel_C/w_pseudo", w_pseudo_t, w_pseudo_l)
    _bounds_check("kernel_C/u", u_t, u_l)
    _bounds_check("kernel_C/kg", kg_t, kg_l)
    _bounds_check("kernel_C/qg", qg_t, qg_l)


# ---------------- Kernel D ----------------
def test_kernel_d_inter_chunk_combine_clip():
    key = jax.random.PRNGKey(5)
    n_chunks = 2
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    Aqk = jax.random.normal(k1, (BSZ, H, n_chunks, BT, BT)) * 1e3
    w_pseudo = jax.random.normal(k2, (BSZ, H, n_chunks, BT, D)) * 1e3
    u = jax.random.normal(k3, (BSZ, H, n_chunks, BT, D)) * 1e3
    kg = jax.random.normal(k4, (BSZ, H, n_chunks, BT, D)) * 1e3
    qg = jax.random.normal(k5, (BSZ, H, n_chunks, BT, D)) * 1e3
    gc_last = jnp.zeros((BSZ, H, n_chunks, D))  # exp(0)=1, no extra blowup from decay
    h0 = jax.random.normal(jax.random.PRNGKey(6), (BSZ, H, D, D)) * 1e3

    o_t, h_t = gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale=1.0, h0=h0, config=CFG_TIGHT)
    o_l, h_l = gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale=1.0, h0=h0, config=CFG_LOOSE)
    _bounds_check("kernel_D/o", o_t, o_l)
    _bounds_check("kernel_D/h_final", h_t, h_l)


# ---------------- B1 ----------------
def test_b1_dhu_backward_clip():
    key = jax.random.PRNGKey(7)
    n_chunks = 2
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    do = jax.random.normal(k1, (BSZ, H, n_chunks, BT, D)) * 1e3
    dv_partial = jax.random.normal(k2, (BSZ, H, n_chunks, BT, D)) * 1e3
    w_pseudo = jax.random.normal(k3, (BSZ, H, n_chunks, BT, D)) * 1e3
    qg = jax.random.normal(k4, (BSZ, H, n_chunks, BT, D)) * 1e3
    kg = jax.random.normal(k5, (BSZ, H, n_chunks, BT, D)) * 1e3
    gc_last = jnp.zeros((BSZ, H, n_chunks, D))
    dht = jax.random.normal(k6, (BSZ, H, D, D)) * 1e3

    dh_all_t, dh0_t, dv_all_t = gdn2_dhu_backward(do, dv_partial, w_pseudo, qg, kg, gc_last, scale=1.0, dht=dht, config=CFG_TIGHT)
    dh_all_l, dh0_l, dv_all_l = gdn2_dhu_backward(do, dv_partial, w_pseudo, qg, kg, gc_last, scale=1.0, dht=dht, config=CFG_LOOSE)
    _bounds_check("B1/dh_all", dh_all_t, dh_all_l)
    _bounds_check("B1/dh0", dh0_t, dh0_l)
    _bounds_check("B1/dv_all", dv_all_t, dv_all_l)


# ---------------- B2 ----------------
def test_b2_dav_backward_clip():
    key = jax.random.PRNGKey(8)
    n_chunks = 1
    k1, k2, k3 = jax.random.split(key, 3)
    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    Aqk = jax.random.normal(k1, (BSZ, H, n_chunks, BT, BT)) * 1e4 * causal[None, None, None]
    v_new = jax.random.normal(k2, (BSZ, H, n_chunks, BT, D)) * 1e4
    do = jax.random.normal(k3, (BSZ, H, n_chunks, BT, D)) * 1e4

    dAqk_t, dv_t = dav_backward_pallas(Aqk, v_new, do, config=CFG_TIGHT)
    dAqk_l, dv_l = dav_backward_pallas(Aqk, v_new, do, config=CFG_LOOSE)
    _bounds_check("B2/dAqk", dAqk_t, dAqk_l)
    _bounds_check("B2/dv_new", dv_t, dv_l)


# ---------------- B3 ----------------
def test_b3_wy_dqkg_backward_clip():
    key = jax.random.PRNGKey(9)
    n_chunks = 1
    ks = jax.random.split(key, 12)
    q, k, b, w, v = (jax.random.normal(ks[i], (BSZ, H, n_chunks, BT, D)) * 30 for i in range(5))
    gc = jnp.zeros((BSZ, H, n_chunks, BT, D))
    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    A = jax.random.normal(ks[5], (BSZ, H, n_chunks, BT, BT)) * 1e4 * causal[None, None, None]
    Akk = jax.random.normal(ks[6], (BSZ, H, n_chunks, BT, BT)) * 1e4 * strict[None, None, None]
    h_pre_all = jax.random.normal(ks[7], (BSZ, H, n_chunks, D, D)) * 1e3
    v_new_all = jax.random.normal(ks[8], (BSZ, H, n_chunks, BT, D)) * 1e3
    do = jax.random.normal(ks[9], (BSZ, H, n_chunks, BT, D)) * 1e3
    dv = jax.random.normal(ks[10], (BSZ, H, n_chunks, BT, D)) * 1e3
    dh_next_all = jax.random.normal(ks[11], (BSZ, H, n_chunks, D, D)) * 1e3

    out_t = wy_dqkg_backward_pallas(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all, scale=1.0, config=CFG_TIGHT)
    out_l = wy_dqkg_backward_pallas(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all, scale=1.0, config=CFG_LOOSE)
    for key_name in ("dq", "dk", "db", "dw", "dv_raw", "dgc", "dAkk"):
        _bounds_check(f"B3/{key_name}", out_t[key_name], out_l[key_name])


# ---------------- B4 ----------------
def test_b4_intra_backward_clip():
    key = jax.random.PRNGKey(10)
    ks = jax.random.split(key, 6)
    q, k, b = [jax.random.normal(ks[i], (BSZ, BT, H, D)) * 30 for i in range(3)]
    g = jnp.zeros((BSZ, BT, H, D))
    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    dAqk = jax.random.normal(ks[4], (BSZ, H, 1, BT, BT)) * 1e4 * causal[None, None, None]
    dAkk = jax.random.normal(ks[5], (BSZ, H, 1, BT, BT)) * 1e4 * strict[None, None, None]

    dq_t, dk_t, db_t, dgc_t = intra_backward_pallas(dAqk, dAkk, q, k, b, g, scale=1.0, config=CFG_TIGHT)
    dq_l, dk_l, db_l, dgc_l = intra_backward_pallas(dAqk, dAkk, q, k, b, g, scale=1.0, config=CFG_LOOSE)
    _bounds_check("B4/dq", dq_t, dq_l)
    _bounds_check("B4/dk", dk_t, dk_l)
    _bounds_check("B4/db", db_t, db_l)
    _bounds_check("B4/dgc", dgc_t, dgc_l)


# ---------------- B5 ----------------
def test_b5_reverse_cumsum_clip():
    key = jax.random.PRNGKey(11)
    n_chunks = 1
    dgc = jax.random.normal(key, (BSZ, H, n_chunks, BT, D)) * 1e4

    dg_t = reverse_cumsum_bwd(dgc, chunk_size=BT, config=CFG_TIGHT)
    dg_l = reverse_cumsum_bwd(dgc, chunk_size=BT, config=CFG_LOOSE)
    _bounds_check("B5/dg_raw", dg_t, dg_l)


# ---------------- Config surface sanity ----------------
@pytest.mark.parametrize("clip_val", [1e2, 5e3, 1e4, 5e4])
def test_config_replace_roundtrip(clip_val):
    """Sanity: dataclasses.replace(KAGGLE_LARGE, clip=X) actually produces
    a usable, independently-clipped config (guards against future
    KernelConfig field additions silently breaking user overrides)."""
    cfg = dc.replace(KAGGLE_LARGE, clip=clip_val)
    assert cfg.clip == clip_val
    assert cfg.bt == KAGGLE_LARGE.bt and cfg.bc == KAGGLE_LARGE.bc  # unrelated fields untouched
    q, k, v, w, b, g = _huge_inputs(jax.random.PRNGKey(42))
    _, akk = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=cfg)
    assert float(jnp.max(jnp.abs(akk))) <= clip_val + TOL
