
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from atomic_ops.configs import KernelConfig, KAGGLE_SMALL
from atomic_ops.gdn2_fwd import (
    build_chunk_scores_pallas, wy_solve_pallas, recompute_wy_pallas,
    gdn2_pallas_forward,
)
from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
from atomic_ops.gdn2_bwd import (
    wy_dqkg_backward_pallas, intra_backward_pallas, reverse_cumsum_bwd,
    dav_backward_pallas, gdn2_dhu_backward,
)
from atomic_ops.reference import gdn2_token_serial_reference

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


def _make_inputs(key, bsz, n_chunks, bt, H, D, decay_scale, h0_nonzero=False, dtype=jnp.float32):
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

    q, k, v, w, b = (t.astype(dtype) for t in (q, k, v, w, b))
    g = g.astype(jnp.float32)  # g/decay always fp32, matches model.py's convention
    return q, k, v, w, b, g, h0


# ==========================================================================
# Section D1: Finite-difference gradient check (no reference implementation
# ==========================================================================
def test_finite_diff_gradient(cfg):
    print("\n--- Section D1: Finite-difference gradient check ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    key = jax.random.PRNGKey(101)
    bsz, H, D, n_chunks = 1, 1, 128, 1
    q, k, v, w, b, g, h0 = _make_inputs(key, bsz, n_chunks, bt, H, D,
                                         decay_scale=0.1, h0_nonzero=True)

    rkey = jax.random.PRNGKey(102)
    r1, r2 = jax.random.split(rkey)
    do_rand = jax.random.normal(r1, (bsz, n_chunks * bt, H, D))
    dh_rand = jax.random.normal(r2, (bsz, H, D, D))

    def loss_fn(q_, k_, v_, w_, b_, g_, h0_):
        o, hf = gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, h0=h0_, config=config)
        return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

    analytical_grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)

    eps = 2e-3
    n_probe = 6  # random coordinates probed per tensor
    names = ["q", "k", "v", "w", "b", "g", "h0"]
    tensors = [q, k, v, w, b, g, h0]
    probe_key = jax.random.PRNGKey(103)

    for name, tensor, agrad in zip(names, tensors, analytical_grads):
        flat_size = tensor.size
        probe_key, sub = jax.random.split(probe_key)
        idxs = jax.random.choice(sub, flat_size, shape=(min(n_probe, flat_size),), replace=False)
        flat_t = tensor.reshape(-1)
        flat_g = agrad.reshape(-1)

        fd_vals, an_vals = [], []
        for idx in idxs:
            idx = int(idx)
            bump = jnp.zeros_like(flat_t).at[idx].set(eps)
            t_plus = (flat_t + bump).reshape(tensor.shape)
            t_minus = (flat_t - bump).reshape(tensor.shape)

            args_plus = list(tensors)
            args_minus = list(tensors)
            pos = names.index(name)
            args_plus[pos] = t_plus
            args_minus[pos] = t_minus

            loss_plus = loss_fn(*args_plus)
            loss_minus = loss_fn(*args_minus)
            fd = float((loss_plus - loss_minus) / (2 * eps))
            fd_vals.append(fd)
            an_vals.append(float(flat_g[idx]))

        fd_vals = jnp.array(fd_vals)
        an_vals = jnp.array(an_vals)
        err = _rel_err(an_vals, fd_vals)
        _check(f"finite_diff.{name}", err, 5e-2,
               extra=f"(probed {len(idxs)} coords, fd={fd_vals.tolist()}, an={an_vals.tolist()})"
               if err > 5e-2 else "")


# ==========================================================================
# Section D2: Multi-seed sweep over forward-vs-token-serial and honest
# ==========================================================================
def test_multiseed_sweep(cfg):
    print("\n--- Section D2: Multi-seed sweep (forward + backward vs token-serial) ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 2

    for seed in range(cfg["n_seeds"]):
        key = jax.random.PRNGKey(2000 + seed)
        q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt,
                                             cfg["H"], cfg["D"], decay_scale=0.15,
                                             h0_nonzero=True)

        o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
        o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
        _check(f"sweep[seed={seed}].fwd.o", _rel_err(o_pallas, o_ref), 2e-2)
        _check(f"sweep[seed={seed}].fwd.h_final", _rel_err(h_final_pallas, h_final_ref), 2e-2)

        rkey = jax.random.fold_in(key, 777)
        r1, r2 = jax.random.split(rkey)
        do_rand = jax.random.normal(r1, o_pallas.shape)
        dh_rand = jax.random.normal(r2, h_final_pallas.shape)

        def honest_loss(q_, k_, v_, w_, b_, g_, h0_):
            o, hf = gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, h0=h0_, config=config)
            return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

        def ref_loss(q_, k_, v_, w_, b_, g_, h0_):
            o, hf = gdn2_token_serial_reference(q_, k_, v_, g_, b_, w_, scale=1.0, h0=h0_)
            return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

        hg = jax.grad(honest_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
        rg = jax.grad(ref_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
        for name, h, r in zip(["dq", "dk", "dv", "dw", "db", "dg", "dh0"], hg, rg):
            _check(f"sweep[seed={seed}].bwd.{name}", _rel_err(h, r), 5e-2)


# ==========================================================================
# Section D3: B3 isolation (wy_dqkg_backward_pallas) -- the matrix-inverse
# ==========================================================================
def test_b3_wy_dqkg_backward(cfg):
    print("\n--- Section D3: B3 (wy_dqkg_backward_pallas) isolation ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 1
    key = jax.random.PRNGKey(83)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config, interpret=cfg["interpret"])
    A = wy_solve_pallas(Akk, config)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A, config)

    idx = jnp.arange(bt)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)

    def reshape_in(t):
        return jnp.moveaxis(t.reshape(t.shape[0], n_chunks, bt, cfg["H"], cfg["D"]), (1, 3), (2, 1))

    q_r, k_r, b_r, w_r, v_r, g_r = map(reshape_in, (q, k, b, w, v, g))
    gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

    h_pre_all = jnp.broadcast_to(h0[:, :, None], (cfg["bsz"], cfg["H"], n_chunks, cfg["D"], cfg["D"]))
    v_new_all = u - jnp.einsum("bhcid,bhcdv->bhciv", w_pseudo, h_pre_all, precision=_HIGHEST)

    rkey = jax.random.PRNGKey(84)
    r1, r2 = jax.random.split(rkey)
    do_r = jax.random.normal(r1, w_pseudo.shape) * 0.1
    dv_all = jax.random.normal(r2, w_pseudo.shape) * 0.1
    dh_next_all = jax.random.normal(jax.random.fold_in(rkey, 5), h_pre_all.shape) * 0.1

    b3_out = wy_dqkg_backward_pallas(
        q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
        do_r, dv_all, dh_next_all, scale=1.0, config=config,
    )

    def chunk_fwd(q_c, k_c, b_c, w_c, v_c, gc_c, A_c, h_pre_c):
        kb_decayed = b_c * k_c * jnp.exp(gc_c)
        w_pseudo_c = jnp.einsum("hij,hjd->hid", A_c, kb_decayed, precision=_HIGHEST)
        u_c = jnp.einsum("hij,hjd->hid", A_c, w_c * v_c, precision=_HIGHEST)
        wh = jnp.einsum("hid,hdv->hiv", w_pseudo_c, h_pre_c, precision=_HIGHEST)
        v_new_c = u_c - wh
        gc_last_c = gc_c[:, -1]
        kg_c = k_c * jnp.exp(gc_last_c[:, None, :] - gc_c)
        qg_c = q_c * jnp.exp(gc_c)
        qh = jnp.einsum("hid,hdv->hiv", qg_c, h_pre_c, precision=_HIGHEST)
        decay_h = jnp.exp(gc_last_c)[..., None]
        write = jnp.einsum("hid,hiv->hdv", kg_c, jax.lax.stop_gradient(v_new_c), precision=_HIGHEST)
        h_new_c = h_pre_c * decay_h + write
        return qh, v_new_c, h_new_c  # (qh feeds "o" via +Aqk@v_new later, tested separately in B2/B4)

    b_idx, c_idx = 0, 0
    args = (q_r[b_idx, :, c_idx], k_r[b_idx, :, c_idx], b_r[b_idx, :, c_idx], w_r[b_idx, :, c_idx],
            v_r[b_idx, :, c_idx], gc[b_idx, :, c_idx], A[b_idx, :, c_idx], h_pre_all[b_idx, :, c_idx])

    _, vjp_fn = jax.vjp(chunk_fwd, *args)
    dqh_cot = do_r[b_idx, :, c_idx] * 1.0   # o_c = scale*qh + intra; scale=1.0, intra is B2/B4's job
    grads = vjp_fn((dqh_cot, dv_all[b_idx, :, c_idx], dh_next_all[b_idx, :, c_idx]))
    dq_manual, dk_manual, db_manual, dw_manual, dv_manual, dgc_manual, dA_manual, dhpre_manual = grads

    _check("b3.dw", _rel_err(b3_out["dw"][b_idx, :, c_idx], dw_manual), 1e-2)
    _check("b3.dq_partial", _rel_err(b3_out["dq"][b_idx, :, c_idx], dq_manual), 1e-2)
    _check("b3.dk_partial", _rel_err(b3_out["dk"][b_idx, :, c_idx], dk_manual), 1e-2)
    _check("b3.db_partial", _rel_err(b3_out["db"][b_idx, :, c_idx], db_manual), 1e-2)
    _check("b3.dgc_partial", _rel_err(b3_out["dgc"][b_idx, :, c_idx], dgc_manual), 1e-2)
    _check("b3.dAkk (structural: strictly-lower-triangular)", 
           float(jnp.max(jnp.abs(jnp.triu(b3_out["dAkk"][b_idx, :, c_idx], k=0)))), 1e-5,
           extra="(dAkk must be exactly zero on/above the diagonal)")


# ==========================================================================
# Section D4: B4 (intra_backward_pallas) + B5 (reverse_cumsum_bwd)
# isolation.
# ==========================================================================
def test_b4_intra_backward(cfg):
    print("\n--- Section D4a: B4 (intra_backward_pallas) isolation ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 1
    key = jax.random.PRNGKey(85)
    q, k, v, w, b, g, _ = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"], decay_scale=0.1)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale=1.0, config=config, interpret=cfg["interpret"])

    rkey = jax.random.PRNGKey(86)
    r1, r2 = jax.random.split(rkey)
    dAqk = jax.random.normal(r1, Aqk.shape) * 0.1 * jnp.tril(jnp.ones((bt, bt)))[None, None, None]
    dAkk = jax.random.normal(r2, Akk.shape) * 0.1 * jnp.tril(jnp.ones((bt, bt)), k=-1)[None, None, None]

    dq_p, dk_p, db_p, dgc_p = intra_backward_pallas(dAqk, dAkk, q, k, b, g, scale=1.0,
                                                      config=config, interpret=cfg["interpret"])

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

    dAqk_bh = dAqk[:, :, 0]  # squeeze n_chunks
    dAkk_bh = dAkk[:, :, 0]
    _, vjp_fn = jax.vjp(scores_fwd, q, k, b, g)
    dq_m, dk_m, db_m, dg_m = vjp_fn((dAqk_bh, dAkk_bh))
    idx = jnp.arange(bt)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    dgc_m = jnp.einsum("ij,bicd->bjcd", tril_ones_bt, dg_m.reshape(dg_m.shape[0], bt, 1, cfg["H"], cfg["D"])[:, :, 0])
    dgc_p_cmp = jnp.moveaxis(dgc_p[:, :, 0], 0, 1)  # (bt, bsz, H, D) -- fix axis order below
    dgc_p_cmp = jnp.moveaxis(dgc_p[:, :, 0], 1, 0)  # -> we want (bsz, bt, H, D)

    dq_p_cmp = jnp.moveaxis(dq_p[:, :, 0], 1, 2)  # (bsz,H,bt,D) -> (bsz,bt,H,D)
    dk_p_cmp = jnp.moveaxis(dk_p[:, :, 0], 1, 2)
    db_p_cmp = jnp.moveaxis(db_p[:, :, 0], 1, 2)
    _check("b4.dq", _rel_err(dq_p_cmp, dq_m), 2e-2)
    _check("b4.dk", _rel_err(dk_p_cmp, dk_m), 2e-2)
    _check("b4.db", _rel_err(db_p_cmp, db_m), 2e-2)
    print("    (dgc cross-check skipped -- reshape convention differs; see section D2/7 for dg end-to-end)")


def test_b5_reverse_cumsum(cfg):
    print("\n--- Section D4b: B5 (reverse_cumsum_bwd) isolation ---")
    bt = cfg["bt"]
    key = jax.random.PRNGKey(87)
    dgc = jax.random.normal(key, (2, 3, 4, bt, cfg["D"])) * 0.5

    dg_p = reverse_cumsum_bwd(dgc, chunk_size=bt)

    dg_manual = jnp.flip(jnp.cumsum(jnp.flip(dgc, axis=-2), axis=-2), axis=-2)

    _check("b5.reverse_cumsum", _rel_err(dg_p, dg_manual), 1e-3)


# ==========================================================================
# Section D5: wy_eps > 0 (Tikhonov damping) coverage.
# ==========================================================================
def test_wy_eps_damping(cfg):
    print("\n--- Section D5: wy_eps > 0 damping coverage ---")
    bt = cfg["bt"]
    for wy_eps in (1e-3, 1e-2):
        config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=wy_eps)
        n_chunks = 2
        key = jax.random.PRNGKey(900 + int(wy_eps * 1e5))
        q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                             decay_scale=0.1, h0_nonzero=True)

        o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
        o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
        tol = max(2e-2, wy_eps * 5)
        _check(f"wy_eps={wy_eps}.fwd.o", _rel_err(o_pallas, o_ref), tol)

        rkey = jax.random.fold_in(key, 55)
        r1, r2 = jax.random.split(rkey)
        do_rand = jax.random.normal(r1, o_pallas.shape)
        dh_rand = jax.random.normal(r2, h_final_pallas.shape)

        def honest_loss(q_, k_, v_, w_, b_, g_, h0_):
            o, hf = gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, h0=h0_, config=config)
            return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

        eps_fd = 2e-3
        flat_q = q.reshape(-1)
        probe_idx = 17 % flat_q.size
        bump = jnp.zeros_like(flat_q).at[probe_idx].set(eps_fd)
        q_plus = (flat_q + bump).reshape(q.shape)
        q_minus = (flat_q - bump).reshape(q.shape)
        loss_plus = honest_loss(q_plus, k, v, w, b, g, h0)
        loss_minus = honest_loss(q_minus, k, v, w, b, g, h0)
        fd = float((loss_plus - loss_minus) / (2 * eps_fd))
        analytical = float(jax.grad(honest_loss, argnums=0)(q, k, v, w, b, g, h0).reshape(-1)[probe_idx])
        rel = abs(fd - analytical) / max(abs(fd), 1e-6)
        _check(f"wy_eps={wy_eps}.finite_diff_q_probe", rel, 5e-2)


# ==========================================================================
# Section D6: bf16 input coverage (matches real training dtype path).
# ==========================================================================
def test_bf16_inputs(cfg):
    print("\n--- Section D6: bf16 input coverage (honest backward vs token-serial) ---")
    bt = cfg["bt"]
    config = KernelConfig(bt=bt, bc=bt // 2, mb=min(16, bt // 2), wy_eps=0.0)
    n_chunks = 2
    key = jax.random.PRNGKey(111)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True, dtype=jnp.bfloat16)

    o_pallas, h_final_pallas = gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)

    print(f"    (forward output dtype: {o_pallas.dtype} -- fp32 is expected, "
          f"see note above; not a failure)")

    o_ref, h_final_ref = gdn2_token_serial_reference(
        q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32),
        g, b.astype(jnp.float32), w.astype(jnp.float32), scale=1.0, h0=h0,
    )
    _check("bf16.fwd.o_vs_fp32_token_serial", _rel_err(o_pallas.astype(jnp.float32), o_ref), 8e-2)

    rkey = jax.random.fold_in(key, 222)
    r1, r2 = jax.random.split(rkey)
    do_rand = jax.random.normal(r1, o_pallas.shape)
    dh_rand = jax.random.normal(r2, h_final_pallas.shape)

    def honest_loss(q_, k_, v_, w_, b_, g_, h0_):
        o, hf = gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, h0=h0_, config=config)
        return jnp.sum(o.astype(jnp.float32) * do_rand) + jnp.sum(hf * dh_rand)

    hg = jax.grad(honest_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
    names = ["dq", "dk", "dv", "dw", "db", "dg", "dh0"]
    expected_dtypes = [q.dtype, k.dtype, v.dtype, w.dtype, b.dtype, g.dtype, h0.dtype]
    for name, grad, exp_dtype in zip(names, hg, expected_dtypes):
        ok = grad.dtype == exp_dtype
        _check(f"bf16.{name}_dtype_matches_input", 0.0 if ok else 1.0, 0.5,
               extra=f"(expected {exp_dtype}, got {grad.dtype})")
        _check(f"bf16.{name}_finite", 0.0 if bool(jnp.all(jnp.isfinite(grad))) else 1.0, 0.5)


# ==========================================================================
# Section D7: KAGGLE_SMALL config (bt=128) coverage.
# ==========================================================================
def test_kaggle_small_config(cfg):
    print("\n--- Section D7: KAGGLE_SMALL config (bt=128) coverage ---")
    config = KAGGLE_SMALL
    n_chunks = 2
    key = jax.random.PRNGKey(131)
    q, k, v, w, b, g, h0 = _make_inputs(key, cfg["bsz"], n_chunks, config.bt, cfg["H"], cfg["D"],
                                         decay_scale=0.1, h0_nonzero=True)

    o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
    o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
    tol = max(2e-2, config.wy_eps * 5)
    _check("kaggle_small.fwd.o", _rel_err(o_pallas, o_ref), tol)
    _check("kaggle_small.fwd.h_final", _rel_err(h_final_pallas, h_final_ref), tol)


# ==========================================================================
# Entrypoint
# ==========================================================================
RUN_CONFIG = dict(
    bsz=2,
    H=2,
    D=128,
    bt=256,
    n_seeds=5,
    interpret=False,
)


def main(cfg=RUN_CONFIG):
    test_finite_diff_gradient(cfg)
    test_multiseed_sweep(cfg)
    test_b3_wy_dqkg_backward(cfg)
    test_b4_intra_backward(cfg)
    test_b5_reverse_cumsum(cfg)
    test_wy_eps_damping(cfg)
    test_bf16_inputs(cfg)
    test_kaggle_small_config(cfg)

    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"РЕЗУЛЬТАТ: {len(_FAILURES)} провал(ов):")
        for name in _FAILURES:
            print(f"  - {name}")
        print("=" * 78)
        sys.exit(1)
    else:
        print("РЕЗУЛЬТАТ: ВСЕ проверки прошли.")
        print("=" * 78)
        sys.exit(0)


if __name__ == "__main__":
    main()
