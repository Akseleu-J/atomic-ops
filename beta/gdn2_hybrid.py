"""
atomic_ops/gdn2_hybrid.py -- Hybrid GDN-2 forward/backward path.

Motivation (measured on B=8, L=4096, H=6, D=128; see README.md benchmarks):
    JAX_REF (pure JAX, checkpointed scan)      fwd ~63ms    bwd ~459ms
    PALLAS  (fused Kernel A/B/C/D + B1-B5)     fwd ~102ms   bwd ~175ms

The pure-JAX forward is faster, most likely because the separate
`pallas_call`s for Kernel A/B/C/D cannot be fused by XLA across call
boundaries, whereas the pure-JAX path is a single XLA graph. The Pallas
backward is faster because of the block-recursive WY solve in Kernel B
(`_block_solve`, `mb`-sized micro-blocks) versus differentiating through
the sequential `bt`-step `jax.lax.scan` used by the pure-JAX `_wy_inverse`.

This module combines both: forward runs the pure-JAX chunked-WY recurrence
(same math as `reference.gdn2_chunked_wy_reference`), but its scan also
extracts the per-chunk residuals (Aqk, Akk, A, w_pseudo, u, kg, qg,
gc_last, h_pre, v_new) into the same layout `(bsz, H, n_chunks, BT, D[,
BT])` expected by the already-validated Pallas backward kernels in
`atomic_ops.gdn2_bwd` (B1-B5). The backward pass does not recompute
Kernel A/B/C the way the fully-fused path in `atomic_ops.gdn2_pipeline`
does -- residuals are already carried over from the forward scan.

STATUS: beta / experimental path. Not wired into any training entrypoint.
Before relying on it in training:
  1. Cross-check gradients against
     `atomic_ops.gdn2_pipeline.gdn2_pallas_forward_trainable` (rel_err)
     across multiple seeds/shapes on real TPU hardware (not interpret mode).
  2. Benchmark the combined fwd+bwd cost, not just the per-stage numbers
     quoted above -- a per-stage win does not automatically imply a
     fwd+bwd win.

Notes on how this differs from a naive JAX-forward/Pallas-backward split:
  - `KernelConfig` is required throughout (`config.bt` replaces a
    hardcoded chunk-size constant), consistent with the rest of the
    package post-refactor.
  - Clipping uses `config.clip` via `sanitize(x, config)` in every
    forward stage, not a hardcoded constant.
  - `wy_eps` is taken from `config.wy_eps` (Tikhonov damping) rather than
    defaulting to 0.0 -- otherwise this path solves a different problem
    than the production `KAGGLE_*` presets and gradients would diverge
    on any near-singular chunk (see `reference.py`'s `_wy_inverse`
    docstring).
  - `custom_vjp` takes `config` as a `nondiff_argnums` entry, matching
    `atomic_ops.gdn2_pipeline._gdn2_core`.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from atomic_ops.configs import KernelConfig, DEFAULT_CONFIG, sanitize, sanitize_h0, validate_inputs
from atomic_ops.gdn2_fwd import (
    _reshape_to_chunks as _r2c,
    _reshape_from_chunks as _r2f,
)
from atomic_ops.gdn2_bwd import (
    gdn2_dhu_backward,
    dav_backward_pallas,
    wy_dqkg_backward_pallas,
    intra_backward_pallas,
    reverse_cumsum_bwd,
)
from atomic_ops.reference import _wy_inverse

_HIGHEST = jax.lax.Precision.HIGHEST
_FINAL_CLIP = 1e4


def _final_sanitize(x):
    return jnp.nan_to_num(
        jnp.clip(x, -_FINAL_CLIP, _FINAL_CLIP),
        nan=0.0, posinf=_FINAL_CLIP, neginf=-_FINAL_CLIP,
    )


def _build_chunk_wy_with_akk(q_c, k_c, v_c, g_raw_c, b_c, w_c, scale, config: KernelConfig):
    """Same computation as `reference._build_chunk_wy`, but also returns
    `Akk`, which the backward kernels (`wy_dqkg_backward_pallas`,
    `intra_backward_pallas`) need for shape/masking and `dAkk`
    accumulation."""
    C = q_c.shape[1]
    f32 = jnp.float32

    gc = jnp.cumsum(g_raw_c.astype(f32), axis=1)
    gc_bhcd = jnp.moveaxis(gc, 2, 1)
    decay_diff = gc_bhcd[:, :, :, None, :] - gc_bhcd[:, :, None, :, :]
    edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))

    causal = jnp.tril(jnp.ones((C, C), dtype=f32))
    strict = jnp.tril(jnp.ones((C, C), dtype=f32), k=-1)

    q_bhcd = jnp.moveaxis(q_c, 2, 1).astype(f32)
    k_bhcd = jnp.moveaxis(k_c, 2, 1).astype(f32)
    b_bhcd = jnp.moveaxis(b_c, 2, 1).astype(f32)

    Aqk = scale * jnp.einsum("bhid,bhijd,bhjd->bhij", q_bhcd, edecay, k_bhcd, precision=_HIGHEST) * causal
    bk_bhcd = b_bhcd * k_bhcd
    Akk = jnp.einsum("bhid,bhijd,bhjd->bhij", bk_bhcd, edecay, k_bhcd, precision=_HIGHEST) * strict

    Aqk = sanitize(Aqk, config)
    Akk = sanitize(Akk, config)

    A = _wy_inverse(Akk, eps=config.wy_eps)
    A = sanitize(A, config)

    kb_decayed = (b_c.astype(f32) * k_c.astype(f32)) * jnp.exp(gc)
    w_pseudo = jnp.einsum("bhij,bjhd->bihd", A, kb_decayed, precision=_HIGHEST)
    u = jnp.einsum("bhij,bjhv->bihv", A, (w_c * v_c).astype(f32), precision=_HIGHEST)
    w_pseudo = sanitize(w_pseudo, config)
    u = sanitize(u, config)

    gc_last = gc[:, -1]
    kg = k_c.astype(f32) * jnp.exp(gc_last[:, None] - gc)
    qg = q_c.astype(f32) * jnp.exp(gc)

    return Aqk, Akk, A, w_pseudo, u, kg, qg, gc_last


def _forward_with_residuals(q, k, v, w, b, g, scale, config: KernelConfig, h0=None):
    """Same algorithm as `reference.gdn2_chunked_wy_reference`, but the
    scan additionally emits per-chunk residuals in the layout expected by
    the Pallas backward kernels in `atomic_ops.gdn2_bwd`
    (`bsz, H, n_chunks, BT, D[, BT]`)."""
    bsz, L, H, D = q.shape
    Dv = v.shape[-1]
    BT = config.bt
    if L % BT != 0:
        raise ValueError(f"seq_len={L} must be divisible by config.bt={BT}.")
    n_chunks = L // BT

    def to_chunks(t):
        shp = t.shape
        t = t.reshape(bsz, n_chunks, BT, *shp[2:])
        return jnp.moveaxis(t, 1, 0)

    q_ch, k_ch, v_ch, g_ch, b_ch, w_ch = map(to_chunks, (q, k, v, g, b, w))

    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, Dv), dtype=jnp.float32)
    h0 = sanitize_h0(h0, config)

    def chunk_step(h_pre, inputs):
        q_c, k_c, v_c, g_c, b_c, w_c = inputs
        Aqk, Akk, A, w_pseudo, u, kg, qg, gc_last = _build_chunk_wy_with_akk(
            q_c, k_c, v_c, g_c, b_c, w_c, scale, config
        )

        wh = jnp.einsum("bihd,bhdv->bihv", w_pseudo, h_pre, precision=_HIGHEST)
        v_new = u - wh
        v_new_bhcv = jnp.moveaxis(v_new, 2, 1)  # (bsz,BT,H,D) -> (bsz,H,BT,D), matches Pallas layout

        qh = jnp.einsum("bihd,bhdv->bihv", qg, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk, v_new_bhcv, precision=_HIGHEST)
        intra = jnp.moveaxis(intra, 1, 2)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gc_last)[..., None]
        write = jnp.einsum("bihd,bihv->bhdv", kg, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        h_new = sanitize(h_new, config)
        o_c = sanitize(o_c, config)

        residuals_c = (Aqk, Akk, A, w_pseudo, u, kg, qg, gc_last, h_pre, v_new_bhcv)
        return h_new, (o_c, residuals_c)

    chunk_step = jax.checkpoint(chunk_step)
    h_final, (o_scanned, residuals_scanned) = jax.lax.scan(
        chunk_step, h0, (q_ch, k_ch, v_ch, g_ch, b_ch, w_ch)
    )
    o = jnp.moveaxis(o_scanned, 0, 1).reshape(bsz, L, H, Dv)

    (Aqk_s, Akk_s, A_s, w_pseudo_s, u_s, kg_s, qg_s, gc_last_s, h_pre_s, v_new_s) = residuals_scanned

    def _leading_to_pos2(x):
        # (n_chunks, ...) -> n_chunks moved to position 2, right after
        # bsz, H. Works for both (n_chunks,bsz,H,BT,BT) and
        # (n_chunks,bsz,H,D,D): index 1/2 is already bsz,H.
        return jnp.moveaxis(x, 0, 2)

    def _bhc(x):
        # (n_chunks,bsz,BT,H,D) -> (bsz,H,n_chunks,BT,D)
        return jnp.moveaxis(x, (0, 3), (2, 1))

    Aqk_r = _leading_to_pos2(Aqk_s)
    Akk_r = _leading_to_pos2(Akk_s)
    A_r = _leading_to_pos2(A_s)
    w_pseudo_r = _bhc(w_pseudo_s)
    u_r = _bhc(u_s)
    kg_r = _bhc(kg_s)
    qg_r = _bhc(qg_s)
    gc_last_r = _leading_to_pos2(gc_last_s)  # (n_chunks,bsz,H,D) -> (bsz,H,n_chunks,D)
    h_pre_r = _leading_to_pos2(h_pre_s)      # (n_chunks,bsz,H,D,D) -> (bsz,H,n_chunks,D,D)
    v_new_r = _leading_to_pos2(v_new_s)      # (n_chunks,bsz,H,BT,D) -> (bsz,H,n_chunks,BT,D)

    residuals = dict(
        Aqk=Aqk_r, Akk=Akk_r, A=A_r,
        w_pseudo=w_pseudo_r, u=u_r, kg=kg_r, qg=qg_r, gc_last=gc_last_r,
        h_pre_all=h_pre_r, v_new_all=v_new_r,
    )
    return o, h_final, residuals


def _build_dh_next_all(dh_all, dht):
    shifted = dh_all[:, :, 1:]
    dht_expanded = dht[:, :, None]
    return jnp.concatenate([shifted, dht_expanded], axis=2)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _gdn2_core_hybrid(q, k, v, w, b, g, scale, config, h0):
    o, h_final, _ = _forward_with_residuals(q, k, v, w, b, g, scale, config, h0=h0)
    return o, h_final


def _gdn2_core_hybrid_fwd(q, k, v, w, b, g, scale, config, h0):
    o, h_final, fwd_res = _forward_with_residuals(q, k, v, w, b, g, scale, config, h0=h0)
    residuals = dict(q=q, k=k, v=v, w=w, b=b, g=g, h0=h0, **fwd_res)
    return (o, h_final), residuals


def _gdn2_core_hybrid_bwd(scale, config, residuals, cotangents):
    q, k, v, w, b, g, h0 = (residuals[key] for key in ("q", "k", "v", "w", "b", "g", "h0"))
    Aqk, Akk, A = residuals["Aqk"], residuals["Akk"], residuals["A"]
    w_pseudo, u, kg, qg, gc_last = (residuals[key] for key in ("w_pseudo", "u", "kg", "qg", "gc_last"))
    h_pre_all, v_new_all = residuals["h_pre_all"], residuals["v_new_all"]

    do, dh_final = cotangents
    bsz, L, H, D = q.shape
    BT = config.bt
    n_chunks = L // BT

    def reshape_in(t):
        return _r2c(t, bsz, n_chunks, H, D, BT)

    g_r = reshape_in(g)
    idx = jnp.arange(BT)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

    q_r = reshape_in(q)
    k_r = reshape_in(k)
    b_r = reshape_in(b)
    w_r = reshape_in(w)
    v_r = reshape_in(v)
    do_r = reshape_in(do)

    # B2
    dAqk, dv_partial = dav_backward_pallas(Aqk, v_new_all, do_r, config)

    # B1
    dh_all, dh0, dv_all = gdn2_dhu_backward(
        do_r, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=dh_final, config=config
    )
    dh_next_all = _build_dh_next_all(dh_all, dh_final)

    # B3
    b3_out = wy_dqkg_backward_pallas(
        q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
        do_r, dv_all, dh_next_all, scale, config,
    )

    # B4
    dq4, dk4, db4, dgc4 = intra_backward_pallas(
        dAqk, b3_out["dAkk"], q, k, b, g, scale, config
    )

    # B5
    dgc_total = b3_out["dgc"] + dgc4
    dg_raw = reverse_cumsum_bwd(dgc_total, chunk_size=BT, config=config)

    dq = _r2f(b3_out["dq"] + dq4, bsz, n_chunks, BT, H, D)
    dk = _r2f(b3_out["dk"] + dk4, bsz, n_chunks, BT, H, D)
    db = _r2f(b3_out["db"] + db4, bsz, n_chunks, BT, H, D)
    dw = _r2f(b3_out["dw"], bsz, n_chunks, BT, H, D)
    dv = _r2f(b3_out["dv_raw"], bsz, n_chunks, BT, H, D)
    dg = _r2f(dg_raw, bsz, n_chunks, BT, H, D)

    dq = _final_sanitize(dq).astype(q.dtype)
    dk = _final_sanitize(dk).astype(k.dtype)
    db = _final_sanitize(db).astype(b.dtype)
    dw = _final_sanitize(dw).astype(w.dtype)
    dv = _final_sanitize(dv).astype(v.dtype)
    dg = _final_sanitize(dg).astype(g.dtype)
    dh0 = _final_sanitize(dh0).astype(h0.dtype)

    return dq, dk, dv, dw, db, dg, dh0


_gdn2_core_hybrid.defvjp(_gdn2_core_hybrid_fwd, _gdn2_core_hybrid_bwd)


def gdn2_hybrid_forward_trainable(q, k, v, w, b, g, scale, h0=None, config: KernelConfig = DEFAULT_CONFIG):
    """Forward = pure JAX (fast at real training shapes, see module
    docstring); backward = the fused Pallas B1-B5 chain from
    `atomic_ops.gdn2_bwd`, without recomputing Kernel A/B/C (residuals are
    already carried over from the forward scan).

    Beta path -- validate gradients against
    `atomic_ops.gdn2_pipeline.gdn2_pallas_forward_trainable` and benchmark
    the combined fwd+bwd cost on real TPU hardware before relying on it.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    validate_inputs(q, k, v, w, b, g, scale, h0, config)
    return _gdn2_core_hybrid(q, k, v, w, b, g, scale, config, h0)
