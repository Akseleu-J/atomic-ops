"""
Pure-JAX reference implementations for GDN-2.
Token-serial (ground truth) and chunked-WY (backward fallback / cross-check).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def _wy_inverse(Akk, eps: float = 0.0):
    """Row-by-row WY inverse via lax.scan. eps>0 damps Akk by (1-eps)
    before inversion -- same Tikhonov-style regularization as the Pallas
    block solve in gdn2_fwd.py's _block_solve, kept here so the reference
    path (used by gdn2_pallas_forward_trainable's fallback on CPU/GPU, and
    as a cross-check target for the Pallas path) solves the SAME damped
    problem instead of the exact one. Without this, gradient/output
    comparisons between the two paths would diverge whenever a chunk's
    Akk is near-singular -- exactly the case damping exists to handle.
    Default eps=0.0 preserves the original exact behavior for anyone
    relying on this function directly.
    """
    C = Akk.shape[-1]
    eye = jnp.eye(C, dtype=Akk.dtype)
    batch_shape = Akk.shape[:-2]

    Akk_damped = Akk * (1.0 - eps)

    def row_step(A_rows, i):
        t_row = jnp.take(Akk_damped, i, axis=-2)
        contrib = jnp.einsum("...j,...jk->...k", t_row, A_rows, precision=_HIGHEST)
        new_row = eye[i] - contrib
        A_rows = jax.lax.dynamic_update_slice_in_dim(A_rows, new_row[..., None, :], i, axis=-2)
        return A_rows, None

    A0 = jnp.zeros(batch_shape + (C, C), dtype=Akk.dtype)
    A_final, _ = jax.lax.scan(row_step, A0, jnp.arange(C))
    return A_final


def gdn2_token_serial_reference(q, k, v, g, b, w, scale, h0=None):
    """Ground-truth token-by-token scan. Slow but numerically exact."""
    bsz, L, H, D = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype

    alpha = jnp.exp(g.astype(jnp.float32))

    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, Dv), dtype=jnp.float32)

    def step(h, inputs):
        q_t, k_t, v_t, alpha_t, b_t, w_t = inputs
        h = h * alpha_t[..., :, None]
        bk_t = (b_t * k_t).astype(jnp.float32)
        erase = jnp.einsum("bhd,bhdv->bhv", bk_t, h, precision=_HIGHEST)
        v_new = (w_t * v_t).astype(jnp.float32) - erase
        h = h + jnp.einsum("bhd,bhv->bhdv", k_t.astype(jnp.float32), v_new, precision=_HIGHEST)
        o_t = jnp.einsum("bhdv,bhd->bhv", h, (q_t * scale).astype(jnp.float32), precision=_HIGHEST)
        return h, o_t

    to_scan = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, alpha, b, w))
    h_final, o_scanned = jax.lax.scan(step, h0, to_scan)
    o = jnp.moveaxis(o_scanned, 0, 1).astype(dtype)
    return o, h_final


def _build_chunk_wy(q_c, k_c, v_c, g_raw_c, b_c, w_c, scale, wy_eps: float = 0.0):
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

    Aqk = jnp.nan_to_num(Aqk, nan=0.0, posinf=1e4, neginf=-1e4)
    Akk = jnp.nan_to_num(Akk, nan=0.0, posinf=1e4, neginf=-1e4)

    A = _wy_inverse(Akk, eps=wy_eps)
    A = jnp.nan_to_num(A, nan=0.0, posinf=1e4, neginf=-1e4)

    kb_decayed = (b_c.astype(f32) * k_c.astype(f32)) * jnp.exp(gc)
    w_pseudo = jnp.einsum("bhij,bjhd->bihd", A, kb_decayed, precision=_HIGHEST)
    u = jnp.einsum("bhij,bjhv->bihv", A, (w_c * v_c).astype(f32), precision=_HIGHEST)
    w_pseudo = jnp.nan_to_num(w_pseudo, nan=0.0, posinf=1e4, neginf=-1e4)
    u = jnp.nan_to_num(u, nan=0.0, posinf=1e4, neginf=-1e4)

    gc_last = gc[:, -1]
    kg = k_c.astype(f32) * jnp.exp(gc_last[:, None] - gc)
    qg = q_c.astype(f32) * jnp.exp(gc)

    return Aqk, w_pseudo, u, kg, qg, gc_last


def gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size, h0=None, wy_eps: float = 0.0):
    """Chunked-WY reference with jax.checkpoint for memory efficiency.

    wy_eps: pass config.wy_eps here when using this as a cross-check
    against the Pallas path (gdn2_pallas_forward_trainable), otherwise the
    two paths solve slightly different problems on near-singular chunks
    and comparisons/asserts between them will show spurious mismatches.
    """
    bsz, L, H, D = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype
    if L % chunk_size != 0:
        raise ValueError(f"seq_len={L} must be divisible by chunk_size={chunk_size}")
    n_chunks = L // chunk_size

    def to_chunks(t):
        shp = t.shape
        t = t.reshape(bsz, n_chunks, chunk_size, *shp[2:])
        return jnp.moveaxis(t, 1, 0)

    q_ch, k_ch, v_ch, g_ch, b_ch, w_ch = map(to_chunks, (q, k, v, g, b, w))

    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, Dv), dtype=jnp.float32)

    def chunk_step(h_pre, inputs):
        q_c, k_c, v_c, g_c, b_c, w_c = inputs
        Aqk, w_pseudo, u, kg, qg, gc_last = _build_chunk_wy(q_c, k_c, v_c, g_c, b_c, w_c, scale, wy_eps=wy_eps)

        wh = jnp.einsum("bihd,bhdv->bihv", w_pseudo, h_pre, precision=_HIGHEST)
        v_new = u - wh

        qh = jnp.einsum("bihd,bhdv->bihv", qg, h_pre, precision=_HIGHEST)
        v_new_bhcv = jnp.moveaxis(v_new, 2, 1)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk, v_new_bhcv, precision=_HIGHEST)
        intra = jnp.moveaxis(intra, 1, 2)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gc_last)[..., None]
        write = jnp.einsum("bihd,bihv->bhdv", kg, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        h_new = jnp.nan_to_num(jnp.clip(h_new, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        o_c = jnp.nan_to_num(o_c, nan=0.0, posinf=1e4, neginf=-1e4)

        return h_new, o_c

    chunk_step = jax.checkpoint(chunk_step)

    h_final, o_scanned = jax.lax.scan(chunk_step, h0, (q_ch, k_ch, v_ch, g_ch, b_ch, w_ch))
    o = jnp.moveaxis(o_scanned, 0, 1).reshape(bsz, L, H, Dv)
    return o, h_final
