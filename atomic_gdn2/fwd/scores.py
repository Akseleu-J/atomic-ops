"""Kernel A: two-level BLR scores (Aqk, Akk). Diagonal via BTL (MXU) or lean (exact, clipped)."""
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from ..precision import make_dot, exp_nonpos, exp_clipped, sanitize
from ..layout import causal_masks, cparams


def _place(block, p0, p1, T):
    parts = []
    if p0 > 0: parts.append(jnp.zeros(block.shape[:-1] + (p0,), jnp.float32))
    parts.append(block)
    if T - p1 > 0: parts.append(jnp.zeros(block.shape[:-1] + (T - p1,), jnp.float32))
    return jnp.concatenate(parts, axis=-1) if len(parts) > 1 else block


def btl_legs(gc, s0, s1):
    """Balanced two-leg: shift by half span so both legs in [-s/2, s/2]; no clip. Valid iff s/2 < 88."""
    r = gc[s0]
    half = 0.5 * (gc[s0] - gc[s1 - 1])
    ea = jnp.exp((gc[s0:s1] - r[None, :]) + half[None, :])
    ec = jnp.exp((r[None, :] - gc[s0:s1]) - half[None, :])
    return ea, ec


def _dia_btl(q, k, bk, gc, s0, s1, scale, dot):
    ea, ec = btl_legs(gc, s0, s1)
    kt = k[s0:s1] * ec
    return scale * dot(q[s0:s1] * ea, kt.T), dot(bk[s0:s1] * ea, kt.T)


def _dia_lean(q, k, bk, gc, s0, s1, scale, dc_):
    gd = gc[s0:s1]
    E, _ = exp_clipped(gd[:, None, :] - gd[None, :, :], -dc_, dc_)
    dq_ = scale * jnp.sum(q[s0:s1][:, None, :] * E * k[s0:s1][None, :, :], axis=-1)
    dk_ = jnp.sum(bk[s0:s1][:, None, :] * E * k[s0:s1][None, :, :], axis=-1)
    return dq_, dk_


def blr_rows(q, k, bk, gc, scale, cfg, dot):
    T, bs, bs2, dc_ = cfg.bt, cfg.score_bs, cfg.bs2, cfg.diff_clip
    idx = jnp.arange(T)
    out_q, out_k = [], []
    for p in range(T // bs):
        p0, p1 = p * bs, (p + 1) * bs
        if p0 > 0:
            r1 = gc[p0]
            a1, _ = exp_nonpos(gc[p0:p1] - r1[None, :])
            c1, _ = exp_nonpos(r1[None, :] - gc)
            kt1 = k * c1
            before = (idx < p0).astype(jnp.float32)[None, :]
            l1q = scale * dot(q[p0:p1] * a1, kt1.T) * before
            l1k = dot(bk[p0:p1] * a1, kt1.T) * before
        else:
            l1q = jnp.zeros((bs, T), jnp.float32); l1k = l1q
        rows_q, rows_k = [], []
        for kk in range(bs // bs2):
            q0 = p0 + kk * bs2; q1 = q0 + bs2
            rq = l1q[kk * bs2:(kk + 1) * bs2]; rk = l1k[kk * bs2:(kk + 1) * bs2]
            if kk > 0:
                r2 = gc[q0]
                a2, _ = exp_nonpos(gc[q0:q1] - r2[None, :])
                c2, _ = exp_nonpos(r2[None, :] - gc[p0:q0])
                kt2 = k[p0:q0] * c2
                rq = rq + _place(scale * dot(q[q0:q1] * a2, kt2.T), p0, q0, T)
                rk = rk + _place(dot(bk[q0:q1] * a2, kt2.T), p0, q0, T)
            if cfg.diag == "btl":
                bq, bkk = _dia_btl(q, k, bk, gc, q0, q1, scale, dot)
            else:
                bq, bkk = _dia_lean(q, k, bk, gc, q0, q1, scale, dc_)
            rows_q.append(rq + _place(bq, q0, q1, T)); rows_k.append(rk + _place(bkk, q0, q1, T))
        out_q.append(jnp.concatenate(rows_q, 0)); out_k.append(jnp.concatenate(rows_k, 0))
    return out_q, out_k


def _kernel(q_ref, k_ref, b_ref, gc_ref, aqk_ref, akk_ref, *, scale, cfg):
    q = q_ref[0, 0, 0].astype(jnp.float32); k = k_ref[0, 0, 0].astype(jnp.float32)
    b = b_ref[0, 0, 0].astype(jnp.float32); gc = gc_ref[0, 0, 0].astype(jnp.float32)
    causal, strict = causal_masks(cfg.bt)
    rq, rk = blr_rows(q, k, b * k, gc, scale, cfg, make_dot(cfg.dot_mode))
    bs = cfg.score_bs
    for p in range(cfg.bt // bs):
        p0, p1 = p * bs, (p + 1) * bs
        aqk_ref[0, 0, 0, p0:p1] = sanitize(rq[p] * causal[p0:p1], cfg.clip)
        akk_ref[0, 0, 0, p0:p1] = sanitize(rk[p] * strict[p0:p1], cfg.clip)


def build_scores(q, k, b, gc, scale, cfg):
    """q,k,b,gc: (B,H,nc,bt,D). Returns Aqk, Akk (B,H,nc,bt,bt)."""
    scale = float(scale)
    bsz, H, nc, bt, D = q.shape
    io = pl.BlockSpec((1, 1, 1, bt, D), lambda i, h, c: (i, h, c, 0, 0))
    sc = pl.BlockSpec((1, 1, 1, bt, bt), lambda i, h, c: (i, h, c, 0, 0))
    return pl.pallas_call(
        lambda *r: _kernel(*r, scale=scale, cfg=cfg), grid=(bsz, H, nc),
        in_specs=[io] * 4, out_specs=[sc, sc],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, bt, bt), jnp.float32)] * 2,
        compiler_params=cparams(cfg.vmem_mb), interpret=cfg.interpret)(q, k, b, gc)
