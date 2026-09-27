"""Fused C+D (recompute WY + inter-chunk scan). Slim residuals: wp/kg/qg are recomputed in backward."""
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from ..precision import make_dot, exp_nonpos, sanitize
from ..layout import cparams


def _kernel(q_ref, k_ref, v_ref, w_ref, b_ref, gc_ref, a_ref, aqk_ref, h0_ref,
            o_ref, hf_ref, hpre_ref, vnew_ref, gcl_ref, *, nc, scale, cfg):
    dot = make_dot(cfg.dot_mode)
    h = h0_ref[0, 0].astype(jnp.float32)
    for c in range(nc):
        q = q_ref[0, 0, c].astype(jnp.float32); k = k_ref[0, 0, c].astype(jnp.float32)
        v = v_ref[0, 0, c].astype(jnp.float32); w = w_ref[0, 0, c].astype(jnp.float32)
        b = b_ref[0, 0, c].astype(jnp.float32); gc = gc_ref[0, 0, c].astype(jnp.float32)
        A = a_ref[0, 0, c].astype(jnp.float32); Aq = aqk_ref[0, 0, c].astype(jnp.float32)
        egc, _ = exp_nonpos(gc)
        wp = sanitize(dot(A, b * k * egc), cfg.clip)
        u = sanitize(dot(A, w * v), cfg.clip)
        gc_last = gc[cfg.bt - 1]
        ekg, _ = exp_nonpos(gc_last[None, :] - gc)
        kg = sanitize(k * ekg, cfg.clip); qg = sanitize(q * egc, cfg.clip)
        hpre_ref[0, 0, c] = h
        v_new = u - dot(wp, h)
        o_c = scale * dot(qg, h) + dot(Aq, v_new)
        dec, _ = exp_nonpos(gc_last)
        h = sanitize(h * dec[:, None] + dot(kg.T, v_new), cfg.clip)
        o_ref[0, 0, c] = sanitize(o_c, cfg.clip)
        vnew_ref[0, 0, c] = v_new
        gcl_ref[0, 0, c] = gc_last
    hf_ref[0, 0] = h


def recompute_and_scan_slim(Aqk, q, k, v, w, b, gc, A, scale, h0, cfg):
    scale = float(scale)
    bsz, H, nc, T, D = q.shape
    io = pl.BlockSpec((1, 1, nc, T, D), lambda i, h: (i, h, 0, 0, 0))
    sc = pl.BlockSpec((1, 1, nc, T, T), lambda i, h: (i, h, 0, 0, 0))
    hs = pl.BlockSpec((1, 1, D, D), lambda i, h: (i, h, 0, 0))
    hp = pl.BlockSpec((1, 1, nc, D, D), lambda i, h: (i, h, 0, 0, 0))
    gl = pl.BlockSpec((1, 1, nc, D), lambda i, h: (i, h, 0, 0))
    o, hf, hpre, vnew, gcl = pl.pallas_call(
        lambda *r: _kernel(*r, nc=nc, scale=scale, cfg=cfg), grid=(bsz, H),
        in_specs=[io] * 6 + [sc, sc, hs], out_specs=[io, hs, hp, io, gl],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32),
                   jax.ShapeDtypeStruct((bsz, H, D, D), jnp.float32),
                   jax.ShapeDtypeStruct((bsz, H, nc, D, D), jnp.float32),
                   jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32),
                   jax.ShapeDtypeStruct((bsz, H, nc, D), jnp.float32)],
        compiler_params=cparams(cfg.vmem_mb), interpret=cfg.interpret)(q, k, v, w, b, gc, A, Aqk, h0)
    return dict(o=o, h_final=hf, h_pre_all=hpre, v_new_all=vnew, gc_last=gcl)
