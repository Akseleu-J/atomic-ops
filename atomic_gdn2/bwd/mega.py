"""Single backward launch: B2+B1+B3+B4+B5, reverse loop over chunks per (batch, head)."""
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from ..precision import make_dot, exp_nonpos, sanitize
from ..layout import causal_masks, cparams
from .b4 import b4_2l_values


def _kernel(q_ref, k_ref, b_ref, w_ref, v_ref, gc_ref, a_ref, aqk_ref, hpre_ref, vnew_ref,
            gcl_ref, do_ref, dht_ref, dq_ref, dk_ref, db_ref, dw_ref, dvraw_ref, dg_ref, dh0_ref,
            *, nc, scale, cfg):
    dot3 = make_dot(cfg.b3_dot_mode); dot_hi = make_dot(cfg.dot_mode)
    dot4 = make_dot("highest")  # fixed: b4_dot_mode removed (T4 phantom audit, v0.2.0)
    T, c_ = cfg.bt, cfg.clip
    idx = jnp.arange(T)
    causal, strict = causal_masks(T)
    triu = (idx[:, None] <= idx[None, :]).astype(jnp.float32)
    last = (idx == (T - 1)).astype(jnp.float32)[:, None]
    dh = dht_ref[0, 0].astype(jnp.float32)
    for c in range(nc - 1, -1, -1):
        f32 = lambda r: r[0, 0, c].astype(jnp.float32)
        q, k, b, w, v, gc = f32(q_ref), f32(k_ref), f32(b_ref), f32(w_ref), f32(v_ref), f32(gc_ref)
        A, Aqk, h_pre, v_new, do = f32(a_ref), f32(aqk_ref), f32(hpre_ref), f32(vnew_ref), f32(do_ref)
        gc_last = gcl_ref[0, 0, c].astype(jnp.float32)
        egc, mgc = exp_nonpos(gc)
        ekg, mkg = exp_nonpos(gc_last[None, :] - gc)
        kb = b * k * egc
        kg = sanitize(k * ekg, c_); qg = sanitize(q * egc, c_)
        wp = sanitize(dot_hi(A, kb), c_)
        dAqk = dot3(do, v_new.T) * causal                                  # B2
        dv_new = sanitize(dot3(Aqk.T, do) + dot3(kg, dh), c_)              # B1
        dec, mdec = exp_nonpos(gc_last)
        dh_next = dh
        dh = sanitize(dot3(qg.T, scale * do) + dh * dec[:, None] - dot3(wp.T, dv_new), c_)
        wv = w * v                                                          # B3
        dqg = dot3(scale * do, h_pre.T)
        dw_pseudo = dot3(-dv_new, h_pre.T)
        dkg = dot3(v_new, dh_next.T)
        dA = sanitize(dot3(dw_pseudo, kb.T) + dot3(dv_new, wv.T), c_)
        dkb = dot3(A.T, dw_pseudo); dwv = dot3(A.T, dv_new)
        tmp = sanitize(dot3(dA, A.T), c_)
        dAkk = sanitize((-dot3(A.T, tmp) * (1.0 - cfg.wy_eps)) * strict, c_)   # the ONLY (1-eps) factor
        dx = dkg * kg
        dq3 = dqg * egc
        dk3 = dkb * egc * b + dkg * ekg
        db3 = dkb * egc * k
        dgc3 = dkb * kb * mgc + dqg * qg * mgc - dx * mkg
        dgc_last = jnp.sum(dx * mkg, axis=0) + dec * jnp.sum(dh_next * h_pre, axis=-1) * mdec
        dgc3 = sanitize(dgc3 + last * dgc_last[None, :], c_)
        dq4, dk4, db4, dgc4 = b4_2l_values(q, k, b, gc, dAqk, dAkk * strict, scale=scale, cfg=cfg, dot=dot4)
        dg_ = dot4(triu, dgc3 + dgc4)                                       # B5
        dq_ref[0, 0, c] = sanitize(dq3 + dq4, c_)
        dk_ref[0, 0, c] = sanitize(dk3 + dk4, c_)
        db_ref[0, 0, c] = sanitize(db3 + db4, c_)
        dw_ref[0, 0, c] = sanitize(dwv * v, c_)
        dvraw_ref[0, 0, c] = sanitize(dwv * w, c_)
        dg_ref[0, 0, c] = sanitize(dg_, c_)
    dh0_ref[0, 0] = dh


def backward_mega(q, k, v, w, b, gc, A, Aqk, h_pre, v_new, gc_last, do, dh_final, scale, cfg):
    """All (B,H,nc,...) chunked. Returns dict dq,dk,db,dw,dv_raw,dg,dh0."""
    scale = float(scale)
    bsz, H, nc, T, D = q.shape
    io = pl.BlockSpec((1, 1, nc, T, D), lambda i, h: (i, h, 0, 0, 0))
    sc = pl.BlockSpec((1, 1, nc, T, T), lambda i, h: (i, h, 0, 0, 0))
    hs = pl.BlockSpec((1, 1, D, D), lambda i, h: (i, h, 0, 0))
    hp = pl.BlockSpec((1, 1, nc, D, D), lambda i, h: (i, h, 0, 0, 0))
    gl = pl.BlockSpec((1, 1, nc, D), lambda i, h: (i, h, 0, 0))
    out = pl.pallas_call(
        lambda *r: _kernel(*r, nc=nc, scale=scale, cfg=cfg), grid=(bsz, H),
        in_specs=[io] * 6 + [sc, sc, hp, io, gl, io, hs],
        out_specs=[io] * 6 + [hs],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, nc, T, D), jnp.float32)] * 6
                  + [jax.ShapeDtypeStruct((bsz, H, D, D), jnp.float32)],
        compiler_params=cparams(cfg.vmem_mb), interpret=cfg.interpret,
    )(q, k, b, w, v, gc, A, Aqk, h_pre, v_new, gc_last, do, dh_final)
    dq, dk, db, dw, dv_raw, dg, dh0 = out
    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dg=dg, dh0=dh0)
