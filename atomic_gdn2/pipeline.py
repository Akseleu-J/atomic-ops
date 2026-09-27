"""custom_vjp pipeline. Head-major inputs (B,H,L,D). Long sequences are segmented (h carried via h0/h_final)."""
import jax, jax.numpy as jnp
from .precision import sanitize
from .layout import to_chunks_hm, from_chunks_hm
from .fwd import build_scores, wy_solve_leafbatched, recompute_and_scan_slim
from .bwd import backward_mega

_FINAL_CLIP = 1e4


def _fin(x, dt, cfg):
    if x.dtype == dt and cfg.clip <= _FINAL_CLIP:
        return x                      # kernel already sanitized at cfg.clip
    return sanitize(x, _FINAL_CLIP).astype(dt)


def make_blr_trainable(cfg, scale):
    scale = float(scale)

    def _fwd(q, k, v, w, b, g, h0):
        gc = jnp.cumsum(g.astype(jnp.float32), axis=-2)
        Aqk, Akk = build_scores(q, k, b, gc, scale, cfg)
        A = wy_solve_leafbatched(Akk, cfg)
        cd = recompute_and_scan_slim(Aqk, q, k, v, w, b, gc, A, scale, h0, cfg)
        return cd["o"], cd["h_final"], dict(gc=gc, Aqk=Aqk, A=A, h_pre_all=cd["h_pre_all"],
                                            v_new_all=cd["v_new_all"], gc_last=cd["gc_last"])

    @jax.custom_vjp
    def core(q, k, v, w, b, g, h0):
        o, hf, _ = _fwd(q, k, v, w, b, g, h0)
        return o, hf

    def core_fwd(q, k, v, w, b, g, h0):
        o, hf, res = _fwd(q, k, v, w, b, g, h0)
        res = dict(res); res.update(q=q, k=k, v=v, w=w, b=b, g=g, h0=h0)
        return (o, hf), res

    def core_bwd(res, cts):
        do, dhf = cts
        out = backward_mega(res["q"], res["k"], res["v"], res["w"], res["b"], res["gc"], res["A"],
                            res["Aqk"], res["h_pre_all"], res["v_new_all"], res["gc_last"],
                            do.astype(jnp.float32), dhf.astype(jnp.float32), scale, cfg)
        q, k, v, w, b, g, h0 = (res[n] for n in ("q", "k", "v", "w", "b", "g", "h0"))
        return (_fin(out["dq"], q.dtype, cfg), _fin(out["dk"], k.dtype, cfg),
                _fin(out["dv_raw"], v.dtype, cfg), _fin(out["dw"], w.dtype, cfg),
                _fin(out["db"], b.dtype, cfg), _fin(out["dg"], g.dtype, cfg),
                _fin(out["dh0"], h0.dtype, cfg))

    core.defvjp(core_fwd, core_bwd)

    def call(q, k, v, w, b, g, h0=None):
        """q..g: (B,H,L,D), L % bt == 0. Returns (o (B,H,L,D), h_final (B,H,D,D))."""
        bsz, H, L, D = q.shape
        if L % cfg.bt:
            raise ValueError(f"L={L} must be divisible by bt={cfg.bt}")
        nc = L // cfg.bt
        h = jnp.zeros((bsz, H, D, D), jnp.float32) if h0 is None else h0
        outs = []
        for s in range(0, nc, cfg.max_chunks_per_call):
            e = min(nc, s + cfg.max_chunks_per_call)
            a, c = s * cfg.bt, e * cfg.bt
            o_ch, h = core(*(to_chunks_hm(t[:, :, a:c], e - s, cfg.bt) for t in (q, k, v, w, b, g)), h)
            outs.append(from_chunks_hm(o_ch))
        return (outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=2)), h

    return call
