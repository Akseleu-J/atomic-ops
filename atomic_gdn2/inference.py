"""Inference kernels: single-token decode step and slim forward-only prefill.

Math follows reference/ref.py::token_serial_ref exactly (per token):
    Sbar = Diag(exp(min(g,0))) S
    erase = (b*k) @ Sbar                       # (b*k) . Sbar, per value channel
    v_new = w*v - erase
    S = Sbar + k (v_new)^T
    o = S^T (scale*q)

Deltas vs the training path (fwd/cd_slim.py): inference keeps only (o, final
state) — no h_pre_all / v_new_all / gc_last residuals are materialized — and
adds the serving features documented per function: per-row reset, segment
(document) resets at chunk boundaries, bfloat16 state storage, and an optional
Frobenius-norm stability cap on the returned state. Mid-chunk document
boundaries are not supported at this level: packed batches should split so a
chunk never spans two documents (the vLLM layer owns that scheduling).

All kernels honor cfg.interpret (CPU emulation) and run on TPU otherwise.
"""
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from .precision import exp_nonpos, sanitize
from .layout import cparams
from .config import BLRConfig

__all__ = ["decode_step", "chunk_forward_inference"]


def _decode_body(state_ref, q_ref, k_ref, v_ref, w_ref, b_ref, g_ref, reset_ref,
                 o_ref, new_ref, *, scale: float, stability_tau,
                 fuse_prologue: bool = False, g_max: float = 5.0):
    S = state_ref[0, 0].astype(jnp.float32)
    q = q_ref[0, 0].astype(jnp.float32)
    k = k_ref[0, 0].astype(jnp.float32)
    v = v_ref[0, 0].astype(jnp.float32)
    w = w_ref[0, 0].astype(jnp.float32)
    b = b_ref[0, 0].astype(jnp.float32)
    g = g_ref[0, 0].astype(jnp.float32)
    if fuse_prologue:
        # activations fused into the kernel (deltanet-tpu's prologue method,
        # adapted to our domain contract): raw q/k are l2-normalized, raw
        # gate logits are squashed here, saving one HBM roundtrip per tensor.
        q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
        b = jax.nn.sigmoid(b)
        w = jax.nn.sigmoid(w)
        g = -jnp.minimum(jax.nn.softplus(g), g_max)
    S = jnp.where(reset_ref[0] > 0, 0.0, S)
    h = S * jnp.exp(jnp.minimum(g, 0.0))[..., None]
    erase = jnp.sum((b * k)[..., None] * h, axis=0)
    v_new = w * v - erase
    h = h + k[..., None] * v_new[None, :]
    o = jnp.sum(h * (q * scale)[:, None], axis=0)
    if stability_tau is not None:
        nrm = jnp.sqrt(jnp.sum(h * h))
        h = h * jnp.minimum(1.0, stability_tau / jnp.maximum(nrm, 1e-12))
    o_ref[0, 0] = o.astype(o_ref.dtype)
    new_ref[0, 0] = h.astype(new_ref.dtype)


def decode_step(q, k, v, g, b, w, state, scale: float = 1.0, reset=None,
                state_dtype=jnp.float32, stability_tau=None,
                fuse_prologue: bool = False, cfg: BLRConfig | None = None):
    """Single-token recurrence for serving.

    q, k, v, w, b, g: (B, H, D); g is THIS token's log-decay (g <= 0 by the
    layer's domain construction; positive values are clamped to 0 with the
    same semantics as token_serial_ref). state: (B, H, D, D) in state_dtype.
    reset: (B,) int/bool — zero the incoming state for those rows (vLLM
    preemption / document boundaries).

    fuse_prologue: when True, q/k are RAW (l2-normalized in-kernel), b/w are
    raw gate logits (sigmoid in-kernel) and g is the pre-softplus projection
    (g = -min(softplus(g), g_max) in-kernel) — activations never materialize
    in HBM (deltanet-tpu's prologue-fusion method adapted to our domain
    contract). Requires cfg.

    Returns (o (B, H, D), new_state (B, H, D, D) in state_dtype). Compute is
    float32; storage dtype is state_dtype ("bfloat16" halves state bytes; the
    loss is a single rounding of the returned state, validated for the
    channel-wise GDN-2 update).
    """
    bsz, H, D = q.shape
    if state.shape != (bsz, H, D, D):
        raise ValueError(f"state must be (B,H,D,D), got {state.shape}")
    st = state.astype(state_dtype)
    rst = (jnp.zeros((bsz,), jnp.int32) if reset is None
           else jnp.asarray(reset).astype(jnp.int32).reshape(bsz))
    grid = (bsz, H)
    ss = pl.BlockSpec((1, 1, D, D), lambda i, h: (i, h, 0, 0))
    vs = pl.BlockSpec((1, 1, D), lambda i, h: (i, h, 0))
    rs = pl.BlockSpec((1,), lambda i, h: (i,))
    if fuse_prologue and cfg is None:
        raise ValueError("fuse_prologue=True requires cfg (needs g_max)")
    o, new = pl.pallas_call(
        lambda *r: _decode_body(*r, scale=float(scale), stability_tau=stability_tau,
                                fuse_prologue=fuse_prologue,
                                g_max=(cfg.g_max if cfg is not None else 5.0)),
        grid=grid,
        in_specs=[ss, vs, vs, vs, vs, vs, vs, rs],
        out_specs=[vs, ss],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, D), q.dtype),
                   jax.ShapeDtypeStruct((bsz, H, D, D), state_dtype)],
        compiler_params=cparams(cfg.vmem_mb if cfg is not None else 150.0),
        interpret=(cfg.interpret if cfg is not None else jax.default_backend() != "tpu"),
    )(st, q, k, v, w, b, g, rst)
    return o, new


def _cd_inference_body(q_ref, k_ref, v_ref, w_ref, b_ref, gc_ref, a_ref, aqk_ref,
                       h0_ref, seg_ref, o_ref, hf_ref, *, nc: int, scale: float,
                       cfg: BLRConfig, seg_mode: bool):
    from .precision import make_dot
    dot = make_dot(cfg.dot_mode)
    h = h0_ref[0, 0].astype(jnp.float32)
    segs = seg_ref[0] if seg_mode else None
    for c in range(nc):
        if seg_mode and c > 0:
            h = jnp.where(segs[c] != segs[c - 1], jnp.zeros_like(h), h)
        q = q_ref[0, 0, c].astype(jnp.float32)
        k = k_ref[0, 0, c].astype(jnp.float32)
        v = v_ref[0, 0, c].astype(jnp.float32)
        w = w_ref[0, 0, c].astype(jnp.float32)
        b = b_ref[0, 0, c].astype(jnp.float32)
        gc = gc_ref[0, 0, c].astype(jnp.float32)
        A = a_ref[0, 0, c].astype(jnp.float32)
        Aq = aqk_ref[0, 0, c].astype(jnp.float32)
        egc, _ = exp_nonpos(gc)
        wp = sanitize(dot(A, b * k * egc), cfg.clip)
        u = sanitize(dot(A, w * v), cfg.clip)
        gc_last = gc[cfg.bt - 1]
        ekg, _ = exp_nonpos(gc_last[None, :] - gc)
        kg = sanitize(k * ekg, cfg.clip)
        qg = sanitize(q * egc, cfg.clip)
        v_new = u - dot(wp, h)
        o_c = scale * dot(qg, h) + dot(Aq, v_new)
        dec, _ = exp_nonpos(gc_last)
        h = sanitize(h * dec[:, None] + dot(kg.T, v_new), cfg.clip)
        o_ref[0, 0, c] = sanitize(o_c, cfg.clip)
    hf_ref[0, 0] = h


def chunk_forward_inference(q, k, v, w, b, g, scale: float, cfg: BLRConfig,
                            h0=None, segment_ids=None, cu_seqlens=None,
                            state_dtype=jnp.float32, stability_tau=None):
    """Forward-only prefill with bounded outputs: returns (o, h_final) only.

    Same math and layouts as pipeline.make_blr_trainable's forward (head-major
    (B,H,L,D), L % cfg.bt == 0, segments of cfg.max_chunks_per_call chunks with
    the state carried between them), but no training residuals are produced and
    the recurrent state is returned in state_dtype.

    segment_ids: optional (B, L) int tensor of document IDs. A document
    boundary resets the state; boundaries are honored at CHUNK granularity in
    this level (a chunk spanning two documents must be split by the caller —
    the vLLM layer owns that scheduling). cu_seqlens: optional 1-D token
    offsets; converted to per-token segment ids internally (same chunk
    granularity caveat). initial_state is assumed to belong
    to the first chunk's document.
    stability_tau: optional cap on the Frobenius norm of the returned state
    (inference-only; the training path is never capped).
    """
    bsz, H, L, D = q.shape
    if L % cfg.bt:
        raise ValueError(f"L={L} must be divisible by bt={cfg.bt}")
    nc = L // cfg.bt
    scale = float(scale)
    h = jnp.zeros((bsz, H, D, D), jnp.float32) if h0 is None else h0.astype(jnp.float32)

    if cu_seqlens is not None:
        # document boundaries at token granularity -> per-token segment ids
        # (chunk-granularity resets as with segment_ids; a chunk spanning a
        # boundary must still be split by the caller)
        if segment_ids is not None:
            raise ValueError("pass either segment_ids or cu_seqlens, not both")
        cs = jnp.asarray(cu_seqlens).reshape(-1)
        pos = jnp.arange(L)[None, :]
        bounds = cs[None, 1:-1]  # interior boundaries only
        segment_ids = (pos >= bounds[:, None]).sum(axis=0)[None, :].astype(jnp.int32)
    seg_chunk = None
    if segment_ids is not None:
        seg = jnp.asarray(segment_ids).reshape(bsz, L)
        seg_chunk = seg[:, ::cfg.bt]  # (B, nc) document id at each chunk start

    outs = []
    for s in range(0, nc, cfg.max_chunks_per_call):
        e = min(nc, s + cfg.max_chunks_per_call)
        n = e - s
        sl = lambda t: t[:, :, s * cfg.bt:e * cfg.bt].reshape(bsz, H, n, cfg.bt, D)
        q_c, k_c, v_c, w_c, b_c = sl(q), sl(k), sl(v), sl(w), sl(b)
        # gc is the PER-CHUNK cumulative decay (pipeline computes it on the
        # chunked tensor, axis=-2 == the bt axis), not over the whole sequence.
        gc_c = jnp.cumsum(sl(g).astype(jnp.float32), axis=-2)
        if seg_chunk is not None:
            seg_cur = seg_chunk[:, s:e]
            if s > 0:
                # document changed at the segment boundary -> zero those rows
                boundary = seg_cur[:, 0] != seg_chunk[:, s - 1]
                h = jnp.where(boundary[:, None, None, None], 0.0, h)
            h_in = h
        else:
            seg_cur = None
            h_in = h
        Aqk, Akk = build_scores_local(q_c, k_c, b_c, gc_c, scale, cfg)
        A = solve_local(Akk, cfg)
        io = pl.BlockSpec((1, 1, n, cfg.bt, D), lambda i, hh: (i, hh, 0, 0, 0))
        sc = pl.BlockSpec((1, 1, n, cfg.bt, cfg.bt), lambda i, hh: (i, hh, 0, 0, 0))
        hs = pl.BlockSpec((1, 1, D, D), lambda i, hh: (i, hh, 0, 0))
        sg = (pl.BlockSpec((1, n), lambda i, hh: (i, 0)) if seg_cur is not None
              else pl.BlockSpec((1, n), lambda i, hh: (0, 0)))
        o_ch, hf = pl.pallas_call(
            lambda *r: _cd_inference_body(*r, nc=n, scale=scale, cfg=cfg,
                                          seg_mode=seg_cur is not None),
            grid=(bsz, H),
            in_specs=[io, io, io, io, io, io, sc, sc, hs, sg],
            out_specs=[io, hs],
            out_shape=[jax.ShapeDtypeStruct((bsz, H, n, cfg.bt, D), q.dtype),
                       jax.ShapeDtypeStruct((bsz, H, D, D), jnp.float32)],
            compiler_params=cparams(cfg.vmem_mb),
            interpret=cfg.interpret,
        )(q_c, k_c, v_c, w_c, b_c, gc_c, A, Aqk, h_in.astype(jnp.float32),
          seg_cur if seg_cur is not None else jnp.zeros((bsz, n), jnp.int32))
        h = hf
        outs.append(from_chunks_local(o_ch))
    o = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=2)
    if stability_tau is not None:
        nrm = jnp.sqrt(jnp.sum(h.astype(jnp.float32) ** 2, axis=(-2, -1), keepdims=True))
        h = h * jnp.minimum(1.0, stability_tau / jnp.maximum(nrm, 1e-12))
    return o, h.astype(state_dtype)


def build_scores_local(q, k, b, gc, scale, cfg):
    from .fwd import build_scores
    return build_scores(q, k, b, gc, scale, cfg)


def solve_local(Akk, cfg):
    from .fwd import wy_solve_leafbatched
    return wy_solve_leafbatched(Akk, cfg)


def from_chunks_local(t):
    b, h, nc, bt, d = t.shape
    return t.reshape(b, h, nc * bt, d)
