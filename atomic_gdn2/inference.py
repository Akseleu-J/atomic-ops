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
import numpy as np

import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from .precision import exp_nonpos, sanitize
from .layout import cparams
from .config import BLRConfig

__all__ = ["decode_step", "chunk_forward_inference", "LayerInputs",
           "check_document_boundaries", "cu_seqlens_to_segment_ids"]

import dataclasses as _dc


@_dc.dataclass(frozen=True)
class LayerInputs:
    """Named input container (audit Stage 3.1, fixes B3 arg-order confusion).

    All tensors are head-ready: q/k (B, H, Dk); v/w (B, Hv, Dv); g (B, H, Dk)
    per-token log-decay (<= 0); b (B, H, Dk). Both entry points accept this
    container instead of six positional tensors, making argument order
    impossible to get wrong.
    """
    q: jax.Array
    k: jax.Array
    v: jax.Array
    w: jax.Array
    b: jax.Array
    g: jax.Array

    def validate(self, Dk: int, Dv: int):
        B = self.q.shape[0]
        want = {"q": (B, self.q.shape[1], Dk), "k": (B, self.q.shape[1], Dk),
                "v": (B, self.v.shape[1], Dv), "w": (B, self.v.shape[1], Dv),
                "b": (B, self.q.shape[1], Dk), "g": (B, self.q.shape[1], Dk)}
        for name, arr in (("q", self.q), ("k", self.k), ("v", self.v),
                          ("w", self.w), ("b", self.b), ("g", self.g)):
            if tuple(arr.shape) != want[name]:
                raise ValueError(f"{name}: expected {want[name]}, got {tuple(arr.shape)}")

    def as_tuple(self):
        return (self.q, self.k, self.v, self.w, self.b, self.g)

PAD_N = 128  # segment-block padding for TPU Mosaic block-shape legality


def _materialize(x, interpret: bool):
    """Turn eager interpret-mode pallas outputs into plain jnp arrays.

    In eager interpret mode the outputs behave like Refs (leading-axis unwrap on
    fancy indexing), so they are round-tripped through numpy. Compiled (TPU)
    outputs are device arrays: no round-trip, which also removes a device->host
    copy per decode step. Under jax.jit the values are tracers: a numpy
    round-trip is impossible (TPU gates A_decode_jittable / A_prefill_jittable
    failed exactly here), so they pass through unchanged.
    """
    if not interpret or isinstance(x, jax.core.Tracer):
        return x
    return jnp.asarray(np.asarray(x))


def _concrete(x):
    """numpy view of x, or None when x is traced (validation then defers to the
    caller: see check_document_boundaries)."""
    if isinstance(x, jax.core.Tracer):
        return None
    try:
        return np.asarray(x)
    except (TypeError, ValueError):
        return None


def cu_seqlens_to_segment_ids(cu_seqlens, bsz: int, L: int):
    """(n_docs+1,) shared or (B, n_docs+1) per-row offsets -> (B, L) int32 ids.

    A boundary token STARTS its document (`>=` semantics, audit A3).
    """
    cs = jnp.asarray(cu_seqlens)
    if cs.ndim == 1:
        cs = jnp.broadcast_to(cs[None, :], (bsz, cs.shape[0]))
    if cs.shape[0] != bsz:
        raise ValueError(f"cu_seqlens rows {cs.shape[0]} != batch {bsz}")
    if cs.shape[1] <= 2:
        return jnp.zeros((bsz, L), jnp.int32)               # single document
    bounds = cs[:, 1:-1]                                     # (B, n_int)
    pos = jnp.arange(L)
    return (pos[None, :, None] >= bounds[:, None, :]).sum(axis=-1).astype(jnp.int32)


def check_document_boundaries(segment_ids, bt: int):
    """Raise ValueError if a document boundary falls inside a FULL chunk.

    Boundaries are honored at chunk granularity by the chunkwise kernel (A2);
    the tail after the last full chunk is decoded token by token and accepts
    any boundary. chunk_forward_inference calls this automatically for concrete
    inputs. Under jax.jit the ids are traced and cannot be inspected, so a
    serving loop that jits prefill with dynamic boundaries must call this on
    the host before dispatch (pass cu_seqlens through cu_seqlens_to_segment_ids
    first).
    """
    ids = _concrete(segment_ids)
    if ids is None:
        raise ValueError("check_document_boundaries needs concrete (non-traced) ids")
    ids = ids.reshape(ids.shape[0], -1)
    bsz, L = ids.shape
    n_full = L // bt
    if n_full == 0:
        return
    chunks = ids[:, :n_full * bt].reshape(bsz, n_full, bt)
    bad = (chunks != chunks[:, :, :1]).any(axis=-1)          # (B, n_full)
    if bad.any():
        b_idx, c_idx = np.argwhere(bad)[0]
        raise ValueError(
            f"document boundary inside chunk {int(c_idx)} of batch row "
            f"{int(b_idx)} (tokens start at {int(c_idx) * bt}): split the "
            "sequence at document boundaries before calling "
            "chunk_forward_inference (chunk-granularity resets only)")


def _decode_body(state_ref, q_ref, k_ref, v_ref, w_ref, b_ref, g_ref,
                 a_log_ref, dt_bias_ref, o_ref, new_ref, *,
                 scale: float, stability_tau, fuse_prologue: bool = False):
    S = state_ref[0].astype(jnp.float32)
    q = q_ref[0].astype(jnp.float32)
    k = k_ref[0].astype(jnp.float32)
    v = v_ref[0].astype(jnp.float32)
    w = w_ref[0].astype(jnp.float32)
    b = b_ref[0].astype(jnp.float32)
    g = g_ref[0].astype(jnp.float32)
    if fuse_prologue:
        # activations fused into the kernel (prologue-fusion method adapted
        # from the deltanet-tpu inference stack) using the CHECKPOINT gate
        # construction: g = -exp(A_log) * softplus(f_proj(x) + dt_bias),
        # b/w are sigmoid logits, q/k are l2-normalized in-kernel.
        q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6)
        k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
        b = jax.nn.sigmoid(b)
        w = jax.nn.sigmoid(w)
        a_log = a_log_ref[0].astype(jnp.float32)      # (H,)
        dt = dt_bias_ref[0].astype(jnp.float32)       # (K,)
        g = -jnp.exp(a_log)[..., None] * jax.nn.softplus(g + dt[None, :])
    h = S * jnp.exp(jnp.minimum(g, 0.0))[..., None]
    erase = jnp.sum((b * k)[..., None] * h, axis=1)
    v_new = w * v - erase
    h = h + k[..., None] * v_new[..., None, :]
    o = jnp.sum(h * (q * scale)[..., None], axis=1)
    if stability_tau is not None:
        # per-head Frobenius norm (B1): heads are independent, one head must
        # not suppress the others; matches the prefill cap semantics.
        nrm = jnp.sqrt(jnp.sum(h * h, axis=(-2, -1)))      # (H,)
        h = h * jnp.minimum(1.0, stability_tau / jnp.maximum(nrm, 1e-12))[..., None, None]
    # A5 policy (owner decision): a non-finite entry never persists — the
    # in-kernel cleanup replaces NaN/Inf state and output entries with zeros
    # (the affected row effectively restarts from a clean state). Done
    # IN-KERNEL: post-pallas jnp.where on interpret-mode outputs trips ref
    # indexing semantics.
    h = jnp.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    o = jnp.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)
    o_ref[0] = o.astype(o_ref.dtype)
    new_ref[0] = h.astype(new_ref.dtype)


def decode_step(q=None, k=None, v=None, w=None, b=None, g=None, state=None,
                scale: float = 1.0, reset=None, state_dtype=jnp.float32,
                stability_tau=None, fuse_prologue: bool = False, a_log=None,
                dt_bias=None, inputs: "LayerInputs | None" = None,
                cfg: BLRConfig | None = None):
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
    if inputs is not None:
        q, k, v, w, b, g = inputs.as_tuple()
    if q is None or state is None:
        raise ValueError("pass either inputs=LayerInputs(...) or q,k,v,w,b,g,state")
    bsz, H, D = q.shape
    if not isinstance(scale, (int, float)) or isinstance(scale, jax.Array):
        raise ValueError(
            "scale must be a concrete Python float (B4): traced values cannot "
            "enter the Pallas kernel; pass it as a compile-time constant")
    if stability_tau is not None and not isinstance(stability_tau, (int, float)):
        raise ValueError(
            "stability_tau must be a concrete Python float (B4)")
    if state.shape != (bsz, H, D, D):
        raise ValueError(f"state must be (B,H,D,D), got {state.shape}")
    st = state.astype(state_dtype)
    if reset is not None:
        # per-row reset happens OUTSIDE the kernel: a (B,) scalar block would
        # violate the TPU Mosaic block-shape rule (the last two dims of a block
        # must divide (8, 128) or equal the array's dims).
        rst = jnp.asarray(reset).reshape(bsz).astype(jnp.int32)
        st = jnp.where(rst[:, None, None, None] > 0, jnp.zeros_like(st), st)
    if fuse_prologue:
        if a_log is None or dt_bias is None:
            raise ValueError(
                "fuse_prologue=True requires a_log (H,) and dt_bias (K,) from "
                "the checkpoint: the kernel then computes "
                "g = -exp(A_log) * softplus(g_raw + dt_bias) in-kernel")
        a_log = jnp.asarray(a_log).reshape(H)
        dt_bias = jnp.asarray(dt_bias).reshape(D)
    # grid (B,); blocks span the FULL (H, .) extent of their last two axes, so
    # they are legal on TPU Mosaic for any H, D (block dims == array dims).
    grid = (bsz,)
    ss = pl.BlockSpec((1, H, D, D), lambda i: (i, 0, 0, 0))
    vs = pl.BlockSpec((1, H, D), lambda i: (i, 0, 0))
    # gate params as (1, X) full-array 2-D blocks (legal: block == array dims);
    # body reads them with the same ref[0] pattern as every other ref
    als = pl.BlockSpec((1, H), lambda i: (0, 0))
    dts = pl.BlockSpec((1, D), lambda i: (0, 0))
    a_log_in = (jnp.asarray(a_log).reshape(1, H) if fuse_prologue
                else jnp.zeros((1, H), jnp.float32))
    dt_in = (jnp.asarray(dt_bias).reshape(1, D) if fuse_prologue
             else jnp.zeros((1, D), jnp.float32))
    interpret = cfg.interpret if cfg is not None else jax.default_backend() != "tpu"
    o, new = pl.pallas_call(
        lambda *r: _decode_body(*r, scale=float(scale), stability_tau=stability_tau,
                                fuse_prologue=fuse_prologue),
        grid=grid,
        in_specs=[ss, vs, vs, vs, vs, vs, vs, als, dts],
        out_specs=[vs, ss],
        out_shape=[jax.ShapeDtypeStruct((bsz, H, D), q.dtype),
                   jax.ShapeDtypeStruct((bsz, H, D, D), state_dtype)],
        compiler_params=cparams(cfg.vmem_mb if cfg is not None else 150.0),
        interpret=interpret,
    )(st, q, k, v, w, b, g, a_log_in, dt_in)
    return _materialize(o, interpret), _materialize(new, interpret)


def _cd_inference_body(q_ref, k_ref, v_ref, w_ref, b_ref, gc_ref, a_ref, aqk_ref,
                       h0_ref, seg_ref, o_ref, hf_ref, *, nc: int, scale: float,
                       cfg: BLRConfig, seg_mode: bool):
    from .precision import make_dot
    dot = make_dot(cfg.dot_mode)
    h = h0_ref[0, 0].astype(jnp.float32)
    segs = seg_ref[0][0] if seg_mode else None  # row 0 of the (8, PAD_N) block = this batch row
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


def chunk_forward_inference(q=None, k=None, v=None, w=None, b=None, g=None,
                            scale: float = 1.0, cfg: BLRConfig | None = None,
                            h0=None, segment_ids=None, cu_seqlens=None,
                            state_dtype=jnp.float32, stability_tau=None,
                            inputs: "LayerInputs | None" = None):
    """Forward-only prefill with bounded outputs: returns (o, h_final) only.

    Same math and layouts as pipeline.make_blr_trainable's forward (head-major
    (B,H,L,D), segments of cfg.max_chunks_per_call chunks with the state
    carried between them), but no training residuals are produced and the
    recurrent state is returned in state_dtype. Any L is accepted: full chunks
    run through the chunkwise kernel, the L % bt tail through decode steps.

    segment_ids: optional (B, L) int tensor of document IDs. A document
    boundary resets the state. Inside the full-chunk region boundaries must sit
    on chunk starts (a boundary inside a chunk raises ValueError — the caller
    splits documents); in the tail any boundary is honored. cu_seqlens:
    optional (n_docs+1,) shared or (B, n_docs+1) per-row token offsets,
    converted with cu_seqlens_to_segment_ids and validated the same way.
    h0 belongs to each row's first document, whatever its id.
    stability_tau: optional cap on the Frobenius norm of the returned state
    (inference-only; the training path is never capped).

    jax.jit: supported. Boundary and g-domain validation need concrete values
    and are skipped for traced inputs — validate on the host with
    check_document_boundaries before dispatching a jitted prefill.
    """
    if inputs is not None:
        q, k, v, w, b, g = inputs.as_tuple()
    bsz, H, L, D = q.shape
    # B7 fix (0.4.0): arbitrary length — full chunks go through the chunkwise
    # kernel, the tail (L % bt tokens) goes through sequential decode steps.
    # The old strict `L % bt == 0` requirement is gone; the vLLM layer can
    # still schedule chunk-aligned sequences for maximum throughput.
    # B2 fix: decode accepts bfloat16 inputs, prefill crashed on ref dtype
    # swap — normalize to float32 at the API boundary (compute is fp32 anyway)
    q, k, v, w, b = (jnp.asarray(t).astype(jnp.float32) for t in (q, k, v, w, b))
    n_full = L // cfg.bt
    r_tail = L - n_full * cfg.bt
    scale = float(scale)
    # A6 fix (audit): prefill never validated the g domain — g=-8 (beyond
    # g_max) gave 58% and g>0 gave 205% error silently. Clamp identically to
    # the decode path (g <= 0) and raise loudly when outside the layer's
    # documented domain [-g_max, 0].
    g = jnp.minimum(jnp.asarray(g).astype(jnp.float32), 0.0)
    g_max_abs = float(cfg.g_max)
    g_np = _concrete(g)                          # concrete only; skipped under jit
    if g_np is not None and bool((g_np < -g_max_abs - 1e-6).any()):
        raise ValueError(
            f"g outside the layer domain: min(g) < -g_max={g_max_abs} "
            "(the layer constructs g = -min(softplus(...), g_max); clamp or "
            "fix the projection before calling)")
    h = jnp.zeros((bsz, H, D, D), jnp.float32) if h0 is None else h0.astype(jnp.float32)

    n_full = L // cfg.bt
    L_full = n_full * cfg.bt
    r_tail = L - L_full
    if cu_seqlens is not None:
        # A3 fix: any number of documents, any batch size. cu_seqlens is either
        # 1-D (shared by all rows) or (B, n_docs+1). A boundary token STARTS a
        # new document (audit: the earlier `>` semantics shifted resets late).
        if segment_ids is not None:
            raise ValueError("pass either segment_ids or cu_seqlens, not both")
        segment_ids = cu_seqlens_to_segment_ids(cu_seqlens, bsz, L)
    seg_chunk = None
    if segment_ids is not None:
        seg = jnp.asarray(segment_ids).reshape(bsz, L).astype(jnp.int32)
        # A2 (owner decision): a boundary INSIDE a full chunk raises. Checked
        # AFTER the cu_seqlens conversion (TPU gate A_reject_cu_seqlen_inside_
        # chunk: the old order validated only explicit segment_ids). The tail
        # is exempt: decode steps handle per-token resets. Traced ids (jit)
        # cannot be inspected here — see check_document_boundaries.
        if _concrete(seg) is not None:
            check_document_boundaries(seg, cfg.bt)
        seg_chunk = seg[:, :L_full:cfg.bt][:, :n_full]  # (B, n_full) seg at chunk start
    if seg_chunk is not None and min(cfg.max_chunks_per_call, n_full) > PAD_N:
        raise ValueError(f"max_chunks_per_call must be <= {PAD_N} with document ids "
                         "(one launch's chunk ids must fit the padded segment block)")

    outs = []
    for s in range(0, n_full, cfg.max_chunks_per_call):
        e = min(n_full, s + cfg.max_chunks_per_call)
        n = e - s
        sl = lambda t: t[:, :, s * cfg.bt:e * cfg.bt].reshape(bsz, H, n, cfg.bt, D)
        q_c, k_c, v_c, w_c, b_c = sl(q), sl(k), sl(v), sl(w), sl(b)
        # gc is the PER-CHUNK cumulative decay (pipeline computes it on the
        # chunked tensor, axis=-2 == the bt axis), not over the whole sequence.
        gc_c = jnp.cumsum(sl(g).astype(jnp.float32), axis=-2)
        # A1 fix: per-row (B, 8, PAD_N) TPU-legal block; row 0 of each block =
        # this batch row's chunk ids, rows 1-7 are padding. Built PER LAUNCH
        # from this launch's chunks (TPU gate A_segment_offsets_in_second_launch:
        # one buffer for chunks 0..n-1 made every later launch read the first
        # launch's ids).
        seg_padded = jnp.zeros((bsz, 8, PAD_N), jnp.int32)
        if seg_chunk is not None:
            seg_cur = seg_chunk[:, s:e]
            seg_padded = seg_padded.at[:, 0, :n].set(seg_cur)
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
        # full-array block: legal on TPU Mosaic for any (B, n) shape
        # A1 fix: per-ROW segment blocks. A (B, n) full-array block forced
        # every program to read row 0's boundaries (audit blocker A1: with
        # B>1 all rows used row 0's document layout, silently mixing states).
        # TPU Mosaic legality: the block must be (1, 8, 128) — axes padded to
        # (8, 128) so the divisibility rule holds; row 0 of the padded block
        # is this program's batch row, rows 1-7 are zero padding.
        sg = pl.BlockSpec((1, 8, PAD_N), lambda i, hh: (i, 0, 0))
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
        )(q_c, k_c, v_c, w_c, b_c, gc_c, A, Aqk, h_in.astype(jnp.float32), seg_padded)
        # materialize eager interpret-mode Refs to plain arrays (see _materialize)
        h = _materialize(hf, cfg.interpret)
        outs.append(from_chunks_local(_materialize(o_ch, cfg.interpret)))
    o = (jnp.concatenate(outs, axis=2) if len(outs) > 1
         else (outs[0] if outs else None))

    # ---- B7: decode the tail (L % bt tokens) with per-token doc resets ----
    if r_tail > 0:
        tail_resets = None
        if seg_chunk is not None:
            seg_full = seg
            # id preceding each tail token. With no full chunks the first tail
            # token is compared with ITSELF: h0 belongs to the first document
            # whatever its id (TPU gate A_tail_only_preserves_nonzero_h0: a
            # constant 0 here reset h0 for any first-document id != 0).
            first_prev = (seg_full[:, L_full - 1:L_full] if L_full > 0
                          else seg_full[:, :1])
            prev = jnp.concatenate([first_prev, seg_full[:, L_full:-1]], axis=1)
            tail_resets = (seg_full[:, L_full:] != prev).astype(jnp.int32)  # (B, r)
        outs_tail = []
        for j in range(r_tail):
            t_abs = L_full + j
            rst = (tail_resets[:, j] if tail_resets is not None else None)
            o_t, h = decode_step(q[:, :, t_abs], k[:, :, t_abs], v[:, :, t_abs],
                                 w[:, :, t_abs], b[:, :, t_abs], g[:, :, t_abs],
                                 h.astype(state_dtype), scale, reset=rst,
                                 state_dtype=state_dtype, cfg=cfg)
            outs_tail.append(o_t)
        o_tail = jnp.stack(outs_tail, axis=2)                    # (B, H, r, D)
        o = (o_tail if o is None else jnp.concatenate([o, o_tail], axis=2))

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
