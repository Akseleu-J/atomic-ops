"""v0.3 inference kernels vs the ground-truth token serial reference.

Everything runs in JAX Pallas interpret mode (cfg.interpret=True -> CPU
emulation), so the suite is hardware-independent; on TPU the same kernels are
compiled. Numerics gates use tight tolerances because interpret mode and the
serial reference share float32 semantics (decode) or the chunked fp32 contract
(prefill, where the chunked pipeline differs from serial only through the
documented fp32 sanitize/clip contract).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atomic_gdn2.config import make_cfg
from atomic_gdn2.inference import decode_step, chunk_forward_inference
from atomic_gdn2.reference import token_serial_ref


def _inputs(key, B, H, D):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    q = jax.random.normal(k1, (B, H, D)) * 0.5
    k = jax.random.normal(k2, (B, H, D)) * 0.5
    v = jax.random.normal(k3, (B, H, D))
    w = jax.nn.sigmoid(jax.random.normal(k4, (B, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(k5, (B, H, D)))
    return q, k, v, w, b


def _g(key, B, H, D, magnitude=1.0):
    return -jnp.exp(jax.random.normal(key, (B, H, D)) * 0.39 - 1.0) * magnitude


def _to_bld(t):  # head-major (B,H,D) -> token serial layout slice (B,1,H,D)
    return t[:, None, :, :]


def test_decode_matches_serial_clean():
    """Decode loop == token_serial_ref, token by token, same initial state."""
    key = jax.random.PRNGKey(1)
    B, H, D, L = 2, 3, 16, 24
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    scale = 0.7
    q = jax.random.normal(key, (B, L, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)

    o_ref, hf_ref = token_serial_ref(q, k, v, g, b, w, scale)

    # head-major per-token slices: (B,H,L,D) -> token t is [:, :, t, :]
    qm = jnp.transpose(q, (0, 2, 1, 3))
    km = jnp.transpose(k, (0, 2, 1, 3))
    vm = jnp.transpose(v, (0, 2, 1, 3))
    wm = jnp.transpose(w, (0, 2, 1, 3))
    bm = jnp.transpose(b, (0, 2, 1, 3))
    gm = jnp.transpose(g, (0, 2, 1, 3))
    state = jnp.zeros((B, H, D, D), jnp.float32)
    outs = []
    for t in range(L):
        o, state = decode_step(qm[:, :, t], km[:, :, t], vm[:, :, t],
                               gm[:, :, t], bm[:, :, t], wm[:, :, t],
                               state, scale, cfg=cfg)
        outs.append(o)
    o_dec = jnp.stack(outs, axis=1)  # (B, L, H, D)
    np.testing.assert_allclose(np.asarray(o_dec), np.asarray(o_ref),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(state), np.asarray(hf_ref),
                               rtol=1e-4, atol=1e-4)


def test_decode_reset_matches_separate_runs():
    """reset=True at a document boundary == running the docs separately."""
    key = jax.random.PRNGKey(2)
    B, H, D, L1, L2 = 2, 2, 16, 10, 14
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    scale = 1.0
    q = jax.random.normal(key, (B, L1 + L2, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L1 + L2, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L1 + L2, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L1 + L2, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L1 + L2, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L1 + L2, H, D)) * 0.39 - 1.0)
    o_ref1, hf_ref1 = token_serial_ref(q[:, :L1], k[:, :L1], v[:, :L1],
                                       g[:, :L1], b[:, :L1], w[:, :L1], scale)
    o_ref2, hf_ref2 = token_serial_ref(q[:, L1:], k[:, L1:], v[:, L1:],
                                       g[:, L1:], b[:, L1:], w[:, L1:], scale)

    qm = jnp.transpose(q, (0, 2, 1, 3))
    km = jnp.transpose(k, (0, 2, 1, 3))
    vm = jnp.transpose(v, (0, 2, 1, 3))
    wm = jnp.transpose(w, (0, 2, 1, 3))
    bm = jnp.transpose(b, (0, 2, 1, 3))
    gm = jnp.transpose(g, (0, 2, 1, 3))
    state = jnp.zeros((B, H, D, D), jnp.float32)
    outs = []
    for t in range(L1 + L2):
        reset = jnp.ones((B,), jnp.int32) if t == L1 else None
        o, state = decode_step(qm[:, :, t], km[:, :, t], vm[:, :, t],
                               gm[:, :, t], bm[:, :, t], wm[:, :, t],
                               state, scale, reset=reset, cfg=cfg)
        outs.append(o)
    o_dec = jnp.stack(outs, axis=1)
    o_ref = jnp.concatenate([o_ref1, o_ref2], axis=1)
    np.testing.assert_allclose(np.asarray(o_dec), np.asarray(o_ref),
                               rtol=1e-4, atol=1e-4)


def test_chunk_inference_matches_serial():
    """Slim prefill (chunked) == token serial reference on the same tokens."""
    key = jax.random.PRNGKey(3)
    B, H, D, bt, nc = 1, 2, 16, 16, 4
    L = bt * nc
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    scale = 1.0
    q = jax.random.normal(key, (B, L, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)
    o_ref, hf_ref = token_serial_ref(q, k, v, g, b, w, scale)

    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))  # (B,L,H,D) -> (B,H,L,D)
    o, hf = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                    scale, cfg)
    # serial vs chunked: the chunked pipeline applies the documented fp32
    # sanitize/clip contract; with this small random draw the outputs agree to
    # the tolerance below (guarded gates stay far from the clip).
    np.testing.assert_allclose(np.asarray(hm(o)), np.asarray(o_ref),
                               rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(hf), np.asarray(hf_ref),
                               rtol=2e-3, atol=2e-3)


def test_segment_ids_match_separate_runs():
    """Packed two-document prefill with segment_ids == two separate prefills."""
    key = jax.random.PRNGKey(4)
    B, H, D, bt, nc = 1, 2, 16, 16, 4
    L = bt * nc
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    scale = 1.0
    q = jax.random.normal(key, (B, L, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    split = 2 * bt  # boundary between chunks 1 and 2
    seg = (jnp.arange(L) >= split).astype(jnp.int32)[None]

    o1, hf1 = chunk_forward_inference(hm(q[:, :split]), hm(k[:, :split]),
                                      hm(v[:, :split]), hm(w[:, :split]),
                                      hm(b[:, :split]), hm(g[:, :split]),
                                      scale, cfg)
    o2, hf2 = chunk_forward_inference(hm(q[:, split:]), hm(k[:, split:]),
                                      hm(v[:, split:]), hm(w[:, split:]),
                                      hm(b[:, split:]), hm(g[:, split:]),
                                      scale, cfg)
    o_pack, hf_pack = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w),
                                              hm(b), hm(g), scale, cfg,
                                              segment_ids=seg)
    o_ref = jnp.concatenate([o1, o2], axis=2)
    np.testing.assert_allclose(np.asarray(o_pack), np.asarray(o_ref),
                               rtol=2e-3, atol=2e-3)
    # the packed run's final state is the LAST document's state (doc 2 started
    # from zero at the boundary), not a concatenation across batch rows:
    np.testing.assert_allclose(np.asarray(hf_pack), np.asarray(hf2),
                               rtol=2e-3, atol=2e-3)


def test_bf16_state_storage():
    """bf16 state storage stays close to fp32 state over a long decode."""
    key = jax.random.PRNGKey(5)
    B, H, D, L = 1, 2, 16, 64
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    scale = 1.0
    q = jax.random.normal(key, (B, L, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)
    qm = jnp.transpose(q, (0, 2, 1, 3))
    km = jnp.transpose(k, (0, 2, 1, 3))
    vm = jnp.transpose(v, (0, 2, 1, 3))
    wm = jnp.transpose(w, (0, 2, 1, 3))
    bm = jnp.transpose(b, (0, 2, 1, 3))
    gm = jnp.transpose(g, (0, 2, 1, 3))
    s32 = jnp.zeros((B, H, D, D), jnp.float32)
    s16 = jnp.zeros((B, H, D, D), jnp.bfloat16)
    for t in range(L):
        o32, s32 = decode_step(qm[:, :, t], km[:, :, t], vm[:, :, t],
                               gm[:, :, t], bm[:, :, t], wm[:, :, t],
                               s32, scale, cfg=cfg)
        o16, s16 = decode_step(qm[:, :, t], km[:, :, t], vm[:, :, t],
                               gm[:, :, t], bm[:, :, t], wm[:, :, t],
                               s16, scale, state_dtype=jnp.bfloat16, cfg=cfg)
    diff = np.abs(np.asarray(o16, np.float32) - np.asarray(o32, np.float32))
    scale_o = np.abs(np.asarray(o32, np.float32)).max() + 1e-6
    assert diff.max() / scale_o < 5e-2, f"bf16-state drift too large: {diff.max()}"


def test_stability_tau_caps_state():
    """Adversarial no-decay tokens: stability_tau caps the returned state norm."""
    key = jax.random.PRNGKey(6)
    B, H, D = 1, 2, 16
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    scale = 1.0
    tau = 4.0
    q = jax.random.normal(key, (B, H, D)) * 3.0
    k = jax.random.normal(key + 1, (B, H, D)) * 3.0
    v = jax.random.normal(key + 2, (B, H, D)) * 3.0
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, H, D)))
    g = jnp.zeros((B, H, D))  # no decay: state grows every step
    state = jnp.zeros((B, H, D, D), jnp.float32)
    for _ in range(6):
        state = decode_step(q, k, v, g, b, w, state, scale,
                            stability_tau=tau, cfg=cfg)[1]
    nrm = np.sqrt((np.asarray(state, np.float64) ** 2).sum(axis=(-2, -1)))
    assert (nrm <= tau * (1.0 + 1e-5)).all(), f"state norm {nrm} exceeds tau {tau}"


def test_decode_fuse_prologue_matches_preactivated():
    """fuse_prologue=True (raw logits in-kernel) == pre-activated inputs."""
    key = jax.random.PRNGKey(7)
    B, H, D, scale = 2, 2, 16, 1.0
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    q = jax.random.normal(key, (B, H, D)) * 2.0          # raw, unnormalized
    k = jax.random.normal(key + 1, (B, H, D)) * 2.0
    v = jax.random.normal(key + 2, (B, H, D))
    w_logit = jax.random.normal(key + 3, (B, H, D)) * 2.0
    b_logit = jax.random.normal(key + 4, (B, H, D)) * 2.0
    g_raw = jax.random.normal(key + 5, (B, H, D)) * 2.0
    state0 = jnp.zeros((B, H, D, D), jnp.float32)

    # pre-activated path (layer does the domain construction)
    g_act = -jnp.minimum(jax.nn.softplus(g_raw), cfg.g_max)
    o_ref, s_ref = decode_step(q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6),
                               k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6),
                               v, g_act, jax.nn.sigmoid(b_logit),
                               jax.nn.sigmoid(w_logit), state0, scale, cfg=cfg)
    # fused path (kernel does the activations)
    o_fus, s_fus = decode_step(q, k, v, g_raw, b_logit, w_logit, state0, scale,
                               fuse_prologue=True, cfg=cfg)
    np.testing.assert_allclose(np.asarray(o_fus), np.asarray(o_ref),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(s_fus), np.asarray(s_ref),
                               rtol=1e-4, atol=1e-4)


def test_cu_seqlens_matches_segment_ids():
    """cu_seqlens conversion == explicit segment_ids (chunk-granularity)."""
    key = jax.random.PRNGKey(8)
    B, H, D, bt, nc, scale = 1, 2, 16, 16, 4, 1.0
    L = bt * nc
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    q = jax.random.normal(key, (B, L, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    split = 2 * bt
    seg = (jnp.arange(L) >= split).astype(jnp.int32)[None]
    cu = jnp.array([0, split, L], jnp.int32)
    o_seg, hf_seg = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b),
                                            hm(g), scale, cfg, segment_ids=seg)
    o_cu, hf_cu = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b),
                                          hm(g), scale, cfg, cu_seqlens=cu)
    np.testing.assert_allclose(np.asarray(o_cu), np.asarray(o_seg),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(hf_cu), np.asarray(hf_seg),
                               rtol=1e-4, atol=1e-4)


def test_multi_segment_prefill_matches_serial():
    """L > max_chunks_per_call: two gates.

    Gate 1 (the segmentation property, exact): multi-segment output is
    bit-identical to a single-call run of the same chunked pipeline — the
    state carry across pallas launches is lossless.
    Gate 2 (documented contract): both agree with the undamped serial
    reference within the chunked pipeline's fp32 contract (wy_eps damping
    changes the solved system by O(eps); sanitize clipping) — the tolerance
    scales with sequence length because the contract difference accumulates.
    """
    key = jax.random.PRNGKey(9)
    B, H, D, bt = 1, 2, 16, 16
    nc = 5  # 5 chunks with max_chunks_per_call=2 -> 3 segments
    L = bt * nc
    base = dict(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                interpret=True, g_max=160.0 / (bt // 4))
    cfg_multi = make_cfg(**base, max_chunks_per_call=2)
    cfg_one = make_cfg(**base, max_chunks_per_call=32)
    scale = 1.0
    q = jax.random.normal(key, (B, L, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)
    o_ref, hf_ref = token_serial_ref(q, k, v, g, b, w, scale)
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    o_multi, hf_multi = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w),
                                                hm(b), hm(g), scale, cfg_multi)
    o_one, hf_one = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w),
                                            hm(b), hm(g), scale, cfg_one)
    # Gate 1: segmentation carry is lossless (bit-identical)
    assert np.array_equal(np.asarray(o_multi), np.asarray(o_one))
    assert np.array_equal(np.asarray(hf_multi), np.asarray(hf_one))
    # Gate 2: serial-equivalence within the documented chunked contract
    np.testing.assert_allclose(np.asarray(hm(o_multi)), np.asarray(o_ref),
                               rtol=1e-2, atol=6e-3)
    np.testing.assert_allclose(np.asarray(hf_multi), np.asarray(hf_ref),
                               rtol=1e-2, atol=3e-3)


def test_decode_long_three_documents_with_resets():
    """Long decode, three documents packed -> reset at each boundary."""
    key = jax.random.PRNGKey(10)
    B, H, D, scale = 2, 2, 16, 1.0
    lens = (23, 41, 36)
    L = sum(lens)
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    q = jax.random.normal(key, (B, L, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, L, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, L, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)
    # reference: each document serially from zero state
    o_parts, offs = [], 0
    for n in lens:
        o_p, _ = token_serial_ref(q[:, offs:offs + n], k[:, offs:offs + n],
                                  v[:, offs:offs + n], g[:, offs:offs + n],
                                  b[:, offs:offs + n], w[:, offs:offs + n], scale)
        o_parts.append(o_p)
        offs += n
    o_ref = jnp.concatenate(o_parts, axis=1)

    qm = jnp.transpose(q, (0, 2, 1, 3))
    km = jnp.transpose(k, (0, 2, 1, 3))
    vm = jnp.transpose(v, (0, 2, 1, 3))
    wm = jnp.transpose(w, (0, 2, 1, 3))
    bm = jnp.transpose(b, (0, 2, 1, 3))
    gm = jnp.transpose(g, (0, 2, 1, 3))
    state = jnp.zeros((B, H, D, D), jnp.float32)
    outs = []
    for t in range(L):
        reset = (jnp.ones((B,), jnp.int32)
                 if t in (lens[0], lens[0] + lens[1]) else None)
        o, state = decode_step(qm[:, :, t], km[:, :, t], vm[:, :, t],
                               gm[:, :, t], bm[:, :, t], wm[:, :, t],
                               state, scale, reset=reset, cfg=cfg)
        outs.append(o)
    o_dec = jnp.stack(outs, axis=1)
    np.testing.assert_allclose(np.asarray(o_dec), np.asarray(o_ref),
                               rtol=1e-4, atol=1e-4)
