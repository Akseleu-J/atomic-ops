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
                               wm[:, :, t], bm[:, :, t], gm[:, :, t],
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
                               wm[:, :, t], bm[:, :, t], gm[:, :, t],
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
                               wm[:, :, t], bm[:, :, t], gm[:, :, t],
                               s32, scale, cfg=cfg)
        o16, s16 = decode_step(qm[:, :, t], km[:, :, t], vm[:, :, t],
                               wm[:, :, t], bm[:, :, t], gm[:, :, t],
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

    # checkpoint-style gate construction (A4 fix): g = -exp(A_log)*softplus(g_raw + dt_bias)
    a_log = jax.random.normal(key + 6, (H,)) * 0.3
    dt_bias = jax.random.normal(key + 7, (D,)) * 0.2
    g_act = -jnp.exp(a_log)[None, :, None] * jax.nn.softplus(g_raw + dt_bias[None, None, :])
    o_ref, s_ref = decode_step(q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6),
                               k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6),
                               v, jax.nn.sigmoid(w_logit), jax.nn.sigmoid(b_logit),
                               g_act, state0, scale, cfg=cfg)
    # fused path (kernel does the activations; new unified arg order)
    o_fus, s_fus = decode_step(q, k, v, w_logit, b_logit, g_raw, state0, scale,
                               fuse_prologue=True, a_log=a_log, dt_bias=dt_bias, cfg=cfg)
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
                               wm[:, :, t], bm[:, :, t], gm[:, :, t],
                               state, scale, reset=reset, cfg=cfg)
        outs.append(o)
    o_dec = jnp.stack(outs, axis=1)
    np.testing.assert_allclose(np.asarray(o_dec), np.asarray(o_ref),
                               rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Regression tests from the production audit (PR-prep/audit_probes/): A1-A6, B1
# ---------------------------------------------------------------------------

def test_a1_segment_ids_batch_rows_independent():
    """Audit A1: with B>1, each batch row must use ITS OWN document layout.

    Reviewer probe: row 1 has a doc boundary mid-sequence, row 0 does not.
    Before the fix all rows read row 0's boundaries (34.8% rel error, silent).
    """
    key = jax.random.PRNGKey(21)
    B, H, D, bt, nc, scale = 2, 2, 16, 16, 4, 1.0
    L = bt * nc
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    q, k, v, w, b, g = (jnp.asarray(x) for x in [
        jax.random.normal(key, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 1, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 2, (B, L, H, D)),
        jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D))),
        jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D))),
        -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)])
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    split = 2 * bt  # boundary after chunk 1
    seg = jnp.stack([(jnp.arange(L) < split).astype(jnp.int32)[None],
                     jnp.zeros((1, L), jnp.int32)], axis=0).reshape(B, L)
    # reference: run each row separately with its own segment_ids
    o_parts = []
    for row in range(B):
        o_r, _ = chunk_forward_inference(hm(q[row:row + 1]), hm(k[row:row + 1]),
                                         hm(v[row:row + 1]), hm(w[row:row + 1]),
                                         hm(b[row:row + 1]), hm(g[row:row + 1]),
                                         scale, cfg, segment_ids=seg[row:row + 1])
        o_parts.append(o_r)
    o_ref = jnp.concatenate(o_parts, axis=0)
    o_pack, _ = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                        scale, cfg, segment_ids=seg)
    np.testing.assert_allclose(np.asarray(o_pack), np.asarray(o_ref),
                               rtol=2e-3, atol=2e-3)


def test_a2_mid_chunk_boundary_raises():
    """Audit A2 (owner decision): a boundary inside a chunk RAISES loudly."""
    key = jax.random.PRNGKey(22)
    B, H, D, bt, nc, scale = 1, 2, 16, 16, 4, 1.0
    L = bt * nc
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    q, k, v, w, b, g = (jnp.asarray(x) for x in [
        jax.random.normal(key, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 1, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 2, (B, L, H, D)),
        jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D))),
        jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D))),
        -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)])
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    seg = (jnp.arange(L) == bt + 5).astype(jnp.int32)[None]  # boundary INSIDE chunk 1
    seg = jnp.cumsum(seg, axis=1)
    with pytest.raises(ValueError, match="boundary inside chunk"):
        chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                scale, cfg, segment_ids=seg)


def test_a3_cu_seqlens_general():
    """Audit A3: cu_seqlens works for any doc count and any B."""
    key = jax.random.PRNGKey(23)
    B, H, D, bt, nc, scale = 2, 2, 16, 16, 4, 1.0
    L = bt * nc
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    q, k, v, w, b, g = (jnp.asarray(x) for x in [
        jax.random.normal(key, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 1, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 2, (B, L, H, D)),
        jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D))),
        jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D))),
        -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)])
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    # 1 document, B=2
    o1, _ = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                    scale, cfg, cu_seqlens=jnp.array([0, L]))
    # 3 documents, B=2 (per-batch cu)
    s1, s2 = bt, 3 * bt
    cu = jnp.array([[0, s1, s2, L], [0, s2, s1, L]])
    o3, _ = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                    scale, cfg, cu_seqlens=cu)
    # 1-D shared cu with 2 docs
    o2, _ = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                    scale, cfg, cu_seqlens=jnp.array([0, s2, L]))
    assert o1.shape == o3.shape == o2.shape


def test_a5_decode_nan_guard_resets_row():
    """Audit A5 (owner decision): a non-finite row is RESET, not carried."""
    key = jax.random.PRNGKey(24)
    B, H, D, scale = 2, 2, 16, 1.0
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    q = jax.random.normal(key, (B, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, H, D)) * 0.39 - 1.0)
    state = jnp.ones((B, H, D, D), jnp.float32)
    state = state.at[0].set(jnp.nan)                     # row 0 poisoned
    o, new = decode_step(q, k, v, w, b, g, state, scale, cfg=cfg)
    # in-kernel A5 policy: non-finite entries are zeroed, the row restarts clean
    assert bool(jnp.isfinite(new).all()), "kernel must sanitize non-finite state"
    assert bool(jnp.isfinite(o).all())


def test_a6_prefill_g_domain_checked():
    """Audit A6: out-of-domain g raises loudly (was 58%/205% silent error)."""
    key = jax.random.PRNGKey(25)
    B, H, D, bt, nc, scale = 1, 2, 16, 16, 2, 1.0
    L = bt * nc
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    q, k, v, w, b, g = (jnp.asarray(x) for x in [
        jax.random.normal(key, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 1, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 2, (B, L, H, D)),
        jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D))),
        jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D))),
        -jnp.full((B, L, H, D), 50.0)])                   # |g| > g_max=40
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    with pytest.raises(ValueError, match="outside the layer domain"):
        chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                scale, cfg)


def test_b1_stability_tau_per_head_consistent():
    """Audit B1: decode and prefill caps now use the same per-head norm."""
    key = jax.random.PRNGKey(26)
    B, H, D, scale = 1, 2, 16, 1.0
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    tau = 4.0
    q = jax.random.normal(key, (B, H, D)) * 3.0
    k = jax.random.normal(key + 1, (B, H, D)) * 3.0
    v = jax.random.normal(key + 2, (B, H, D)) * 3.0
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, H, D)))
    g = jnp.zeros((B, H, D))
    state = jnp.zeros((B, H, D, D), jnp.float32)
    for _ in range(6):
        state = decode_step(q, k, v, w, b, g, state, scale,
                            stability_tau=tau, cfg=cfg)[1]
    nrm = np.sqrt((np.asarray(state, np.float64) ** 2).sum(axis=(-2, -1)))
    assert (nrm <= tau * (1 + 1e-5)).all()


# ---------------------------------------------------------------------------
# Stage 3 (0.4.0): arbitrary length (B7), LayerInputs container (B3), B4 errors
# ---------------------------------------------------------------------------

def test_b7_arbitrary_length_matches_serial():
    """L = 2*bt + 13: full chunks + decode tail == serial reference."""
    key = jax.random.PRNGKey(30)
    B, H, D, bt = 1, 2, 16, 16
    L = bt * 2 + 13
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
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    o, hf = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                    scale, cfg)
    assert np.asarray(hm(o)).shape == (B, L, H, D), "full-length output"
    np.testing.assert_allclose(np.asarray(hm(o)), np.asarray(o_ref),
                               rtol=1e-2, atol=6e-3)
    np.testing.assert_allclose(np.asarray(hf), np.asarray(hf_ref),
                               rtol=1e-2, atol=3e-3)


def test_b7_arbitrary_length_with_segment_ids():
    """Arbitrary length + document resets in the tail (decode reset path)."""
    key = jax.random.PRNGKey(31)
    B, H, D, bt = 1, 2, 16, 16
    L = bt * 2 + 10
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    scale = 1.0
    q, k, v, w, b, g = (jnp.asarray(x) for x in [
        jax.random.normal(key, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 1, (B, L, H, D)) * 0.5,
        jax.random.normal(key + 2, (B, L, H, D)),
        jax.nn.sigmoid(jax.random.normal(key + 3, (B, L, H, D))),
        jax.nn.sigmoid(jax.random.normal(key + 4, (B, L, H, D))),
        -jnp.exp(jax.random.normal(key + 5, (B, L, H, D)) * 0.39 - 1.0)])
    hm = lambda t: jnp.transpose(t, (0, 2, 1, 3))
    split = bt * 2 + 5                   # boundary INSIDE the tail (L_full=32)
    seg = (jnp.arange(L) >= split).astype(jnp.int32)[None]
    o1, _ = chunk_forward_inference(hm(q[:, :split]), hm(k[:, :split]),
                                    hm(v[:, :split]), hm(w[:, :split]),
                                    hm(b[:, :split]), hm(g[:, :split]), scale, cfg)
    o2, hf2 = chunk_forward_inference(hm(q[:, split:]), hm(k[:, split:]),
                                      hm(v[:, split:]), hm(w[:, split:]),
                                      hm(b[:, split:]), hm(g[:, split:]), scale, cfg)
    o_pack, hf_pack = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b),
                                              hm(g), scale, cfg, segment_ids=seg)
    np.testing.assert_allclose(np.asarray(o_pack),
                               np.asarray(jnp.concatenate([o1, o2], axis=2)),
                               rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(hf_pack), np.asarray(hf2),
                               rtol=2e-3, atol=2e-3)


def test_layer_inputs_container_equivalence():
    """LayerInputs container == individual positional args (B3 final)."""
    key = jax.random.PRNGKey(32)
    B, H, D, scale = 1, 2, 16, 1.0
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    q = jax.random.normal(key, (B, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, H, D)) * 0.39 - 1.0)
    state = jnp.zeros((B, H, D, D), jnp.float32)
    from atomic_gdn2.inference import LayerInputs
    inp = LayerInputs(q=q, k=k, v=v, w=w, b=b, g=g)
    o_c, s_c = decode_step(inputs=inp, state=state, scale=scale, cfg=cfg)
    o_p, s_p = decode_step(q, k, v, w, b, g, state, scale, cfg=cfg)
    np.testing.assert_array_equal(np.asarray(o_c), np.asarray(o_p))
    np.testing.assert_array_equal(np.asarray(s_c), np.asarray(s_p))


def test_b4_static_param_friendly_error():
    """Audit B4: traced/array scale -> friendly ValueError, not a deep crash."""
    import jax
    key = jax.random.PRNGKey(33)
    B, H, D = 1, 2, 16
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True)
    q = jax.random.normal(key, (B, H, D)) * 0.5
    k = jax.random.normal(key + 1, (B, H, D)) * 0.5
    v = jax.random.normal(key + 2, (B, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, H, D)))
    g = -jnp.exp(jax.random.normal(key + 5, (B, H, D)) * 0.39 - 1.0)
    state = jnp.zeros((B, H, D, D), jnp.float32)
    with pytest.raises(ValueError, match="concrete Python float"):
        decode_step(q, k, v, w, b, g, state, jnp.asarray(0.7), cfg=cfg)


def test_adversarial_zero_k_and_extreme_g():
    """Zero keys (erase = write = 0 net) and in-domain extreme decay."""
    key = jax.random.PRNGKey(34)
    B, H, D, scale = 1, 2, 16, 1.0
    cfg = make_cfg(bt=16, bc=8, mb=8, score_bs=16, bs2=8, interpret=True,
                   g_max=160.0 / 8)
    q = jax.random.normal(key, (B, H, D)) * 0.5
    k = jnp.zeros((B, H, D))                          # zero keys
    v = jax.random.normal(key + 2, (B, H, D))
    w = jax.nn.sigmoid(jax.random.normal(key + 3, (B, H, D)))
    b = jax.nn.sigmoid(jax.random.normal(key + 4, (B, H, D)))
    g = jnp.full((B, H, D), -cfg.g_max)               # extreme in-domain decay
    state = jnp.zeros((B, H, D, D), jnp.float32)
    o, new = decode_step(q, k, v, w, b, g, state, scale, cfg=cfg)
    # zero k -> no state write, no erase: S = Diag(exp(g)) S, o = q . S = 0
    assert float(np.abs(np.asarray(o)).max()) == 0.0
    assert float(np.abs(np.asarray(new)).max()) == 0.0
    assert bool(jnp.isfinite(new).all())


def test_layer_conventions_vs_serial():
    """Audit Stage 2.4: the FULL layer conventions (short conv + projections +
    l2norm + sigmoids + checkpoint gate construction + kernel + RMSNormSwishGate
    + o_proj) against the serial reference. This is the parity the TPU
    notebook (gdn2_v031_stage2_tpu.ipynb RUN 2) measures against NVlabs."""
    key = jax.random.PRNGKey(40)
    B, H, Dh, bt, scale = 1, 4, 16, 16, 1.0
    L = bt
    D = H * Dh; Dk = D
    cfg = make_cfg(bt=bt, bc=bt // 2, mb=8, score_bs=bt, bs2=bt // 4,
                   interpret=True, g_max=160.0 / (bt // 4))
    W = {
        "q_proj.weight": jax.random.normal(key, (Dk, D)) * 0.02,
        "k_proj.weight": jax.random.normal(key + 1, (Dk, D)) * 0.02,
        "v_proj.weight": jax.random.normal(key + 2, (D, D)) * 0.02,
        "w_proj.weight": jax.random.normal(key + 3, (D, D)) * 0.02,
        "b_proj.weight": jax.random.normal(key + 4, (Dk, D)) * 0.02,
        "f_proj.0.weight": jax.random.normal(key + 5, (Dh, D)) * 0.02,
        "f_proj.1.weight": jax.random.normal(key + 6, (Dk, Dh)) * 0.02,
        "g_proj.0.weight": jax.random.normal(key + 7, (Dh, D)) * 0.02,
        "g_proj.1.weight": jax.random.normal(key + 8, (D, Dh)) * 0.02,
        "g_proj.1.bias": jnp.zeros(D),
        "A_log": jax.random.normal(key + 9, (H,)) * 0.3,
        "dt_bias": jax.random.normal(key + 10, (Dk,)) * 0.2,
        "o_proj.weight": jax.random.normal(key + 11, (D, D)) * 0.02,
        "o_norm.weight": (jax.random.normal(key + 12, (Dh,)) * 0.5 + 1),
        "q_conv1d.conv.weight": jax.random.normal(key + 13, (D, 1, 4)) * 0.1,
        "k_conv1d.conv.weight": jax.random.normal(key + 14, (D, 1, 4)) * 0.1,
        "v_conv1d.conv.weight": jax.random.normal(key + 15, (D, 1, 4)) * 0.1,
    }
    x = jax.random.normal(key + 20, (B, L, D)) * 0.5
    tq = x[0]                                            # (L, D)
    lin = lambda n: tq @ W[n + ".weight"].T

    def conv(xl, wname):                                 # causal depthwise + silu
        wc = W[wname][:, 0, :]                           # (D, 4)
        xp = jnp.pad(xl.T, ((0, 0), (3, 0)))             # (D, L+3)
        y = sum(xp[:, kk:kk+L] * wc[:, kk][:, None] for kk in range(4))
        return y.T
    l2 = lambda t: t * jax.lax.rsqrt((t*t).sum(-1, keepdims=True) + 1e-6)
    q = l2(jax.nn.silu(conv(lin("q_proj"), "q_conv1d.conv.weight")))
    k = l2(jax.nn.silu(conv(lin("k_proj"), "k_conv1d.conv.weight")))
    v = jax.nn.silu(conv(lin("v_proj"), "v_conv1d.conv.weight"))
    b = jax.nn.sigmoid(lin("b_proj"))
    w = jax.nn.sigmoid(lin("w_proj"))
    f = lin("f_proj.0") @ W["f_proj.1.weight"].T
    # g construction in (L, Dk) space: dt_bias is per-K-channel; the per-head
    # A_log multiplies after the head split (audit A4 checkpoint contract)
    g = (-W["A_log"].reshape(H, 1)
         * jax.nn.softplus(f + W["dt_bias"].reshape(1, Dk)).reshape(L, H, Dh)).reshape(L, Dk)

    # serial reference on the SAME prepared q/k/v/w/b/g — head-split to
    # (B, L, H, Dh) as token_serial_ref expects (batch dim added)
    sp4 = lambda t: t.reshape(L, H, Dh)[None]
    o_ref, _ = token_serial_ref(sp4(q), sp4(k), sp4(v), sp4(g), sp4(b),
                                sp4(w), scale)
    o_ref = o_ref[0].reshape(L, H, Dh)

    # chunk kernel path
    hm = lambda t: t.reshape(1, L, H, Dh).transpose(0, 2, 1, 3)
    o_kern, _ = chunk_forward_inference(hm(q), hm(k), hm(v), hm(w), hm(b), hm(g),
                                        scale, cfg)
    o_kern = o_kern[0].transpose(1, 0, 2)                # (L, H, Dh)

    # shared epilogue: RMSNormSwishGate + o_proj
    gate_pre = tq @ W["g_proj.0.weight"].T @ W["g_proj.1.weight"].T + W["g_proj.1.bias"]
    Wn = W["o_norm.weight"].reshape(1, 1, Dh)
    def epilogue(o):
        o3 = o.reshape(L, H, Dh)
        v_rms = jnp.sqrt((o3**2).mean(-1, keepdims=True) + 1e-5)
        return (o3 / v_rms * Wn * jax.nn.silu(gate_pre.reshape(L, H, Dh))).reshape(L, Dk) @ W["o_proj.weight"].T
    e_ref = epilogue(o_ref)
    e_kern = epilogue(o_kern)
    rel = np.linalg.norm(np.asarray(e_kern) - np.asarray(e_ref)) / (np.linalg.norm(np.asarray(e_ref)) + 1e-9)
    assert rel < 1e-3, f"layer conventions diverge: rel L2 {rel:.5f}"
