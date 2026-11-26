# Production contract — atomic_gdn2 inference kernels

**Version**: 0.3.1 (audit Stage 0 artifact; stages per
`V0.3.0_PRODUCTION_AUDIT_AND_PLAN.md` §4)
**Date**: 2026-11-20
**Owner decisions**: per audit §5 (recommendations accepted 2026-11-19:
pure-JAX/TPU first; mid-chunk boundary raises; fuse_prologue requires
checkpoint A_log/dt_bias; non-finite rows reset; per-head stability cap).

## Target shape (frozen for 0.3.x validation)
- Dense GDN-2 layer stack: `B` variable, `H=16`, `D=128`, `bt=128`, 18 layers
  (matches the published 1.3B GDN-2 checkpoint geometry).
- Batch rows are independent sessions; document boundaries at chunk
  granularity (mid-chunk boundaries are rejected at the API).

## SLO (to be measured in audit Stage 6; placeholders until then)
- Decode: p50/p99 ms per token per row (steady state, jit-batched loop) —
  placeholder replaced 136 ms/step figure (dispatch-bound).
- Prefill: tokens/s on the target shape, steady state, bf16 compute config.
- Memory: peak HBM per session-state slot; state cache budget bounded.
- Correctness: relative L2 vs `token_serial_ref` ≤ 1e-3 on TPU (audit Stage
  5 replaces the 0.15 placeholder).

## Behavioural contracts
1. **Domain**: `g ∈ [-g_max, 0]` (layer constructs it; API clamps and raises
   outside — audit A6). `b, w ∈ (0, 1)` via sigmoid (Stage 2.1 fix).
2. **Documents**: boundary token starts its document (`>=` semantics);
   boundaries inside a full chunk raise (for `segment_ids` and `cu_seqlens`
   alike); tail boundaries are honored per token; `h0` belongs to each row's
   first document whatever its id; `cu_seqlens` 1-D (shared) or
   `(B, n_docs+1)`.
7. **JIT**: `decode_step` and `chunk_forward_inference` are jittable. Traced
   inputs skip the boundary and g-domain checks; the caller validates on the
   host with `check_document_boundaries` before a jitted dispatch.
3. **Numerics**: float32 compute; bfloat16 optional state storage and
   compute config; `wy_eps` default 1e-3 for training, `wy_eps=0` measured
   ~500× closer to serial for inference (audit B8 — pick per release).
4. **Non-finite policy**: in-kernel cleanup — NaN/Inf state and output
   entries are zeroed; rows restart clean; a metric counter is planned for
   0.4.0.
5. **Stability**: per-head Frobenius cap `stability_tau` on inference-returned
   state (off by default).
6. **Gate compatibility**: `fuse_prologue` consumes the checkpoint
   construction (`a_log`, `dt_bias`); the layer applies sigmoids before the
   kernel (both paths consistent since 0.3.1).

## Release gates (checklist owner: audit §6)
- [ ] A1–A6 closed by regression tests (DONE 2026-11-19, commit 69647b9)
- [ ] B1–B8 fixed or accepted in writing (B1/B2/B3/B5 DONE; B4/B7/B8 planned 0.4.0)
- [ ] Real weights vs external reference (Stage 2)
- [ ] All serving modes compiled and validated on TPU at the target shape (Stage 5)
      (2026-11-24 Colab v5e: 5 serving gates red — fixed in 0.4.0-alpha.2 with
      CPU regression tests; TPU re-run pending)
- [ ] Attestation stores numbers + kernel sha256 (Stage 5)
- [ ] Inference tests mandatory in CI (DONE, ci.yml updated)
- [ ] SLO on the full 18-layer model (Stage 6)
- [ ] Soak test + session isolation (Stage 7)
- [ ] License/attraction check + signed release (Stage 8)
