# Known Limitations

This document lists deliberate restrictions and known gaps of the current
release. Each item includes the reason, current evidence, and the
recommended workaround. Items marked [planned] have a targeted fix in a
future release.

## 1. Pairwise decay computation is VPU-bound, not MXU-bound

Kernel A (`build_chunk_scores_pallas`) and its backward counterpart B4 (`intra_backward_pallas`) compute the pairwise decay-weighted product via explicit broadcast + elementwise multiply + manual reduction (`_weighted_pair_sum` / `_dL_pair_sum` / `_dR_pair_sum` / `_dgc_pair_sum`), which runs on the VPU rather than the MXU. This is the current, shipped implementation and is not user-configurable.

An MXU-factorized alternative has been explored as an isolated, off-by-default experiment and shows a large speedup in isolation, but has not been validated end-to-end and is not present in this codebase's kernels. See `ROADMAP.md` for the investigation, its current status, and the gates required before any such change would ship.


## 2. Fused forward is slower than the pure-JAX WY forward

**Status:** known performance gap, root cause now identified (see section
6). Mitigated in (planned, unreleased) by an experimental hybrid path (section 5); real
fix targeted for v0.2.0 (section 1).

**Numbers (TPU v5e-8, B=8, L=4096):** Pallas fwd 102.08 ms (FP32) / 101.51 ms
(BF16) vs JAX_REF fwd 63.24 ms / 62.38 ms -- about 0.62x (i.e. Pallas is
~1.6x slower).

**Root cause (confirmed via isolated kernel-by-kernel timing + cost
analysis, not just theory):** it is **not** dispatch/scheduling overhead
between separate `pallas_call` graphs. An isolated per-kernel benchmark
(sum of Kernel A+B+C+D measured independently) matches the full pipeline
time within measurement noise (gap = -2.5%, i.e. within noise budget, see
section 6) -- ruling out the "XLA builds separate graphs with barrier
overhead between them" hypothesis. The actual cost is concentrated in two
kernels:
- **Kernel A** (`build_chunk_scores_pallas`, scores): ~47.6 ms -- VPU-bound
  broadcast/reduce instead of MXU matmul (see section 1).
- **Kernel B** (`wy_solve_pallas`, WY block solve): ~51.2 ms -- forward
  substitution operates on `MB=16` micro-blocks, far below the TPU MXU's
  efficient `128x128` tile size, so matmuls inside `_micro_forward_substitution`
  are likely heavily padded/underutilized.
Kernel C (recompute) and Kernel D (inter-chunk scan) are cheap (~2.3 ms /
~3.4 ms) and not a concern.

**Workaround:** for inference-only workloads use `gdn2_forward` (dispatches
to the pure-JAX reference off-TPU) or `gdn2_chunked_wy_reference` directly.
For training, see the experimental hybrid path (section 5).

## 3. Fused kernels are TPU-only and require `d_head = 128`

The Pallas path assumes TPU MXU tiling. On CPU/GPU, or with `d_head != 128`,
the public API automatically falls back to the pure-JAX chunked-WY reference
(slower, correct). Only Kernel A (`build_chunk_scores_pallas`) and Kernel B4
(`intra_backward_pallas`) accept `interpret=True` and can execute on CPU;
the remaining kernels lower via Mosaic and require a TPU regardless of shape.

## 4. Shape constraints

- `seq_len` must be divisible by `config.bt` (256 by default, 128 for
`KAGGLE_SMALL`).
- `KernelConfig.bt` must equal `2 * config.bc`; vary `mb` for solver
granularity. Other `bt/bc` ratios raise `ValueError` by design (the
top-level WY solve supports only the 2-block split).
- `KernelConfig.mb` (currently 16 across all presets) is the likely cause
of Kernel B's MXU underutilization noted in section 2; this is a candidate
for tuning alongside the `use_centering` fix in v0.2.0, but has not been
benchmarked independently yet -- treat as a hypothesis, not a confirmed
cause, until isolated.

## 5. [NEW, (planned, unreleased), EXPERIMENTAL/BETA] Hybrid JAX-forward + Pallas-backward path

**Status:** landed in `beta/` as an opt-in experimental path, not wired
into `model.py` or the public `gdn2_forward_trainable` dispatcher. Not a
long-term architectural direction -- a stopgap pending section 1.
**Target release: v0.1.5.** Isolated correctness/speed gates already pass
(see numbers below); the remaining blocker before v0.1.5 is validating the
path on a **real training run** (not just the isolated fwd+bwd benchmark),
since only end-to-end training exercises long-run numerical stability that
a single-batch gate cannot catch.

**What it is:** forward uses a plain-JAX chunked-WY scan (same algorithm
and residual layout as `gdn2_chunked_wy_reference`, but additionally
returns the intermediate residuals -- `Aqk, Akk, A, w_pseudo, u, kg, qg,
gc_last, h_pre_all, v_new_all` -- in the exact layout the existing Pallas
backward kernels (B1-B5) expect), and backward reuses the fused Pallas
B1-B5 chain unchanged, without recomputing forward.

**Why it works:** JAX-forward is ~1.6x faster than Pallas-forward (see
section 2), while the Pallas B1-B5 backward chain is already 2.6-3.9x
faster than the JAX_REF backward (see README benchmarks). The hybrid
combines the faster half of each path.

**Measured speedup vs PALLAS_PROD (fwd+bwd, TPU v5e-8, FP32, median of 15
iters):**

| Config | JAX_REF (ms) | PALLAS_PROD (ms) | HYBRID (ms) | HYBRID vs PALLAS_PROD |
| --- | --- | --- | --- | --- |
| KAGGLE_SMALL (B=4, L=2048) | 119.53 | 30.74 | 26.30 | 1.17x |
| KAGGLE_MEDIUM (B=4, L=4096) | 268.53 | 87.51 | 77.27 | 1.13x |
| train shape (B=8, L=4096) | 459.94 | 173.97 | 137.03 | 1.27x |

Gradient correctness gate (finite-diff + tol=5e-2 vs JAX_REF autodiff):
PASSED, all tensors finite.

**Known gaps before this can be considered release-ready (not yet
closed as of this writing):**
- **Residual-parity test missing.** All current gates compare final
  outputs/gradients through a full loss, not the individual residual
  tensors (`Aqk, Akk, A, w_pseudo, u, kg, qg, h_pre_all, v_new_all`)
  element-by-element against the Pallas-forward-produced residuals. Given
  that B1-B5 were only ever tested against Pallas-forward residuals, a
  silent layout/dtype mismatch in the JAX-forward residual harvesting
  could pass the coarse gradient gate while being subtly wrong. This is
  the single highest-priority item before wider review.
- **`jax.checkpoint(chunk_step)` left in the hybrid forward scan.** It has
  no effect here (backward does not re-differentiate through this scan --
  residuals are consumed directly by `custom_vjp`'s explicit backward
  rule), so it should be removed; leaving it in reads as unintentional to
  a reviewer.
- **No isolated (forward-excluded) backward-only comparison published
  yet.** The fwd/bwd split in the speed table above double-counts forward
  cost inside the naive `jax.vjp(loss, ...)`-based "bwd" column (this was
  independently confirmed via the kernel-gap diagnostic in section 6:
  Δfwd and Δbwd between PALLAS_PROD and HYBRID track almost exactly). The
  fwd+bwd total is valid; the fwd/bwd split as currently presented is not
  and should be re-measured via a direct call to the backward rule on
  pre-computed residuals before publishing outside this repo.
- **No BF16 numbers yet.** All hybrid numbers above are FP32 only; BF16 is
  the production training dtype.
- **No memory (peak HBM) numbers yet.** `benchmarks/run_memory.py` does
  not yet have a `"HYBRID"` path entry.
- **KAGGLE_LARGE (clip=5e3) not yet benchmarked** for the hybrid path.
- **Full deep-correctness suite not yet run.** Only a coarse finite-diff
  gate has been checked so far; multi-seed sweep against
  `gdn2_token_serial_reference` (the derivation-independent reference, per
  `docs/TESTING_STRATEGY.md` Layer 2/3) is still pending.

**Recommended framing for the v0.1.5 release:** ship as `beta/`, explicitly
labeled experimental, not default-wired, pending the items above -- most
importantly the real-training validation run. Do not recommend for
production training pipelines until that run is published.

## 6. Kernel-gap diagnostic (TPU v5e-8, KAGGLE_MEDIUM, B=8 L=4096, FP32)

Run to test (and rule out) the hypothesis that Pallas-forward's slowness
relative to JAX_REF comes from scheduling/barrier overhead between
separate `pallas_call` graphs, as opposed to the cost of the kernels
themselves.

**Method:** each forward kernel (A/B/C/D) and each backward kernel
(B1-B5) was benchmarked in isolation (own `jax.jit`, real intermediate
values chained from the previous stage) and the sum was compared against
the full pipeline's measured time. Backward was measured with forward
cost explicitly excluded (direct call to `_gdn2_core_bwd` on pre-harvested
residuals, not via `jax.vjp(loss, ...)`).

**Results:**

| stage | sum(isolated) | full pipeline | gap | gap % | proxy busy % |
| --- | --- | --- | --- | --- | --- |
| fwd | 104.563 ms | 102.023 ms | -2.540 ms | -2.5% | 102.5% |
| bwd | 71.995 ms | 72.704 ms | +0.709 ms | +1.0% | 99.0% |

**Conclusion:** gap is within measurement noise on both fwd and bwd (in
fact slightly negative on fwd, meaning the fused pipeline is marginally
*faster* than the sum of its isolated parts, consistent with XLA doing
some cross-kernel scheduling even across `pallas_call` boundaries). The
"disconnected graphs" hypothesis is **ruled out**. Per-kernel breakdown:

fwd: Kernel A 47.576 ms <- expensive, VPU pair-sum (see section 1/2)
Kernel B 51.234 ms <- expensive, MB=16 sub-MXU-tile solve (see section 2/4)
Kernel C 2.311 ms
Kernel D 3.442 ms

bwd: B2 1.547 ms
B1 3.145 ms
B3 4.555 ms
B4 61.959 ms <- expensive, mirrors Kernel A's VPU pair-sum pattern
B5 0.790 ms


Kernel A and its backward counterpart B4 use the same
broadcast-multiply-reduce pattern (`_weighted_pair_sum` /
`_dL_pair_sum`/`_dR_pair_sum`/`_dgc_pair_sum`) instead of a true MXU
matmul. An MXU-factorized alternative (tentatively `use_centering`) is a
documented hypothesis in `ROADMAP.md`, not yet implemented anywhere in
this codebase. This VPU/MXU gap is currently the leading, evidence-backed
hypothesis for the majority of the fwd/bwd slowdown vs JAX_REF -- not
kernel-count/dispatch overhead.

**Not yet done:** `compiled.cost_analysis()` (flops / bytes-accessed) on
Kernel A/B in isolation, to directly confirm VPU- vs MXU-bound and rule
out simple HBM-bandwidth explanations for Kernel B specifically (its
slowness could be sub-tile MXU padding, or could be something else in the
recursive block-solve structure -- not yet isolated from the `use_centering`
hypothesis, which only directly explains Kernel A/B4, not Kernel B).
xplane/TensorBoard trace visual confirmation was not obtained (no
forwarded port in the current environment); the numeric isolated-vs-pipeline
method above was used as the primary evidence instead and is considered
sufficient to reject the dispatch-overhead hypothesis.

---

## Roadmap

**(planned, unreleased) (current):**
- Ship `beta/` hybrid JAX-forward + Pallas-backward path as opt-in,
  experimental, not wired into `model.py`. See section 5 for exact status
  and open items.
- Document this file's findings (this update).

**v0.2.0 (planned):**
- Validate the section 5 hybrid path fully end-to-end (see its open
  items list); this is a precondition, not optional.
- Only if the hybrid path does not fully close the fwd/bwd gap: prototype
  and validate the `use_centering` MXU-factorization hypothesis from
  scratch (see `ROADMAP.md` -- no code for it exists yet), including its
  own isolated correctness/speed tests and the full deep-correctness
  suite (`tests/extended/test_gdn2_deep_correctness.py`).
- If validated and shown to close most of the forward gap: this becomes
  the primary fix, and the section 5 hybrid path is downgraded to a
  documented alternative rather than the default recommendation.
- Independently investigate Kernel B (`wy_solve_pallas`, `MB=16`
  sub-tile solve) via `cost_analysis()` and a micro-block-size sweep --
  not yet confirmed to share the same root cause as Kernel A/B4.
- Close the open items under section 5 (residual-parity test, remove
  stray `jax.checkpoint`, isolated bwd-only speed table, BF16 numbers,
  memory numbers, KAGGLE_LARGE coverage, multi-seed sweep) regardless of
  whether the hybrid remains a recommended path, since `beta/` code should
  still meet the project's normal evidentiary bar before any wider
  promotion.

> **Note on bwd vs fwd+bwd timings:** the `bwd` column is measured via `jax.vjp(loss, ...)`, which re-runs the forward pass internally to build the VJP closure before the backward pass executes. This is why `bwd` and `fwdbwd` numbers are nearly identical in the tables above/below -- it is an artifact of the measurement method (the forward cost is unavoidably included in both), not a claim that backward alone costs the same as forward+backward combined.
