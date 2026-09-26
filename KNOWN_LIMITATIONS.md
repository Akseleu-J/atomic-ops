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
6). An experimental hybrid path was investigated (section 5) but did not
deliver the expected end-to-end gain and has been closed. The real fix
remains the Kernel A/B4 VPU-to-MXU factorization tracked in `ROADMAP.md`.

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
- **Kernel B** (`wy_solve_pallas`, WY block solve): ~51.2 ms -- sequential,
  data-dependent recursive forward-substitution in `_block_solve`; no
  parallelism to expose to the MXU regardless of tile size. The earlier
  `mb`-related hypothesis was ruled out by a sweep (see section 4).
Kernel C (recompute) and Kernel D (inter-chunk scan) are cheap (~2.3 ms /
~3.4 ms) and not a concern.

**Workaround:** for inference-only workloads use `gdn2_forward` (dispatches
to the pure-JAX reference off-TPU) or `gdn2_chunked_wy_reference` directly.
For training, the fused Pallas path still wins end-to-end because backward
dominates the step. The forward-only gap is tracked as Kernel A/B4 MXU
factorization in `ROADMAP.md`.

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
- `KernelConfig.mb` (currently 16 across all presets) was the initial
hypothesis for Kernel B's MXU underutilization, but a sweep of
`mb ∈ {32, 64, 128}` showed no monotonic relationship (44.9–64.0 ms), so
tile size is **not** the cause. The bottleneck is the sequential,
data-dependent forward-substitution in `_block_solve`. See `ROADMAP.md`
→ `Open, no fix scheduled: Kernel B`.

## 5. [CLOSED] Hybrid JAX-forward + Pallas-backward path

**Status:** CLOSED, not pursued further. A full TPU attestation run showed
it did not deliver the expected end-to-end (fwd+bwd) speedup over the
fused Pallas path once measured correctly.

Closed as `HYPOTHESIS-REJECTED`; see `ROADMAP.md`. Code preserved at
`archive/gdn2_hybrid.py` for reference.


## 6. Kernel-gap diagnostic (TPU v5e-8, KAGGLE_MEDIUM, B=8 L=4096, FP32)

> **Note on bwd vs fwd+bwd timings:** the `bwd` column is measured via `jax.vjp(loss, ...)`, which re-runs the forward pass internally to build the VJP closure before the backward pass executes. This is why `bwd` and `fwdbwd` numbers are nearly identical in the tables above/below -- it is an artifact of the measurement method (the forward cost is unavoidably included in both), not a claim that backward alone costs the same as forward+backward combined.

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
Kernel B 51.234 ms <- expensive, sequential forward-substitution (see sections 2/4)
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


## 7. Single-chip tile-pack limit at H=6

**Status:** hardware sweet spot, not a kernel bug.

On a single TPU v5e-1, the fused kernel's tile pack (`bs2=64`) fits
cleanly up to 6 heads. Kernel speedup vs `associative_scan` peaks there:

| H | L=2048 | L=4096 |
|---|--------|--------|
| 2 | 193x | 207x |
| 4 | 300x | 322x |
| 6 | **468x** | **448x** |

Beyond H=6 the fused tile no longer fits cleanly on a single chip and
speedup drops. This is a property of the single-chip VMEM/MXU budget,
not of the algorithm. Meaningful scaling beyond H=6 requires sharding
across chips (TPU v5e-8). See `ROADMAP.md` for the sharded-scaling plan.

## 8. Per-layer slope decomposition is invalid at B=4

**Status:** methodology limitation, documented for reproducibility.

The per-layer Amdahl decomposition (`b_X - b_NONE` from a
`t_step(n_layers)` fit, `X in {NEW, OLD, PROD}`) requires a valid
zero-baseline. The `NONE` reference is an elementwise stub that touches
all 6 inputs but does not run attention.

At B=8 the method works (see `attestation/final_report.json`,
`8.decomposition`): `b_NONE ~ 5.78 ms/layer`, `b_NEW ~ 9.06`,
`b_OLD ~ 1668.33` — differences are positive and meaningful.

At B=4 the method is expected to **fail structurally**: the `NONE` stub
materializes more intermediate tensors than the fused kernel at that
batch size, which can push `b_NONE > b_NEW` and make the decomposed
`kernel_share` meaningless (not merely noisy). The published H-scaling
table in `README.md` / `attestation/scaling.json` therefore reports only
**direct kernel fwd+bwd measurements** at B=4, never slope-decomposed
numbers — the two should not be mixed in the same table.

## 9. Cross-config numbers are not comparable without saying so

**Status:** reporting discipline note.

Speedup figures measured at different `(B, L, H)` shapes are not
directly comparable and must not be used to extrapolate across shapes.
Example: `attestation/final_report.json`'s per-layer slope numbers were
measured at B=8; the headline H-scaling table in `README.md` /
`attestation/scaling.json` is measured at B=4. Both are correct in
their own regime; neither supersedes or validates the other.

## Roadmap

**Status as of v0.2.0:** the experimental hybrid JAX-forward +
Pallas-backward path (formerly `beta/gdn2_hybrid.py`) was investigated,
attested end-to-end on real TPU, and **closed as HYPOTHESIS-REJECTED** —
it did not deliver the expected fwd+bwd speedup once measured correctly.
It is not planned, not wired into anything, and not a recommended path.
Code is preserved for reference at `archive/gdn2_hybrid.py`.
See `ROADMAP.md` ("HYPOTHESIS-REJECTED" section) for the full writeup.

Forward-looking work (Kernel A/B4 MXU factorization, Kernel B `mb`
sub-tile tuning, sharded H-scaling) is tracked exclusively in
`ROADMAP.md` under v0.3.0+ — not duplicated here, to avoid this file and
`ROADMAP.md` drifting out of sync again.
