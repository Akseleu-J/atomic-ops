# Roadmap

Status legend:
- **VALIDATED** — measured end-to-end through the public API
  (`gdn2_pallas_forward_trainable` / `gdn2_forward_trainable`), with a
  correctness gate that passed and numbers that are published in
  `benchmarks/raw/` or `README.md`.
- **ISOLATED-ONLY** — a specific kernel or code path has been measured
  or proven correct in isolation (its own test file, its own benchmark
  script), but has NOT yet been measured end-to-end through the full
  pipeline it will ship in. Isolated validation is necessary but not
  sufficient for promotion to a default config.
- **HYPOTHESIS-REJECTED** — an investigated fix direction that measurement
  showed does not work. Kept here so it isn't re-attempted without new
  evidence.
- **OPEN** — a known cost or limitation with no fix in progress.

Nothing in this document is a claim about the current `KAGGLE_*` preset
defaults unless explicitly marked VALIDATED and cross-referenced to a
CHANGELOG entry. Isolated-kernel speedups are real numbers from real
benchmarks, but they are not a substitute for the end-to-end gate.

---

## Hypothesis (post-beta, not implemented): MXU-factorized pairwise decay (`use_centering`)

**Status:** HYPOTHESIS ONLY. No code for this exists anywhere in the
package -- no `KernelConfig` field, no kernel branch in `gdn2_fwd.py` /
`gdn2_bwd.py`, no `NotImplementedError` gate. Nothing below has been
measured in this repository; it is written down here so the idea isn't
lost or re-invented from scratch, and so it isn't attempted before its
listed prerequisite.

**Why this is being tracked at all:** `KNOWN_LIMITATIONS.md` section 1/6
identifies the pairwise decay computation in Kernel A
(`build_chunk_scores_pallas`) and its backward counterpart B4
(`intra_backward_pallas`) as VPU-bound (`_weighted_pair_sum` /
`_dL_pair_sum` / `_dR_pair_sum` / `_dgc_pair_sum`: broadcast + elementwise
multiply + manual reduction) rather than MXU-bound. In principle,
centering the pairwise decay term `exp(gc_i - gc_j)` around a shared
per-chunk reference point `gn` (e.g. `gn = gc[bt // 2]`) factors it into
two real matmuls (`q_scaled @ k_scaled.T`) instead of a VPU reduction,
which is the kind of change that could meaningfully close the forward gap
described in `KNOWN_LIMITATIONS.md` section 2.

**Explicit precondition -- do not start this before it is met:** the
`beta/gdn2_hybrid.py` path (JAX-forward + fused Pallas-backward, see
`KNOWN_LIMITATIONS.md` section 5) must be fully validated end-to-end
first (residual-parity test, BF16 numbers, memory numbers, full
deep-correctness suite -- see that section's open-items list). The
hybrid path is a smaller, already-working change; if it turns out to
close the forward/backward gap on its own, an MXU-factorized rewrite of
Kernel A/B4 may not be worth its implementation and validation cost. This
hypothesis is the fallback plan **if and only if** the hybrid path is
validated and still leaves a meaningful gap versus JAX_REF/PALLAS.

**What "validating this hypothesis" would require, if pursued (none of
this exists yet):**
1. A from-scratch implementation of the centered factorization in
   `_kernel_a_body`, gated behind a new, explicitly-named opt-in
   `KernelConfig` field (with its own `NotImplementedError` safety gate,
   matching how every other experimental knob in this codebase is
   introduced) -- not assumed to already exist.
2. Isolated correctness test (vs. the default/non-centered path) and
   isolated speed benchmark for Kernel A, then the same for the B4
   backward counterpart, including the backward gradient contribution
   through the shared reference point `gn` (chain rule through
   `eq_i = exp(clip(gc_i - gn))`, `ek_j = exp(clip(gn - gc_j))`) --
   this is exactly the kind of shared-variable backward term that is
   easy to compute but easy to forget to write back; any implementation
   must have an explicit isolated test for it, independently re-derived
   (not copy-pasted from the forward kernel), before it is trusted.
3. Full `custom_vjp` pipeline correctness (multi-seed vs.
   `gdn2_token_serial_reference`, finite-difference, `wy_eps` damping
   interaction, bf16 coverage, `KAGGLE_SMALL` blocking) -- per the
   layered strategy in `docs/TESTING_STRATEGY.md`.
4. End-to-end fwd/bwd/fwdbwd wall-clock through
   `gdn2_pallas_forward_trainable`, not just isolated kernel calls.
5. Peak HBM (`run_memory_benchmark.py`) for the new path.
6. A repeat of the kernel-gap diagnostic (sum of isolated per-kernel
   timings vs. full pipeline) to rule out a new dispatch/scheduling gap
   from the changed intermediate shapes.

**Do not** add a `use_centering` (or similarly named) field to
`KernelConfig`, add branches to `_kernel_a_body`/`_kernel_b4_body`, or
reference this hypothesis as an existing/gated/tested code path in
`README.md`, `CHANGELOG.md`, or `KNOWN_LIMITATIONS.md` until steps 1-6
above have actually been done. Until then this section is the only place
in the repo where this idea should be mentioned.

---

## Open, no fix scheduled: Kernel B (WY-solve)

**Status:** OPEN. Confirmed not a tile-size (`mb`) issue via sweep
(32/64/128 gave 44.9-64.0ms, no monotonic relationship) — bottleneck is
the sequential, data-dependent recursive block-forward-substitution in
`_block_solve` (`N_MICRO` sequential steps with data dependency between
them), which has no parallelism to expose to the MXU regardless of block
size.

**Why this could matter later:** *if* the post-beta `use_centering`
hypothesis above is ever implemented and validated, Kernel B would likely
become the dominant forward cost by a wide margin (an estimated ~51ms of
~59.6ms forward, i.e. ~86%, extrapolated from today's isolated Kernel
A/B4 VPU-vs-MXU numbers) -- itself unconfirmed and entirely contingent on
that hypothesis being pursued at all (see the section above).

**Candidate directions (none investigated yet):**
- Alternative block-triangular-solve factorization that exposes more
  independent work across micro-blocks (e.g. block-cyclic reduction
  instead of pure forward substitution).
- Investigate whether `N_MICRO` can be reduced by fusing multiple
  micro-block solves into fewer, larger MXU-friendly operations even
  if some redundant computation is introduced.
- `compiled.cost_analysis()` (flops / bytes-accessed) on Kernel B in
  isolation was flagged as "not yet done" as far back as the original
  kernel-gap diagnostic — still not done; would help distinguish
  HBM-bandwidth-bound from genuinely serialization-bound.

No target release. Treat as a standing research item, not a roadmap
milestone with a date.

---

## Completed (VALIDATED, shipped)

- Fused forward + backward Pallas kernels (Kernel A/B/C/D, B1-B5),
  `custom_vjp` trainable wrapper — v0.1.0.
- Kernel B4 fix (`_kernel_b4_body` performance bug) — v0.1.1, ~50%
  backward improvement, measured end-to-end and published in
  `benchmarks/raw/`.

(Nothing from the `use_centering` hypothesis appears in this section --
see the hypothesis note above; no code for it exists yet.)
