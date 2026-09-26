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
- **HYPOTHESIS-ONLY** — a proposed direction that exists only on paper
  (no implementation, no tests, no benchmarks in this repo). It is
  tracked for prioritization and to prevent duplicate re-invention,
  but it must not be referenced as an existing or planned feature until
  the preconditions listed in its section are met and it has been
  promoted to at least ISOLATED-ONLY.
- **HYPOTHESIS-REJECTED** — an investigated fix direction that measurement
  showed does not work. Kept here so it isn't re-attempted without new
  evidence.
- **OPEN** — a known cost or limitation with no fix in progress.

Nothing in this document is a claim about the current `KAGGLE_*` preset
defaults unless explicitly marked VALIDATED and cross-referenced to a
CHANGELOG entry. Isolated-kernel speedups are real numbers from real
benchmarks, but they are not a substitute for the end-to-end gate.

---

## Completed (VALIDATED, shipped)

- Fused forward + backward Pallas kernels (Kernel A/B/C/D, B1-B5),
  `custom_vjp` trainable wrapper — v0.1.0.
- Direct kernel H-scaling measurements on TPU v5e-1 (B=4, D=128, `bs2=64`)
  — v0.2.0. Peak 468x at H=6, L=2048. See `attestation/scaling.json`.

(Nothing from the `use_centering` hypothesis appears in this section —
no code for it exists yet.)

---

## HYPOTHESIS-REJECTED

### Hybrid JAX-forward + Pallas-backward path (rejected 2026-09-27)

Investigated as an opt-in experimental path (originally `beta/gdn2_hybrid.py`).
A full TPU attestation run showed no end-to-end (fwd+bwd) speedup over the
fused Pallas path once measured correctly. Closed; not pursued further.
Code archived at `archive/gdn2_hybrid.py`.

---

## Open, no fix scheduled: Kernel B (WY-solve)

**Status:** OPEN. Confirmed not a tile-size (`mb`) issue via sweep
(32/64/128 gave 44.9–64.0 ms, no monotonic relationship) — bottleneck is
the sequential, data-dependent recursive block-forward-substitution in
`_block_solve` (`N_MICRO` sequential steps with data dependency between
them), which has no parallelism to expose to the MXU regardless of block
size.

**Why this could matter later:** *if* the `use_centering` hypothesis below
is ever implemented and validated, Kernel B would likely become the
dominant forward cost by a wide margin (an estimated ~51 ms of ~59.6 ms
forward, i.e. ~86%, extrapolated from today's isolated Kernel A/B4
VPU-vs-MXU numbers) — itself unconfirmed and entirely contingent on that
hypothesis being pursued at all.

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

## Research hypothesis: MXU-factorized pairwise decay (`use_centering`)

> **Author's estimate:** back-of-envelope FLOP/tiling calculations suggest
> this could further accelerate the fused forward at very large head counts.
> For `atomic_gdn2` v0.2.0 the forward is already faster than both
> `atomic_ops` v0.1.0 Pallas forward and pure-JAX WY reference — this is
> a marginal-gain optimization, not a gap-closing fix. This estimate is **not yet backed by any
> implementation or benchmark** in this repository; treat it as a strong
> prior for prioritization, not as a validated result.

**Status:** HYPOTHESIS-ONLY. No code for this exists anywhere in the
package — no `KernelConfig` field, no kernel branch in `gdn2_fwd.py` /
`gdn2_bwd.py`, no `NotImplementedError` gate. Nothing below has been
measured in this repository; it is written down here so the idea isn't
lost or re-invented from scratch.

**Why this is being tracked at all:** `KNOWN_LIMITATIONS.md` section 1/6
identifies the pairwise decay computation in Kernel A
(`build_chunk_scores_pallas`) and its backward counterpart B4
(`intra_backward_pallas`) as VPU-bound (`_weighted_pair_sum` /
`_dL_pair_sum` / `_dR_pair_sum` / `_dgc_pair_sum`: broadcast + elementwise
multiply + manual reduction) rather than MXU-bound. In principle,
centering the pairwise decay term `exp(gc_i - gc_j)` around a shared
per-chunk reference point `gn` (e.g. `gn = gc[bt // 2]`) factors it into
two real matmuls (`q_scaled @ k_scaled.T`) instead of a VPU reduction,
which is the kind of change that could further accelerate the forward pass
at large head counts (marginal-gain optimization).

**Precondition:** for `atomic_gdn2` v0.2.0 the fused forward is already
faster than both `atomic_ops` v0.1.0 Pallas forward and pure-JAX WY
reference on the measured shapes. The earlier hybrid path (JAX-forward +
Pallas-backward) was investigated and **closed as HYPOTHESIS-REJECTED**.
This `use_centering` factorization remains a candidate for further
forward-only optimization at larger head counts, but is not required to
close any current v0.2.0 gap.

**What "validating this hypothesis" would require, if pursued (none of
this exists yet):**
1. A from-scratch implementation of the centered factorization in
   `_kernel_a_body`, gated behind a new, explicitly-named opt-in
   `KernelConfig` field (with its own `NotImplementedError` safety gate,
   matching how every other experimental knob in this codebase is
   introduced) — not assumed to already exist.
2. Isolated correctness test (vs. the default/non-centered path) and
   isolated speed benchmark for Kernel A, then the same for the B4
   backward counterpart, including the backward gradient contribution
   through the shared reference point `gn` (chain rule through
   `eq_i = exp(clip(gc_i - gn))`, `ek_j = exp(clip(gn - gc_j))`) —
   this is exactly the kind of shared-variable backward term that is
   easy to compute but easy to forget to write back; any implementation
   must have an explicit isolated test for it, independently re-derived
   (not copy-pasted from the forward kernel), before it is trusted.
3. Full `custom_vjp` pipeline correctness (multi-seed vs.
   `gdn2_token_serial_reference`, finite-difference, `wy_eps` damping
   interaction, bf16 coverage, `KAGGLE_SMALL` blocking) — per the
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
`README.md`, `CHANGELOG.md`, or `KNOWN_LIMITATIONS.md` until steps 1–6
above have actually been done. Until then this section is the only place
in the repo where this idea should be mentioned.

---

## v0.3.0 — planned

### Sharded H-scaling on TPU v5e-8

**Status:** VALIDATED-HYPOTHESIS, not yet run.

Single-chip measurements on TPU v5e-1 (B=4, D=128, `bs2=64`) show kernel
speedup peaks at H=6 (468x at L=2048, 448x at L=4096) because the fused
kernel's tile pack fits cleanly up to 6 heads, after which it stops
fitting and speedup drops.

On TPU v5e-8, H is a natural sharding axis (each chip holds H/8 heads
locally; state is per-head independent, no cross-chip sync in the scan).
The per-chip tile-pack sweet spot of 6 heads predicts a **global peak
near H=48** (H_local = 6), with speedup roughly flat from H=32 to H=64.
This is a prediction, not a measurement — nothing below has been run.

**Plan:**
- Mesh: `(8,)` shard over the `h` axis
- Config: B=4 (or B=2 if HBM-bound), L=2048, `bs2=64`, D=128
- Sweep: H in {16, 24, 32, 48, 64}
- Baseline: same `associative_scan` OLD at identical mesh
- Success criterion: kernel speedup at H=48 >= 400x with correctness
  `rel_l2(NEW, JAX_REF) < 5e-2`

**Why it might not peak at 48:**
- `jax.lax.associative_scan` may behave differently under sharding
  (all-gather on cross-chip state) — could push the peak higher.
- ICI latency per chunk-scan sync could pull the peak lower.
- 2D sharding (H x B) may be required to keep per-chip batch >= 2.

### 2D sharding (H x B) exploration

**Status:** HYPOTHESIS-ONLY.

At large H with B=4, H-only sharding leaves per-chip batch = 4/B_shard.
For `bs2=64` granularity, per-chip batch >= 2 is preferred. Candidate
meshes with `H_shard * B_shard = 8`: (8,1), (4,2), (2,4), (1,8).
Not yet implemented or measured.

### E2E scaling on sharded setup

**Status:** HYPOTHESIS-ONLY.

Kernel-only scaling is what the section above measures. End-to-end
(`t_step(n_layers)`) sharded scaling is separate and not yet run.
Prediction: E2E peaks earlier than the kernel (H ~ 24–32) because
projections/MLP/optimizer do not scale with H, growing the Amdahl
residual. Needs its own measurement before being stated as fact.

### Kernel A / B4 MXU factorization

See the `Research hypothesis: MXU-factorized pairwise decay` section
above for the full precondition, gates, and status.

---

## v0.4.0 — exploratory

### Multi-chip chunk-scan (pipeline parallel)

**Status:** HYPOTHESIS-ONLY, not designed.

`num_chunks` is another natural sharding axis: chunks depend on each
other only through the recurrent state. Pipeline-parallel chunk-scan
across chips could allow much longer sequences (L=16384+) without
increasing per-chip memory.

### Flash-attention-style pairwise decay

**Status:** HYPOTHESIS-ONLY, not designed.

Streaming tiled computation of the pairwise decay term (instead of
materializing the full matrix per chunk) could reduce VMEM pressure and
possibly enable larger `bs2`.

---

## v0.5.0+ — research

### Forward-pass parity with pure-JAX WY

**Status:** open question, contingent on v0.3.0 items above.

For `atomic_ops` v0.1.0 the fused forward is ~0.62x of the pure-JAX WY
reference (see `KNOWN_LIMITATIONS.md` section 2). `atomic_gdn2` v0.2.0
closes this gap: on the measured train shape its forward is faster than
both the v0.1.0 Pallas path and the pure-JAX WY reference. Remaining
forward-only work is tracked as the `use_centering` research hypothesis
above, targeting very large head counts rather than closing any current
v0.2.0 gap. Nothing here should be read as re-opening the closed hybrid
path without new evidence.
