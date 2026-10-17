# Known limitations

Evidence status: 17 October 2026. This documentation/evidence update leaves both kernel packages unchanged. Current measurements and numerical checks are separate from earlier attestation records.

## Hardware and shapes

- Fused execution targets TPU and head dimension 128. CPU interpretation or a reference fallback is not a TPU performance result. Consult each package's dispatcher and configuration before changing backend or layout.
- Current speed measurements cover one TPU v5e-1, H in {2,4,6}, L in {2048,4096}, D=128 and the documented batches. H>6, multi-chip scaling and other TPU generations are not measured here. There is no evidence here for a universal six-head hardware limit.
- `atomic_gdn2` requires `bt=2*bc`, compatible score/sub-block divisions, and power-of-two ladder geometry; see [config.py](atomic_gdn2/config.py). Sequence lengths must fit chunk/segmentation requirements. Its API uses B,H,L,D; the older package uses B,L,H,D.
- BTL pairwise decay has an FP32 exponential range boundary. The configured half-span `g_max*bs2/2` must respect the documented numerical envelope (88 is the overflow boundary, not a recommended target). Clipping and non-finite guards affect semantics outside the intended domain.
- `wy_eps` rescales triangular coupling. It is not diagonal regularization. All current comparisons use zero damping; nonzero damping results cannot be substituted into that comparison.

## Timing scope and comparator

- The peak **411.75 ± 5.13×** is a six-input gradient microbenchmark, including required forward work, relative to the supplied OLD affine associative scan. It is not forward-only latency or all seven input gradients.
- The **117.76 ± 0.15×** ratio is a full synthetic 34,873,952-parameter LM step relative to OLD with block rematerialization. NEW does not use the same rematerialization policy. This must accompany the headline comparison.
- OLD and JAX_REF are the author's implementations. The results do not establish superiority over every available external implementation. JAX_REF/NEW is **4.531 ± 0.004×** on the synthetic step.
- Kernel settings differ between generations (NEW ladder leaf 8; PALLAS_V1 leaf 16). The experiment compares the recorded configurations, not an exhaustive tuning search.
- Fixed input and initial optimizer state are replayed on every call. Compilation, dataset I/O, validation and checkpointing are excluded. This is not sustained training throughput of the larger published models.
- Full output trees are synchronized inside timing. Finite checks and file I/O occur outside it; surrounding host work can still influence execution. Paths run in separate processes, with order rotated between cases rather than every round.
- SD describes three input-seed summaries on the recorded VM, not cross-VM reproducibility or a confidence interval. Timing ratios use paired round ratios, so they need not equal ratios of displayed mean latencies.
- No new peak-HBM measurement is claimed. Historical memory and slope-decomposition numbers have different workloads/protocols and must not be mixed with these direct timings.

## Numerical agreement and optimizer sensitivity

### Parameter-update gate failures: seeds 0 and 1

The saved gradient snapshots pass the recorded relative-L2 threshold of 0.005 per tensor. Full-step parameters, optimizer states and loss also pass their recorded gates. **NEW/OLD parameter-update differences fail for seeds 0 and 1; NEW/JAX_REF update differences pass all three.** A passing parameter norm can hide a difference in a small update. The original comparison failures remain in the raw archives.

An update comparison first subtracts the initial parameters, so a small difference
that was negligible in a full parameter norm can be large relative to the update.
For AdamW with initially zero moments, the first bias-corrected adaptive term is
`g_c / (abs(g_c) + epsilon)`, where `g_c` is the gradient after global clipping.
With `epsilon=1e-8`, opposite gradients near that scale can yield visibly different
updates despite tiny absolute gradient differences. Weight decay adds its own term;
epsilon does not generate an adaptive update from an exactly zero gradient.

For the selected seed-0 coordinate `block_6/q_conv_w[236,2]`, OLD and NEW have opposite signs near 1e-8. Severe cancellation is present in the captured convolution factors. Exact summation retains **98.2542%** of their probed gradient difference, locating most of that difference before the final contraction. This does not establish the unique upstream operation responsible for it.

This supports a concrete cancellation/optimizer-sensitivity case, not an exhaustive
explanation of every seed-0/1 update failure. In particular, the separate seed-1
`decay_a[4]` discrepancy has not been localized by this capture. All current speed
measurements use zero damping, so damping alone cannot explain these failures.
The exact captured sums do not prove that every upstream difference is rounding
rather than an implementation issue. See the [numerical evidence](numerical/README.md)
for reports, controls and original archives.

- The exact oracle sums already-rounded captured FP32 factors; it is not the mathematically exact gradient of the full network.
- The fresh exact TPU run operates on frozen captures, plus an isolated convolution check. It does not rerun full-model Pallas backward or measure speed.
- Zero-probe instrumentation changes OLD seed 0 slightly while retaining its sign; it is not bitwise transparent for every path. Uninstrumented/probed controls are retained.
- The selected coordinate is not the separate failing seed-1 `decay_a[4]` coordinate. That separate discrepancy is not localized by this case study.
- Cross-factor percentage decomposition depends on the chosen ordering and is not a unique causal attribution.
- AdamW can amplify opposite small gradients through `g/(abs(g)+eps)` on its first bias-corrected step. Epsilon itself produces no adaptive update from exactly zero gradient and zero moments. Weight decay is a separate contribution.
- The controlled Decimal/FP64 experiment contains a deliberately constructed cancellation stress objective. It runs on CPU without Pallas; it does not measure the prevalence of this effect in natural training.
- Completed model training does not prove trajectory equivalence, universal correctness, or absence of a quality effect. Those claims require separate evidence.

## Artifact completeness and historical records

The current speed archive preserves timing samples, sources, environment metadata and tensor diagnostics. Large speed-run snapshot arrays were omitted from the compact archive; their hashes remain. Actual-coordinate arrays are included in the later numerical archive.

The controlled-rounding numerical worker completed, but its remote publication-export step exited 1. Tables and plots were recovered locally from the unchanged report; the recovery receipt is retained. Earlier incomplete/failed runs remain categorized rather than silently promoted to current results.

Historical attestation, MQAR, memory, and older performance records remain available under [attestation](attestation/README.md) and [artifacts](artifacts/README.md). They are not current headline speed evidence. For methods and raw data see [benchmarks](benchmarks/README.md) and [numerical](numerical/README.md).
