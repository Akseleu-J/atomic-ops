# atomic-ops

**Gated DeltaNet-2 kernels for TPU, written in JAX/Pallas.**

[Implementation evidence](docs/PUBLICATION_EVIDENCE.md) · [Historical attestation](attestation/) ·
[Corrected reproduction protocol](docs/REPRODUCTION_PROTOCOL.md) ·
[Language-model experiment and paper](https://github.com/Akseleu-J/atomic-gdn2)

## Two implementations, one distribution

| Import | Generation | Scope |
|---|---|---|
| `atomic_ops` | v0.1 | Legacy Pallas port of the NVlabs Triton implementation |
| `atomic_gdn2` | v0.2 | TPU-oriented block-local-rescaling implementation used by the current models |

The current 2026-10-16 Baseline/hybrid training runs used `atomic_gdn2` from immutable commit
`0dbc59cf74883315d7adbebe9229c520663e0d58`. This publication update changes documentation
and the benchmark harness, not the numerical kernel code. Do not mix v0.1 and v0.2 timings or APIs.

## Current v0.2 design

- Three Pallas forward stages: decay-aware scores; leaf-batched WY ladder; fused factor reconstruction and chunk scan.
- Fused B2/B1/B3/B4/B5 custom backward, one Pallas call per segment, reverse recurrence over chunks.
- Saved chunk-level residuals plus selective reconstruction of local factors. Training storage grows with chunk count.
- Balanced two-leg (BTL) decay factorization and configurable numerical envelope.
- Head-major API `(B,H,L,D)`, initial/final recurrent state, and segmentation for long sequences.

```python
from atomic_gdn2 import make_cfg, make_blr_trainable
cfg = make_cfg(bt=128, bc=64, score_bs=128, bs2=32, mb=8, interpret=False)
fwd = make_blr_trainable(cfg, scale=128 ** -0.5)
# q,k,v,w,b,g: (B,H,L,D), L divisible by bt; b in [0,1], g <= 0.
# h0: (B,H,D,D), or None for zeros.
o, h_final = fwd(q, k, v, w, b, g, h0=None)
```

Install with `pip install atomic-ops`, or pin the recorded Git commit for a specific experiment.
The demonstrated accelerated configuration is TPU v5e with D=128. `interpret=True` is a CPU debugging
mode for Pallas; it is not the same claim as automatic dispatch to a pure-JAX WY fallback.
The old `atomic_ops.fallback` API belongs to the legacy generation.

## Numerics and memory

Defaults: `dot_mode=highest`, `solve_dot_mode=bf16x3`, `b3_dot_mode=bf16x3`; B4 explicitly
uses highest. bf16x3 uses two hi/lo operand components and three products. The solve is
`(I + (1-eps) Akk)^-1`, default eps=1e-3, not eps-I diagonal loading.

`make_cfg` defaults g_max to `160/bs2`; for bs2=32 the intended half-span bound is 80<88.
User overrides/inputs must respect the envelope; arbitrary configurations are not guaranteed safe.
Clipping and NaN replacement occur at kernel boundaries. Their presence is not proof that invalid
inputs are harmless. The forward saves score/solve matrices, chunk-start states and updated values;
the backward selectively recomputes local factors. Neither constant-memory training nor a whole
forward/backward in one launch is claimed. Default max_chunks_per_call=32.

## Historical measurements: reported, not newly rerun

`attestation/scaling.json` reports the following ratios against its supplied associative-scan
wrapper, single TPU v5e-1, B=4, D=128, bs2=64:

| Heads | L=2048 | L=4096 |
|---:|---:|---:|
| 2 | 192.83× | 206.90× |
| 4 | 299.59× | 321.69× |
| 6 | 467.64× | 448.10× |

Forward relative L2 values in that record are 2.486e-6 and 2.481e-6 for two L=2048 shapes.
The historical aggregate contains 145 PASS entries and one OLD MQAR OOM; entries are not all
numerical correctness tests. Raw per-notebook logs are not committed with the aggregate.

**Reproduction qualification:** the previous scaling notebook contained call-arity and workload-
equivalence defects (h0, scale, final state and normalized OLD composition). It had no saved outputs.
A corrected harness now gates timing on forward and gradient parity under equal settings.
It has not yet been timed on TPU, so the table is not a new equivalent-workload speedup claim.

The previously documented 200.17× small-LM figure lacks its matching raw artifact in the inspected
aggregate. A different L=4096, 27.49M aggregate records NEW=169.6 ms, OLD=26764.55 ms (157.81×),
PROD=757.65 ms (4.47× vs NEW). These are distinct historical records, not the speedup of the
current 307M hybrid over the 334M Baseline. Detailed scope and discrepancies are documented in
[PUBLICATION_EVIDENCE.md](docs/PUBLICATION_EVIDENCE.md).

## Reproduce with a parity gate

```bash
python scripts/reproduce_kernel_bench.py --shape 4,2048,6,128 --output benchmark.json
```

Run on a TPU from an explicitly pinned source revision. The harness uses identical nonzero h0,
real final states, output scale/objective, equal precision/repeats, and zero damping for equivalence.
It emits a timing ratio only after forward/gradient gates pass; failed/OOM baselines are recorded.
See [protocol](docs/REPRODUCTION_PROTOCOL.md). No new TPU benchmark was run in this publication update.

## Model-level evidence

The four recent language-model runs use the same kernel for both architectures. Their measured
BPB and threshold results are in [Atomic-GDN2](https://github.com/Akseleu-J/atomic-gdn2) and
the [model card](https://huggingface.co/Akseleu/Atomic-GDN2). Those results evaluate a complete
hybrid architecture, not an ablation of the kernel or DAR.

## License and attribution

See LICENSE and CITATION.cff. GDN-2 recurrence and the legacy port retain their upstream attribution;
the v0.2 implementation is positioned by this repository as a distinct TPU-oriented design.
