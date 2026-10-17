# atomic-ops

**Gated DeltaNet-2 kernels for TPU, implemented in JAX/Pallas.**

Author: **Akseleu Omirbay**. This repository contains two generations:

| Package / benchmark name | Role |
|---|---|
| `atomic_gdn2` / NEW | Original TPU implementation: chunked WY, fused backward, custom VJP and configurable precision. |
| `atomic_ops` / PALLAS_V1 | Earlier Pallas port of the NVlabs Triton reference; retained as an intermediate implementation. |

The benchmark also includes OLD (the author's affine `associative_scan` implementation) and JAX_REF (the author's pure-JAX chunked-WY implementation). These are implementation comparisons, not comparisons against an external optimized library.

## Current measured performance

Single **TPU v5e-1**, three seeds, zero damping; benchmarked kernel revision
`11616acd8a061768a38feb12c58e5e329952f5d5`. The October 2026 update adds evidence and documentation; it changes neither kernel implementation nor package version.

### Gradient microbenchmark

**Peak measured speedup: 411.75 ± 5.13× relative to OLD**, at B=4, H=6, L=2048, D=128.
The timed workload differentiates `sum(o**2) + sum(h_final**2)` with respect to the six sequence inputs; initial state is fixed. These are gradient-evaluation timings, including required forward computation.

| Heads | Length | OLD (ms) | NEW (ms) | OLD / NEW, mean ± SD |
|---:|---:|---:|---:|---:|
| 2 | 2048 | 339.686 | 1.701 | 199.70 ± 3.01× |
| 2 | 4096 | 678.852 | 2.943 | 230.73 ± 2.45× |
| 4 | 2048 | 751.936 | 2.702 | 278.34 ± 4.32× |
| 4 | 4096 | 1511.075 | 10.493 | 144.02 ± 1.14× |
| 6 | 2048 | 1188.874 | 2.888 | 411.75 ± 5.13× |
| 6 | 4096 | 2384.967 | 13.818 | 172.80 ± 7.18× |

### Complete synthetic language-model step

Eight layers, **34,873,952 parameters**, B=8, L=2048, width 512, H=6, D=128,
vocabulary 256; next-token cross entropy, global gradient clipping and AdamW.

| Path | Step latency, mean ± SD (ms) | OLD / path, mean ± SD |
|---|---:|---:|
| OLD | 24101.386 ± 0.855 | 1.00× |
| JAX_REF | 927.546 ± 0.879 | 25.98 ± 0.03× |
| PALLAS_V1 | 627.458 ± 0.131 | 38.41 ± 0.01× |
| NEW (`atomic_gdn2`) | **204.659 ± 0.262** | **117.76 ± 0.15×** |

NEW is **4.531 ± 0.004×** faster than JAX_REF in this step. OLD uses block rematerialization to fit memory; the other paths do not. This is a fixed-state replay benchmark, not an observed throughput measurement of the larger trained models or a data-loading pipeline.

### Measurement and numerical status

Compilation is excluded; every timed call synchronizes the entire returned tree. Each path has three seeds × three rounds × two warmups and four timed calls per round: **1008 timed calls** across 28 cases. Latency uses per-round medians, then the median across rounds for each seed. Reported SD is sample SD across three seed summaries. Speedup uses paired per-round ratios, summarized by seed; it is not the ratio of displayed mean latencies. This SD is neither a confidence interval nor variation across machines.

Kernel gradient snapshots pass the recorded relative-L2 threshold. Full-step parameter, optimizer-state and loss snapshots pass their recorded gates. **Parameter-update differences between NEW and OLD fail the recorded gate for seeds 0 and 1**; NEW/JAX_REF update comparisons pass all three. These separate outcomes are retained, not averaged away.

See [complete benchmark protocol and raw timings](benchmarks/README.md),
[numerical evidence](numerical/README.md), and [known limitations](KNOWN_LIMITATIONS.md).

## Numerical evidence: cancellation and AdamW sensitivity

The audit traces a real full-model coordinate, `block_6/q_conv_w[236,2]`, through saved inputs and cotangents. For seed 0, OLD and NEW give opposite signs near AdamW's epsilon scale. **98.2542%** of their probed gradient difference remains when the last contraction is replaced with an exact sum of the captured FP32 factors. The sign difference is therefore already present before the final reduction.

A fresh TPU integer-bin audit confirms **27 exact cross contractions, nine decompositions and nine permutation checks**, with eight controls. An independent standard-library verifier reconstructs exact rational results from IEEE binary32 values and checks device bins.

This establishes cancellation and sensitivity in the captured computation. It does not identify a unique upstream cause, prove the true full-network gradient is zero, or establish unchanged training quality. The exact TPU audit reuses frozen factors; it does not rerun the full network. See [methods, controls, logs and scope](numerical/README.md).

## Implementation and usage

`atomic_gdn2` uses a residual-carrying custom VJP, selective recomputation, chunked WY solves, and fused backward stages. Long sequences may require multiple segmented calls. `wy_eps` rescales triangular coupling; it is not diagonal Tikhonov loading. The current measurements use `wy_eps=0`.

Install the source checkout:

```bash
git clone https://github.com/Akseleu-J/atomic-ops.git
cd atomic-ops
pip install -e .
```

For exact benchmark dependencies and source snapshots, use the [self-contained Colab notebook](benchmarks/notebooks/all-paths-speed.ipynb). The measured environment is Python 3.13.16, JAX/JAXlib 0.11.2, libtpu 0.0.48.1, NumPy 2.1.3, Flax 0.12.9 and Optax 0.2.8. This documentation update does not publish a new distribution.

```python
import math
import jax.numpy as jnp
from atomic_gdn2 import make_cfg, make_blr_trainable

cfg = make_cfg(wy_eps=0.0)
fn = make_blr_trainable(cfg, 1.0 / math.sqrt(128))
shape = (1, 2, 256, 128)  # batch, heads, sequence, head dimension
q, k, v, w = (0.01 * jnp.ones(shape, jnp.float32) for _ in range(4))
b = 0.5 * jnp.ones(shape, jnp.float32)
g = -0.05 * jnp.ones(shape, jnp.float32)
o, h_final = fn(q, k, v, w, b, g)
```

The older `atomic_ops` API uses `(batch, sequence, heads, head_dimension)` instead. The examples contain both package generations; use the API/layout stated by each file.
See [examples](examples) and [configuration](atomic_gdn2/config.py); the package APIs and layouts should not be interchanged.

## Earlier validation and provenance

The [historical attestation](attestation/README.md) preserves multilayer correctness, finite differences, MQAR and training-scale experiments. Its aggregated report records 145 PASS and one OLD baseline OOM. These are historical results with their own configurations, not a rerun of every gate in the current audit. Historical MQAR includes NEW easy 0.9999 and hard 0.9986; see the original report for task definitions and controls.

Model-training provenance and the October speed experiment have different recorded kernel revisions. Existing model training used `0dbc59cf74883315d7adbebe9229c520663e0d58`; the present benchmark pins the revision stated above. Completed training supports practical use but does not prove equality of optimizer trajectories between implementations.

## Repository structure

```text
atomic_gdn2/                 current kernel implementation (unchanged)
atomic_ops/                  earlier Pallas implementation (unchanged)
benchmarks/
  README.md                  current timing protocol and scope
  results/2026-10-17/         aggregate and individual timing CSVs
  notebooks/                 self-contained four-path speed experiment
numerical/
  README.md                  findings, controls and limits
  experiments/2026-10-17/     rounding, actual-coordinate and exact-TPU evidence
  notebooks/                 reproduction notebooks
  tools/                     exact TPU accumulator and independent verifier
artifacts/
  INDEX.json                 original archive roles and SHA256 hashes
  current-evidence/          four current raw experiment archives
  supporting-diagnostics/   earlier root-cause investigations
  historical-not-for-current-claims/  superseded or partial experiments
  historical-documentation/ prior documentation preserved verbatim
attestation/                original historical validation records
tests/ and tests_gdn2/       existing validation suites
KNOWN_LIMITATIONS.md        hardware, measurement and numerical boundaries
```

[Original archive index](artifacts/README.md) · [Testing strategy](docs/TESTING_STRATEGY.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md).

## Citation and license

```bibtex
@software{akseleu_atomic_gdn2_2026,
  author = {Akseleu Omirbay},
  title = {atomic_gdn2: Gated DeltaNet-2 kernels for TPU in JAX/Pallas},
  year = {2026},
  url = {https://github.com/Akseleu-J/atomic-ops}
}
```

The earlier `atomic_ops` publication has [Zenodo DOI 10.5281/zenodo.22706659](https://doi.org/10.5281/zenodo.22706659); that archival record is distinct from the current evidence update.

Both packages use the [MIT License](LICENSE). `atomic_ops` preserves attribution to the NVlabs Gated DeltaNet-2 Triton reference. `atomic_gdn2` is Akseleu Omirbay's original TPU implementation of the published recurrence. See [contribution guidance](CONTRIBUTING.md) and [issues](https://github.com/Akseleu-J/atomic-ops/issues).
