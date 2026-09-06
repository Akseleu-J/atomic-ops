# atomic_ops

**Fused Gated DeltaNet-2 (GDN-2) kernels for TPU v5e, written in JAX/Pallas.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TPU v5e](https://img.shields.io/badge/TPU-v5e-orange.svg)](https://cloud.google.com/tpu/docs/v5e)

A from-scratch port of the [NVlabs Gated DeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) Triton kernels
to `jax.experimental.pallas`, targeting **TPU v5e-8**. The backward pass is a single fused `custom_vjp`
that reuses forward residuals instead of recomputing them.

> **Headline numbers (measured, see [Benchmarks](#-benchmarks)):** on the full training shape
> (batch 8, seq 4096, 6 heads, d_head 128) the fused backward makes the training step
> **2.6× (FP32) / 3.4× (BF16) faster than the best pure-JAX WY baseline** and
> **27.2× (FP32) / 13.3× (BF16) faster than the widely used `associative_scan` baseline**.
> Across all measured shapes the gain vs `associative_scan` reaches **up to 38.8×** (FP32).
> The fused forward alone is currently **~1.6× slower** than the pure-JAX WY forward on TPU —
> training steps are backward-dominated, so end-to-end is still a clear win.
> A hybrid `JAX forward + Pallas backward` mode is planned (see [Limitations](#-limitations)).

---

## Features

- **Fully fused kernels** — forward (chunk scores → WY block solve → recompute → inter-chunk scan)
and backward (B1–B5 stage kernels), all in Pallas with TPU MXU tiling.
- **One-call training API** — `gdn2_forward_trainable` via `jax.custom_vjp`; no recomputation
of forward activations in the backward pass.
- **Automatic fallback** — on CPU/GPU or when `d_head != 128`, dispatches to a checkpointed
pure-JAX chunked-WY reference with identical damping semantics.
- **Numerical safety by default** — clipping + `nan_to_num` at every kernel boundary,
optional `wy_eps` Tikhonov-style damping of the WY solve, per-stage diagnostics
via the `GDN2_FWD_DIAG=1` environment flag.
- **Tunable configs** — `KernelConfig(bt, bc, mb, clip, wy_eps)` with presets tuned for
Kaggle TPU v5e (`KAGGLE_SMALL` / `KAGGLE_MEDIUM` / `KAGGLE_LARGE`) and an
`estimate_memory` / `get_recommended_config` helper.

## Installation

```bash
pip install atomic_ops

# or from source
pip install git+https://github.com/Akseleu-J/atomic_ops.git

# development
git clone https://github.com/Akseleu-J/atomic_ops.git
cd atomic_ops
pip install jax==0.11.1 jaxlib==0.11.1 libtpu==0.0.46 flax==0.12.9 optax==0.2.4
```

Requires Python ≥ 3.10 and `jax>=0.4.20`. For TPU, install the matching `jaxlib`/`libtpu`
builds (see [JAX TPU installation](https://jax.readthedocs.io/en/latest/installation.html)).

## Quick start

```python
import jax.numpy as jnp
from atomic_ops import gdn2_forward_trainable

# (batch, seq_len, heads, d_head); d_head must be 128 on TPU
shape = (4, 2048, 6, 128)
q = jnp.ones(shape, dtype=jnp.float32)
k = jnp.ones(shape, dtype=jnp.float32)
v = jnp.ones(shape, dtype=jnp.float32)
w = jnp.ones(shape, dtype=jnp.float32)
b = jnp.ones(shape, dtype=jnp.float32)
g = jnp.ones(shape, dtype=jnp.float32)   # log-decay gate, keep g <= 0
scale = shape[-1] ** -0.5

# Forward + backward (custom_vjp under the hood)
out, h_final = gdn2_forward_trainable(q, k, v, w, b, g, scale)

print(out.shape)      # (4, 2048, 6, 128)
print(h_final.shape)  # (4, 6, 128, 128)
```

### Runnable examples

Both examples work out of the box after `pip install atomic_ops`:

- [`examples/minimal_usage.py`](examples/minimal_usage.py) — 20-line minimal script:
build tensors, run `gdn2_forward_trainable`, print output shapes. Includes the
`is_tpu_available()` check so you can see which backend path was taken.
- [`examples/train_gdn2_layer.py`](examples/train_gdn2_layer.py) — a complete Flax +
Optax training step: a `GDN2Layer` module that projects `x` into `q,k,v,w,b,g`,
applies the gate via `-softplus(g)` (keeps `g <= 0`), auto-picks a config with
`get_recommended_config`, and runs one AdamW update step.

```bash
python examples/minimal_usage.py       # forward+backward, prints backend + shapes
python examples/train_gdn2_layer.py    # full training step with Flax/Optax
```

## API

| Function | Description |
| --- | --- |
| `gdn2_forward_trainable(q, k, v, w, b, g, scale, h0=None, config=None)` | Forward **+** gradients. Dispatches to Pallas on TPU with `d_head=128`, otherwise to the pure-JAX reference. This is what you want for training. |
| `gdn2_forward(q, k, v, w, b, g, scale, h0=None, config=None)` | Forward only, same auto-dispatch. |
| `gdn2_pallas_forward_trainable(...)` | Forces the fused Pallas path (raises on non-TPU / `d_head != 128`). |
| `gdn2_pallas_forward(...)` / `gdn2_pallas_forward_with_residuals(...)` | Inference-oriented Pallas forward; the `_with_residuals` variant also returns the intermediate tensors the backward pass needs. |
| `gdn2_token_serial_reference(...)` | Ground-truth token-by-token scan. Slow, numerically exact — used as an independent check in tests. |
| `gdn2_chunked_wy_reference(...)` | Chunked-WY reference (the fallback path). |
| `KernelConfig(bt, bc, mb, clip, wy_eps)` | Kernel blocking / numerical-safety config. Constraints: `bt = 2*bc`, `bc % mb == 0`. |
| `KAGGLE_SMALL` / `KAGGLE_MEDIUM` / `KAGGLE_LARGE` | Presets for TPU v5e-8 (default: `KAGGLE_MEDIUM`). |
| `estimate_memory(...)` / `get_recommended_config(...)` | Rough per-chip HBM estimate and preset auto-selection. |
| `is_tpu_available()` | `True` if a TPU device is visible to JAX (used by the fallback dispatcher). |

All tensor arguments have shape `(batch, seq_len, heads, d_head)` with `seq_len % config.bt == 0`.
The initial recurrent state `h0` is optional, shape `(batch, heads, d_head, d_head)`, float32.

## Benchmarks

Measured on **TPU v5e-8** with `jax==0.11.1`, `jaxlib==0.11.1`, `libtpu==0.0.46.1`.
Baselines: **OLD** = `jax.lax.associative_scan` GDN-2 (the formulation used in many research
codebases); **JAX_REF** = chunked WY recurrence in pure JAX. Mean steady-state over repeats, ms.

### Full training step — batch 8, seq 4096, 6 heads, d_head 128

| Dtype | Stage | OLD (ms) | JAX_REF (ms) | **Pallas (ms)** | vs OLD | vs JAX_REF |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 | fwd | 1067.24 | 63.24 | 102.08 | 10.5× | 0.62× |
| FP32 | bwd | 4767.24 | 458.99 | **175.33** | **27.2×** | **2.6×** |
| FP32 | fwd+bwd | 4762.70 | 460.03 | **175.20** | **27.2×** | **2.6×** |
| BF16 | fwd | 585.49 | 62.38 | 101.51 | 5.8× | 0.61× |
| BF16 | bwd | 2322.97 | 594.17 | **174.25** | **13.3×** | **3.4×** |
| BF16 | fwd+bwd | 2320.27 | 594.86 | **174.20** | **13.3×** | **3.4×** |

### All measured shapes — full fwd+bwd cycle, speedup vs baselines

| Config (batch, seq_len) | Dtype | Pallas (ms) | vs OLD (scan) | vs JAX_REF |
| --- | --- | --- | --- | --- |
| small (B=1, L=1024) | FP32 | 5.92 | 24.3× | 3.7× |
| small (B=1, L=1024) | BF16 | 6.00 | 11.5× | 3.8× |
| medium (B=4, L=4096) | FP32 | 88.07 | 30.2× | 3.1× |
| medium (B=4, L=4096) | BF16 | 87.60 | 14.9× | 3.8× |
| **train shape (B=8, L=4096)** | FP32 | **175.20** | **27.2×** | **2.6×** |
| **train shape (B=8, L=4096)** | BF16 | **174.20** | **13.3×** | **3.4×** |
| KAGGLE_SMALL preset (B=4, L=2048) | FP32 | 30.86 | **38.8×** | 3.9× |
| KAGGLE_SMALL preset (B=4, L=2048) | BF16 | 30.66 | 18.7× | 3.9× |
| KAGGLE_LARGE preset (B=8, L=4096) | FP32 | 175.34 | 27.2× | 2.6× |
| KAGGLE_LARGE preset (B=8, L=4096) | BF16 | 174.12 | 13.3× | 3.4× |

Read it honestly: the fused **forward is ~1.6× slower than the pure-JAX WY forward** today;
the fused **backward is 2.6–3.9× faster** than JAX_REF, and since real training steps are
backward-dominated, the full cycle wins by **2.6–3.9×** over the fastest pure-JAX baseline and
by **11.5–38.8×** over the `associative_scan` baseline depending on shape and dtype.
Full per-stage tables for all configs and both dtypes:
[`benchmarks/raw/benchmark_speed_final_averaged.md`](benchmarks/raw/benchmark_speed_final_averaged.md).

### Peak HBM — train shape (B=8, L=4096), forward+backward

| Dtype | OLD (MB) | JAX_REF (MB) | Pallas (MB) |
| --- | --- | --- | --- |
| FP32 | 1252.7 | 1250.0 | **1238.1** |
| BF16 | 915.5 | 915.5 | 915.5 |

Memory is on par with the pure-JAX reference (the backward reuses forward residuals instead of
recomputing them). Details:
[`benchmarks/raw/benchmark_memory_final_averaged.md`](benchmarks/raw/benchmark_memory_final_averaged.md).

### Reproduce

```bash
python benchmarks/run_speed_benchmark.py     # writes JSON + markdown tables
python benchmarks/run_memory_benchmark.py    # fork-isolated peak HBM measurement
```

Every timing run is correctness-gated before measurement (Pallas output must match the reference
within tolerance, otherwise the row is rejected).

## Correctness & testing

The test suite is deliberately layered — comparing implementations that share the same algebraic
derivation can hide derivation bugs, so the strongest checks are derivation-independent
(finite-difference gradients, token-serial scan). Full rationale:
[`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md).

- **CPU smoke tests** (seconds, no TPU needed, `interpret=True`):

```bash
  pytest tests/test_gdn2_full_math_correctness.py -v
```

- **Full suite** (requires TPU): multi-seed sweeps, finite-difference gradient checks,
isolated B3–B5 backward-stage tests, BF16 dtype-contract checks, `wy_eps` damping coverage,
alternative `KAGGLE_SMALL` blocking:

```bash
  pytest tests/extended/test_gdn2_deep_correctness.py -v
```

## Limitations

- **TPU-only fused kernels.** The Pallas path assumes TPU MXU tiling and `d_head = 128`;
other backends/dtypes automatically fall back to the pure-JAX reference (slower, correct).
- **Fused forward is currently slower than the pure-JAX WY forward** (~0.6×). If your workload
is inference-only, use `gdn2_forward` / `gdn2_chunked_wy_reference` until the hybrid
`JAX forward + Pallas backward` mode lands (planned for v0.3.0).
- `seq_len` must be divisible by `config.bt` (256 by default, 128 for `KAGGLE_SMALL`).
- `KernelConfig.bt` must equal `2 * config.bc`; vary `mb` for solver granularity.
- The pairwise decay computation (Kernel A / B4) currently uses a VPU-bound broadcast-reduce
pattern rather than an MXU matmul. An MXU-factorized alternative has been sketched as a
post-beta hypothesis but is not implemented in this release; see
[`ROADMAP.md`](ROADMAP.md) and [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md).

## Debugging

Set `GDN2_FWD_DIAG=1` to get per-stage reports of non-finite or suspiciously large
(`>1e6`) activations at every kernel boundary. Diagnostic only — never changes values.

```bash
GDN2_FWD_DIAG=1 python your_training_script.py
```

## Project layout

```javascript
atomic_ops/
├── atomic_ops/                  # the package
│   ├── configs.py               # KernelConfig, presets, sanitize/validate helpers
│   ├── gdn2_fwd.py              # forward kernels: A (scores), B (WY solve), C (recompute), D (scan)
│   ├── gdn2_bwd.py              # backward kernels B1–B5
│   ├── gdn2_pipeline.py         # custom_vjp trainable wrapper
│   ├── reference.py             # token-serial + chunked-WY pure-JAX references
│   ├── fallback.py              # auto-dispatch (TPU+d_head=128 -> Pallas, else reference)
│   └── utils.py                 # is_tpu_available, estimate_memory, get_recommended_config
├── benchmarks/                  # speed & memory benchmarks + raw results
├── tests/                       # CPU smoke tests
│   └── extended/                # full TPU correctness suite
├── examples/                    # minimal_usage.py + Flax training step
├── docs/TESTING_STRATEGY.md     # why the tests are built this way
└── .github/workflows/           # CI (tests, lint), publish to PyPI
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `pytest tests/test_gdn2_full_math_correctness.py`
must pass, Ruff must be green, kernel changes require the full TPU suite referenced in the PR.

## License

MIT — see [LICENSE](LICENSE). Kernels ported from the NVlabs Gated DeltaNet-2 Triton reference.

## Citation

```bibtex
@software{atomic_ops,
  author = {Omirbay, Akseleu},
  title  = {atomic_ops: Fused Gated DeltaNet-2 kernels for TPU v5e in JAX/Pallas},
  url    = {https://github.com/Akseleu-J/atomic_ops},
  license = {MIT},
  year   = {2026}
}
```

## Support

If this package is useful in your research, consider giving it a ⭐ — it helps other researchers find it.
Bug reports and questions go to [Issues](https://github.com/Akseleu-J/atomic_ops/issues).
# atomic-ops
