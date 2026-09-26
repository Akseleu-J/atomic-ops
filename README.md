# atomic_ops

## Packages

This repository ships **two generations** of GDN-2 Pallas kernels:

### `atomic_ops` v0.1.0 — stable
Fused forward + backward GDN-2 kernels. **10–39×** faster than
`associative_scan`. PyPI published.

### `atomic_gdn2` v0.2.0 — next generation (shipped inside `atomic-ops` 0.2.0)
- Unified **mega-backward** (B2+B1+B3+B4+B5 in one Pallas launch)
- **T-22 numerical boundary** documented (`half_span < 88`)
- **7 specialized pytest suites** (`tests_gdn2/`, T1–T6 + T9) — finite
  differences, domain guard, canary, phantom-param audit, grad-vs-ref,
  train smoke. Plus a separate **T8 bench script** at
  `tools_gdn2/bench.py` (manual, not part of CI).
- Zero-shot MQAR to **2048 tokens** at 0.9995

```bash
pip install -e .
pytest tests_gdn2/ -v -m "not slow"
```

> `atomic_gdn2` is research-grade. `atomic_ops` remains stable production path.


**Fused Gated DeltaNet-2 (GDN-2) kernels for TPU v5e, written in JAX/Pallas.**
<p align="center">
  <a href="https://pypi.org/project/atomic-ops/">
    <img src="https://img.shields.io/pypi/v/atomic-ops?style=flat-square&color=blue" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/atomic-ops/">
    <img src="https://img.shields.io/pypi/pyversions/atomic-ops?style=flat-square" alt="Python">
  </a>
  <a href="https://github.com/Akseleu-J/atomic-ops/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Akseleu-J/atomic-ops?style=flat-square" alt="License">
  </a>
  <a href="https://cloud.google.com/tpu/docs/v5e">
    <img src="https://img.shields.io/badge/TPU-v5e-orange?style=flat-square" alt="TPU v5e">
  </a>
  <a href="https://doi.org/10.5281/zenodo.22706659">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22706659.svg" alt="DOI">
  </a>
</p>

---

A from-scratch port of the [NVlabs Gated DeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) Triton kernels to `jax.experimental.pallas`, targeting **TPU v5e-8**. The backward pass is implemented as a single fused `custom_vjp` that reuses forward residuals instead of recomputing them.
**Kernel speedup vs `associative_scan`** — direct measurement, single TPU v5e-1,
B=4, D=128, `bs2=64`:

| H | L=2048 | L=4096 |
|---|--------|--------|
| 2 | 193x | 207x |
| 4 | 300x | 322x |
| 6 | **468x** | **448x** |

Correctness: `rel_l2(NEW, JAX_REF) = 2.5e-06` at H in {4,6}, L=2048.

**Attestation note.** The correctness/training cells referenced in
this README are an **external TPU attestation suite** (16 notebooks,
real v5e-1 / v5e-8), **not** the pytest suites in `tests_gdn2/`.
CI in this repo runs CPU-only `interpret=True` smoke tests.
See [`attestation/`](./attestation/) and [`docs/ATTESTATION.md`](./docs/ATTESTATION.md).

Speedup grows with head count and peaks at H=6 on a single chip: the fused
kernel's tile pack (`bs2=64`) fits cleanly up to 6 heads, while
`associative_scan` pays an O(H*L*d^2) materialization cost that grows
linearly with H. **Beyond H=6 the fused tile stops fitting cleanly on a
single chip and sharding across chips (TPU v5e-8) is required for
meaningful further scaling.** A sharded H>6 study is future work.

**v0.1.0 (`atomic_ops`, released):** 2.6x (FP32) / 3.4x (BF16) vs best
pure-JAX WY baseline; 27.2x vs `associative_scan` on train shape
(B=8, L=4096, H=6, D=128); peak 38.8x on the smaller KAGGLE_SMALL preset
(B=4, L=2048).

**Other v0.2.0 results:** 145/146 correctness gates PASS; MQAR 0.9999 easy /
0.9986 hard / 0.9995 zero-shot @ 2048; scaling 6M->142M monotone on enwik8;
T-22 numerical boundary documented (`half_span < 88`).

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
# atomic-ops >= 0.2.0 ships BOTH packages from one PyPI distribution
pip install atomic-ops

# atomic_ops   -> stable, v0.1.x-generation kernels
# atomic_gdn2  -> research-grade, v0.2.0-generation kernels
# both are importable right after the single install above.

# For editable / from-source development instead:
git clone https://github.com/Akseleu-J/atomic-ops.git
cd atomic-ops
pip install -e .

# TPU stack (v5e)
pip install jax==0.11.1 jaxlib==0.11.1 libtpu==0.0.46 flax==0.12.9 optax==0.2.4
```

Requires Python ≥ 3.10 and `jax>=0.4.20`. For TPU, install the matching `jaxlib`/`libtpu`
builds (see [JAX TPU installation](https://jax.readthedocs.io/en/latest/installation.html)).

## Quick start

### v0.2.0 (`atomic_gdn2`)

```python
from atomic_gdn2 import make_cfg, make_blr_trainable
import jax.numpy as jnp, math

cfg = make_cfg()                                   # bt=128, bs2=32, BTL
fn  = make_blr_trainable(cfg, 1.0 / math.sqrt(128))

B, H, L, D = 8, 6, 2048, 128
q, k, v = (jnp.ones((B, H, L, D), jnp.float32) for _ in range(3))
w = jnp.ones_like(q); b = 0.5 * jnp.ones_like(q); g = -0.05 * jnp.ones_like(q)
o, h_final = fn(q, k, v, w, b, g)
```

### v0.1.0 (`atomic_ops`)

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

Both examples work out of the box after `pip install atomic-ops`:

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

## Citation

**v0.1.0 (`atomic_ops`, published, Zenodo DOI):**

```bibtex
@software{akseleu_atomic_ops_2026,
  author    = {Omirbay Akseleu},
  title     = {Atomic Ops: Fused Gated DeltaNet-2 Kernels for TPU v5e in JAX/Pallas},
  year      = {2026},
  version   = {v0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22706659},
  url       = {https://doi.org/10.5281/zenodo.22706659}
}
```

**v0.2.0 (`atomic_gdn2`, in development, source only):**

```bibtex
@software{akseleu_atomic_gdn2_2026,
  author  = {Omirbay Akseleu},
  title   = {atomic_gdn2: Next-generation GDN-2 Pallas kernels for TPU v5e},
  year    = {2026},
  version = {v0.2.0},
  url     = {https://github.com/Akseleu-J/atomic-ops}
}
```

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
is inference-only, use `gdn2_forward` / `gdn2_chunked_wy_reference`. The experimental
hybrid `JAX forward + Pallas backward` path was closed in v0.2.0
(`HYPOTHESIS-REJECTED`, see `ROADMAP.md`).
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
├── atomic_ops/                  # v0.1.0 stable
│   ├── configs.py
│   ├── gdn2_fwd.py
│   ├── gdn2_bwd.py
│   ├── gdn2_pipeline.py
│   ├── reference.py
│   └── fallback.py
├── atomic_gdn2/                 # v0.2.0 next generation
│   ├── config.py                # BLRConfig
│   ├── fwd/                     # scores, solve, cd_slim
│   ├── bwd/                     # b4, mega (unified backward)
│   ├── reference/               # forward_ref, token_serial_ref
│   ├── training/                # model, data, train, canary
│   └── pipeline.py              # make_blr_trainable
├── benchmarks/                  # speed / memory
├── tests/                       # atomic_ops CPU smoke
├── tests_gdn2/                  # atomic_gdn2 (T1–T9)
├── docs/ATTESTATION.md          # v0.2.0 numbers
└── .github/workflows/           # CI
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `pytest tests/test_gdn2_full_math_correctness.py`
must pass, Ruff must be green, kernel changes require the full TPU suite referenced in the PR.

## License

MIT — see [LICENSE](LICENSE). Kernels ported from the NVlabs Gated DeltaNet-2 Triton reference.

## Support

If this package is useful in your research, consider giving it a ⭐ — it helps other researchers find it.
Bug reports and questions go to [Issues](https://github.com/Akseleu-J/atomic-ops/issues).
# atomic-ops

> **Note on bwd vs fwd+bwd timings:** the `bwd` column is measured via `jax.vjp(loss, ...)`, which re-runs the forward pass internally to build the VJP closure before the backward pass executes. This is why `bwd` and `fwdbwd` numbers are nearly identical in the tables above/below -- it is an artifact of the measurement method (the forward cost is unavoidably included in both), not a claim that backward alone costs the same as forward+backward combined.
