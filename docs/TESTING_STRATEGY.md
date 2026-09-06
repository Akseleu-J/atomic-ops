# Testing Strategy for GDN‑2 Pallas Kernels

Why the test suite is structured the way it is, and how to interpret failures.

---

## 1. The core problem

Most unit tests compare an implementation against a “reference”. In this project, however, almost every reference is **another implementation of the same math**:

- `gdn2_pallas_forward` (Pallas kernels on TPU)
- `gdn2_chunked_wy_reference` (chunked WY recurrence in pure JAX)
- `gdn2_token_serial_reference` (token‑by‑token scan in pure JAX)

If the bug is in the **derivation of the formulas** (not in their implementation on a specific backend), it will likely appear in **all three** implementations equally – and comparing Pallas against JAX_REF would show no difference, even though both are wrong.

That is why the test suite is built in **layers**, each catching a different class of errors, rather than duplicating the same checks.

---

## 2. Test layers (from weakest to strongest evidence)

### Layer 0 – Shape / dtype / config sanity
- Checks `KernelConfig` invariants (`bt % bc == 0`, `D` matches, etc.)
- Cheap, CPU‑only; runs in CI on every push.
- If this fails, no other layer is meaningful.

### Layer 1 – Reference‑vs‑reference smoke test
- Verifies that `gdn2_chunked_wy_reference` runs and returns the correct shapes.
- Not a correctness test – just a sanity check before using it as a golden reference.

### Layer 2 – Pallas vs token‑serial (forward + backward)
- Compares `gdn2_pallas_forward[_trainable]` against `gdn2_token_serial_reference` (an independent scan, **not** the chunked WY reference – because chunked WY shares the same algebraic derivation and would hide systematic errors).
- Runs with a single random seed – good but not enough (see Layer 3).

### Layer 3 – Multi‑seed sweep
- Repeats Layer 2 across several seeds.
- Motivation: we have seen cases where a specific (batch, head, chunk) combination with periodic patterns produced `cond(Akk) ~ 1.45e8` – a single “lucky” seed would never catch that.

### Layer 4 – Finite‑difference gradient check – **the strongest test**
- Compares analytical gradients from `custom_vjp` against numerical gradients computed as `(f(x+eps) - f(x-eps)) / (2*eps)`.
- This check **shares no derivation** with either chunked‑WY or token‑serial implementations – finite differences are derived independently from all other code.
- If there is a bug in the gradient formula (not in its Pallas implementation), only this test will catch it.
- Expensive: each probed coordinate requires an extra forward+backward pass. Therefore we probe a small random subset of coordinates per tensor (see `n_probe` in the test config) rather than the whole array.

### Layer 5 – Isolated backward‑stage checks (B3, B4, B5)
The first test suite (`test_gdn2_full_math_correctness.py`) isolated only B1/B2. This extended suite adds:

- **B3 (`wy_dqkg_backward_pallas`)** – the matrix‑inverse gradient step, explicitly called out in the docstrings as the highest‑amplification point in the backward chain (`dAkk_raw = -A^T @ dA_total @ A^T`). Never tested in isolation before.
- **B4 (`intra_backward_pallas`)** – cross‑checked with a freshly written `scores_fwd` function differentiated via `jax.vjp` using the same cotangents (`dAqk`, `dAkk`) passed to the Pallas kernel.
- **B5 (`reverse_cumsum_bwd`)** – verified against `jnp.flip(cumsum(flip(x)))` instead of reusing the `tril_ones @ x` trick that the kernel itself uses (otherwise the test would only prove “the code matches itself”).

Isolation is methodologically important: if Layers 2–4 fail, isolated B3/B4/B5 tests tell you **which of the five backward stages** contains the bug, instead of forcing you to debug the entire pipeline.

### Layer 6 – Numerical stability: `wy_eps` damping
All tests above default to `wy_eps=0.0`, but production configs (`KAGGLE_SMALL/MEDIUM/LARGE`) use `wy_eps=1e-3`.  
The chain rule through damping (`(1 - wy_eps)`) is a separate piece of logic that was recently fixed (see `gdn2_bwd.py` – “FIX: chain rule through damping”).  
Tolerance is scaled with `wy_eps` (`tol = max(2e-2, wy_eps * 5)`) because the damped forward **deliberately** solves a slightly different problem than the undamped token‑serial reference – so comparison cannot be as tight as for `wy_eps=0`.

### Layer 7 – BF16 input coverage
Real training feeds `q/k/v/w/b` as `bfloat16` (only `g` stays `float32`). All tests above use `float32` everywhere. This layer:
- Runs the honest backward vs the `float32` token‑serial reference with loosened tolerance (BF16 has ~3 decimal digits of precision – that is expected noise, not a bug).
- Verifies the **dtype contract** of `custom_vjp`: returned gradients must have the same dtype as the corresponding input tensor, otherwise `optax` / optimisers will break on `apply_updates`.

### Layer 8 – Alternative config: `KAGGLE_SMALL`
All tests default to `bt=256`. `KAGGLE_SMALL` uses `bt=128`, with a different `n_sub`/`n_micro` in `_block_solve`.  
Catches bugs that only appear with a different blocking grid (e.g., earlier we discovered that `bc < bt/2` produced structurally zero blocks – a class of errors specific to block size).

---

## 3. What these tests do NOT cover (by design)

- **Performance / speed** – that is the job of `benchmarks/run_speed_benchmark.py` (which compares OLD, JAX_REF, and PALLAS, with correctness gates before timing).
- **Memory (HBM) usage** – peak memory is measured in a separate, process‑isolated benchmark (`benchmarks/run_memory_benchmark.py`) that uses forking to get an honest peak without interference from previous runs. Memory numbers are published separately and are not part of the correctness suite.
- **Full pipeline on CPU** – `wy_solve_pallas`, `recompute_wy_pallas`, `dav_backward_pallas`, and `wy_dqkg_backward_pallas` call `pl.pallas_call` without `interpret=True`, so they expect TPU‑specific lowering. Only `build_chunk_scores_pallas` (Kernel A) and `intra_backward_pallas` (B4) support `interpret=True` and can run on CPU. That determines what goes into the CPU smoke test.

---

## 4. How to run

```bash
# Fast CPU smoke test (seconds, no TPU required):
python tests/extended/test_gdn2_deep_correctness_mini.py
```

# Full suite (requires a TPU, with interpret=False):
python tests/extended/test_gdn2_deep_correctness.py

---

The full suite is mandatory before:

publishing the package,

any change to gdn2_bwd.py / gdn2_fwd.py / configs.py,

adding a new KernelConfig preset.
