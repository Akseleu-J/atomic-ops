# Contributing to atomic_ops

Thank you for your interest in contributing! This project is a research-grade
implementation of fused Gated DeltaNet-2 kernels in JAX/Pallas for TPU v5e.

## Quick start

```bash
git clone https://github.com/Akseleu-J/atomic-ops.git
cd atomic-ops
pip install -e ".[dev]"
pre-commit install
```

## Running tests

CI runs on CPU with `interpret=True`. Does NOT reproduce TPU-specific
results (sharding parity, Mosaic lowering, timing).

Current timing results are in [benchmarks/README.md](benchmarks/README.md); numerical
controls and recorded update failures are in [numerical/README.md](numerical/README.md).
The historical attestation has its own revisions and mixed gate types; its 145 PASS and
one OLD OOM are not a count of current independent numerical proofs.

### CPU smoke tests (no TPU required)

```bash
pytest tests/test_gdn2_full_math_correctness.py -v
python tests/extended/test_gdn2_deep_correctness_mini.py
```

These use `interpret=True` / plain JAX and run in seconds on any machine.

### Full correctness suite (requires TPU)

```bash
pytest tests/extended/test_gdn2_deep_correctness.py -v
```

This includes finite-difference gradient checks, multi-seed sweeps, bf16
coverage, and isolated backward-kernel tests (B3-B5).

### Benchmarks

Use [the frozen four-path notebook](benchmarks/notebooks/all-paths-speed.ipynb) for
the current timing protocol. Other root-level scripts are development tools; their
configuration and workload differ. No current peak-HBM result is claimed. See the
[tool status](docs/BENCHMARK_TOOL_STATUS.md) before using a legacy entry point.

## Code style

We use **Ruff** for linting and formatting:

```bash
ruff check atomic_ops tests benchmarks examples --fix
ruff format atomic_ops tests benchmarks examples
```

Pre-commit hooks are configured to run these checks automatically.

## Pull request checklist

- [ ] Tests pass locally (`pytest tests/test_gdn2_full_math_correctness.py`)
- [ ] No new lint/format debt (`python ci/check_lint.py`); see [CI scope](docs/CI_SCOPE.md).
- [ ] Docstrings follow [Google style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
- [ ] If you changed kernels (`gdn2_fwd.py`, `gdn2_bwd.py`), the full TPU test
suite was run and results are referenced in the PR description.
- [ ] CHANGELOG.md is updated for user-facing changes.

## Reporting bugs

Please use the [Bug Report](https://github.com/Akseleu-J/atomic-ops/issues/new?template=bug_report.md)
template and include:

- JAX version (`jax.__version__`)
- Hardware (CPU / GPU / TPU v5e-8 / etc.)
- Minimal reproducer script
- Expected vs actual output

## Architecture questions

For questions about the WY-formulation, kernel blocking strategy, or the
backward chain (B1-B5), see `docs/TESTING_STRATEGY.md` and inline docstrings
in `atomic_ops/gdn2_fwd.py` and `atomic_ops/gdn2_bwd.py`.

## JAX version policy

Three sources pin JAX, for different purposes — do not confuse them:

- `pyproject.toml`: `jax>=0.4.20,<0.13` — declared dependency constraints;
  this is not evidence that every JAX/Python combination was tested.
- `requirements.txt`: `jax==0.11.1` — **Kaggle lockfile** for the
  reproducible notebook/attestation runs. Not a support boundary.
- `.github/workflows/ci.yml`: `jax==0.11.2` — CI snapshot. Green
  CI means the code passes on this specific version; it does not
  assert that 0.11.2 is the only supported version.

JAX 0.11.2 and Flax 0.12.9 require Python >=3.12. The current CI matrix
uses Python 3.12/3.13; changing the matrix does not itself establish a passing run.

If you hit a version-specific bug, note the exact `jax.__version__`
in your issue.
