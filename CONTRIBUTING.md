# Contributing to atomic_ops

Thank you for your interest in contributing! This project is a research-grade
implementation of fused Gated DeltaNet-2 kernels in JAX/Pallas for TPU v5e.

## Quick start

```bash
git clone https://github.com/Akseleu-J/atomic_ops.git
cd atomic_ops
pip install -e ".[dev]"
pre-commit install
```

## Running tests

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

```bash
python benchmarks/run_speed_benchmark.py
python benchmarks/run_memory_benchmark.py
```

## Code style

We use **Ruff** for linting and formatting:

```bash
ruff check atomic_ops tests benchmarks examples --fix
ruff format atomic_ops tests benchmarks examples
```

Pre-commit hooks are configured to run these checks automatically.

## Pull request checklist

- [ ] Tests pass locally (`pytest tests/test_gdn2_full_math_correctness.py`)
- [ ] Ruff is green (`ruff check` and `ruff format --check`)
- [ ] Docstrings follow [Google style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
- [ ] If you changed kernels (`gdn2_fwd.py`, `gdn2_bwd.py`), the full TPU test
suite was run and results are referenced in the PR description.
- [ ] CHANGELOG.md is updated for user-facing changes.

## Reporting bugs

Please use the [Bug Report](https://github.com/Akseleu-J/atomic_ops/issues/new?template=bug_report.md)
template and include:

- JAX version (`jax.__version__`)
- Hardware (CPU / GPU / TPU v5e-8 / etc.)
- Minimal reproducer script
- Expected vs actual output

## Architecture questions

For questions about the WY-formulation, kernel blocking strategy, or the
backward chain (B1-B5), see `docs/TESTING_STRATEGY.md` and inline docstrings
in `atomic_ops/gdn2_fwd.py` and `atomic_ops/gdn2_bwd.py`.
