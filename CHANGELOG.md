# Changelog

All notable user-facing changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-05

First public release under the `atomic_ops` name (renamed from
`atomic_ops` prior to release; no prior published history).

### Added

- Fused GDN-2 forward kernels for TPU v5e in JAX/Pallas:
  Kernel A (chunk scores), Kernel B (WY block solve), Kernel C (recompute),
  Kernel D (inter-chunk scan).
- Fused backward kernels B1-B5 with a `jax.custom_vjp` trainable wrapper
  (`gdn2_pallas_forward_trainable`) that reuses forward residuals instead of
  recomputing them.
- Automatic fallback: CPU/GPU or `d_head != 128` dispatches to a checkpointed
  pure-JAX chunked-WY reference with identical `wy_eps` damping semantics.
- `KernelConfig` with TPU v5e presets `KAGGLE_SMALL` / `KAGGLE_MEDIUM` /
  `KAGGLE_LARGE`, plus `estimate_memory` / `get_recommended_config` helpers.
- Numerical-safety sanitizers at every kernel boundary and a per-stage
  diagnostic mode via the `GDN2_FWD_DIAG=1` environment flag.
- Layered correctness suite: CPU smoke tests, multi-seed TPU sweeps,
  finite-difference gradient checks, isolated B3-B5 backward-stage tests,
  BF16 dtype-contract checks, `wy_eps` damping coverage
  (see `docs/TESTING_STRATEGY.md`).
- Speed and memory benchmarks with raw results (`benchmarks/`).
- `beta/gdn2_hybrid.py`: experimental, opt-in hybrid JAX-forward +
  fused-Pallas-backward path. Not wired into the public dispatcher
  (`gdn2_forward_trainable`); see `KNOWN_LIMITATIONS.md` section 5 and
  `ROADMAP.md` for status and open validation items before this is
  recommended for production training.
- `ROADMAP.md`: tracked hypotheses and open performance items (MXU-factorized
  pairwise decay / `use_centering`, Kernel B block-solve investigation),
  explicitly marked as not-yet-implemented where applicable.
- `KNOWN_LIMITATIONS.md`: documented gaps (forward slower than pure-JAX WY
  forward, VPU-bound pairwise decay, TPU-only / `d_head=128` requirement,
  shape constraints, hybrid-path open items, kernel-gap diagnostic).
- MIT `LICENSE`.

### Known limitations

- The fused forward is currently slower than the pure-JAX WY forward
  (~0.62x on TPU v5e-8, forward-only). Training steps are backward-dominated,
  so the full cycle is still faster than the best pure-JAX baseline end-to-end.
- The pairwise decay computation (Kernel A / B4) is VPU-bound rather than
  MXU-bound; an MXU-factorized `use_centering` alternative is a documented
  hypothesis in `ROADMAP.md`, not implemented in this release.
- See `KNOWN_LIMITATIONS.md` for the full list, including the experimental
  hybrid path's open validation items.

### Performance (TPU v5e-8, train shape B=8, L=4096, 6 heads, d_head=128)

| Metric | FP32 | BF16 |
| --- | --- | --- |
| fwd+bwd vs associative_scan | 27.2x | 13.3x |
| fwd+bwd vs pure-JAX chunked WY | 2.6x | 3.4x |
| Best-case vs associative_scan (all shapes) | 38.8x | 18.7x |

[Unreleased]: https://github.com/Akseleu-J/atomic_ops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Akseleu-J/atomic_ops/releases/tag/v0.1.0
