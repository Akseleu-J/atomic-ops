# Changelog

## [0.2.0] — 2026-09-27

### Added — `atomic_gdn2` package
- Unified mega-backward: B2+B1+B3+B4+B5 in one Pallas launch
- T-22 numerical boundary documented (`half_span < 88`, `BTL_STOP=80`)
- Test suite T1–T9 under `tests_gdn2/`
- Reference implementations: `forward_ref`, `token_serial_ref`, `b4_exact_f64_np`
- Tools `tools_gdn2/`: `bench.py`, `t0_diagnose.py`
- Per-layer Amdahl decomposition (R² > 0.9999)

### Fixed
- O-1: removed clamp masks that gated `dgc`

### Removed
- `b4_dot_mode` config field. T4 phantom-parameter audit found no effect
  for `diag='btl'` at proj_dim>=768. Removed rather than fixed.
  B4 diagonal matmuls now always use `precision=HIGHEST`.
- `beta/gdn2_hybrid.py` hybrid path. Closed as `HYPOTHESIS-REJECTED`.
  Code preserved at `archive/gdn2_hybrid.py`.

### Headline

Kernel speedup vs `associative_scan` — direct measurement, single TPU v5e-1,
B=4, D=128, `bs2=64`:

| H | L=2048 | L=4096 |
|---|--------|--------|
| 2 | 193x | 207x |
| 4 | 300x | 322x |
| 6 | **468x** | **448x** |

Speedup peaks at H=6 on a single chip; beyond H=6 requires sharding (TPU v5e-8).

## [Unreleased]

## [0.1.0] - 2026-09-06

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

[Unreleased]: https://github.com/Akseleu-J/atomic-ops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Akseleu-J/atomic-ops/releases/tag/v0.1.0
