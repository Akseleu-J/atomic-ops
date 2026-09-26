# Attestation summary

External TPU v5e attestation, separate from this repo's CPU CI.

## Correctness (v5e-1)

| Test | Result | Tol |
|---|---|---|
| forward_ref vs token_serial | 1.5e-4 | 5e-3 |
| Kernel A vs f64 | 2.4e-6 | 5e-5 |
| Kernel B4 vs f64 | 2.5e-6 | 5e-5 |
| Grad-gate (6 seeds) | <=3.4e-5 | 5e-3 |
| T4 finite-diff, serial fp64 | **1.4e-10** | 5e-5 |
| T4 finite-diff, kernel bf16x3 | 6.5e-4 | 2e-3 |
| Causality | 0.0 | 1e-6 |
| O-1 old (positive control) | 6.3e-1 | expected-FAIL |
| O-1 fixed | 2.5e-6 | 1e-5 |
| Adversarial (8) | 8/8 PASS | -- |

## MQAR (v5e-1)

| Kernel | Easy | Hard | steps to 0.99 |
|---|---|---|---|
| NEW | 0.9999 | 0.9986 | 2000 / 5000 |
| PROD | 1.0000 | -- | 1500 |
| Attention | 0.9997 | 0.9999 | 1000 |
| OLD | OOM (234 GB) | -- | -- |

Zero-shot @ 2048: 0.9995.

## Speed (v5e-1)

| Kernel | B=8, L=4096, H=4 |
|---|---|
| OLD | 3384.6 ms |
| PROD | 81.85 ms |
| NEW (Pallas) | **9.53 ms** |

Kernel speedup: 355x vs OLD, 8.59x vs PROD.
E2E (27.5M LM, B=8, L=2048, H=6): 200.17x vs OLD (103.29 ms vs 20674.62 ms),
4.5x vs PROD.

## Amdahl (v5e-1)

| Kernel | share (n=8) | E2E | Amdahl | residual |
|---|---|---|---|---|
| NEW | 35.2% | 1x | -- | -- |
| PROD | 86.9% | 4.90x | 4.93x | 0.7% |
| OLD | 99.5% | 179.7x | 139.8x | 22.2% |

## Scaling (v5e-8, 2000 steps, bsz=8)

| Model | Params | ppl |
|---|---|---|
| 6M | 5.2M | 3.67 |
| 25M | 20.6M | 2.94 |
| 80M | 69.4M | 2.74 |
| 142M | 123M | **2.59** |

## Reproducibility

Cross-session hash identical. 100 runs -> 1 unique hash.

## T-22

bf16 soft bound at half_span~=88. bf16x3 tolerates ~1.7x over, ~5% loss cost.

## Distinct from `use_centering`

`diag='btl'` uses balanced two-leg (`btl_legs`, no clip) + `half_span<88`
guard + runtime canary. Not the same mechanism as the earlier `use_centering`
three-leg scheme (which had a clip-saturation leak).

## Raw artifacts

See [`../attestation/`](../attestation/) for `final_report.json`, `final_report.md`, and `provenance.json`.

## Direct kernel H-scaling (single TPU v5e-1)

B=4, D=128, `bs2=64`. `associative_scan` baseline vs `atomic_gdn2` NEW:

| H | L=2048 | L=4096 |
|---|--------|--------|
| 2 | 193x | 207x |
| 4 | 300x | 322x |
| 6 | **468x** | **448x** |

Peak at H=6 on single chip. Correctness `rel_l2(NEW, JAX_REF) = 2.5e-06`
at H in {4,6}, L=2048. Beyond H=6 requires sharding (TPU v5e-8); sharded
study is future work.

Raw: [`../attestation/scaling.json`](../attestation/scaling.json).

## Raw artifacts

See [`../attestation/`](../attestation/) for `scaling.json`,
`final_report.json`, `final_report.md`, and `provenance.json`.
