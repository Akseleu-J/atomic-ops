# Numerical validation and sensitivity audit

This directory contains a reproducible numerical case study for the GDN2 kernel implementations. It accompanies the benchmark results and records its own inputs, precision, provenance and limitations.

## Experiments

1. controlled-rounding: independent Decimal 80/110-digit sequential recurrence/tangent and finite-difference controls; FP32/FP64 serial, OLD and pure-JAX WY comparisons. The cancellation stress objective is deliberately constructed; all seeds and the ordinary objective are included.
2. actual-coordinate: full-LM coordinate block_6/q_conv_w[236,2], OLD/JAX_REF/NEW, seeds 0/1/2, uninstrumented/zero-probe controls, actual saved inputs and cotangents, exact contractions and cross-factor decompositions.
3. exact-TPU-confirmation: fresh TPU integer-bin audit of the frozen captures. Eight controls, 27 exact cross contractions, nine decompositions and nine permutation checks pass. Source identity, raw device bins and an independent stdlib verifier are provided.

For the selected seed-0 OLD/NEW pair, 98.2542% of the recorded gradient difference remains after replacing the final contraction with an exact sum of captured factors. The sign difference is already present in those factors. Severe cancellation and AdamW epsilon sensitivity are documented.

## Scope

The exact sum is an oracle for captured, already-rounded factors, not the unrounded full-LM gradient. The TPU confirmation does not re-execute the full network. These data do not establish the exclusive upstream cause, universal correctness or unchanged model quality. Recorded norm-based PASSs do not erase separate update-difference FAILs. Instrumentation effects and historical/incomplete executions remain explicit.

## Navigation

- experiments/2026-10-17/: compact arrays, tables, reports and logs.
- tools/: exact TPU accumulator and independent verifier.
- notebooks/: self-contained reproduction notebooks.
- ../artifacts/: unchanged original experiment ZIPs, source snapshots and manifests.
