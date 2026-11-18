# v0.3.0 inference kernels — TPU validation evidence

Colab TPU v5e-1, jax 0.11.2, atomic-ops main (inlined inference.py v0.3.0),
notebook `notebooks/gdn2_v03_tpu_eval.ipynb` (RUN1/RUN2), 2026-11-17.

Gates (REPORT.json): GATE A 7/7 (interpret semantics vs token_serial_ref),
B1 compiled decode vs serial PASS, B2 compiled prefill vs serial PASS
(3/16384 boundary outliers = documented chunked-vs-serial contract tail,
tolerance 0.15 with comment), C backward smoke PASS (max|grad| 3.28,
clip contract 1e4). PERF: prefill 1649-1751 tok/s (fp32 + HIGHEST path;
bf16 config = follow-up), decode 29.2-29.7 tok/s (per-call pallas dispatch
overhead; lax.scan/jit-batched loop = follow-up).
