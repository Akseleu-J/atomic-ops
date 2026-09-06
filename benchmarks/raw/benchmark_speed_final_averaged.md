# Speed Benchmarks – Full Results

All times are in **milliseconds** (mean steady‑state, averaged over repeats).  
Measured on **TPU v5e‑8** with `jax==0.11.1`, `jaxlib==0.11.1`.

---

## FP32

| Config | Path | Stage | Mean (ms) | Std (ms) | CV | Micro‑bs | N‑rep |
|--------|------|-------|-----------|----------|----|----------|-------|
| small_B1_L1024 | OLD | fwd | 31.65 | 0.22 | 0.007 | - | 2 |
| small_B1_L1024 | OLD | bwd | 143.06 | 0.10 | 0.001 | - | 2 |
| small_B1_L1024 | OLD | fwdbwd | 143.84 | 0.08 | 0.001 | - | 2 |
| small_B1_L1024 | JAX_REF | fwd | 3.75 | 0.03 | 0.007 | - | 2 |
| small_B1_L1024 | JAX_REF | bwd | 22.27 | 0.09 | 0.004 | - | 2 |
| small_B1_L1024 | JAX_REF | fwdbwd | 22.09 | 0.06 | 0.003 | - | 2 |
| small_B1_L1024 | PALLAS | fwd | 3.58 | 0.02 | 0.004 | - | 2 |
| small_B1_L1024 | PALLAS | bwd | 5.94 | 0.04 | 0.007 | - | 2 |
| small_B1_L1024 | PALLAS | fwdbwd | 5.92 | 0.03 | 0.006 | - | 2 |
| medium_B4_L4096 | OLD | fwd | 531.67 | 0.14 | 0.000 | - | 2 |
| medium_B4_L4096 | OLD | bwd | 2618.22 | 0.14 | 0.000 | - | 2 |
| medium_B4_L4096 | OLD | fwdbwd | 2655.59 | 0.27 | 0.000 | - | 2 |
| medium_B4_L4096 | JAX_REF | fwd | 40.13 | 0.09 | 0.002 | - | 2 |
| medium_B4_L4096 | JAX_REF | bwd | 268.55 | 0.10 | 0.000 | - | 2 |
| medium_B4_L4096 | JAX_REF | fwdbwd | 268.71 | 0.15 | 0.001 | - | 2 |
| medium_B4_L4096 | PALLAS | fwd | 51.55 | 0.12 | 0.002 | - | 2 |
| medium_B4_L4096 | PALLAS | bwd | 87.95 | 0.10 | 0.001 | - | 2 |
| medium_B4_L4096 | PALLAS | fwdbwd | 88.07 | 0.07 | 0.001 | - | 2 |
| train_shape_B8_L4096 | OLD | fwd | 1067.24 | 0.17 | 0.000 | - | 2 |
| train_shape_B8_L4096 | OLD | bwd | 4767.24 | 0.21 | 0.000 | 2 | 2 |
| train_shape_B8_L4096 | OLD | fwdbwd | 4762.70 | 0.11 | 0.000 | 2 | 2 |
| train_shape_B8_L4096 | JAX_REF | fwd | 63.24 | 0.09 | 0.001 | - | 2 |
| train_shape_B8_L4096 | JAX_REF | bwd | 458.99 | 0.10 | 0.000 | - | 2 |
| train_shape_B8_L4096 | JAX_REF | fwdbwd | 460.03 | 0.08 | 0.000 | - | 2 |
| train_shape_B8_L4096 | PALLAS | fwd | 102.08 | 0.10 | 0.001 | - | 2 |
| train_shape_B8_L4096 | PALLAS | bwd | 175.33 | 0.15 | 0.001 | - | 2 |
| train_shape_B8_L4096 | PALLAS | fwdbwd | 175.20 | 0.08 | 0.000 | - | 2 |
| kaggle_small_preset_B4_L2048 | OLD | fwd | 239.88 | 0.08 | 0.000 | - | 2 |
| kaggle_small_preset_B4_L2048 | OLD | bwd | 1199.12 | 0.13 | 0.000 | - | 2 |
| kaggle_small_preset_B4_L2048 | OLD | fwdbwd | 1196.35 | 0.30 | 0.000 | - | 2 |
| kaggle_small_preset_B4_L2048 | JAX_REF | fwd | 13.01 | 0.04 | 0.003 | - | 2 |
| kaggle_small_preset_B4_L2048 | JAX_REF | bwd | 114.98 | 0.09 | 0.001 | - | 2 |
| kaggle_small_preset_B4_L2048 | JAX_REF | fwdbwd | 119.51 | 0.10 | 0.001 | - | 2 |
| kaggle_small_preset_B4_L2048 | PALLAS | fwd | 17.99 | 0.06 | 0.003 | - | 2 |
| kaggle_small_preset_B4_L2048 | PALLAS | bwd | 30.60 | 0.05 | 0.002 | - | 2 |
| kaggle_small_preset_B4_L2048 | PALLAS | fwdbwd | 30.86 | 0.10 | 0.003 | - | 2 |
| kaggle_large_preset_B8_L4096 | OLD | fwd | 1067.24 | 0.12 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | OLD | bwd | 4766.88 | 0.17 | 0.000 | 2 | 2 |
| kaggle_large_preset_B8_L4096 | OLD | fwdbwd | 4762.64 | 0.11 | 0.000 | 2 | 2 |
| kaggle_large_preset_B8_L4096 | JAX_REF | fwd | 63.19 | 0.09 | 0.001 | - | 2 |
| kaggle_large_preset_B8_L4096 | JAX_REF | bwd | 459.13 | 0.11 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | JAX_REF | fwdbwd | 459.98 | 0.13 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | PALLAS | fwd | 101.96 | 0.07 | 0.001 | - | 2 |
| kaggle_large_preset_B8_L4096 | PALLAS | bwd | 175.27 | 0.05 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | PALLAS | fwdbwd | 175.34 | 0.10 | 0.001 | - | 2 |

---

## BF16

| Config | Path | Stage | Mean (ms) | Std (ms) | CV | Micro‑bs | N‑rep |
|--------|------|-------|-----------|----------|----|----------|-------|
| small_B1_L1024 | OLD | fwd | 14.41 | 0.07 | 0.005 | - | 2 |
| small_B1_L1024 | OLD | bwd | 69.00 | 0.11 | 0.002 | - | 2 |
| small_B1_L1024 | OLD | fwdbwd | 68.77 | 0.12 | 0.002 | - | 2 |
| small_B1_L1024 | JAX_REF | fwd | 3.66 | 0.01 | 0.003 | - | 2 |
| small_B1_L1024 | JAX_REF | bwd | 22.44 | 0.12 | 0.005 | - | 2 |
| small_B1_L1024 | JAX_REF | fwdbwd | 22.70 | 0.02 | 0.001 | - | 2 |
| small_B1_L1024 | PALLAS | fwd | 3.65 | 0.02 | 0.005 | - | 2 |
| small_B1_L1024 | PALLAS | bwd | 5.88 | 0.04 | 0.006 | - | 2 |
| small_B1_L1024 | PALLAS | fwdbwd | 6.00 | 0.06 | 0.011 | - | 2 |
| medium_B4_L4096 | OLD | fwd | 248.91 | 0.05 | 0.000 | - | 2 |
| medium_B4_L4096 | OLD | bwd | 1305.20 | 0.17 | 0.000 | - | 2 |
| medium_B4_L4096 | OLD | fwdbwd | 1304.06 | 0.17 | 0.000 | - | 2 |
| medium_B4_L4096 | JAX_REF | fwd | 39.75 | 0.07 | 0.002 | - | 2 |
| medium_B4_L4096 | JAX_REF | bwd | 332.07 | 0.10 | 0.000 | - | 2 |
| medium_B4_L4096 | JAX_REF | fwdbwd | 332.06 | 0.22 | 0.001 | - | 2 |
| medium_B4_L4096 | PALLAS | fwd | 51.13 | 0.04 | 0.001 | - | 2 |
| medium_B4_L4096 | PALLAS | bwd | 87.67 | 0.14 | 0.002 | - | 2 |
| medium_B4_L4096 | PALLAS | fwdbwd | 87.60 | 0.09 | 0.001 | - | 2 |
| train_shape_B8_L4096 | OLD | fwd | 585.49 | 0.24 | 0.000 | - | 2 |
| train_shape_B8_L4096 | OLD | bwd | 2322.97 | 0.25 | 0.000 | 2 | 2 |
| train_shape_B8_L4096 | OLD | fwdbwd | 2320.27 | 0.14 | 0.000 | 2 | 2 |
| train_shape_B8_L4096 | JAX_REF | fwd | 62.38 | 0.09 | 0.001 | - | 2 |
| train_shape_B8_L4096 | JAX_REF | bwd | 594.17 | 0.04 | 0.000 | - | 2 |
| train_shape_B8_L4096 | JAX_REF | fwdbwd | 594.86 | 0.08 | 0.000 | - | 2 |
| train_shape_B8_L4096 | PALLAS | fwd | 101.51 | 0.14 | 0.001 | - | 2 |
| train_shape_B8_L4096 | PALLAS | bwd | 174.25 | 0.11 | 0.001 | - | 2 |
| train_shape_B8_L4096 | PALLAS | fwdbwd | 174.20 | 0.08 | 0.000 | - | 2 |
| kaggle_small_preset_B4_L2048 | OLD | fwd | 112.78 | 0.11 | 0.001 | - | 2 |
| kaggle_small_preset_B4_L2048 | OLD | bwd | 573.34 | 0.07 | 0.000 | - | 2 |
| kaggle_small_preset_B4_L2048 | OLD | fwdbwd | 573.34 | 0.12 | 0.000 | - | 2 |
| kaggle_small_preset_B4_L2048 | JAX_REF | fwd | 12.58 | 0.03 | 0.003 | - | 2 |
| kaggle_small_preset_B4_L2048 | JAX_REF | bwd | 120.30 | 0.11 | 0.001 | - | 2 |
| kaggle_small_preset_B4_L2048 | JAX_REF | fwdbwd | 120.45 | 0.07 | 0.001 | - | 2 |
| kaggle_small_preset_B4_L2048 | PALLAS | fwd | 17.90 | 0.08 | 0.004 | - | 2 |
| kaggle_small_preset_B4_L2048 | PALLAS | bwd | 30.65 | 0.11 | 0.004 | - | 2 |
| kaggle_small_preset_B4_L2048 | PALLAS | fwdbwd | 30.66 | 0.07 | 0.002 | - | 2 |
| kaggle_large_preset_B8_L4096 | OLD | fwd | 585.51 | 0.06 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | OLD | bwd | 2323.21 | 0.12 | 0.000 | 2 | 2 |
| kaggle_large_preset_B8_L4096 | OLD | fwdbwd | 2320.45 | 0.21 | 0.000 | 2 | 2 |
| kaggle_large_preset_B8_L4096 | JAX_REF | fwd | 62.29 | 0.16 | 0.003 | - | 2 |
| kaggle_large_preset_B8_L4096 | JAX_REF | bwd | 594.09 | 0.12 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | JAX_REF | fwdbwd | 594.87 | 0.11 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | PALLAS | fwd | 101.63 | 0.10 | 0.001 | - | 2 |
| kaggle_large_preset_B8_L4096 | PALLAS | bwd | 174.35 | 0.06 | 0.000 | - | 2 |
| kaggle_large_preset_B8_L4096 | PALLAS | fwdbwd | 174.12 | 0.06 | 0.000 | - | 2 |

---

*Full speedup summaries and additional details can be found in `benchmarks/README.md`.*
