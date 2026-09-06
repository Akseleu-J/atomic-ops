# Memory Benchmarks – Peak HBM (MB)

All numbers are **peak memory usage** (in MB) measured during the actual call, process‑isolated (fork) to avoid interference.  
Measured on **TPU v5e‑8** with `jax==0.11.1`, `jaxlib==0.11.1`.

Only three representative configs are shown; full raw data is available in `raw/`.

---

## FP32

| config (batch) | stage | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |
|---|---|---|---|---|
| small_B1_L1024 (B=1) | fwd | 32.822 | 27.454 | 31.646 |
| small_B1_L1024 (B=1) | bwd | 76.303 | 52.842 | 56.386 |
| small_B1_L1024 (B=1) | fwdbwd | 76.337 | 52.900 | 56.443 |
| train_shape_B8_L4096 (B=8) | fwd | 725.899 | 726.384 | 724.706 |
| train_shape_B8_L4096 (B=8) | bwd | 1252.688 | 1249.912 | 1238.035 |
| train_shape_B8_L4096 (B=8) | fwdbwd | 1252.746 | 1250.011 | 1238.111 |

---

## BF16

| config (batch) | stage | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |
|---|---|---|---|---|
| small_B1_L1024 (B=1) | fwd | 31.046 | 31.046 | 31.046 |
| small_B1_L1024 (B=1) | bwd | 61.088 | 33.092 | 37.544 |
| small_B1_L1024 (B=1) | fwdbwd | 61.144 | 33.148 | 37.601 |
| train_shape_B8_L4096 (B=8) | fwd | 915.486 | 915.486 | 915.486 |
| train_shape_B8_L4096 (B=8) | bwd | 915.486 | 915.486 | 915.486 |
| train_shape_B8_L4096 (B=8) | fwdbwd | 915.486 | 915.486 | 915.486 |


---

*Note: memory values are the mean peak over 3 seeds; standard deviations were negligible.*

