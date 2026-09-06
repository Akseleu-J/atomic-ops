# GDN‑2 Pallas: Performance Benchmarks

All numbers below were measured on **TPU v5e‑8** (Kaggle/Google Cloud) with  
`jax==0.11.1`, `jaxlib==0.11.1`, `libtpu==0.0.46.1`.

We compare three implementations:
- **OLD** – Associative scan baseline.
- **JAX_REF** – Chunked WY recurrence in pure JAX.
- **PALLAS** – Our custom Pallas kernels (this library).

---

## ⚡ Speed (Full forward+backward cycle)

The most important metric for training. Measured in **milliseconds (ms)**.

| Config | Dtype | OLD (ms) | JAX_REF (ms) | PALLAS (ms) | Speedup vs OLD | Speedup vs JAX_REF |
|--------|-------|----------|--------------|-------------|----------------|---------------------|
| **train_shape (B=8,L=4096)** | FP32 | 4762.70 | 460.03 | **175.20** | **27.18×** | **2.63×** |
| **train_shape (B=8,L=4096)** | BF16 | 2320.27 | 594.86 | **174.20** | **13.32×** | **3.41×** |

> **Key takeaway:** In BF16 (the real-world dtype for TPU training), PALLAS is **3.4× faster** than the pure-JAX WY reference and **13.3× faster** than the legacy scan.

---

## 🧠 Memory (Peak HBM)

Peak memory usage measured in **Megabytes (MB)** during the full forward+backward pass (process-isolated).  
Lower is better.

| Config | Dtype | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |
|--------|-------|----------|--------------|-------------|
| train_shape_B8_L4096 (B=8) | fp32 | 1252.746 | 1250.011 | 1238.111 |
| train_shape_B8_L4096 (B=8) | bf16 | 915.486 | 915.486 | 915.486 |




---

## 📊 Full Results & Raw Data

For complete per‑stage breakdowns (fwd / bwd / fwdbwd) and all configs:

- **Speed (detailed tables):** [`benchmark_speed_final_averaged.md`](raw/benchmark_speed_final_averaged.md)
- **Speed (machine‑readable JSON):** [`benchmark_speed_final_averaged.json`](raw/benchmark_speed_final_averaged.json)
- **Memory (detailed tables):** [`benchmark_memory_final_averaged.md`](raw/benchmark_memory_final_averaged.md)
- **Memory (machine‑readable JSON):** [`benchmark_memory_final_averaged.json`](raw/benchmark_memory_final_averaged.json)

---

## 🔁 Reproducibility

To reproduce these numbers yourself, run:

```bash
# Speed benchmarks
python run_speed_benchmark.py

# Memory benchmarks (requires a clean process fork)
python run_memory_benchmark.py
