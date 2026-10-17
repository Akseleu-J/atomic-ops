# Benchmark entry-point status

The current measured protocol is the self-contained notebook
[`benchmarks/notebooks/all-paths-speed.ipynb`](../benchmarks/notebooks/all-paths-speed.ipynb).
Its executed source, immutable kernel pin, all timing samples and raw logs are linked
from [benchmarks/README.md](../benchmarks/README.md).

## Distinguish the published snapshot from this local working copy

At audited public commit `24f99004da1af2c7d5b079eeec913170fac49790`, the older
`notebooks/speedup_scaling.ipynb` contained duplicate h0 calls and scale/state mismatch.
The old `benchmarks/run_speed_benchmark.py` OLD wrapper normalized affine transitions,
ignored h0/scale and returned a zero final state. The older single-path
`scripts/reproduce_kernel_bench.py` hid a removed b4_dot_mode keyword in a broad exception.
These are findings about that historical source, not the frozen four-path notebook.

This working copy already had separate benchmark repairs before the hygiene update.
The speed and reproduction entry points route to `benchmarks/honest_gdn2.py`;
the development scaling notebook embeds that earlier diagnostic suite. The memory
benchmark uses spawned workers. Those local repairs are preserved, not overwritten
with historical source. They have not been rerun as part of this hygiene update.

These development protocols differ from the current four-path experiment. A successful
run does not reproduce its published table unless the source, inputs, precision,
timing and aggregation protocol match. No new peak-HBM measurement is claimed.
`tools_gdn2/bench.py` is a separate timing smoke tool, not a cross-device regression gate.

Original public source remains available at the immutable commit above. Kernel packages,
raw archives and original measured samples are unchanged by this documentation update.
