# Current benchmark evidence: 17 October 2026

Pinned kernel revision: 11616acd8a061768a38feb12c58e5e329952f5d5. Four paths: OLD affine associative_scan, the author's pure-JAX WY JAX_REF, earlier PALLAS_V1 and current atomic_gdn2. Single TPU v5e; zero damping; fixed-state replay, full-tree synchronization. Three seeds, three rounds, two warmups and four timed calls per round for every path. Compilation, host checks and file IO are outside timing.

Peak gradient-microbenchmark OLD/atomic_gdn2 ratio: 411.75 +/- 5.13x, B4 H6 L2048 D128. Complete synthetic 34,873,952-parameter LM step: 117.76 +/- 0.15x, B8 L2048; OLD block remat. All six kernel shapes and all four implementations are in results/2026-10-17/speed-summary.csv. Sample-level data contain 1008 timed calls. SD is across three seed summaries, not a confidence interval or cross-VM variation.

Raw source/metadata/logs: [four-path archive](../artifacts/current-evidence/Atomic-gdn2-all-paths-speed-results-20261017T121040Z.zip).

Numerical snapshot PASS and parameter-update-difference FAIL are both retained. See ../numerical/. These timings do not prove optimizer-trajectory equivalence or a measured speedup of the released 307M/334M models.

## Reproduce and inspect

Open [the four-path Colab notebook](notebooks/all-paths-speed.ipynb). It embeds the executed harness and pins the benchmarked source revision. Earlier root-level benchmark entry points and notebooks are development/historical tools, not substitutes for this frozen protocol.

- [All paths and shapes](results/2026-10-17/speed-summary.csv)
- [Every timed sample](results/2026-10-17/speed-all-timing-samples.csv)
- [Verification receipt](results/2026-10-17/final-publication-evidence-20261017-verification.json)
- [Detailed scope and caveats](../KNOWN_LIMITATIONS.md)
- [Earlier v0.1 raw timing records](raw/benchmark_speed_final_averaged.md)

Latency: median of timed calls per round, then median of round medians per seed. Speedup: paired round ratios summarized by seed; aggregate mean and sample SD across seeds. Kernel objective: sum of squared outputs and final state, differentiated with respect to six sequence inputs. E2E: eight-layer synthetic LM, fixed initial state, block-rematerialized OLD. NEW uses bt=128, bc=64, score_bs=128, bs2=64, mb=8, BTL, score precision highest and solve/B3 bf16x3; PALLAS_V1 uses mb=16. No new memory or forward-only headline is claimed.
