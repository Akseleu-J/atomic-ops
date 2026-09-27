# Attestation artifacts

External TPU v5e attestation for `atomic_gdn2` v0.2.0.

## Files

- `scaling.json` — direct kernel H-scaling, single TPU v5e-1, H in {2,4,6}
- `final_report.json` — aggregated correctness + training runs
- `final_report.md` — human-readable summary
- `provenance.json` — commit hash, date, hardware, package versions

## Headline: kernel speedup vs `associative_scan`

Direct measurement, single TPU v5e-1, B=4, D=128, `bs2=64`:

| H | L=2048 | L=4096 |
|---|--------|--------|
| 2 | 193x | 207x |
| 4 | 300x | 322x |
| 6 | **468x** | **448x** |

Speedup grows with head count and peaks at H=6: the fused kernel's tile
pack (`bs2=64`) fits cleanly up to 6 heads, while `associative_scan` pays
an O(H*L*d^2) materialization cost that grows linearly with H.

Beyond H=6 the fused tile stops fitting cleanly on a single chip and
sharding across chips (TPU v5e-8) is required for meaningful further
scaling. A sharded H>6 study is future work.

## Correctness

`rel_l2(NEW, JAX_REF)` at L=2048:

| H | rel_l2 |
|---|--------|
| 4 | 2.486e-06 |
| 6 | 2.481e-06 |

Both within tolerance.

## Reproduce

```bash
python scripts/reproduce_kernel_bench.py --shape 4,2048,6,128
```

## Provenance

- Commit: `c6862ced7da70782` (short hash of 33 `.py` files under `pkg/`)
- Date: `2026-09-27`
- Hardware: TPU v5e-1 (single chip, B=4)
- JAX `0.11.2`, libtpu `0.0.48`, Python `3.12.13` / `3.13.15`
- Raw scaling data: `scaling.json`

## Not included

Slope-decomposed per-layer numbers (NB16b, B=8) are not included: the
NONE-baseline method is not valid at B=4 because the elementwise NONE
stub materializes more intermediates than the fused kernel. Only direct
measurements are reported.
