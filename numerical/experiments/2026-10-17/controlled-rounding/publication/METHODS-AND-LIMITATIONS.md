# Numerical sensitivity: methods and scope

This artifact tests a mechanism, not a predetermined conclusion. Stored float32 inputs are compared without changing their values. OLD and WY are derived transparently from archived benchmark sources; source derivations and hashes accompany the report. FP64 runs on CPU, not TPU. The independent Decimal oracle implements the literal sequential recurrence and forward tangent, without reusing scan/WY code. Decimal precision refinement and central finite differences check the oracle.

Both an ordinary quadratic objective and an intentionally constructed near-cancelling linear objective are reported for every predeclared seed. One stress coefficient is chosen using the oracle tangent and rounded to float32 before all algorithms run. This construction establishes possibility, not event frequency. Absolute errors are primary near zero; sign disagreement alone is not an accuracy metric.

The optimizer illustration uses zero initial moments. Its analytic update uses bias-corrected moments, excludes clipping and decay, and varies epsilon. A separate Optax replay starts from the previously observed clipped gradients; it tests amplification only. Epsilon does not generate an adaptive update from an exactly zero gradient with zero moments. Weight decay can still update parameters.

## Interpretation rule

If FP64 OLD/WY approach the independent oracle while FP32 differences persist, the data support finite-precision sensitivity on these inputs. If this does not happen, investigate algorithmic or implementation differences. Do not silently relabel a failed check as passed. Inspect evidence-summary.json, all cases and logs before writing a conclusion.

## Limits for publication

This reduced experiment does not establish the mathematically exact gradient of the original LM coordinate, exclude every Pallas implementation defect, establish long-run optimizer equivalence, or demonstrate unchanged validation quality. Previously completed training runs provide separate evidence of practical training behavior. Combine this experiment with the independent TPU NEW/JAX_REF comparison, but label their scopes separately. A controlled demonstration and an observed full-model discrepancy must not be described as the same experiment.

No independent statistical sampling claim is made from three predetermined seeds. Report all per-seed values rather than a significance test. These numbers are numerical diagnostics, not speedup measurements.

## Files

CSV files contain all cases, oracle checks, precision convergence and the observed-coordinate replay. The LaTeX table includes every algorithm, precision, objective and seed. Vector PDF/SVG and raster PNG show all stress cases. Original input arrays, exact executed sources, environment freeze, raw report and archive manifest support reproduction. Incomplete execution remains explicitly incomplete.
