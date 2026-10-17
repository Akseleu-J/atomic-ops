# CI scope and existing lint debt

CPU jobs use Python 3.12/3.13 with JAX/JAXlib 0.11.2. These are unit/reference and
Pallas-interpretation checks, not TPU timing or a new evaluation of model quality.
TPU-only lowering tests carry explicit skip reasons on CPU. Their absence from a
CPU pass must not be reported as a fresh TPU pass.

The legacy score-oracle test now retains its singleton chunk axis before comparison.
Previously `[B,H,1,C,C]` versus `[B,H,C,C]` broadcasting could compare different heads
and batches. Shapes are now asserted equal; the numerical threshold is unchanged.

## Lint policy

`Lint (no new debt)` pins Ruff 0.14.0 and checks tracked active Python files, including
both kernel namespaces, tests and tools. Historical `archive/` code is excluded.
Existing unresolved diagnostics and format
debt are listed in `.github/lint-baseline.json`. This includes style and Bugbear
warnings; recording a baseline does not fix them or certify their harmlessness.

CI rejects any new diagnostic fingerprint or newly unformatted file. Syntax errors
and undefined/local-before-assignment names cannot be baselined. The job summary
reports the remaining counts explicitly. A green result means **no new lint debt**,
not that the whole repository passes a strict unbaselined Ruff audit.

Baseline refresh is an explicit reviewed maintenance operation:

```bash
python ci/check_lint.py --write-baseline
```

CI never runs that command. Do not refresh the baseline merely to bypass a newly
introduced issue; inspect and fix the change first.

The historical measured timing archives and kernel implementation are unchanged by
these CI/test-harness corrections.
