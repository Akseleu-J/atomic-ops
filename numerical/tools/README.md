# Verification tools

tpu_exact_confirmation.py matches the executed worker from the recorded TPU run. It accepts --capture and --out. The delivered notebook refuses CPU fallback.

verify_exact_tpu_archive.py uses only the Python standard library. It independently decodes IEEE binary32 and checks raw device integer bins. CLI defaults were replaced by required path arguments for portability; its arithmetic is unchanged.

Example from the extracted bundle root:

```bash
python numerical/tools/verify_exact_tpu_archive.py --archive artifacts/current-evidence/Atomic-gdn2-exact-TPU-confirmation-results-20261017T150441Z.zip --expected-source numerical/tools/tpu_exact_confirmation.py --out verification-output
```
