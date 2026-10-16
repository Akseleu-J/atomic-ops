# Current limitations and historical scope

## v0.2 atomic_gdn2

- The accelerated/tested envelope is TPU v5e, with D=128 in the headline historical measurements.
  Other shapes/backends are not covered by those timing claims. CPU interpret mode is debugging.
- L must be divisible by bt, bt=2*bc, and score/inner/leaf dimensions must satisfy configuration checks.
- BTL factorization relies on a valid exponential half-span envelope. The default g_max factory
  binding is not a proof for arbitrary user overrides or invalid g inputs.
- Stage precision differs: highest and bf16x3 are both used. Clipping/damping alter the numerical
  computation; safeguard activation frequency was not characterized in the model runs.
- The forward saves per-chunk matrices/states/values; memory is not O(1) in training sequence length.
  Long sequences split into segments; one fused backward call is per segment.
- Historical speedup values use a supplied scan wrapper with known reproduction/equivalence defects.
  The corrected parity-gated harness has not been timed on TPU. No universal 468x or model-level
  200x speedup is established by the current publication update.
- The attestation is aggregated and lacks its original per-notebook raw logs. Content hashes and
  Git commits must be distinguished. The old 200.17x documentation record is not in that aggregate.
- No claim that a 307M hybrid/334M baseline ratio is explained by the kernel: both use this kernel.
- Numerical agreement under a reported shape/config is not proof of all backward/optimizer behavior.

## Legacy atomic_ops v0.1

Earlier VPU-bound pairwise scores and slow leaf solve measurements refer to the old/lean path;
they must not be presented as the only current v0.2 BTL implementation. Likewise the legacy
fallback dispatcher must not be attributed to the unconditional v0.2 make_blr_trainable API.
The rejected JAX-forward/Pallas-backward experiment is preserved in archive/ and not shipped
as the current path. Historical details remain available in Git history and ROADMAP.md.

See docs/PUBLICATION_EVIDENCE.md and docs/REPRODUCTION_PROTOCOL.md for the publication scope.
