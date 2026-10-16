# Kernel evidence for the language-model publication

## Source and scope

The four 2026-10-16 model runs used `atomic_gdn2` from immutable atomic-ops commit
`0dbc59cf74883315d7adbebe9229c520663e0d58`. This documentation revision does not alter kernel math.
The `atomic_ops` v0.1 port and `atomic_gdn2` v0.2 implementation are separate generations.
The language-model comparison uses v0.2 for both architectures, not different backends.

## Implementation facts

- Three forward Pallas stages: scores, leaf-batched WY solve, fused factor reconstruction/chunk scan.
- One fused B2/B1/B3/B4/B5 backward call per segment, not one call for all forward/backward or unlimited context.
- Custom VJP retains chunk-start states, updated values, score/solve matrices and inputs;
  backward selectively reconstructs local factors. Saved training memory grows with chunk count.
- The default segment cap is 32 chunks. The API carries h0/h_final across segments.
- `dot_mode=highest`, `solve_dot_mode=bf16x3`, `b3_dot_mode=bf16x3`; B4 explicitly uses highest.
- bf16x3 uses hi/lo components and three products, not three components per operand.
- Solve: `A=(I+(1-eps)Akk)^-1`; eps=1e-3 rescales triangular coupling, not eps-I loading.
- With b_s2=32, g_max=160/32=5 and the BTL half-span bound is 80<88.
- Kernel clip=1e4 and hybrid mixer clip=1e3 are different safeguards. Their activation rate is unmeasured.

## Recurrence implemented by the reference

    Sbar = Diag(exp(g)) S_prev
    u = b * k
    v_new = w * v - Sbar.T @ u
    S = Sbar + outer(k, v_new)
    o = scale * S.T @ q

The erase/write gates are channel-wise. Omitting w or replacing b by an unexplained scalar
does not specify the current API. Damping and sanitization require an explicit numerical qualification.

## Historical reports

`attestation/scaling.json` reports 192.83x--467.64x versus the supplied associative-scan wrapper,
single v5e chip, B=4, D=128, b_s2=64, H=2/4/6 and L=2048/4096. It cites repo_commit b8a7687.
Forward relative L2 values 2.486e-6 and 2.481e-6 refer to two shapes, not universal backward error.
The `c6862ced...` identifier in the attestation README is described as a content hash and
must not automatically be treated as a Git commit. Original per-notebook raw logs are not committed.

The aggregate has 145 PASS / 1 FAIL entries mixing numerical, training and status gates.
The FAIL is an OLD MQAR OOM, not a NEW numerical failure. Do not call this 146 numerical tests.

Older README reports 103.29 vs 20674.62 ms (200.17x) for a 27.5M L=2048 LM. The matching
raw artifact is not in the inspected aggregate. `final_report.json` instead contains a different
27.49M L=4096 run: NEW 169.6 ms, PROD 757.65 ms, OLD 26764.55 ms (157.81x and 4.47x).
These records must not be merged, transferred to the 307M hybrid, or selected as if from one protocol.

## Reproducer correction

The previous notebook had duplicate h0 arguments, unequal h0/final-state/scale behavior,
transition normalization inside OLD composition, a removed b4_dot_mode field hidden by a broad
exception, and different timing repeat counts. It had no saved execution outputs.
These defects do not prove historical measurements false, but prevent treating the old wrapper
as an equivalent-workload reproducer. See `docs/REPRODUCTION_PROTOCOL.md` in atomic-ops.

The updated harness compares identical nonzero h0, actual final state, output scale and loss,
uses pure affine composition, zero damping for equivalence, equal precision and repeats,
and gates timings on forward and gradient parity. It has not yet been timed on TPU.
No new performance number is claimed by this publication update.

## Repositories

- Kernel and historical attestation: https://github.com/Akseleu-J/atomic-ops
- Current model/code/paper and four run records: https://github.com/Akseleu-J/atomic-gdn2
- Current parameter exports/model card: https://huggingface.co/Akseleu/Atomic-GDN2
