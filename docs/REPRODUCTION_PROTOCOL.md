# Corrected reproduction protocol (not yet timed)

The historical `speedup_scaling.ipynb` passed h0 twice, ignored h0/scale in OLD and returned
a zero final state, while NEW used the supplied h0/scale and returned the final state.
The OLD combine also normalized affine transitions. Those wrappers were not identical tasks.
The old notebook had no saved outputs; historical timings remain in attestation unchanged.

The corrected script uses pure affine associative composition, actual initial/final states,
the same output scale and objective, identical precision and zero WY damping for the equivalence
gate, equal warmup/repeats, full source/environment metadata and synchronized timing.
Forward and gradient parity gates must pass before a speedup is emitted. Gate tolerances and
configuration are recorded in the output. A failed/OOM baseline is reported, not converted
into an infinite speedup. It does not promise to reproduce the old 468× headline.

Run `python scripts/reproduce_kernel_bench.py --shape 4,2048,6,128 --output benchmark.json`
on a TPU. This documentation update did not run the script or consume TPU quota.
It benchmarks the kernel only, not the entire 307M language model. Model evidence is in
https://github.com/Akseleu-J/atomic-gdn2 .
