# Actual-coordinate numerical mechanism audit

The target is block_6/q_conv_w[236,2], seed 0 as originally observed, with seeds 1 and 2 as predeclared controls. The full original 34,873,952-parameter model, B8 L2048, is evaluated on TPU by OLD, JAX_REF and atomic_gdn2. OLD uses block remat. No timings are reported.

An additive zero probe at the selected convolution preactivation records its cotangent. The model also returns that convolution's input channel. For left padding K-1, tap j, the exact chain-rule contraction is sum(input[b,t+j-(K-1)] * cotangent[b,t]). Its captured-input value is computed with exact rational arithmetic: finite binary32 factors are exact rational numbers. Both factors already contain upstream rounding. This is not a high-precision oracle for the whole network.

The independent uninstrumented full-model gradient is rerun for each path and seed. A zero-valued probe changes the AD graph and can change XLA rounding. Differences must be disclosed; the instrumentation is not assumed numerically transparent. All full gradients are returned from executables; only compact coordinate evidence is archived.

Cross contractions hold either the captured input or cotangent fixed. An exact algebraic decomposition separates their contributions to the local gradient discrepancy. This locates where differences enter; it does not prove their exclusive upstream cause. FP32/FP64/fsum contractions quantify local rounding; cancellation ratios quantify conditioning of this sum. Epsilon sensitivity uses first-step bias-corrected Adam and separately reports weight decay. It does not establish equivalence of optimizer trajectories.

All arrays needed to reproduce the coordinate contractions are in the ZIP. Include every seed and any incomplete phase in reporting. Strong cancellation plus convergence of local FP64 to the exact dot supports a local rounding explanation; persistent differences between exact captured contractions locate upstream discrepancies and must not be silently dismissed. The preceding independent Decimal recurrence experiment is complementary evidence, not an oracle for the real full model.
