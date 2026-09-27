# atomic_gdn2 attestation — 2026-09-27T06:51:37.349638

**PASS=145  FAIL=1  SUSPECT=0  SKIP=0**

## Sources
- **nb1**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb10**: commit=c6862ced7da70782 backend=tpu ndev=8
- **nb11**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb12**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb13**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb16**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb16b**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb2**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb3**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb4**: commit=c6862ced7da70782 backend=tpu ndev=8
- **nb5**: commit=c6862ced7da70782 backend=tpu ndev=8
- **nb7**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb8**: commit=c6862ced7da70782 backend=tpu ndev=8
- **nb9**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb_hbm**: commit=c6862ced7da70782 backend=tpu ndev=1
- **nb_scale**: commit=c6862ced7da70782 backend=tpu ndev=1

## Open items (перечислить честно, не скрывать)
- T-22 verdict: INFO (base_first_nf=None)
- T4 finite-diff (tol=5e-3): PASS rel=0.0006531
- b4_dot phantom root-cause: b4_dot_mode phantom: _diag_btl uses passed dot, _diag_lean ignores it entirely
- MQAR zero-shot: done