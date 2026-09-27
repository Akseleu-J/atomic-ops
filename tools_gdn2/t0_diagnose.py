"""T0: arbiter of the enwik8 blow-up. Usage on TPU:
  python tools/t0_diagnose.py --steps 900 --branch base        # trains, saves ckpt every 50
  python tools/t0_diagnose.py --resume ckpt/step800.pkl --branch bs2_32 --steps 400
Branches (restart from the SAME checkpoint): base(bs2=128,gmax off) | bs2_32 | lean | highest | xla_ref
Read logs: half_span_max crossing 88 right before the spike => T-22; spike only in base but fixed by
highest => precision; xla_ref also spikes => model/optimizer, not the kernel."""
import argparse, os, sys, pickle, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax, jax.numpy as jnp, numpy as np, optax
from atomic_gdn2 import make_cfg
from atomic_gdn2.training import train as T
from atomic_gdn2.training.data import load_enwik8

ap = argparse.ArgumentParser()
ap.add_argument("--branch", default="base"); ap.add_argument("--steps", type=int, default=900)
ap.add_argument("--L", type=int, default=4096); ap.add_argument("--bsz", type=int, default=8)
ap.add_argument("--resume", default=None); ap.add_argument("--old_param", action="store_true",
    help="reproduce the failing setup: g=-softplus unclamped, no l2norm, bs2=128")
a = ap.parse_args()

kw = dict(bt=128, bc=64, score_bs=128, mb=8)
if a.branch == "base":      cfg = make_cfg(bs2=128, g_max=1e3, **kw)
elif a.branch == "bs2_32":  cfg = make_cfg(bs2=32, **kw)
elif a.branch == "lean":    cfg = make_cfg(bs2=128, g_max=1e3, diag="lean", **kw)
elif a.branch == "highest": cfg = make_cfg(bs2=128, g_max=1e3, b3_dot_mode="highest", solve_dot_mode="highest", **kw)
else: raise SystemExit("xla_ref: run tests/xla_ref_train.py (L=1024) — see README")
data = load_enwik8(a.L, "train"); val = load_enwik8(a.L, "val")
T.train(cfg, data, val, bsz=a.bsz, steps=a.steps, stats_every=25, ckpt_every=50,
        ckpt_dir=f"ckpt_{a.branch}", l2norm_k=not a.old_param, g_init=0.69 if a.old_param else 0.05)
