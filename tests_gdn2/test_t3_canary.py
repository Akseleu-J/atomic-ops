"""T3: canary — BTL vs lean on the same chunk sample; deviation over threshold => alarm."""
import jax.numpy as jnp
from atomic_gdn2 import make_cfg
from atomic_gdn2.training.canary import btl_vs_lean_canary


def test_canary_quiet_in_domain():
    cfg = make_cfg(bt=64, bc=32, score_bs=64, bs2=32, mb=8)
    r = btl_vs_lean_canary(cfg, D=64, g_level=0.3)
    assert r["rel_err"] < 1e-4 and not r["alarm"]


def test_canary_fires_out_of_domain():
    cfg = make_cfg(bt=64, bc=32, score_bs=64, bs2=32, mb=8, g_max=1e3)
    r = btl_vs_lean_canary(cfg, D=64, g_level=8.0)
    assert r["alarm"]
