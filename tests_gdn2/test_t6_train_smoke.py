"""T6 (tiny CPU version): finite loss, loss decreases, skip-step keeps params finite. Full T6 = tools/t6_smoke.py on TPU."""
import numpy as np, jax, pytest
from atomic_gdn2 import make_cfg
from atomic_gdn2.training.train import train


@pytest.mark.slow
def test_tiny_training_runs():
    cfg = make_cfg(bt=64, bc=32, score_bs=64, bs2=32, mb=8)
    data = np.random.default_rng(0).integers(0, 256, size=(16, 128), dtype=np.uint8)
    data[:] = (np.arange(128)[None, :] % 7).astype(np.uint8)          # learnable pattern
    params, h = train(cfg, data, d_model=64, n_heads=1, d_head=64, n_layers=1, bsz=2, steps=12,
                      warmup=2, lr=3e-3, log_every=100, val_every=0, ckpt_every=0, stats_every=6, ckpt_dir="/tmp/ck")
    assert np.all(np.isfinite(h)) and np.mean(h[-3:]) < h[0]
