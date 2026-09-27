"""T2: BTL validity boundary at half_span ~ 88 and guard quality (recall must be 100%, no late alarm)."""
import jax, jax.numpy as jnp, numpy as np, pytest
from atomic_gdn2 import make_cfg
from atomic_gdn2.fwd import build_scores
from atomic_gdn2.domain import half_span_max, check_preconditions, BTL_STOP
from conftest import rel_l2

D, T = 64, 64


def akk(cfg, c):
    rng = np.random.default_rng(0)
    q, k, b = (jnp.asarray(rng.normal(size=(1, 1, 1, T, D)).astype(np.float32) * s) for s in (0.3, 0.3, 0.5))
    b = jnp.abs(b)
    g = -jnp.full((1, 1, 1, T, D), c, jnp.float32)
    gc = jnp.cumsum(g, axis=-2)
    return build_scores(q, k, b, gc, 1 / 8.0, cfg)[1], gc


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0])
def test_btl_boundary_and_guard(c):
    bs2 = 32
    cb = make_cfg(bt=T, bc=32, score_bs=T, bs2=bs2, mb=8, diag="btl", g_max=1e3)
    cl = cb.with_(diag="lean")
    Ab, gc = akk(cb, c); Al, _ = akk(cl, c)
    err = rel_l2(Ab, Al)
    hs = float(half_span_max(gc, bs2))
    alarm = hs > BTL_STOP
    wrong = err > 1e-3
    assert not (wrong and not alarm), f"SILENT FAILURE c={c}: err={err:.2e} half_span={hs:.1f}"
    if hs < 60:
        assert not wrong, f"false failure inside domain: err={err:.2e}"


def test_precondition_check_raises():
    g = -jnp.full((1, 1, 128, 8), 6.0)
    with pytest.raises(RuntimeError):
        check_preconditions(g, make_cfg(bs2=32, g_max=1e3), strict=True)
