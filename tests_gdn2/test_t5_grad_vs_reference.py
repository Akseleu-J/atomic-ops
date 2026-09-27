"""T5/T7: analytic custom_vjp vs autodiff of forward_ref; causality, determinism, dtype, batch independence."""
import jax, jax.numpy as jnp, numpy as np, pytest
from atomic_gdn2 import make_blr_trainable
from atomic_gdn2.reference import forward_ref
from conftest import mk_inputs, rel_l2, DOMAIN

D, L = 64, 128
SC = 1 / 8.0
sw = lambda t: jnp.swapaxes(t, 1, 2)


def _both(cfg, args_blhd):
    fn = make_blr_trainable(cfg, SC)
    h0 = jnp.zeros((1, 1, D, D), jnp.float32)
    def ln(*a):
        o, hf = fn(*map(sw, a), h0); return jnp.sum(o * o) + jnp.sum(hf * hf)
    def lr(*a):
        o, hf = forward_ref(*a, SC, h0, cfg); return jnp.sum(sw(o) * sw(o)) + jnp.sum(hf * hf)
    ix = (0, 1, 2, 3, 4, 5)
    return jax.grad(ln, argnums=ix)(*args_blhd), jax.grad(lr, argnums=ix)(*args_blhd)


@pytest.mark.parametrize("name,seed,g,k,b,dom", DOMAIN[:3] + [DOMAIN[3]])
def test_grad_vs_forward_ref(small_cfg, name, seed, g, k, b, dom):
    if name == "strong_decay":
        pytest.skip("reference underflows; covered by test_t2_domain")
    a = mk_inputs(1, L, 1, D, seed, g, k, b)
    gn, gr = _both(small_cfg.with_(g_max=1e3, bs2=32), a)
    errs = {n: rel_l2(x, y) for n, x, y in zip("qkvwbg", gn, gr)}
    assert max(errs.values()) < 3e-3, errs


def test_forward_matches_token_serial(small_cfg):
    from atomic_gdn2.reference import token_serial_ref
    q, k, v, w, b, g = mk_inputs(1, L, 1, D, 0)
    o, _ = make_blr_trainable(small_cfg, SC)(*map(sw, (q, k, v, w, b, g)))
    ots, _ = token_serial_ref(q, k, v, g, b, w, SC)
    assert rel_l2(sw(o), ots) < 5e-3


def test_causality_dq(small_cfg):
    q, k, v, w, b, g = mk_inputs(1, L, 1, D, 0, g=0.05)
    fn = make_blr_trainable(small_cfg, SC)
    def dq_of(g_):
        return jax.grad(lambda a: jnp.sum(fn(sw(a), *map(sw, (k, v, w, b, g_)))[0] ** 2))(q)
    p = L * 3 // 4
    g2 = g.at[:, p].add(-1.0)
    assert float(jnp.max(jnp.abs(dq_of(g)[:, :p] - dq_of(g2)[:, :p]))) == 0.0


def test_determinism_and_dtype(small_cfg):
    args = mk_inputs(1, L, 1, D, 0, layout="bhld")
    fn = make_blr_trainable(small_cfg, SC)
    f = jax.grad(lambda *a: jnp.sum(fn(*a)[0] ** 2), argnums=(0, 1, 2, 3, 4, 5))
    g1, g2 = f(*args), f(*args)
    assert all(bool(jnp.all(x == y)) for x, y in zip(g1, g2))
    bf = tuple(t.astype(jnp.bfloat16) for t in args)
    gb = f(*bf)
    assert all(x.dtype == jnp.bfloat16 for x in gb)


def test_gate_can_fail():
    x = jnp.ones((4, 4))
    assert rel_l2(x * 1.001, x) > 1e-4
