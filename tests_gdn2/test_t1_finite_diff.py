"""T1: finite differences through token_serial_ref in float64 (shares NO formula with BLR).
Directional derivative of loss(o) computed by FD on the f64 recurrence must match the kernel's analytic gradient."""
import jax, jax.numpy as jnp, numpy as np
from atomic_gdn2 import make_blr_trainable
from atomic_gdn2.reference import token_serial_ref
from conftest import mk_inputs

D, L, SC = 64, 128, 1 / 8.0
sw = lambda t: jnp.swapaxes(t, 1, 2)


def test_fd_directional(small_cfg):
    args = mk_inputs(1, L, 1, D, 0, g=0.05)             # q,k,v,w,b,g (B,L,H,D)
    fn = make_blr_trainable(small_cfg, SC)
    g_an = jax.grad(lambda *a: jnp.sum(fn(*map(sw, a))[0] ** 2), argnums=(0, 1, 2, 4, 5))(*args)
    rng = np.random.default_rng(1)
    dirs = [rng.normal(size=args[i].shape) for i in (0, 1, 2, 4, 5)]
    an = sum(float(np.sum(np.asarray(g, np.float64) * d)) for g, d in zip(g_an, dirs))
    eps = 1e-5
    with jax.enable_x64(True):
        A = [jnp.asarray(np.asarray(a), jnp.float64) for a in args]
        def f(s):
            q, k, v, w, b, g = list(A)
            for idx, d in zip((0, 1, 2, 4, 5), dirs):
                if idx == 0: q = A[0] + s * d
                if idx == 1: k = A[1] + s * d
                if idx == 2: v = A[2] + s * d
                if idx == 4: b = A[4] + s * d
                if idx == 5: g = A[5] + s * d
            o, _ = token_serial_ref(q, k, v, g, b, w, SC)
            return float(jnp.sum(o * o))
        fd = (f(eps) - f(-eps)) / (2 * eps)
    assert abs(an - fd) / max(abs(fd), 1e-12) < 5e-3, (an, fd)
