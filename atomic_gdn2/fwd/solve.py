"""Kernel B: A = (I + (1-eps) Akk)^-1 by block-doubling ladder with leaf-batched base inverses."""
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from ..precision import make_einsum, sanitize
from ..layout import cparams
from ..config import effective_group


def _leaf_inverse(T_mb, mb, N):
    idx = jnp.arange(mb)
    def body(i, A):
        oh = (idx == i).astype(jnp.float32)
        t_row = jnp.sum(T_mb * oh[None, :, None], axis=1)
        contrib = jnp.sum(t_row[:, :, None] * A, axis=1)
        new_row = oh[None, :] - contrib
        m = oh[None, :, None]
        return A * (1.0 - m) + m * new_row[:, None, :]
    return jax.lax.fori_loop(0, mb, body, jnp.zeros((N, mb, mb), jnp.float32))


def ladder_inverse_leafbatched(S, eps, C, base, mode="bf16x3"):
    nb = C // base
    assert C % base == 0 and nb & (nb - 1) == 0 and base & (base - 1) == 0
    ein = make_einsum(mode)
    Se = S * (1.0 - eps)
    leaves = jnp.stack([Se[..., m*base:(m+1)*base, m*base:(m+1)*base] for m in range(nb)], axis=-3)
    lead = leaves.shape[:-3]
    flat = leaves.reshape((-1,) + leaves.shape[-3:])
    X0 = jax.vmap(lambda t: _leaf_inverse(t, base, nb))(flat).reshape(lead + (nb, base, base))
    X = [X0[..., m, :, :] for m in range(nb)]
    b = base
    while b < C:
        n2, new = 2 * b, []
        for m in range(C // n2):
            top, bot = X[2*m], X[2*m+1]
            i0 = m * n2
            S_l = Se[..., i0+b:i0+n2, i0:i0+b]
            mid = ein("...ij,...jk->...ik", S_l, top)
            ll = -ein("...ij,...jk->...ik", bot, mid)
            z = jnp.zeros_like(ll)
            new.append(jnp.concatenate([jnp.concatenate([top, z], -1), jnp.concatenate([ll, bot], -1)], -2))
        X, b = new, n2
    return X[0]


def _kernel(akk_ref, a_ref, *, cfg):
    S = akk_ref[0, 0].astype(jnp.float32)
    a_ref[0, 0] = sanitize(ladder_inverse_leafbatched(S, cfg.wy_eps, cfg.bt, cfg.mb, cfg.solve_dot_mode), cfg.clip)


def wy_solve_leafbatched(Akk, cfg, group=None):
    bsz, H, nc = Akk.shape[:3]
    g = effective_group(nc, group if group is not None else cfg.group)
    spec = pl.BlockSpec((1, 1, g, cfg.bt, cfg.bt), lambda i, h, c: (i, h, c, 0, 0))
    return pl.pallas_call(
        lambda *r: _kernel(*r, cfg=cfg), grid=(bsz, H, nc // g),
        in_specs=[spec], out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(Akk.shape, jnp.float32),
        compiler_params=cparams(cfg.vmem_mb), interpret=cfg.interpret)(Akk)
