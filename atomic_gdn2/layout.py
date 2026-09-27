"""Head-major layout (B,H,L,D) <-> chunks (B,H,nc,bt,D): pure reshapes."""
import jax.numpy as jnp

def to_chunks_hm(t, nc, bt):
    b, h, L, d = t.shape
    assert L == nc * bt, f"L={L} != nc*bt={nc*bt}"
    return t.reshape(b, h, nc, bt, d)

def from_chunks_hm(t):
    b, h, nc, bt, d = t.shape
    return t.reshape(b, h, nc * bt, d)

def causal_masks(T):
    i = jnp.arange(T)
    return (i[:, None] >= i[None, :]).astype(jnp.float32), (i[:, None] > i[None, :]).astype(jnp.float32)

def cparams(vmem_mb):
    from jax.experimental.pallas import tpu as pltpu
    for nm in ("CompilerParams", "TPUCompilerParams"):
        c = getattr(pltpu, nm, None)
        if c is not None:
            return c(vmem_limit_bytes=int(vmem_mb * 1024 * 1024))
    return None
