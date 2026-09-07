"""
Minimal usage example for atomic_ops.

Install:
    pip install atomic-ops

Run:
    python minimal_usage.py
"""
import jax.numpy as jnp
from atomic_ops import gdn2_forward_trainable, is_tpu_available

# Shapes: (batch, seq_len, heads, d_head). d_head must be 128 (MXU tile size).
batch, seq_len, heads, d_head = 4, 2048, 6, 128
shape = (batch, seq_len, heads, d_head)

q = jnp.ones(shape, dtype=jnp.float32)
k = jnp.ones(shape, dtype=jnp.float32)
v = jnp.ones(shape, dtype=jnp.float32)
w = jnp.ones(shape, dtype=jnp.float32)
b = jnp.ones(shape, dtype=jnp.float32)
g = jnp.ones(shape, dtype=jnp.float32)  # log-decay (recommend g <= 0)

scale = d_head ** -0.5

# Forward + backward (custom_vjp under the hood).
# On TPU with d_head=128 this dispatches to the fused Pallas kernels;
# otherwise it falls back to a pure-JAX reference implementation.
print(f"TPU available: {is_tpu_available()}")
out, h_final = gdn2_forward_trainable(q, k, v, w, b, g, scale)

print("output:     ", out.shape)      # (4, 2048, 6, 128)
print("final state:", h_final.shape)  # (4, 6, 128, 128)
