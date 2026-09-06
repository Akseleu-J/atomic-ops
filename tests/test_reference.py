import pytest
import jax
import jax.numpy as jnp
from atomic_ops.reference import gdn2_chunked_wy_reference
@pytest.mark.parametrize("batch, seq, heads, d_head", [(2, 512, 4, 128)])
def test_reference_forward(batch, seq, heads, d_head):
    """Проверяем, что reference-функция работает и возвращает правильные формы."""
    shape = (batch, seq, heads, d_head)
    q = jnp.ones(shape, dtype=jnp.float32)
    k = jnp.ones(shape, dtype=jnp.float32)
    v = jnp.ones(shape, dtype=jnp.float32)
    w = jnp.ones(shape, dtype=jnp.float32)
    b = jnp.ones(shape, dtype=jnp.float32)
    g = jnp.ones(shape, dtype=jnp.float32)
    scale = 0.1
    chunk_size = 256

    out, h_final = gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size)
    assert out.shape == shape
    assert h_final.shape == (batch, heads, d_head, d_head)
