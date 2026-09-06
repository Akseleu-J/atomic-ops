import pytest
import jax
import jax.numpy as jnp
from atomic_ops.reference import gdn2_chunked_wy_reference
from atomic_ops import gdn2_pallas_forward_trainable, gdn2_pallas_forward
def is_tpu_available():
    """Проверяем, доступен ли TPU."""
    try:
        jax.devices('tpu')
        return True
    except:
        return False

@pytest.mark.skipif(not is_tpu_available(), reason="TPU not available")
def test_pallas_forward_backward():
    shape = (2, 512, 4, 128)
    q = jnp.ones(shape, dtype=jnp.float32)
    k = jnp.ones(shape, dtype=jnp.float32)
    v = jnp.ones(shape, dtype=jnp.float32)
    w = jnp.ones(shape, dtype=jnp.float32)
    b = jnp.ones(shape, dtype=jnp.float32)
    g = jnp.ones(shape, dtype=jnp.float32)
    scale = 0.1

    out, h_final = gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale)
    assert out.shape == shape
    assert h_final.shape == (shape[0], shape[2], shape[3], shape[3])

@pytest.mark.skipif(not is_tpu_available(), reason="TPU not available")
def test_pallas_forward_only():
    shape = (2, 512, 4, 128)
    q = jnp.ones(shape, dtype=jnp.float32)
    k = jnp.ones(shape, dtype=jnp.float32)
    v = jnp.ones(shape, dtype=jnp.float32)
    w = jnp.ones(shape, dtype=jnp.float32)
    b = jnp.ones(shape, dtype=jnp.float32)
    g = jnp.ones(shape, dtype=jnp.float32)
    scale = 0.1

    out, _ = gdn2_pallas_forward(q, k, v, w, b, g, scale)
    assert out.shape == shape
