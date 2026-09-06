from __future__ import annotations
import jax.numpy as jnp
from .utils import is_tpu_available
from .gdn2_fwd import gdn2_pallas_forward as _pallas_fwd
from .gdn2_pipeline import gdn2_pallas_forward_trainable as _pallas_trainable
from .reference import gdn2_chunked_wy_reference


def gdn2_forward(q, k, v, w, b, g, scale, h0=None, config=None):
    from .configs import DEFAULT_CONFIG
    config = config or DEFAULT_CONFIG
    if is_tpu_available() and q.shape[-1] == 128:
        return _pallas_fwd(q, k, v, w, b, g, scale, h0=h0, config=config)
    return gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size=config.bt, h0=h0)


def gdn2_forward_trainable(q, k, v, w, b, g, scale, h0=None, config=None):
    from .configs import DEFAULT_CONFIG
    config = config or DEFAULT_CONFIG
    if is_tpu_available() and q.shape[-1] == 128:
        return _pallas_trainable(q, k, v, w, b, g, scale, h0=h0, config=config)
    return gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size=config.bt, h0=h0)
