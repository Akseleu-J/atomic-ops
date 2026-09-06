def test_imports():
    from atomic_ops import (
        gdn2_forward_trainable,
        gdn2_forward,
        gdn2_pallas_forward_trainable,
        gdn2_chunked_wy_reference,
        KernelConfig,
        is_tpu_available,
        estimate_memory,
    )
    assert gdn2_forward_trainable is not None

def test_version():
    import atomic_ops
    assert hasattr(atomic_ops, "__version__")
