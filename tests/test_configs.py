import jax.numpy as jnp

from atomic_ops.configs import (
    KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE, DEFAULT_CONFIG,
    sanitize, clip_acc, validate_inputs,
)


def test_default_config_is_medium():
    assert DEFAULT_CONFIG == KAGGLE_MEDIUM


def test_sanitize_removes_nan_and_clips():
    x = jnp.array([float("nan"), float("inf"), -float("inf"), 5.0, 1e9])
    out = sanitize(x, KAGGLE_SMALL)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert float(out[3]) == 5.0
    assert float(out[4]) == KAGGLE_SMALL.clip


def test_clip_acc_matches_sanitize():
    x = jnp.array([2e5, -2e5, 1.0])
    assert bool(jnp.all(clip_acc(x, KAGGLE_LARGE) == sanitize(x, KAGGLE_LARGE)))


def test_n_sub_n_micro_consistency():
    for cfg in (KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE):
        assert cfg.bt % cfg.bc == 0
        assert cfg.bc % cfg.mb == 0


def test_config_rejects_bad_shapes():
    import pytest
    with pytest.raises(ValueError):
        from atomic_ops.configs import KernelConfig
        KernelConfig(bt=100, bc=64)  # 100 % 64 != 0


def test_validate_inputs_shape_mismatch():
    import pytest
    q = jnp.ones((2, 256, 4, 128))
    k = jnp.ones((2, 256, 4, 64))  # wrong D
    with pytest.raises(ValueError):
        validate_inputs(q, k, q, q, q, q, 0.1, None, KAGGLE_MEDIUM)


def test_validate_inputs_ok():
    q = jnp.ones((2, 256, 4, 128))
    out = validate_inputs(q, q, q, q, q, q, 0.1, None, KAGGLE_MEDIUM)
    assert out == (2, 256, 4, 128, 1)
