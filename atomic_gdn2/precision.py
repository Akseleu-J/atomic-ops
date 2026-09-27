"""Matmul precision modes and safe exponentials (invariant g<=0, alpha in (0,1])."""
import jax, jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST
DEFAULT = jax.lax.Precision.DEFAULT


def split_bf16(x):
    hi = x.astype(jnp.bfloat16).astype(jnp.float32)
    lo = (x - hi).astype(jnp.bfloat16).astype(jnp.float32)
    return hi, lo


def dot_bf16x3(a, b):
    a_hi, a_lo = split_bf16(a); b_hi, b_lo = split_bf16(b)
    return (jnp.dot(a_hi, b_hi, precision=DEFAULT) + jnp.dot(a_hi, b_lo, precision=DEFAULT)
            + jnp.dot(a_lo, b_hi, precision=DEFAULT))


def einsum_bf16x3(spec, a, b):
    a_hi, a_lo = split_bf16(a); b_hi, b_lo = split_bf16(b)
    return (jnp.einsum(spec, a_hi, b_hi, precision=DEFAULT)
            + jnp.einsum(spec, a_hi, b_lo, precision=DEFAULT)
            + jnp.einsum(spec, a_lo, b_hi, precision=DEFAULT))


def make_dot(mode: str):
    if mode == "bf16x3":
        return dot_bf16x3
    p = HIGHEST if mode == "highest" else DEFAULT
    return lambda a, b: jnp.dot(a, b, precision=p)


def make_einsum(mode: str):
    if mode == "bf16x3":
        return einsum_bf16x3
    p = HIGHEST if mode == "highest" else DEFAULT
    return lambda spec, a, b: jnp.einsum(spec, a, b, precision=p)


def exp_nonpos(x):
    """exp(min(x,0)) and mask 1{x<0}. NOTE: mask must NOT be used to gate
    dgc contributions (bug O-1: x==0 on flat gc segments)."""
    return jnp.exp(jnp.minimum(x, 0.0)), (x < 0.0).astype(jnp.float32)


def exp_clipped(x, lo, hi):
    return jnp.exp(jnp.clip(x, lo, hi)), ((x >= lo) & (x <= hi)).astype(jnp.float32)


def sanitize(x, clip: float):
    return jnp.nan_to_num(jnp.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)
