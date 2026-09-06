
from __future__ import annotations

import gc
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from atomic_ops.configs import KernelConfig, KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
from atomic_ops.reference import gdn2_chunked_wy_reference


# ==========================================================================
# (OLD) associative_scan baseline -- verbatim algorithm, unchanged.
# ==========================================================================
def _gdn2_recurrence_impl(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype):
    def _combine(state1, state2):
        m1, c1 = state1
        m2, c2 = state2
        m_new = m2 @ m1
        fro_norm = jnp.sqrt(jnp.sum(jnp.square(m_new), axis=(-2, -1), keepdims=True))
        scale = jnp.minimum(1.0, 1.0 / (fro_norm + 1e-6))
        m_new = m_new * scale
        c_new = m2 @ c1 + c2
        c_new = jnp.nan_to_num(c_new, nan=0.0, posinf=1e4, neginf=-1e4)
        return m_new, c_new

    def _to_chunks(t):
        t = t.reshape(b, num_chunks, chunk_size, n_heads, d_head)
        return jnp.moveaxis(t, 1, 0)

    k_ch, ea_ch, z_ch, alpha_ch, q_ch = map(_to_chunks, (k, ea, z, alpha, q))

    eye_bh = jnp.broadcast_to(jnp.eye(d_head, dtype=dtype), (b, n_heads, d_head, d_head))
    zero_bh = jnp.zeros((b, n_heads, d_head, d_head), dtype=dtype)

    def _chunk_step(carry, chunk_inputs):
        carry_M, carry_S = carry
        k_c, ea_c, z_c, alpha_c, q_c = chunk_inputs
        eye = jnp.eye(d_head, dtype=dtype)[None, None, None, :, :]
        M_c = eye * alpha_c[:, :, :, None, :] - k_c[:, :, :, :, None] @ ea_c[:, :, :, None, :]
        C_c = k_c[:, :, :, :, None] @ z_c[:, :, :, None, :]
        P_local, S_local = jax.lax.associative_scan(_combine, (M_c, C_c), axis=1)
        global_M = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_M)
        global_S = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_S) + S_local
        global_S = jnp.nan_to_num(global_S, nan=0.0, posinf=1e4, neginf=-1e4)
        out_c = jnp.einsum("bchij,bchi->bchj", global_S, q_c)
        new_carry = (global_M[:, -1], global_S[:, -1])
        return new_carry, out_c

    _chunk_step = jax.checkpoint(_chunk_step)
    _, out_chunks = jax.lax.scan(
        _chunk_step, (eye_bh, zero_bh), (k_ch, ea_ch, z_ch, alpha_ch, q_ch)
    )
    return jnp.moveaxis(out_chunks, 0, 1).reshape(b, num_chunks * chunk_size, n_heads * d_head)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10))
def gdn2_recurrence_safe(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype):
    return _gdn2_recurrence_impl(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype)


def _gdn2_recurrence_safe_fwd(k, ea, z, alpha, q, chunk_size, num_chunks, b, n_heads, d_head, dtype):
    primal_fn = jax.checkpoint(
        lambda k_, ea_, z_, alpha_, q_: _gdn2_recurrence_impl(
            k_, ea_, z_, alpha_, q_, chunk_size, num_chunks, b, n_heads, d_head, dtype
        )
    )
    out, vjp_fn = jax.vjp(primal_fn, k, ea, z, alpha, q)
    return out, vjp_fn


def _gdn2_recurrence_safe_bwd(chunk_size, num_chunks, b, n_heads, d_head, dtype, vjp_fn, g):
    grads = vjp_fn(g)
    _GRAD_CLIP = 1e3
    return tuple(
        jnp.nan_to_num(jnp.clip(gr, -_GRAD_CLIP, _GRAD_CLIP), nan=0.0, posinf=_GRAD_CLIP, neginf=-_GRAD_CLIP)
        for gr in grads
    )


gdn2_recurrence_safe.defvjp(_gdn2_recurrence_safe_fwd, _gdn2_recurrence_safe_bwd)


def old_associative_scan_forward(q, k, v, w, b, g, scale, h0, chunk_size, n_heads, d_head, bsz=None):
    actual_bsz = q.shape[0]
    alpha = jnp.exp(g.astype(jnp.float32))
    ea = ((b * k) * alpha).astype(q.dtype)
    z = (w * v).astype(q.dtype)
    num_chunks = q.shape[1] // chunk_size
    out = gdn2_recurrence_safe(
        k, ea, z, alpha.astype(q.dtype), q, chunk_size, num_chunks, actual_bsz, n_heads, d_head, q.dtype
    )
    out = out.reshape(actual_bsz, out.shape[1], n_heads, d_head)
    return out, jnp.zeros((actual_bsz, n_heads, d_head, d_head), dtype=q.dtype)


# ==========================================================================
# Config / result data model (memory fields removed)
# ==========================================================================
@dataclass(frozen=True)
class BenchConfig:
    name: str
    B: int
    L: int
    H: int
    D: int
    kernel_config: KernelConfig
    old_micro_bs: Optional[int] = None


@dataclass
class StageTiming:
    compile_s: float
    steady_state_s: list
    micro_bs: Optional[int] = None

    @property
    def mean(self):
        return statistics.mean(self.steady_state_s)

    @property
    def std(self):
        return statistics.pstdev(self.steady_state_s) if len(self.steady_state_s) > 1 else 0.0

    @property
    def min(self):
        return min(self.steady_state_s)

    @property
    def max(self):
        return max(self.steady_state_s)

    @property
    def cv(self):
        return (self.std / self.mean) if self.mean > 0 else float("nan")


@dataclass
class PathResult:
    path_name: str
    fwd: StageTiming
    bwd: StageTiming
    fwdbwd: StageTiming
    correctness_ok: bool
    correctness_max_abs_diff: float
    grad_correctness_ok: Optional[bool] = None
    grad_correctness_max_rel_err: Optional[float] = None


@dataclass
class ConfigResult:
    config_name: str
    dtype: str
    B: int
    L: int
    H: int
    D: int
    n_seeds: int
    n_iters_per_seed: int
    paths: dict = field(default_factory=dict)


# ==========================================================================
# Environment / version stamping (kept -- needed to interpret timings)
# ==========================================================================
def collect_env_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
    }
    try:
        import jaxlib
        info["jaxlib_version"] = jaxlib.__version__
    except Exception:
        info["jaxlib_version"] = "unknown"
    try:
        devs = jax.local_devices()
        info["jax_backend"] = jax.default_backend()
        info["device_count"] = len(devs)
        info["device_kind"] = devs[0].device_kind if devs else "unknown"
        info["device_platform"] = devs[0].platform if devs else "unknown"
    except Exception as e:
        info["device_error"] = str(e)
    try:
        import libtpu
        info["libtpu_version"] = getattr(libtpu, "__version__", "unknown")
    except Exception:
        info["libtpu_version"] = "n/a (not importable / not TPU)"
    return info


def print_env_info(info: dict):
    print("\n" + "=" * 88)
    print("ENVIRONMENT")
    print("=" * 88)
    for k, v in info.items():
        print(f"  {k:>20}: {v}")


def _reset_between_measurements():
    """Best-effort flush of transient buffers between path measurements
    (compile-cache / gc only -- no memory measurement is taken)."""
    gc.collect()
    jax.clear_caches()
    jax.block_until_ready(jnp.zeros((1,)))


# ==========================================================================
# Timing harness
# ==========================================================================
def _time_repeated(fn, args_per_seed, n_iters_per_seed):
    first_args = args_per_seed[0]
    t0 = time.perf_counter()
    out = fn(*first_args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    steady = []
    for args in args_per_seed:
        for _ in range(n_iters_per_seed):
            t0 = time.perf_counter()
            out = fn(*args)
            jax.block_until_ready(out)
            steady.append(time.perf_counter() - t0)
    return compile_s, steady


def _loss_fn(fwd_fn):
    def loss(q, k, v, w, b, g, h0):
        o, h_final = fwd_fn(q, k, v, w, b, g, h0)
        return jnp.sum(o.astype(jnp.float32) ** 2) + jnp.sum(h_final.astype(jnp.float32) ** 2)
    return loss


# --------------------------------------------------------------------
# Micro-batched gradient accumulation (OOM fix for OLD) -- unchanged
# --------------------------------------------------------------------
def make_microbatched_grad_fns(loss_fn, micro_bs):
    def micro_bwd_step(*micro_args):
        _, vjp_fn = jax.vjp(loss_fn, *micro_args)
        cot = jnp.array(1.0, dtype=jnp.float32)
        return vjp_fn(cot)

    def micro_fwdbwd_step(*micro_args):
        val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(*micro_args)
        return val, grads

    micro_bwd_jit = jax.jit(micro_bwd_step)
    micro_fwdbwd_jit = jax.jit(micro_fwdbwd_step)

    def _iter_microbatches(args):
        B = args[0].shape[0]
        for start in range(0, B, micro_bs):
            end = min(start + micro_bs, B)
            yield tuple(a[start:end] for a in args)

    def bwd_only(*args):
        grad_chunks = None
        for micro_args in _iter_microbatches(args):
            g = micro_bwd_jit(*micro_args)
            if grad_chunks is None:
                grad_chunks = [[gi] for gi in g]
            else:
                for lst, gi in zip(grad_chunks, g):
                    lst.append(gi)
        return tuple(jnp.concatenate(lst, axis=0) for lst in grad_chunks)

    def fwdbwd(*args):
        grad_chunks = None
        val_total = 0.0
        for micro_args in _iter_microbatches(args):
            val, g = micro_fwdbwd_jit(*micro_args)
            val_total = val_total + val
            if grad_chunks is None:
                grad_chunks = [[gi] for gi in g]
            else:
                for lst, gi in zip(grad_chunks, g):
                    lst.append(gi)
        grads = tuple(jnp.concatenate(lst, axis=0) for lst in grad_chunks)
        return val_total, grads

    return bwd_only, fwdbwd, micro_bwd_jit, micro_fwdbwd_jit


def bench_path(path_name, fwd_raw, args_per_seed, n_iters_per_seed,
                micro_bs: Optional[int] = None):
    label = path_name if micro_bs is None else f"{path_name} (micro_bs={micro_bs})"
    print(f"  -- benchmarking path: {label}")
    try:
        fwd_jit = jax.jit(fwd_raw)
        compile_fwd, steady_fwd = _time_repeated(fwd_jit, args_per_seed, n_iters_per_seed)
        print(f"     fwd:     mean={statistics.mean(steady_fwd)*1000:.2f}ms")

        loss = _loss_fn(fwd_raw)
        _reset_between_measurements()

        if micro_bs is None:
            def bwd_only(*args):
                _, vjp_fn = jax.vjp(loss, *args)
                cot = jnp.array(1.0, dtype=jnp.float32)
                return vjp_fn(cot)
            bwd_jit = jax.jit(bwd_only)
            compile_bwd, steady_bwd = _time_repeated(bwd_jit, args_per_seed, n_iters_per_seed)

            fwdbwd_fn = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5, 6))
            fwdbwd_jit = jax.jit(fwdbwd_fn)
            compile_fb, steady_fb = _time_repeated(fwdbwd_jit, args_per_seed, n_iters_per_seed)
        else:
            bwd_only, fwdbwd_fn, micro_bwd_jit, micro_fwdbwd_jit = make_microbatched_grad_fns(loss, micro_bs)
            compile_bwd, steady_bwd = _time_repeated(bwd_only, args_per_seed, n_iters_per_seed)
            compile_fb, steady_fb = _time_repeated(fwdbwd_fn, args_per_seed, n_iters_per_seed)

        print(f"     bwd:     mean={statistics.mean(steady_bwd)*1000:.2f}ms")
        print(f"     fwd+bwd: mean={statistics.mean(steady_fb)*1000:.2f}ms")

        return PathResult(
            path_name=path_name,
            fwd=StageTiming(compile_fwd, steady_fwd),
            bwd=StageTiming(compile_bwd, steady_bwd, micro_bs=micro_bs),
            fwdbwd=StageTiming(compile_fb, steady_fb, micro_bs=micro_bs),
            correctness_ok=True,
            correctness_max_abs_diff=float("nan"),
        )
    except Exception as e:
        print(f"    [FAILED] path {label} raised {type(e).__name__} -- "
              f"skipping this path for this config. Detail: {str(e)[:200]}")
        return None


# ==========================================================================
# Correctness gates (kept: they gate whether timings are meaningful)
# ==========================================================================
def _check_correctness(reference_out, candidate_out, tol=1e-2):
    ref_o, ref_h = reference_out
    cand_o, cand_h = candidate_out
    if not (bool(jnp.all(jnp.isfinite(ref_o))) and bool(jnp.all(jnp.isfinite(cand_o)))):
        return False, float("inf")
    diff = float(jnp.max(jnp.abs(ref_o.astype(jnp.float32) - cand_o.astype(jnp.float32))))
    ok = diff < tol
    return ok, diff


def _check_grad_correctness(fwd_ref, fwd_candidate, args, rel_tol=5e-2):
    loss_ref = _loss_fn(fwd_ref)
    loss_cand = _loss_fn(fwd_candidate)
    try:
        grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2, 3, 4, 5, 6))(*args)
        grads_cand = jax.grad(loss_cand, argnums=(0, 1, 2, 3, 4, 5, 6))(*args)
    except Exception as e:
        return False, float("inf"), f"grad computation raised {type(e).__name__}: {str(e)[:200]}"

    names = ("dq", "dk", "dv", "dw", "db", "dg", "dh0")
    max_rel = 0.0
    per_tensor = {}
    all_finite = True
    for name, gr, gc_ in zip(names, grads_ref, grads_cand):
        gr32 = gr.astype(jnp.float32)
        gc32 = gc_.astype(jnp.float32)
        finite = bool(jnp.all(jnp.isfinite(gr32))) and bool(jnp.all(jnp.isfinite(gc32)))
        all_finite = all_finite and finite
        if not finite:
            per_tensor[name] = float("inf")
            max_rel = float("inf")
            continue
        denom = float(jnp.max(jnp.abs(gr32))) + 1e-6
        rel = float(jnp.max(jnp.abs(gc32 - gr32))) / denom
        per_tensor[name] = rel
        max_rel = max(max_rel, rel)

    ok = all_finite and (max_rel < rel_tol)
    detail = ", ".join(f"{k}={v:.2e}" for k, v in per_tensor.items())
    return ok, max_rel, detail


# ==========================================================================
# Input generation
# ==========================================================================
def make_inputs(key, B, L, H, D, dtype=jnp.float32):
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    eps = 1e-6
    q = jax.random.normal(k1, (B, L, H, D), dtype=jnp.float32) * 0.3
    k = jax.random.normal(k2, (B, L, H, D), dtype=jnp.float32) * 0.3
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + eps)
    v = jax.random.normal(k3, (B, L, H, D), dtype=jnp.float32) * 0.3
    g = -jnp.abs(jax.random.normal(k4, (B, L, H, D), dtype=jnp.float32)) * 0.05
    b = jax.random.uniform(k5, (B, L, H, D), minval=0.2, maxval=1.0, dtype=jnp.float32)
    w = jax.random.uniform(k6, (B, L, H, D), minval=0.2, maxval=1.0, dtype=jnp.float32)
    h0 = jax.random.normal(k7, (B, H, D, D), dtype=jnp.float32) * 0.1

    if dtype != jnp.float32:
        q, k, v, g, b, w = (t.astype(dtype) for t in (q, k, v, g, b, w))
    return q, k, v, w, b, g, h0


# ==========================================================================
# Main benchmark loop (single repeat, single dtype)
# ==========================================================================
N_SEEDS = 2
N_ITERS_PER_SEED = 2


def bench_one_config(bench_cfg: BenchConfig, base_key, dtype, n_seeds=N_SEEDS, n_iters=N_ITERS_PER_SEED):
    B, L, H, D = bench_cfg.B, bench_cfg.L, bench_cfg.H, bench_cfg.D
    kcfg = bench_cfg.kernel_config
    chunk_size = kcfg.bt
    assert L % chunk_size == 0, f"L={L} must be divisible by bt={chunk_size}"

    scale = D ** -0.5
    dtype_name = "bf16" if dtype != jnp.float32 else "fp32"
    args_per_seed = [make_inputs(jax.random.fold_in(base_key, s), B, L, H, D, dtype=dtype)
                      for s in range(n_seeds)]

    fwd_old = lambda q, k, v, w, b, g, h0: old_associative_scan_forward(
        q, k, v, w, b, g, scale, h0, chunk_size, H, D, bsz=None)
    fwd_ref = lambda q, k, v, w, b, g, h0: gdn2_chunked_wy_reference(
        q, k, v, g, b, w, scale, chunk_size=chunk_size, h0=h0, wy_eps=kcfg.wy_eps)
    fwd_pallas = lambda q, k, v, w, b, g, h0: gdn2_pallas_forward_trainable(
        q, k, v, w, b, g, scale, h0=h0, config=kcfg)

    print(f"\n=== {bench_cfg.name} [{dtype_name}]: B={B} L={L} H={H} D={D} "
          f"(bt={kcfg.bt}, bc={kcfg.bc}, mb={kcfg.mb}, wy_eps={kcfg.wy_eps}, "
          f"old_micro_bs={bench_cfg.old_micro_bs}) ===")

    print("  forward correctness gate (vs JAX_REF, tol=1e-2, informational only):")
    ref_out = fwd_ref(*args_per_seed[0])
    old_out = fwd_old(*args_per_seed[0])
    pallas_out = fwd_pallas(*args_per_seed[0])
    old_ok, old_diff = _check_correctness(ref_out, old_out)
    pallas_ok, pallas_diff = _check_correctness(ref_out, pallas_out)
    print(f"    OLD    vs JAX_REF: max_abs_diff={old_diff:.4e} (OLD ignores h0 -- not a gate)")
    print(f"    PALLAS vs JAX_REF: max_abs_diff={pallas_diff:.4e}  {'OK' if pallas_ok else 'FAILED GATE'}")

    print("  gradient correctness gate (vs JAX_REF autodiff, rel_tol=5e-2):")
    pallas_grad_ok, pallas_grad_relerr, pallas_grad_detail = _check_grad_correctness(
        fwd_ref, fwd_pallas, args_per_seed[0])
    print(f"    PALLAS grads vs JAX_REF: max_rel_err={pallas_grad_relerr:.4e}  "
          f"{'OK' if pallas_grad_ok else 'FAILED GATE'}  [{pallas_grad_detail}]")

    if not pallas_ok:
        print("    [!!] PALLAS failed the FORWARD correctness gate -- SKIPPING timing for this config.")
        return None
    if not pallas_grad_ok:
        print("    [!!] PALLAS failed the GRADIENT correctness gate -- SKIPPING timing for this config.")
        return None

    result = ConfigResult(
        config_name=bench_cfg.name, dtype=dtype_name, B=B, L=L, H=H, D=D,
        n_seeds=n_seeds, n_iters_per_seed=n_iters,
    )

    _reset_between_measurements()
    old_res = bench_path("OLD (associative_scan)", fwd_old, args_per_seed, n_iters,
                          micro_bs=bench_cfg.old_micro_bs)
    if old_res is not None:
        old_res.correctness_ok, old_res.correctness_max_abs_diff = True, old_diff

    _reset_between_measurements()
    ref_res = bench_path("JAX_REF (chunked-WY, plain JAX)", fwd_ref, args_per_seed, n_iters)
    if ref_res is not None:
        ref_res.correctness_ok, ref_res.correctness_max_abs_diff = True, 0.0

    _reset_between_measurements()
    pallas_res = bench_path("PALLAS (atomic_ops)", fwd_pallas, args_per_seed, n_iters)
    if pallas_res is not None:
        pallas_res.correctness_ok, pallas_res.correctness_max_abs_diff = pallas_ok, pallas_diff
        pallas_res.grad_correctness_ok = pallas_grad_ok
        pallas_res.grad_correctness_max_rel_err = pallas_grad_relerr

    if ref_res is None or pallas_res is None:
        print("    [!!] JAX_REF or PALLAS crashed -- skipping config.")
        return None
    if old_res is None:
        print("    [NOTE] OLD still crashed even with micro-batching for this config.")

    result.paths = {k: v for k, v in
                     (("OLD", old_res), ("JAX_REF", ref_res), ("PALLAS", pallas_res))
                     if v is not None}
    return result


CONFIGS = [
    BenchConfig("small_B1_L1024", B=1, L=1024, H=6, D=128, kernel_config=KAGGLE_MEDIUM,
                old_micro_bs=None),
    BenchConfig("medium_B4_L4096", B=4, L=4096, H=6, D=128, kernel_config=KAGGLE_MEDIUM,
                old_micro_bs=None),
    BenchConfig("train_shape_B8_L4096", B=8, L=4096, H=6, D=128, kernel_config=KAGGLE_MEDIUM,
                old_micro_bs=2),
    BenchConfig("kaggle_small_preset_B4_L2048", B=4, L=2048, H=6, D=128, kernel_config=KAGGLE_SMALL,
                old_micro_bs=None),
    BenchConfig("kaggle_large_preset_B8_L4096", B=8, L=4096, H=6, D=128, kernel_config=KAGGLE_LARGE,
                old_micro_bs=2),
]

DTYPES = [jnp.float32, jnp.bfloat16]

REPEATS = 2


@dataclass
class AggStage:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    compile_ms: float
    micro_bs: Optional[int]
    n_repeats: int

    @property
    def cv(self):
        return (self.std_ms / self.mean_ms) if self.mean_ms > 0 else float("nan")


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else float("nan")


def aggregate_repeats(all_repeat_results, dtype_name):
    agg = {}
    n_configs = len(CONFIGS)
    for ci in range(n_configs):
        cfg_name = CONFIGS[ci].name
        per_path = {}
        for path_name in ("OLD", "JAX_REF", "PALLAS"):
            per_stage = {}
            for stage_name in ("fwd", "bwd", "fwdbwd"):
                means, stds, mins, maxs, compiles, mbs = [], [], [], [], [], None
                for repeat_results in all_repeat_results:
                    r = repeat_results[ci] if ci < len(repeat_results) else None
                    if r is None or r.dtype != dtype_name or path_name not in r.paths:
                        continue
                    pr = r.paths[path_name]
                    st = getattr(pr, stage_name)
                    means.append(st.mean)
                    stds.append(st.std)
                    mins.append(st.min)
                    maxs.append(st.max)
                    compiles.append(st.compile_s)
                    mbs = st.micro_bs
                if not means:
                    continue
                per_stage[stage_name] = AggStage(
                    mean_ms=_avg(means) * 1000,
                    std_ms=_avg(stds) * 1000,
                    min_ms=min(mins) * 1000,
                    max_ms=max(maxs) * 1000,
                    compile_ms=_avg(compiles) * 1000,
                    micro_bs=mbs,
                    n_repeats=len(means),
                )
            if per_stage:
                per_path[path_name] = per_stage
        if per_path:
            agg[cfg_name] = per_path
    return agg


def print_final_speed_table(agg, dtype_name):
    """Prints ONE table for this dtype: config x path x stage -> mean(ms),
    plus a compact speedup block."""
    print("\n" + "=" * 100)
    print(f"SPEED TABLE [{dtype_name}]  (mean steady-state time; n_rep = repeats contributing)")
    print("=" * 100)
    header = (f"{'config':>28} {'path':>10} {'stage':>8} {'mean(ms)':>10} "
              f"{'std(ms)':>9} {'cv':>6} {'micro_bs':>9} {'n_rep':>6}")
    print(header)
    print("-" * len(header))
    for cfg_name, per_path in agg.items():
        for path_name, per_stage in per_path.items():
            for stage_name in ("fwd", "bwd", "fwdbwd"):
                if stage_name not in per_stage:
                    continue
                s = per_stage[stage_name]
                mb_str = str(s.micro_bs) if s.micro_bs else "-"
                print(f"{cfg_name:>28} {path_name:>10} {stage_name:>8} {s.mean_ms:>10.2f} "
                      f"{s.std_ms:>9.2f} {s.cv:>6.3f} {mb_str:>9} {s.n_repeats:>6}")

    print(f"\n  {'config':>28} {'OLD->REF fwd/bwd/fb':>22} {'REF->PALLAS fwd/bwd/fb':>24} {'OLD->PALLAS fwd/bwd/fb':>24}")
    for cfg_name, per_path in agg.items():
        old, ref, pallas = per_path.get("OLD"), per_path.get("JAX_REF"), per_path.get("PALLAS")
        if not (ref and pallas):
            continue
        def trip(a, b):
            return f"{a['fwd'].mean_ms/b['fwd'].mean_ms:.2f}x/{a['bwd'].mean_ms/b['bwd'].mean_ms:.2f}x/{a['fwdbwd'].mean_ms/b['fwdbwd'].mean_ms:.2f}x"
        old_ref = trip(old, ref) if old else "n/a"
        ref_pallas = trip(ref, pallas)
        old_pallas = trip(old, pallas) if old else "n/a"
        print(f"  {cfg_name:>28} {old_ref:>22} {ref_pallas:>24} {old_pallas:>24}")


def dump_final_json(agg_by_dtype, path, env_info, repeats_requested):
    serializable = {
        "env": env_info,
        "repeats_requested": repeats_requested,
        "n_seeds": N_SEEDS,
        "n_iters_per_seed": N_ITERS_PER_SEED,
        "by_dtype": {},
    }
    for dtype_name, agg in agg_by_dtype.items():
        serializable["by_dtype"][dtype_name] = {
            cfg_name: {
                path_name: {stage: asdict(s) for stage, s in per_stage.items()}
                for path_name, per_stage in per_path.items()
            }
            for cfg_name, per_path in agg.items()
        }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[OUTPUT] Final averaged speed results dumped to {path}")


if __name__ == "__main__":
    env_info = collect_env_info()
    print_env_info(env_info)

    master_key = jax.random.PRNGKey(6)
    all_repeat_results_by_dtype = {"fp32": [], "bf16": []}

    for dtype in DTYPES:
        dtype_name = "bf16" if dtype != jnp.float32 else "fp32"
        print("\n" + "#" * 96)
        print(f"# DTYPE PASS: {dtype_name}")
        print("#" * 96)

        for rep in range(REPEATS):
            print("\n" + "#" * 88)
            print(f"# [{dtype_name}] REPEAT {rep + 1}/{REPEATS}")
            print("#" * 88)
            repeat_key = jax.random.fold_in(master_key, rep)
            results = []
            for i, cfg in enumerate(CONFIGS):
                subkey = jax.random.fold_in(repeat_key, i)
                try:
                    r = bench_one_config(cfg, subkey, dtype)
                    results.append(r)
                except Exception as e:
                    print(f"[FAILED] {cfg.name} [{dtype_name}]: {type(e).__name__}: {e}")
                    results.append(None)
            all_repeat_results_by_dtype[dtype_name].append(results)

    agg_by_dtype = {}
    for dtype_name, all_repeat_results in all_repeat_results_by_dtype.items():
        if not all_repeat_results:
            continue
        agg = aggregate_repeats(all_repeat_results, dtype_name)
        agg_by_dtype[dtype_name] = agg

    if "fp32" in agg_by_dtype:
        print_final_speed_table(agg_by_dtype["fp32"], "fp32")
    if "bf16" in agg_by_dtype:
        print_final_speed_table(agg_by_dtype["bf16"], "bf16")

    dump_final_json(agg_by_dtype, "benchmark_speed_final_averaged.json", env_info, REPEATS)

    print("\n" + "=" * 96)
    print("[SUMMARY] Done. Speed-only benchmark (no memory/HBM measurement).")
    print("  REMINDER: N_SEEDS/N_ITERS_PER_SEED/REPEATS reduced for a quick run -- "
          "widen before treating as final publishable numbers.")
    print("=" * 96)
