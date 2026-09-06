from __future__ import annotations

import gc
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
from functools import partial


# ==========================================================================
# ==== КОНФИГУРАЦИЯ =====
# ==========================================================================
SELECTED_CONFIG_NAMES = [
    "small_B1_L1024",
    "train_shape_B8_L4096",
]
N_SEEDS = 2

SELECTED_DTYPES = ["fp32", "bf16"]
SELECTED_PATHS = ["OLD", "JAX_REF", "PALLAS"]
SELECTED_STAGES = ["fwd", "bwd", "fwdbwd"]

GATE_TOLERANCES = {
    "fp32": dict(fwd_tol=1e-2, grad_rel_tol=5e-2),
    "bf16": dict(fwd_tol=5e-2, grad_rel_tol=1e-1),
}

FORK_JOIN_TIMEOUT_S = 900
FORCE_XLA_PYTHON_CLIENT_PREALLOCATE = None
FORCE_XLA_PYTHON_CLIENT_MEM_FRACTION = None

_OOM_ERROR_SUBSTRINGS = (
    "RESOURCE_EXHAUSTED", "Out of memory", "out of memory",
    "failed to allocate", "Compilation failure", "Attempting to allocate",
)


def _looks_like_oom(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    return any(s in msg for s in _OOM_ERROR_SUBSTRINGS)


_TPU_RACE_ERROR_SUBSTRINGS = (
    "Device or resource busy", "TPU initialization failed",
    "Unable to initialize backend 'tpu'", "Couldn't open iommu group",
)
TPU_RACE_MAX_RETRIES = 3
TPU_RACE_BACKOFF_BASE_S = 2.0


def _looks_like_tpu_init_race(result: dict) -> bool:
    if result.get("ok"):
        return False
    err = str(result.get("error", ""))
    return any(s in err for s in _TPU_RACE_ERROR_SUBSTRINGS)


def _with_tpu_race_retry(fn, *args, label=""):
    last_result = None
    for attempt in range(TPU_RACE_MAX_RETRIES + 1):
        last_result = fn(*args)
        if not _looks_like_tpu_init_race(last_result):
            return last_result
        if attempt < TPU_RACE_MAX_RETRIES:
            wait_s = TPU_RACE_BACKOFF_BASE_S * (2 ** attempt)
            print(f"    [TPU-RACE] {label}: похоже на гонку освобождения TPU-устройства -- "
                  f"жду {wait_s:.0f}с, попытка {attempt + 1}/{TPU_RACE_MAX_RETRIES}...")
            time.sleep(wait_s)
    print(f"    [TPU-RACE] {label}: race не устранилась после {TPU_RACE_MAX_RETRIES} "
          f"повторов -- считаю реальной ошибкой.")
    return last_result


@dataclass(frozen=True)
class RunSpec:
    config: str
    dtype: str
    path: str
    stage: str
    seed: int


def _build_run_grid():
    grid = []
    for config in SELECTED_CONFIG_NAMES:
        for dtype in SELECTED_DTYPES:
            for path in SELECTED_PATHS:
                for stage in SELECTED_STAGES:
                    for seed in range(N_SEEDS):
                        grid.append(RunSpec(config, dtype, path, stage, seed))
    return grid


RUN_GRID = _build_run_grid()

CONFIGS = {
    "small_B1_L1024": dict(B=1, L=1024, H=6, D=128, kernel_config="KAGGLE_MEDIUM"),
    "train_shape_B8_L4096": dict(B=8, L=4096, H=6, D=128, kernel_config="KAGGLE_MEDIUM"),
}


def _set_and_log_memory_env():
    import os
    if FORCE_XLA_PYTHON_CLIENT_PREALLOCATE is not None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = FORCE_XLA_PYTHON_CLIENT_PREALLOCATE
    if FORCE_XLA_PYTHON_CLIENT_MEM_FRACTION is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = FORCE_XLA_PYTHON_CLIENT_MEM_FRACTION
    return dict(
        XLA_PYTHON_CLIENT_PREALLOCATE=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<unset, JAX default=true>"),
        XLA_PYTHON_CLIENT_MEM_FRACTION=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "<unset, JAX default=0.75>"),
    )


def _compile_time_stats_from_jitted(jitted_fn, args):
    compiled = jitted_fn.lower(*args).compile()
    analysis = compiled.memory_analysis()
    temp = int(getattr(analysis, "temp_size_in_bytes", 0) or 0)
    argument = int(getattr(analysis, "argument_size_in_bytes", 0) or 0)
    output = int(getattr(analysis, "output_size_in_bytes", 0) or 0)
    alias = int(getattr(analysis, "alias_size_in_bytes", 0) or 0)
    return dict(peak_est_bytes=temp + argument + output - alias)


def _child_main(run_idx: int, result_queue: "mp.Queue"):
    mem_env = _set_and_log_memory_env()
    import os
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/kaggle/working/.jax_cache")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
    from atomic_ops.configs import KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
    from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
    from atomic_ops.reference import gdn2_chunked_wy_reference

    KERNEL_CONFIGS = {"KAGGLE_SMALL": KAGGLE_SMALL, "KAGGLE_MEDIUM": KAGGLE_MEDIUM, "KAGGLE_LARGE": KAGGLE_LARGE}

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
        _, out_chunks = jax.lax.scan(_chunk_step, (eye_bh, zero_bh), (k_ch, ea_ch, z_ch, alpha_ch, q_ch))
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

    def old_associative_scan_forward(q, k, v, w, b, g, scale, h0, chunk_size, n_heads, d_head):
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
        jax.block_until_ready((q, k, v, g, b, w, h0))  # так генерация входов не попадёт в измеряемый пик
        return q, k, v, w, b, g, h0

    def build_fn(path_name, H, D, kcfg, scale):
        chunk_size = kcfg.bt
        if path_name == "OLD":
            return lambda q, k, v, w, b, g, h0: old_associative_scan_forward(
                q, k, v, w, b, g, scale, h0, chunk_size, H, D)
        if path_name == "JAX_REF":
            return lambda q, k, v, w, b, g, h0: gdn2_chunked_wy_reference(
                q, k, v, g, b, w, scale, chunk_size=chunk_size, h0=h0, wy_eps=kcfg.wy_eps)
        if path_name == "PALLAS":
            return lambda q, k, v, w, b, g, h0: gdn2_pallas_forward_trainable(
                q, k, v, w, b, g, scale, h0=h0, config=kcfg)
        raise ValueError(f"unknown path {path_name}")

    def _loss_fn(fwd_fn):
        def loss(q, k, v, w, b, g, h0):
            o, h_final = fwd_fn(q, k, v, w, b, g, h0)
            return jnp.sum(o.astype(jnp.float32) ** 2) + jnp.sum(h_final.astype(jnp.float32) ** 2)
        return loss

    def _get_memory_stats():
        try:
            return jax.local_devices()[0].memory_stats()
        except Exception:
            return None

    spec = RUN_GRID[run_idx]
    result = dict(run_idx=run_idx, config=spec.config, dtype=spec.dtype, path=spec.path,
                  stage=spec.stage, seed=spec.seed, ok=False, error=None, mem_env=mem_env)
    try:
        cfg = CONFIGS[spec.config]
        kcfg = KERNEL_CONFIGS[cfg["kernel_config"]]
        B, L, H, D = cfg["B"], cfg["L"], cfg["H"], cfg["D"]
        scale = D ** -0.5
        dtype = jnp.bfloat16 if spec.dtype == "bf16" else jnp.float32
        key = jax.random.fold_in(jax.random.PRNGKey(6), spec.seed)

        inputs = make_inputs(key, B, L, H, D, dtype=dtype)
        fwd_fn = build_fn(spec.path, H, D, kcfg, scale)
        loss_fn = _loss_fn(fwd_fn)

        if spec.stage == "fwd":
            call_fn, call_args = jax.jit(fwd_fn), inputs
        elif spec.stage == "bwd":
            def bwd_only(*a):
                _, vjp_fn = jax.vjp(loss_fn, *a)
                return vjp_fn(jnp.array(1.0, dtype=jnp.float32))
            call_fn, call_args = jax.jit(bwd_only), inputs
        elif spec.stage == "fwdbwd":
            call_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6)))
            call_args = inputs
        else:
            raise ValueError(spec.stage)

        try:
            compile_time = _compile_time_stats_from_jitted(call_fn, call_args)
        except Exception as e:
            if _looks_like_oom(e):
                result.update(ok=False, oom=True, error=f"OOM at compile time: {type(e).__name__}: {e}"[:500])
                result_queue.put(result)
                return
            raise

        try:
            warm_out = call_fn(*call_args)
            jax.block_until_ready(warm_out)
            del warm_out
            gc.collect()
        except Exception as e:
            if _looks_like_oom(e):
                result.update(ok=False, oom=True, error=f"OOM at warm-up: {type(e).__name__}: {e}"[:500])
                result_queue.put(result)
                return
            raise

        settle_stats = _get_memory_stats()
        settle_peak = settle_stats.get("peak_bytes_in_use", -1) if settle_stats else -1

        # --- ЕДИНСТВЕННОЕ измеряемое число: honest high-water-mark ---
        try:
            out = call_fn(*call_args)
            jax.block_until_ready(out)
        except Exception as e:
            if _looks_like_oom(e):
                result.update(ok=False, oom=True, error=f"OOM at measured call: {type(e).__name__}: {e}"[:500])
                result_queue.put(result)
                return
            raise

        stats_after = _get_memory_stats()
        peak_bytes_in_use = stats_after.get("peak_bytes_in_use", -1) if stats_after else -1
        bytes_limit = stats_after.get("bytes_limit", -1) if stats_after else -1
        steady_state_confirmed = (
            settle_peak == peak_bytes_in_use if (settle_peak >= 0 and peak_bytes_in_use >= 0) else None
        )
        near_pool_limit = (
            bytes_limit > 0 and peak_bytes_in_use > 0 and (peak_bytes_in_use / bytes_limit) > 0.95
        )

        result.update(
            ok=True,
            oom=False,
            peak_bytes_in_use=peak_bytes_in_use,
            compile_time_peak_est_bytes=compile_time["peak_est_bytes"],
            settle_peak_bytes=settle_peak,
            steady_state_confirmed=steady_state_confirmed,
            bytes_limit=bytes_limit,
            near_pool_limit=near_pool_limit,
            batch_size=B,
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    result_queue.put(result)


def _child_gate_check(config_name: str, dtype_name: str, path_name: str, result_queue: "mp.Queue"):
    _set_and_log_memory_env()
    import jax
    import jax.numpy as jnp
    from atomic_ops.configs import KAGGLE_SMALL, KAGGLE_MEDIUM, KAGGLE_LARGE
    from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
    from atomic_ops.reference import gdn2_chunked_wy_reference

    KERNEL_CONFIGS = {"KAGGLE_SMALL": KAGGLE_SMALL, "KAGGLE_MEDIUM": KAGGLE_MEDIUM, "KAGGLE_LARGE": KAGGLE_LARGE}
    tol = GATE_TOLERANCES[dtype_name]
    result = dict(config=config_name, dtype=dtype_name, path=path_name, ok=False, error=None)
    try:
        cfg = CONFIGS[config_name]
        kcfg = KERNEL_CONFIGS[cfg["kernel_config"]]
        B, L, H, D = cfg["B"], cfg["L"], cfg["H"], cfg["D"]
        scale = D ** -0.5
        dtype = jnp.bfloat16 if dtype_name == "bf16" else jnp.float32
        key = jax.random.fold_in(jax.random.PRNGKey(6), 0)

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

        inputs = make_inputs(key, B, L, H, D, dtype=dtype)

        if path_name == "JAX_REF":
            result.update(ok=True, note="JAX_REF is the reference itself -- no gate needed.")
            result_queue.put(result)
            return
        if path_name == "OLD":
            result.update(ok=True, note="OLD ignores h0 by design -- correctness vs JAX_REF not meaningful.")
            result_queue.put(result)
            return

        ref_fn = lambda q, k, v, w, b, g, h0: gdn2_chunked_wy_reference(
            q, k, v, g, b, w, scale, chunk_size=kcfg.bt, h0=h0, wy_eps=kcfg.wy_eps)
        cand_fn = lambda q, k, v, w, b, g, h0: gdn2_pallas_forward_trainable(
            q, k, v, w, b, g, scale, h0=h0, config=kcfg)

        def _loss(fn):
            def loss(q, k, v, w, b, g, h0):
                o, hf = fn(q, k, v, w, b, g, h0)
                return jnp.sum(o.astype(jnp.float32) ** 2) + jnp.sum(hf.astype(jnp.float32) ** 2)
            return loss

        ref_o, ref_h = ref_fn(*inputs)
        cand_o, cand_h = cand_fn(*inputs)
        fwd_diff = float(jnp.max(jnp.abs(ref_o.astype(jnp.float32) - cand_o.astype(jnp.float32))))
        fwd_ok = fwd_diff < tol["fwd_tol"] and bool(jnp.all(jnp.isfinite(cand_o)))

        grads_ref = jax.grad(_loss(ref_fn), argnums=(0, 1, 2, 3, 4, 5, 6))(*inputs)
        grads_cand = jax.grad(_loss(cand_fn), argnums=(0, 1, 2, 3, 4, 5, 6))(*inputs)
        max_rel = 0.0
        for gr, gcd in zip(grads_ref, grads_cand):
            gr32, gcd32 = gr.astype(jnp.float32), gcd.astype(jnp.float32)
            if not (bool(jnp.all(jnp.isfinite(gr32))) and bool(jnp.all(jnp.isfinite(gcd32)))):
                max_rel = float("inf")
                continue
            denom = float(jnp.max(jnp.abs(gr32))) + 1e-6
            max_rel = max(max_rel, float(jnp.max(jnp.abs(gcd32 - gr32))) / denom)
        grad_ok = max_rel < tol["grad_rel_tol"]

        result.update(ok=True, fwd_correctness_ok=fwd_ok, fwd_correctness_max_abs_diff=fwd_diff,
                      grad_correctness_ok=grad_ok, grad_correctness_max_rel_err=max_rel)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    result_queue.put(result)


def run_gate_check(config_name, dtype_name, path_name):
    ctx = mp.get_context("fork")
    qres = ctx.Queue()
    p = ctx.Process(target=_child_gate_check, args=(config_name, dtype_name, path_name, qres))
    p.start()
    try:
        r = qres.get(timeout=FORK_JOIN_TIMEOUT_S)
    except Exception:
        r = dict(config=config_name, dtype=dtype_name, path=path_name, ok=False,
                  error="gate-check process produced no result")
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        p.join()
    return r


def run_isolated(run_idx: int) -> dict:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child_main, args=(run_idx, q))
    p.start()
    got = None
    try:
        got = q.get(timeout=FORK_JOIN_TIMEOUT_S)
    except Exception:
        pass
    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        p.join()
    if got is None:
        spec = RUN_GRID[run_idx]
        return dict(run_idx=run_idx, config=spec.config, dtype=spec.dtype, path=spec.path,
                    stage=spec.stage, seed=spec.seed, ok=False,
                    error=f"child process produced no result (exitcode={p.exitcode})")
    return got


# ==========================================================================
# Оркестрация + отчёт
# ==========================================================================
def run_all():
    total = len(RUN_GRID)
    print("=" * 96)
    print(f"MEMORY BENCHMARK (FINAL, no microbatching, peak_bytes_in_use primary) -- "
          f"{total} isolated runs")
    print(f"started_utc={datetime.now(timezone.utc).isoformat()}")
    print("=" * 96)

    raw_results, gate_results = [], []
    seen_gate_keys = set()
    for spec in RUN_GRID:
        gkey = (spec.config, spec.dtype, spec.path)
        if gkey in seen_gate_keys:
            continue
        seen_gate_keys.add(gkey)
        print(f"\n[GATE] {spec.config} [{spec.dtype}] {spec.path} ...")
        g = _with_tpu_race_retry(run_gate_check, spec.config, spec.dtype, spec.path,
                                  label=f"gate {spec.config}[{spec.dtype}]{spec.path}")
        gate_results.append(g)
        if g.get("ok"):
            if "grad_correctness_max_rel_err" in g:
                print(f"    grad max_rel_err={g['grad_correctness_max_rel_err']:.4e} "
                      f"{'OK' if g.get('grad_correctness_ok') else 'FAILED'}")
            else:
                print(f"    {g.get('note', 'gate skipped/not applicable')}")
        else:
            print(f"    [GATE ERROR] {g.get('error')}")

    failed_gate_keys = {
        (g["config"], g["dtype"], g["path"]) for g in gate_results
        if (not g.get("ok") or g.get("grad_correctness_ok") is False) and not _looks_like_tpu_init_race(g)
    }

    for run_idx in range(total):
        spec = RUN_GRID[run_idx]
        gkey = (spec.config, spec.dtype, spec.path)
        label = f"[{spec.config}][{spec.dtype}][{spec.path}][{spec.stage}][seed={spec.seed}]"
        print(f"\n[{run_idx + 1}/{total}] >>> {label}")
        if gkey in failed_gate_keys:
            print("    [SKIPPED] failed correctness/gradient gate.")
            raw_results.append(dict(run_idx=run_idx, config=spec.config, dtype=spec.dtype, path=spec.path,
                                     stage=spec.stage, seed=spec.seed, ok=False, oom=False,
                                     error="skipped: failed correctness/gradient gate"))
            continue
        r = _with_tpu_race_retry(run_isolated, run_idx, label=f"run {label}")
        raw_results.append(r)
        if r.get("ok"):
            peak = r.get("peak_bytes_in_use", -1)
            ct = r.get("compile_time_peak_est_bytes", -1)
            steady = r.get("steady_state_confirmed")
            steady_flag = "" if steady is True else (" [STEADY-STATE NOT CONFIRMED]" if steady is False else "")
            limit_flag = " [NEAR POOL LIMIT!]" if r.get("near_pool_limit") else ""
            print(f"    [OK] peak_bytes_in_use={peak/1e6:.3f}MB  batch={r.get('batch_size')}  "
                  f"(diag: compile_time_est={ct/1e6:.3f}MB){steady_flag}{limit_flag}")
        elif r.get("oom"):
            print(f"    [OOM] {r.get('error')}  -- честно публикуется как OOM, без экстраполяции")
        else:
            race_note = " (TPU-init race, retries exhausted)" if _looks_like_tpu_init_race(r) else ""
            print(f"    [FAILED/GATED] {r.get('error')}{race_note}")

    grouped = defaultdict(list)
    oom_groups = set()
    for r in raw_results:
        key = (r["config"], r["dtype"], r["path"], r["stage"])
        if r.get("ok"):
            grouped[key].append(r)
        elif r.get("oom"):
            oom_groups.add(key)

    agg = defaultdict(lambda: defaultdict(dict))
    for (config, dtype, path, stage), runs in grouped.items():
        peaks = [r["peak_bytes_in_use"] for r in runs if r.get("peak_bytes_in_use", -1) >= 0]
        unsteady_any = any(r.get("steady_state_confirmed") is False for r in runs)
        agg[(config, dtype)][path][stage] = dict(
            n_seeds_ok=len(runs),
            peak_mean_bytes=(sum(peaks) / len(peaks)) if peaks else None,
            unsteady_any=unsteady_any,
            oom=False,
            batch_size=runs[0].get("batch_size") if runs else None,
        )
    for (config, dtype, path, stage) in oom_groups:
        if stage not in agg[(config, dtype)].get(path, {}):
            agg[(config, dtype)].setdefault(path, {})[stage] = dict(
                n_seeds_ok=0, peak_mean_bytes=None, unsteady_any=False, oom=True,
                batch_size=CONFIGS[config]["B"],
            )

    gate_summary = {}
    for r in raw_results:
        if r.get("path") == "PALLAS" and "grad_correctness_ok" in r:
            gate_summary.setdefault((r["config"], r["dtype"], r["path"]), r)

    failed = [r for r in raw_results if not r.get("ok") and not r.get("oom")]
    ooms = [r for r in raw_results if r.get("oom")]

    print("\n" + "=" * 96)
    print("FINAL REPORT -- MEMORY BENCHMARK (no microbatching)")
    print("=" * 96)

    print("\n-- gradient correctness gate (vs JAX_REF autodiff) --")
    for (config, dtype, path), r in sorted(gate_summary.items()):
        relerr = r.get("grad_correctness_max_rel_err")
        relerr_str = f"{relerr:.2e}" if relerr is not None else "n/a"
        print(f"  {config:>30} [{dtype}] {path:>8}  max_rel_err={relerr_str:>10}  "
              f"{'OK' if r.get('grad_correctness_ok') else 'FAILED'}")

    if ooms:
        print(f"\n-- honest OOMs ({len(ooms)}), published as OOM, NOT extrapolated --")
        for r in ooms:
            print(f"  {r.get('config')} [{r.get('dtype')}] {r.get('path')} {r.get('stage')} "
                  f"seed={r.get('seed')}: {str(r.get('error'))[:150]}")

    if failed:
        print(f"\n-- other failed/gated-out runs ({len(failed)}) --")
        for r in failed:
            print(f"  {r.get('config')} [{r.get('dtype')}] {r.get('path')} {r.get('stage')} "
                  f"seed={r.get('seed')}: {str(r.get('error'))[:150]}")

    print(f"\n[SUMMARY] {len([r for r in raw_results if r.get('ok')])}/{len(raw_results)} runs succeeded, "
          f"{len(ooms)} honest OOMs, {len(failed)} other failures.")

    for dtype_name in SELECTED_DTYPES:
        print_publish_table({k: v for k, v in agg.items() if k[1] == dtype_name}, dtype_name)

    return raw_results, agg, gate_summary


def print_publish_table(agg, dtype_name):
    print("\n" + "=" * 100)
    print(f"PUBLISH TABLE -- peak HBM (MB, memory_stats().peak_bytes_in_use), {dtype_name}, "
          f"process-isolated (fork), n_seeds={N_SEEDS}, SAME batch size across all paths")
    print("=" * 100)
    print("\n| config (batch) | stage | OLD (MB) | JAX_REF (MB) | PALLAS (MB) |")
    print("|---|---|---|---|---|")
    for (config, dtype), per_path in sorted(agg.items()):
        b = CONFIGS[config]["B"]
        for stage in ("fwd", "bwd", "fwdbwd"):
            def cell(path):
                s = per_path.get(path, {}).get(stage)
                if not s:
                    return "n/a"
                if s.get("oom"):
                    return "OOM"
                if s['peak_mean_bytes'] is None:
                    return "n/a"
                flag = " *" if s.get("unsteady_any") else ""
                return f"{s['peak_mean_bytes']/1e6:.3f}{flag}"
            print(f"| {config} (B={b}) | {stage} | {cell('OLD')} | {cell('JAX_REF')} | {cell('PALLAS')} |")
    print("\n`*` = steady-state not confirmed on at least one seed (post-call peak != post-warmup settle peak).")
    print("`OOM` = ran out of HBM at this exact batch size; NOT extrapolated from a smaller batch.")


if __name__ == "__main__":
    raw_results, agg, gate_summary = run_all()
