"""Training loop with: skip-step on non-finite/spiky grads, no weight-decay on bias/embed,
periodic domain stats (half_span, |g|, beta*|k|^2, norms), canary, checkpoints."""
import math, time, pickle, os
import numpy as np, jax, jax.numpy as jnp, optax
from .model import init_model, model_fwd
from .canary import btl_vs_lean_canary


def _wd_mask(params):
    def f(path, x):
        name = "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        return x.ndim >= 2 and "embed" not in name
    return jax.tree_util.tree_map_with_path(f, params)


def make_optimizer(params, lr, warmup, total, wd=0.1, clip=1.0):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, total, lr * 0.1)
    return optax.chain(optax.clip_by_global_norm(clip),
                       optax.adamw(sched, b1=0.9, b2=0.95, weight_decay=wd, mask=_wd_mask(params)))


def train(cfg, data_train, data_val=None, *, d_model=256, n_heads=4, d_head=64, n_layers=6, bsz=8,
          steps=2000, warmup=200, lr=3e-4, seed=0, log_every=50, val_every=500, ckpt_every=100,
          ckpt_dir="ckpt", spike_factor=5.0, stats_every=50, l2norm_k=True, g_init=0.05):
    from .data import Sampler
    sampler = Sampler(data_train, bsz, seed)
    params = init_model(jax.random.PRNGKey(seed), d_model, n_heads, d_head, n_layers, g_init=g_init)
    opt = make_optimizer(params, lr, warmup, steps)
    ost = opt.init(params)
    os.makedirs(ckpt_dir, exist_ok=True)

    def loss_fn(p, ids):
        lg = model_fwd(p, ids, cfg, n_heads, d_head, l2norm_k=l2norm_k)[:, :-1]
        return optax.softmax_cross_entropy_with_integer_labels(lg, ids[:, 1:]).mean()

    @jax.jit
    def step(p, s, ids, gn_ema):
        loss, gr = jax.value_and_grad(loss_fn)(p, ids)
        gn = optax.global_norm(gr)
        ok = jnp.isfinite(loss) & jnp.isfinite(gn) & ((gn_ema <= 0) | (gn < spike_factor * gn_ema))
        upd, s2 = opt.update(gr, s, p)
        p2 = optax.apply_updates(p, upd)
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, old)
        gn_new = jnp.where(ok, jnp.where(gn_ema > 0, 0.98 * gn_ema + 0.02 * gn, gn), gn_ema)
        return keep(p2, p), keep(s2, s), loss, gn, ok, gn_new

    @jax.jit
    def stats_fn(p, ids):
        _, st = model_fwd(p, ids, cfg, n_heads, d_head, stats=True, l2norm_k=l2norm_k)
        return st

    @jax.jit
    def eval_fn(p, ids): return loss_fn(p, ids)

    gn_ema = jnp.float32(0.0); skipped = 0; hist = []
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        ids = jnp.asarray(sampler.next())
        params, ost, loss, gn, ok, gn_ema = step(params, ost, ids, gn_ema)
        hist.append(float(loss)); skipped += int(not bool(ok))
        if i % stats_every == 0:
            st = stats_fn(params, ids)
            hs = max(float(s["half_span"]) for s in st); bk = max(float(s["bk2_max"]) for s in st)
            print(f"  [domain] half_span_max={hs:.1f} (limit 88)  mean|g|={np.mean([float(s['mean_abs_g']) for s in st]):.3f}"
                  f"  beta|k|2_max={bk:.2f}  skipped={skipped}")
            if hs > 80: print("  [ALARM] half_span > 80")
        if i % log_every == 0:
            dt = (time.perf_counter() - t0) / i
            print(f"{i:6d} loss={hist[-1]:.4f} gnorm={float(gn):.2f} {dt*1e3:.1f} ms/step")
        if val_every and data_val is not None and i % val_every == 0:
            print(f"       [val] {float(eval_fn(params, jnp.asarray(data_val[:bsz]))):.4f}")
        if ckpt_every and i % ckpt_every == 0:
            with open(f"{ckpt_dir}/step{i}.pkl", "wb") as f:
                pickle.dump(jax.device_get(params), f)
    return params, hist
