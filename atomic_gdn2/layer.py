"""GDN-2 layer: projections -> BLR kernel -> out proj (+ MLP). Domain enforced by construction:
 g = -min(softplus(g_raw + bias), g_max)   (I-1 and BTL half_span bound)
 beta = sigmoid(.), k optionally L2-normalised  => beta*||k||^2 < 1   (I-2)"""
import math
import jax, jax.numpy as jnp
from .domain import init_g_bias, bk2_stat, half_span_from_g


def dense(key, din, dout, scale=0.02):
    return {"w": (jax.random.normal(key, (din, dout)) * (scale / math.sqrt(din))).astype(jnp.float32),
            "b": jnp.zeros(dout, jnp.float32)}


def lnorm(x, eps=1e-5):
    m = jnp.mean(x, -1, keepdims=True); v = jnp.var(x, -1, keepdims=True)
    return (x - m) * jax.lax.rsqrt(v + eps)


def init_layer(key, d_model, n_heads, d_head, expansion, g_init=0.05):
    ks = jax.random.split(key, 9); dh = n_heads * d_head
    p = {n: dense(ks[i], d_model, dh) for i, n in enumerate("qkvwbg")}
    p["g"]["b"] = jnp.full((dh,), init_g_bias(g_init), jnp.float32)
    p["o"] = dense(ks[6], dh, d_model)
    p["up"] = dense(ks[7], d_model, expansion * d_model)
    p["down"] = dense(ks[8], expansion * d_model, d_model)
    return p


def layer_fwd(x, p, blr_fn, n_heads, d_head, cfg, l2norm_k=True, stats=False):
    B, L, _ = x.shape
    h = lnorm(x)
    proj = lambda n: h @ p[n]["w"] + p[n]["b"]
    q, k, v, w = proj("q"), proj("k"), proj("v"), proj("w")
    hm = lambda t: t.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)
    q, k, v, w = map(hm, (q, k, v, w))
    if l2norm_k:
        k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
        q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6)
    b = jax.nn.sigmoid(hm(proj("b")))
    g = -jnp.minimum(jax.nn.softplus(hm(proj("g"))), cfg.g_max)
    o, _ = blr_fn(q, k, v, w, b, g)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, n_heads * d_head)
    x = x + o @ p["o"]["w"] + p["o"]["b"]
    h2 = jax.nn.gelu(lnorm(x) @ p["up"]["w"] + p["up"]["b"])
    x = x + h2 @ p["down"]["w"] + p["down"]["b"]
    if not stats:
        return x
    bk_max, bk_mean = bk2_stat(k, b)
    return x, dict(half_span=half_span_from_g(g, cfg.bt, cfg.bs2), mean_abs_g=jnp.mean(jnp.abs(g)),
                   bk2_max=bk_max, bk2_mean=bk_mean, q_norm=jnp.mean(jnp.linalg.norm(q, axis=-1)),
                   k_norm=jnp.mean(jnp.linalg.norm(k, axis=-1)), v_norm=jnp.mean(jnp.linalg.norm(v, axis=-1)))
