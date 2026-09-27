import math, jax, jax.numpy as jnp
from ..layer import init_layer, layer_fwd, lnorm, dense
from ..pipeline import make_blr_trainable


def init_model(key, d_model, n_heads, d_head, n_layers, vocab=256, expansion=4, g_init=0.05):
    ks = jax.random.split(key, n_layers + 2)
    return {"embed": (jax.random.normal(ks[0], (vocab, d_model)) * 0.02).astype(jnp.float32),
            "layers": [init_layer(ks[i + 1], d_model, n_heads, d_head, expansion, g_init) for i in range(n_layers)],
            "head": dense(ks[-1], d_model, vocab)}


def model_fwd(params, tokens, cfg, n_heads, d_head, d_model_scale=None, stats=False, l2norm_k=True):
    blr = make_blr_trainable(cfg, 1.0 / math.sqrt(d_head))
    x = params["embed"][tokens]; st = []
    for p in params["layers"]:
        r = layer_fwd(x, p, blr, n_heads, d_head, cfg, l2norm_k=l2norm_k, stats=stats)
        x, s = (r if stats else (r, None)); st.append(s)
    logits = lnorm(x) @ params["head"]["w"] + params["head"]["b"]
    return (logits, st) if stats else logits
