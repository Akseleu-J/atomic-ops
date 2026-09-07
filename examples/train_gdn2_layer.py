"""
Install:
    pip install atomic-ops
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from atomic_ops import gdn2_forward_trainable, get_recommended_config


class GDN2Layer(nn.Module):
    heads: int
    d_head: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, seq_len, d_model) -> project to q,k,v,w,b,g
        d_model = self.heads * self.d_head
        q, k, v, w, b, g = jnp.split(
            nn.Dense(6 * d_model, use_bias=False)(x), 6, axis=-1
        )

        def split_heads(t):
            b_, l_, _ = t.shape
            return t.reshape(b_, l_, self.heads, self.d_head)

        q, k, v, w, b = map(split_heads, (q, k, v, w, b))
        g = -jax.nn.softplus(split_heads(g))  # log-decay: держим g <= 0

        bsz, seq_len = x.shape[0], x.shape[1]
        config = get_recommended_config(bsz, seq_len, self.heads, self.d_head)

        out, _ = gdn2_forward_trainable(q, k, v, w, b, g, scale=self.d_head ** -0.5,
                                         config=config)
        return out.reshape(bsz, seq_len, d_model)


model = GDN2Layer(heads=6, d_head=128)
params = model.init(jax.random.PRNGKey(0), jnp.ones((2, 512, 6 * 128)))
opt = optax.adamw(3e-4)
opt_state = opt.init(params)


def loss_fn(params, x, targets):
    pred = model.apply(params, x)
    return jnp.mean((pred - targets) ** 2)


@jax.jit
def train_step(params, opt_state, x, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, targets)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


x = jnp.ones((2, 512, 6 * 128))
targets = jnp.zeros_like(x)
params, opt_state, loss = train_step(params, opt_state, x, targets)
print("loss:", loss)
