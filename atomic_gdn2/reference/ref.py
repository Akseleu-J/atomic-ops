"""Pure-JAX / numpy references. forward_ref is autodiff-transparent (gradient reference).
token_serial_ref is the ground-truth recurrence (shares no BLR formula)."""
import jax, jax.numpy as jnp, numpy as np
from ..precision import HIGHEST, make_einsum, exp_nonpos, exp_clipped, sanitize
from ..layout import causal_masks


def to_chunks(t, bsz, nc, H, D, bt):           # (B,L,H,D) -> (B,H,nc,bt,D)
    return jnp.moveaxis(t.reshape(bsz, nc, bt, H, D), (1, 3), (2, 1))

def from_chunks(t, bsz, nc, bt, H, D):
    return jnp.moveaxis(t, (1, 2, 3), (3, 1, 2)).reshape(bsz, nc * bt, H, D)

def chunk_gc(g_r): return jnp.cumsum(g_r.astype(jnp.float32), axis=-2)


def _place(block, p0, p1, T):
    parts = []
    if p0 > 0: parts.append(jnp.zeros(block.shape[:-1] + (p0,), jnp.float32))
    parts.append(block)
    if T - p1 > 0: parts.append(jnp.zeros(block.shape[:-1] + (T - p1,), jnp.float32))
    return jnp.concatenate(parts, -1) if len(parts) > 1 else block


def blr_scores_ref(q, k, b, gc, scale, cfg):
    """One-level BLR, non-centered clipped diagonal (exact formula, no BTL)."""
    T, bs, dc_ = cfg.bt, cfg.score_bs, cfg.diff_clip
    ein = make_einsum("highest")
    causal, strict = causal_masks(T)
    idx = jnp.arange(T); bk = b * k
    rq, rk = [], []
    for p in range(T // bs):
        p0, p1 = p * bs, (p + 1) * bs
        r = gc[..., p0, :]
        a, _ = exp_nonpos(gc[..., p0:p1, :] - r[..., None, :])
        if p0 > 0:
            c, _ = exp_nonpos(r[..., None, :] - gc)
            kt = k * c; before = (idx < p0).astype(jnp.float32)
            oq = scale * ein("...id,...jd->...ij", q[..., p0:p1, :] * a, kt) * before
            ok = ein("...id,...jd->...ij", bk[..., p0:p1, :] * a, kt) * before
        else:
            oq = jnp.zeros(q[..., p0:p1, :].shape[:-1] + (T,), jnp.float32); ok = oq
        gd = gc[..., p0:p1, :]
        E, _ = exp_clipped(gd[..., :, None, :] - gd[..., None, :, :], -dc_, dc_)
        dq_ = scale * jnp.einsum("...id,...ijd,...jd->...ij", q[..., p0:p1, :], E, k[..., p0:p1, :], precision=HIGHEST)
        dk_ = jnp.einsum("...id,...ijd,...jd->...ij", bk[..., p0:p1, :], E, k[..., p0:p1, :], precision=HIGHEST)
        rq.append(oq + _place(dq_, p0, p1, T)); rk.append(ok + _place(dk_, p0, p1, T))
    return (sanitize(jnp.concatenate(rq, -2) * causal, cfg.clip),
            sanitize(jnp.concatenate(rk, -2) * strict, cfg.clip))


def _micro_inv(T_mb, mb):
    idx = jnp.arange(mb)
    def body(i, A):
        oh = (idx == i).astype(jnp.float32)
        t_row = jnp.sum(T_mb * oh[:, None], axis=-2)
        contrib = jnp.sum(t_row[..., :, None] * A, axis=-2)
        col = oh[:, None]
        return A * (1.0 - col) + col * (oh - contrib)[..., None, :]
    return jax.lax.fori_loop(0, mb, body, jnp.zeros(T_mb.shape, jnp.float32))


def ladder_inverse(S, eps, C, base, mode="highest"):
    ein = make_einsum(mode); nb = C // base
    Se = S * (1.0 - eps)
    X = [_micro_inv(Se[..., m*base:(m+1)*base, m*base:(m+1)*base], base) for m in range(nb)]
    b = base
    while b < C:
        n2, new = 2 * b, []
        for m in range(C // n2):
            top, bot = X[2*m], X[2*m+1]; i0 = m * n2
            mid = ein("...ij,...jk->...ik", Se[..., i0+b:i0+n2, i0:i0+b], top)
            ll = -ein("...ij,...jk->...ik", bot, mid); z = jnp.zeros_like(ll)
            new.append(jnp.concatenate([jnp.concatenate([top, z], -1), jnp.concatenate([ll, bot], -1)], -2))
        X, b = new, n2
    return X[0]


def recompute_wy_ref(q, k, v, w, b, gc, A, cfg):
    ein = make_einsum("highest")
    egc, _ = exp_nonpos(gc)
    wp = ein("...ij,...jd->...id", A, b * k * egc)
    u = ein("...ij,...jd->...id", A, w * v)
    gl = gc[..., -1, :]
    ekg, _ = exp_nonpos(gl[..., None, :] - gc)
    c = cfg.clip
    return sanitize(wp, c), sanitize(u, c), sanitize(k * ekg, c), sanitize(q * egc, c), gl


def inter_chunk_scan_ref(Aqk, wp, u, kg, qg, gc_last, scale, h0, cfg):
    ein = make_einsum("highest")
    xs = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, wp, u, kg, qg, gc_last)); c = cfg.clip
    def step(h, x):
        Aq, wp_, u_, kg_, qg_, gl = x
        v_new = u_ - ein("...id,...dv->...iv", wp_, h)
        o = scale * ein("...id,...dv->...iv", qg_, h) + ein("...ij,...jv->...iv", Aq, v_new)
        dec, _ = exp_nonpos(gl)
        h_new = sanitize(h * dec[..., :, None] + ein("...id,...iv->...dv", kg_, v_new), c)
        return h_new, (sanitize(o, c), h, v_new)
    hf, (o, hp, vn) = jax.lax.scan(step, h0, xs)
    return jnp.moveaxis(o, 0, 2), hf, jnp.moveaxis(hp, 0, 2), jnp.moveaxis(vn, 0, 2)


def forward_ref(q, k, v, w, b, g, scale, h0, cfg):
    """(B,L,H,D) inputs. Autodiff-transparent."""
    bsz, L, H, D = q.shape; nc = L // cfg.bt
    qr, kr, vr, wr, br, gr = (to_chunks(t, bsz, nc, H, D, cfg.bt) for t in (q, k, v, w, b, g))
    gc = chunk_gc(gr)
    Aqk, Akk = blr_scores_ref(qr, kr, br, gc, scale, cfg)
    A = sanitize(ladder_inverse(Akk, cfg.wy_eps, cfg.bt, cfg.mb, "highest"), cfg.clip)
    wp, u, kg, qg, gl = recompute_wy_ref(qr, kr, vr, wr, br, gc, A, cfg)
    o, hf, _, _ = inter_chunk_scan_ref(Aqk, wp, u, kg, qg, gl, scale, h0, cfg)
    return from_chunks(o, bsz, nc, cfg.bt, H, D), hf


def token_serial_ref(q, k, v, g, b, w, scale, h0=None):
    """Ground truth token recurrence. (B,L,H,D)."""
    bsz, L, H, D = q.shape
    dt = jnp.result_type(q.dtype, jnp.float32)
    if h0 is None: h0 = jnp.zeros((bsz, H, D, D), dt)
    alpha = jnp.exp(jnp.minimum(g.astype(dt), 0.0))
    def step(h, xs):
        q_t, k_t, v_t, a_t, b_t, w_t = xs
        h = h * a_t[..., :, None]
        erase = jnp.einsum("bhd,bhdv->bhv", (b_t * k_t).astype(dt), h, precision=HIGHEST)
        v_new = (w_t * v_t).astype(dt) - erase
        h = h + jnp.einsum("bhd,bhv->bhdv", k_t.astype(dt), v_new, precision=HIGHEST)
        return h, jnp.einsum("bhdv,bhd->bhv", h, (q_t * scale).astype(dt), precision=HIGHEST)
    hf, o = jax.lax.scan(step, h0, tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, alpha, b, w)))
    return jnp.moveaxis(o, 0, 1), hf


def b4_exact_f64_np(q, k, b, gc, dAq, dAk, scale, clip=1e4, dc=20.0, dblk=8):
    """Exact non-factorized B4 in float64 numpy (single chunk (T,D))."""
    f = lambda x: np.asarray(jax.device_get(x), np.float64)
    q, k, b, gc, dAq, dAk = map(f, (q, k, b, gc, dAq, dAk))
    T, D = q.shape; i = np.arange(T)
    dAq = dAq * (i[:, None] >= i[None, :]); dAk = dAk * (i[:, None] > i[None, :]); bk = b * k
    dq = np.zeros((T, D)); dbk = np.zeros((T, D)); dk = np.zeros((T, D)); dgc = np.zeros((T, D))
    for d0 in range(0, D, dblk):
        d1 = min(d0 + dblk, D)
        diff = gc[:, None, d0:d1] - gc[None, :, d0:d1]
        cm = ((diff >= -dc) & (diff <= dc)).astype(np.float64); E = np.exp(np.clip(diff, -dc, dc))
        kd = k[None, :, d0:d1]
        dq[:, d0:d1] = scale * np.sum(dAq[:, :, None] * E * kd, axis=1)
        dbk[:, d0:d1] = np.sum(dAk[:, :, None] * E * kd, axis=1)
        dk[:, d0:d1] = (scale * np.sum(dAq[:, :, None] * E * q[:, None, d0:d1], axis=0)
                        + np.sum(dAk[:, :, None] * E * bk[:, None, d0:d1], axis=0))
        wgt = ((dAq[:, :, None] * (scale * q)[:, None, d0:d1] + dAk[:, :, None] * bk[:, None, d0:d1])
               * k[None, :, d0:d1] * E * cm)
        dgc[:, d0:d1] = np.sum(wgt, axis=1) - np.sum(wgt, axis=0)
    san = lambda x: np.nan_to_num(np.clip(x, -clip, clip), nan=0.0, posinf=clip, neginf=-clip)
    return san(dq), san(dk + dbk * b), san(dbk * k), san(dgc)
