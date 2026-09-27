"""B4 (intra-chunk backward). NO clamp masks on level-1/2 (fix of O-1) and no dr scatter.
Concatenate-based placement only (Mosaic has no scatter-add)."""
import jax.numpy as jnp
from ..precision import exp_nonpos, sanitize


def local_scatter(x, lo, bs, D):
    n = x.shape[0]
    parts = []
    if lo: parts.append(jnp.zeros((lo, D), jnp.float32))
    parts.append(x)
    if bs - lo - n: parts.append(jnp.zeros((bs - lo - n, D), jnp.float32))
    return jnp.concatenate(parts, 0) if len(parts) > 1 else x


def local_place(vals, T, offset):
    return local_scatter(vals, offset, T, vals.shape[-1])


def _diag_lean(q, k, bk, gc, dAqk, dAkk, s0, s1, scale, dc_):
    gd = gc[s0:s1]
    E = jnp.exp(jnp.clip(gd[:, None, :] - gd[None, :, :], -dc_, dc_))
    dMq, dMk = dAqk[s0:s1, s0:s1], dAkk[s0:s1, s0:s1]
    qd, kd, bkd = q[s0:s1], k[s0:s1], bk[s0:s1]
    Ek = E * kd[None, :, :]
    dq_d = scale * jnp.sum(dMq[:, :, None] * Ek, axis=1)
    dbk_d = jnp.sum(dMk[:, :, None] * Ek, axis=1)
    M = dMq[:, :, None] * (scale * qd)[:, None, :] + dMk[:, :, None] * bkd[:, None, :]
    dk_d = jnp.sum(M * E, axis=0)
    return dq_d, dbk_d, dk_d, qd * dq_d + bkd * dbk_d - kd * dk_d


def _diag_btl(q, k, bk, gc, dAqk, dAkk, s0, s1, scale, dot):
    from ..fwd.scores import btl_legs
    ea, ec = btl_legs(gc, s0, s1)
    qt, bkt, kt = q[s0:s1] * ea, bk[s0:s1] * ea, k[s0:s1] * ec
    dMq, dMk = dAqk[s0:s1, s0:s1], dAkk[s0:s1, s0:s1]
    dqt = scale * dot(dMq, kt); dbkt = dot(dMk, kt)
    dkt = scale * dot(dMq.T, qt) + dot(dMk.T, bkt)
    dq_d, dbk_d, dk_d = dqt * ea, dbkt * ea, dkt * ec
    qd, kd, bkd = q[s0:s1], k[s0:s1], bk[s0:s1]
    return dq_d, dbk_d, dk_d, qd * dq_d + bkd * dbk_d - kd * dk_d


def b4_2l_values(q, k, b, gc, dAqk, dAkk, *, scale, cfg, dot):
    """dAqk/dAkk already causal/strict-masked. Returns dq, dk, db, dgc for one chunk."""
    T, bs, bs2, dc_, c_ = cfg.bt, cfg.score_bs, cfg.bs2, cfg.diff_clip, cfg.clip
    idx = jnp.arange(T); bk = b * k; D = q.shape[-1]
    dk_global = jnp.zeros((T, D), jnp.float32); dgc_global = jnp.zeros((T, D), jnp.float32)
    dq_parts, dbk_parts = [], []
    for p in range(T // bs):
        p0, p1 = p * bs, (p + 1) * bs
        z = jnp.zeros((bs, D), jnp.float32)
        dq_loc = dbk_loc = dk_loc = dgc_loc = z
        before = (idx < p0).astype(jnp.float32)
        if p0 > 0:
            r1 = gc[p0]
            a1, _ = exp_nonpos(gc[p0:p1] - r1[None, :]); c1, _ = exp_nonpos(r1[None, :] - gc)
            qt1, bkt1 = q[p0:p1] * a1, bk[p0:p1] * a1
            kt1 = k * c1 * before[:, None]
            dMq1 = dAqk[p0:p1] * before[None, :]; dMk1 = dAkk[p0:p1] * before[None, :]
            dqt1 = scale * dot(dMq1, kt1); dbkt1 = dot(dMk1, kt1)
            dkt1 = scale * dot(dMq1.T, qt1) + dot(dMk1.T, bkt1)
            dq_loc = dq_loc + dqt1 * a1; dbk_loc = dbk_loc + dbkt1 * a1
            dk_global = sanitize(dk_global + dkt1 * c1 * before[:, None], c_)
            dgc_global = sanitize(dgc_global - dkt1 * kt1, c_)          # no mask (O-1)
            dgc_loc = dgc_loc + (dqt1 * qt1 + dbkt1 * bkt1)
        for kk in range(bs // bs2):
            lo, hi = kk * bs2, (kk + 1) * bs2
            q0, q1 = p0 + lo, p0 + hi
            if kk > 0:
                r2 = gc[q0]
                a2, _ = exp_nonpos(gc[q0:q1] - r2[None, :]); c2, _ = exp_nonpos(r2[None, :] - gc[p0:q0])
                qt2, bkt2 = q[q0:q1] * a2, bk[q0:q1] * a2
                kt2 = k[p0:q0] * c2
                dMq2, dMk2 = dAqk[q0:q1, p0:q0], dAkk[q0:q1, p0:q0]
                dqt2 = scale * dot(dMq2, kt2); dbkt2 = dot(dMk2, kt2)
                dkt2 = scale * dot(dMq2.T, qt2) + dot(dMk2.T, bkt2)
                dq_loc = dq_loc + local_scatter(dqt2 * a2, lo, bs, D)
                dbk_loc = dbk_loc + local_scatter(dbkt2 * a2, lo, bs, D)
                dk_loc = dk_loc + local_scatter(dkt2 * c2, 0, bs, D)
                dgc_loc = dgc_loc + local_scatter(-(dkt2 * kt2), 0, bs, D)
                dgc_loc = dgc_loc + local_scatter(dqt2 * qt2 + dbkt2 * bkt2, lo, bs, D)
            if cfg.diag == "btl":
                dq_d, dbk_d, dk_d, dgc_d = _diag_btl(q, k, bk, gc, dAqk, dAkk, q0, q1, scale, dot)
            else:
                dq_d, dbk_d, dk_d, dgc_d = _diag_lean(q, k, bk, gc, dAqk, dAkk, q0, q1, scale, dc_)
            dq_loc = dq_loc + local_scatter(dq_d, lo, bs, D)
            dbk_loc = dbk_loc + local_scatter(dbk_d, lo, bs, D)
            dk_loc = dk_loc + local_scatter(dk_d, lo, bs, D)
            dgc_loc = dgc_loc + local_scatter(dgc_d, lo, bs, D)
        dq_parts.append(dq_loc); dbk_parts.append(dbk_loc)
        dk_global = sanitize(dk_global + local_place(dk_loc, T, p0), c_)
        dgc_global = sanitize(dgc_global + local_place(dgc_loc, T, p0), c_)
    dq_v = sanitize(jnp.concatenate(dq_parts, 0), c_)
    dbk_v = sanitize(jnp.concatenate(dbk_parts, 0), c_)
    return dq_v, sanitize(dk_global + dbk_v * b, c_), sanitize(dbk_v * k, c_), sanitize(dgc_global, c_)
