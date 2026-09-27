"""Single source of truth for kernel configuration."""
from __future__ import annotations
import dataclasses as dc
import math
import jax

MB = 1024 * 1024
DOT_MODES = ("highest", "bf16x3", "default")
DIAGS = ("btl", "lean")
BTL_HARD_LIMIT = 88.0     # exp overflow of one BTL leg (f32)
BTL_WARN = 60.0
BTL_STOP = 80.0


def _default_interpret() -> bool:
    return jax.default_backend() != "tpu"


def effective_group(n_chunks: int, requested):
    if requested is None or requested >= n_chunks:
        return n_chunks
    g = int(requested)
    while g > 1 and n_chunks % g:
        g -= 1
    return max(1, g)


@dc.dataclass(frozen=True)
class BLRConfig:
    # geometry
    bt: int = 128
    bc: int = 64            # bt == 2*bc (kept for reference ladder API)
    mb: int = 8             # ladder leaf size
    score_bs: int = 128     # outer BLR block
    bs2: int = 32           # inner BLR block (BTL diagonal size!). half_span = bs2/2*|g|
    diag: str = "btl"       # 'btl' (MXU) | 'lean' (exact non-centered, slower)
    diff_clip: float = 20.0
    # numerics
    clip: float = 1e4
    wy_eps: float = 1e-3
    dot_mode: str = "highest"        # scores / cd / B5 / wp recompute
    solve_dot_mode: str = "bf16x3"   # ladder
    b3_dot_mode: str = "bf16x3"      # B1/B2/B3
    # execution
    group: int | None = None         # None -> all chunks
    max_chunks_per_call: int = 32    # longer sequences are segmented (h carried)
    interpret: bool = dc.field(default_factory=_default_interpret)
    vmem_mb: float = 150.0
    # domain
    g_max: float = 160.0 / 32        # clamp on |g| per token; must satisfy g_max*bs2/2 < 88

    def __post_init__(self):
        if self.bt != 2 * self.bc:
            raise ValueError(f"bt={self.bt} must equal 2*bc")
        if self.bt % self.score_bs or self.score_bs % self.bs2:
            raise ValueError("need bt % score_bs == 0 and score_bs % bs2 == 0")
        if self.mb & (self.mb - 1) or self.bt % self.mb or (self.bt // self.mb) & (self.bt // self.mb - 1):
            raise ValueError("mb and bt/mb must be powers of two dividing bt")
        if not 0.0 <= self.wy_eps < 1.0:
            raise ValueError("wy_eps in [0,1)")
        for m in (self.dot_mode, self.solve_dot_mode, self.b3_dot_mode):
            if m not in DOT_MODES:
                raise ValueError(f"dot mode {m} not in {DOT_MODES}")
        if self.diag not in DIAGS:
            raise ValueError(f"diag must be in {DIAGS}")

    @property
    def n_sub(self): return self.bt // self.score_bs
    @property
    def n_inner(self): return self.score_bs // self.bs2
    @property
    def btl_half_span_limit(self): return self.g_max * self.bs2 / 2.0
    def with_(self, **kw): return dc.replace(self, **kw)


def make_cfg(**kw) -> BLRConfig:
    """PROD preset: bt=128, sbs=128, bs2=32, mb=8, btl, bf16x3 (B3), g_max tied to bs2."""
    bs2 = kw.get("bs2", 32)
    kw.setdefault("g_max", 160.0 / bs2)
    return BLRConfig(**kw)


PROD = make_cfg
