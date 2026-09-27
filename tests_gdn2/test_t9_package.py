"""T9: clean import, config validation, no monkeypatching, single mega-kernel source of truth."""
import inspect, pytest
import atomic_gdn2
from atomic_gdn2 import make_cfg, BLRConfig
from atomic_gdn2.bwd import b4


def test_import_and_prod_preset():
    c = make_cfg()
    assert (c.bt, c.score_bs, c.bs2, c.mb, c.diag) == (128, 128, 32, 8, "btl")
    assert c.btl_half_span_limit < 88.0


@pytest.mark.parametrize("bs2", [8, 16, 32, 64, 128])
def test_gmax_tied_to_bs2(bs2):
    assert make_cfg(bs2=bs2).btl_half_span_limit < 88.0


def test_bad_config_rejected():
    for kw in (dict(bt=100), dict(bs2=48), dict(mb=6), dict(diag="x"), dict(b3_dot_mode="fp8")):
        with pytest.raises(ValueError):
            BLRConfig(**kw)


def test_b4_has_no_clamp_masks():
    """O-1 regression: masks m_a/m_c must never gate dgc again."""
    src = inspect.getsource(b4.b4_2l_values)
    for banned in ("m_a", "m_c", "dr ="):
        assert banned not in src
