import numpy as np

from himadri.features import polarimetry as P


def _stokes(s0, s1, s2, s3):
    arr = np.array([s0, s1, s2, s3], dtype=np.float32).reshape(4, 1, 1)
    return arr


def _val(x):
    return float(np.asarray(x).ravel()[0])


def test_cpr_closed_form():
    # CPR = (S0 - S3)/(S0 + S3); for S3 = -0.2*S0 -> CPR = 1.2/0.8 = 1.5
    st = _stokes(1.0, 0.0, 0.0, -0.2)
    assert abs(_val(P.cpr(st)) - 1.5) < 1e-4


def test_dop_closed_form():
    st = _stokes(1.0, 0.3, 0.4, 0.0)  # |pol| = 0.5
    assert abs(_val(P.dop(st)) - 0.5) < 1e-4


def test_mchi_partition_sums_to_total_polarised():
    st = _stokes(1.0, 0.2, 0.1, -0.3)
    d = P.m_chi_decomposition(st)
    total = _val(d["single"]) + _val(d["double"]) + _val(d["volume"])
    assert abs(total - 1.0) < 1e-3  # single+double+volume == S0


def test_ice_vs_rock_separation(feats):
    # On synthetic data, ice region must show low DOP and volume dominance.
    pass  # covered by test_separation.py with the truth map
