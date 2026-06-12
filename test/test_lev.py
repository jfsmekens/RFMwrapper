"""
test_lev.py — pytest tests for *LEV (intermediate output-level) parsing.

A zenith run with ``levels=[...]`` makes RFM emit one extra spectrum per
requested altitude (filename suffix ``_up<alt×1000>``), all from a single
integration.  These tests check that the wrapper:

  - surfaces those outputs on the ``'altitude'`` axis (one row per level plus
    the observer-altitude column), rather than dropping them or misfiling them
    as Jacobian outputs;
  - preserves the physics: the per-altitude optical depths are cumulative cuts
    of one integration, so layer ODs recovered by differencing sum to the
    observer column exactly;
  - leaves plain (non-LEV) and Jacobian runs unchanged.

Requires the RFM executable; skipped automatically when it is not configured.

Run with:
    pytest test/test_lev.py -v
"""
import os
import tempfile

import numpy as np
import pytest


def _rfm_available() -> bool:
    try:
        from rfmwrapper import config
        return os.path.isfile(config.get_exec_path())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _rfm_available(), reason="RFM executable not configured"
)

_OBS, _L1, _L2 = 0.5, 1.5, 2.5


def _write_std_atm() -> str:
    from rfmwrapper import readAtm, writeAtm, config
    atm = readAtm(os.path.join(config.get_atm_dir(), "std.atm"))
    with tempfile.NamedTemporaryFile(suffix=".atm", delete=False) as tf:
        path = tf.name
    writeAtm(atm, path)
    return path


@pytest.fixture(scope="module")
def lev_run():
    from rfmwrapper import RFM
    atm_path = _write_std_atm()
    try:
        m = RFM(type="zenith", wn_start=1000, wn_stop=1050, nperwn=5,
                gas="H2O CO2", elev=90.0, h_obs=_OBS, levels=[_L1, _L2],
                atm_path=atm_path, out="RAD OPT", silent=True)
        m.run()
        yield m
    finally:
        if os.path.exists(atm_path):
            os.remove(atm_path)


def test_lev_surfaces_altitude_axis(lev_run):
    opt = lev_run.getSpectra("opt")
    assert opt.label_type == "altitude"
    np.testing.assert_allclose(np.sort(opt.labels), [_OBS, _L1, _L2])
    assert opt.data.shape[0] == 3
    # Radiance output is surfaced the same way.
    assert lev_run.getSpectra("rad").label_type == "altitude"


def test_lev_optical_depths_are_cumulative_cuts(lev_run):
    # Each `_up<alt>` output is the integrated column from that altitude to
    # space, so differencing recovers the layer ODs and they must sum exactly
    # to the observer column (one integration → no discretization gap).
    opt = lev_run.getSpectra("opt")
    idx = {round(a, 3): i for i, a in enumerate(opt.labels)}
    OD = opt.data
    OD_prox  = OD[idx[_OBS]] - OD[idx[_L1]]
    OD_slice = OD[idx[_L1]]  - OD[idx[_L2]]
    OD_dist  = OD[idx[_L2]]
    full     = OD[idx[_OBS]]

    rel = np.abs(OD_prox + OD_slice + OD_dist - full).max() / full.max()
    assert rel < 1e-10, f"layer ODs do not sum to the observer column (rel {rel:.1e})"
    # Each layer OD is non-negative (within numerical noise).
    for name, od in (("prox", OD_prox), ("slice", OD_slice), ("dist", OD_dist)):
        assert od.min() > -1e-9, f"{name} OD has a spurious negative ({od.min():.1e})"


def test_plain_run_still_labels_by_elevation():
    # A run without `levels` must be unaffected by the LEV parsing.
    from rfmwrapper import RFM
    atm_path = _write_std_atm()
    try:
        m = RFM(type="zenith", wn_start=1000, wn_stop=1050, nperwn=5, gas="H2O",
                elev=90.0, h_obs=_OBS, atm_path=atm_path, out="OPT", silent=True)
        m.run()
        opt = m.getSpectra("opt")
        assert opt.label_type == "elevation"
        np.testing.assert_allclose(opt.labels, [90.0])
    finally:
        if os.path.exists(atm_path):
            os.remove(atm_path)
