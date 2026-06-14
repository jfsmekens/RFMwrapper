"""
test_jacobians.py — getJacobians() grouping and label-consistency.

Regression coverage for the multi-output column Jacobian bug: a column
Jacobian (``{param: None}``) run with several output flags writes one Jacobian
file per output quantity (∂RAD/∂T, ∂BBT/∂T, …).  Those are distinct quantities
and must never be stacked together; in every case ``labels`` must have exactly
one entry per data row.

Runs RFM (needs the binary), like test_lev.py.  Kept to tiny spectral windows.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfmwrapper import RFM


def _run(**kw):
    base = dict(type="zenith", wn_start=800, wn_stop=815, nperwn=4,
                gas="H2O CO2", elev=90.0)
    base.update(kw)
    rfm = RFM(**base, silent=True)
    rfm.run(silent=True)
    return rfm


def test_column_jac_multi_output_not_merged():
    # The crash case: one elevation, column TEM Jacobian, four output flags.
    rfm = _run(out="RAD BBT OPT TRA", jacobians={"TEM": None})
    try:
        rad = rfm.getJacobians("TEM")                 # default → rad
        bbt = rfm.getJacobians("TEM", output="bbt")
        # Each output's Jacobian is a single column spectrum, not a 4-row stack.
        assert rad.data.shape[0] == 1 == rad.labels.shape[0]
        assert bbt.data.shape[0] == 1 == bbt.labels.shape[0]
        assert rad.label_type == "column"
        # Distinct physical quantities.
        assert not np.allclose(rad.data, bbt.data)
    finally:
        rfm.cleanup()


def test_jac_labels_match_rows_all_cases():
    # Layer Jacobian → one row per perturbation altitude, label_type 'altitude'.
    rfm = _run(out="RAD", jacobians={"TEM": [0, 5, 10, 20, 40, 60]})
    try:
        jac = rfm.getJacobians("TEM")
        assert jac.data.shape[0] == jac.labels.shape[0] >= 2
        assert jac.label_type == "altitude"
    finally:
        rfm.cleanup()


def test_multi_elevation_column_jac_labelled_by_elevation():
    rfm = _run(elev=[30.0, 60.0, 90.0], out="RAD", jacobians={"TEM": None})
    try:
        jac = rfm.getJacobians("TEM")
        assert jac.data.shape[0] == jac.labels.shape[0] == 3
        assert jac.label_type == "elevation"
        assert np.allclose(np.sort(jac.labels), [30.0, 60.0, 90.0])
    finally:
        rfm.cleanup()


def test_unknown_output_raises():
    rfm = _run(out="RAD", jacobians={"TEM": None})
    try:
        with pytest.raises(KeyError):
            rfm.getJacobians("TEM", output="bbt")   # bbt not in this run
    finally:
        rfm.cleanup()
