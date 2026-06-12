"""
test_xsc.py — pytest tests for XSC Python-side utilities.

Tests only the Python logic: header parsing, readXSCHeaders, concatXSC.
No RFM executable required.

Run with:
    pytest test/test_xsc.py -v
"""
import os
import glob
import tempfile
import logging
import pytest

from rfmwrapper import readXSCHeaders, concatXSC, __XSCPATH__
from rfmwrapper.functions import _parseXSCHeader

SF6_DIR = '/Users/js4247/Downloads/SF6'

_HDR_NARROW = "                 SF6  925.0012  955.0020   5477 180.00  20.3 2.099E-16 0.01            SF6    air 15"
_HDR_WIDE   = "                 SF6  779.9997 1100.0002 127439  189.2 201.4 1.435E-160.022            SF6    air113"


def _sf6_source_files():
    return sorted(f for f in glob.glob(f'{SF6_DIR}/*.xsc') if 'combined' not in f)


def _wide_files():
    """Wide-band (780–1100 cm⁻¹) SF6 tabulations."""
    df = readXSCHeaders(_sf6_source_files())
    mask = (df['wn_start'] > 770) & (df['wn_start'] < 790) & \
           (df['wn_end']   > 1090) & (df['wn_end']   < 1110)
    return df[mask]['file'].tolist()


# ---------------------------------------------------------------------------
# 1. Header parsing
# ---------------------------------------------------------------------------

def test_parse_narrow_band_header():
    r = _parseXSCHeader(_HDR_NARROW)
    assert r['name'] == 'SF6'
    assert abs(r['wn_start'] - 925.0012) < 1e-3
    assert abs(r['wn_end']   - 955.0020) < 1e-3
    assert r['npts'] == 5477
    assert abs(r['T'] - 180.0) < 0.1
    assert abs(r['p_torr'] - 20.3) < 0.1
    assert abs(r['p_mbar'] - 20.3 * 1.33322) < 0.1


def test_parse_wide_band_header():
    r = _parseXSCHeader(_HDR_WIDE)
    assert r['npts'] == 127439
    assert abs(r['T'] - 189.2) < 0.1
    assert abs(r['spacing'] - (1100.0002 - 779.9997) / (127439 - 1)) < 1e-6


# ---------------------------------------------------------------------------
# 2. readXSCHeaders
# ---------------------------------------------------------------------------

def test_readxscheaders_row_count():
    if not os.path.isdir(SF6_DIR):
        pytest.skip(f"SF6 source directory not found: {SF6_DIR}")
    files = _sf6_source_files()
    df = readXSCHeaders(files)
    assert len(df) == len(files)
    assert {'T', 'p_mbar', 'spacing'} <= set(df.columns)


def test_readxscheaders_all_sf6():
    if not os.path.isdir(SF6_DIR):
        pytest.skip(f"SF6 source directory not found: {SF6_DIR}")
    df = readXSCHeaders(_sf6_source_files())
    assert (df['name'] == 'SF6').all()


def test_readxscheaders_multiple_spectral_ranges():
    if not os.path.isdir(SF6_DIR):
        pytest.skip(f"SF6 source directory not found: {SF6_DIR}")
    df = readXSCHeaders(_sf6_source_files())
    assert df.groupby(['wn_start', 'wn_end']).ngroups > 3


# ---------------------------------------------------------------------------
# 3. concatXSC — explicit output path
# ---------------------------------------------------------------------------

def test_concatxsc_explicit_output():
    if not os.path.isdir(SF6_DIR):
        pytest.skip(f"SF6 source directory not found: {SF6_DIR}")
    wide = _wide_files()
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'sf6_test.xsc')
        df = concatXSC(wide, output=out, verbose=False)
        assert os.path.exists(out)
        assert len(df) == len(wide)
        with open(out) as f:
            assert _parseXSCHeader(f.readline())['name'] == 'SF6'


# ---------------------------------------------------------------------------
# 4. concatXSC — default output path
# ---------------------------------------------------------------------------

def test_concatxsc_default_output_path():
    if not os.path.isdir(SF6_DIR):
        pytest.skip(f"SF6 source directory not found: {SF6_DIR}")
    expected = os.path.join(__XSCPATH__, 'sf6.xsc')
    concatXSC(_wide_files(), verbose=False)
    assert os.path.exists(expected)
    with open(expected) as f:
        assert _parseXSCHeader(f.readline())['name'] == 'SF6'


# ---------------------------------------------------------------------------
# 5. concatXSC — output-in-glob protection
# ---------------------------------------------------------------------------

def test_concatxsc_skips_output_file_in_input_list():
    if not os.path.isdir(SF6_DIR):
        pytest.skip(f"SF6 source directory not found: {SF6_DIR}")
    existing = os.path.join(__XSCPATH__, 'sf6.xsc')
    if not os.path.exists(existing):
        pytest.skip("sf6.xsc not yet written — run test_concatxsc_default_output_path first")
    wide = _wide_files()
    df = concatXSC(wide + [existing], verbose=False)
    assert len(df) == len(wide)


# ---------------------------------------------------------------------------
# 6. concatXSC — mixed spectral range warning
# ---------------------------------------------------------------------------

def test_concatxsc_mixed_range_warning(caplog):
    if not os.path.isdir(SF6_DIR):
        pytest.skip(f"SF6 source directory not found: {SF6_DIR}")
    df_all = readXSCHeaders(_sf6_source_files())
    narrow = df_all[df_all['wn_start'] > 900]['file'].tolist()[:2]
    wide   = df_all[df_all['wn_start'] < 790]['file'].tolist()[:2]
    with tempfile.TemporaryDirectory() as td:
        with caplog.at_level(logging.WARNING, logger='rfmwrapper.functions'):
            concatXSC(narrow + wide, output=os.path.join(td, 'mixed.xsc'), verbose=False)
    assert 'spectral range' in caplog.text.lower()
