"""
test_filters.py — Tests for filterConvolve, applyFilters, listSensors,
listPlatforms, and _loadFilters.

No RFM executable required — all tests operate on synthetic spectra.

Run with:
    pytest test/test_filters.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rfmwrapper import listSensors, listPlatforms, Spectra
from rfmwrapper.functions import _loadFilters, filterConvolve
from rfmwrapper.objects import applyFilters


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _flat_wn(wn_min, wn_max, n=500, value=100.0):
    """Flat spectrum in wavenumber space [cm⁻¹]."""
    wn = np.linspace(wn_min, wn_max, n)
    return wn, np.ones(n) * value


def _make_spectra(wn_start, wn_stop, n=500, value=100.0, output='rad'):
    """Flat Spectra object in wavenumber space."""
    wn   = np.linspace(wn_start, wn_stop, n)
    data = np.ones((1, n)) * value
    return Spectra(wn=wn, data=data, labels=np.array([0.0]),
                   label_type='elevation', output=output)


@pytest.fixture
def aster_filters():
    return _loadFilters('ASTER')


# ---------------------------------------------------------------------------
# 1. listSensors
# ---------------------------------------------------------------------------

def test_list_sensors_returns_list():
    sensors = listSensors()
    assert isinstance(sensors, list)
    assert len(sensors) > 0


def test_list_sensors_contains_aster():
    assert 'ASTER' in listSensors()


def test_list_sensors_contains_multiplatform():
    sensors = listSensors()
    assert any('/' in s for s in sensors), "Expected at least one multi-platform entry"
    assert 'GOES-ABI/GOES-16' in sensors


# ---------------------------------------------------------------------------
# 2. listPlatforms
# ---------------------------------------------------------------------------

def test_listplatforms_goes_abi():
    platforms = listPlatforms('GOES-ABI')
    assert isinstance(platforms, list)
    assert 'GOES-16' in platforms
    assert 'GOES-17' in platforms
    assert 'GOES-18' in platforms
    assert 'GOES-19' in platforms


def test_listplatforms_avhrr():
    platforms = listPlatforms('AVHRR')
    assert 'METOP-1' in platforms
    assert 'NOAA-19' in platforms


def test_listplatforms_single_platform_raises():
    """ASTER has no sub-directories; listPlatforms must raise."""
    with pytest.raises(ValueError):
        listPlatforms('ASTER')


def test_listplatforms_unknown_raises():
    with pytest.raises(ValueError):
        listPlatforms('NONEXISTENT')


# ---------------------------------------------------------------------------
# 3. _loadFilters
# ---------------------------------------------------------------------------

def test_load_filters_valid():
    filters = _loadFilters('ASTER')
    assert isinstance(filters, dict)
    assert len(filters) > 0
    for name, f in filters.items():
        assert 'wn' in f and 'wl' in f and 'T' in f
        assert len(f['wn']) == len(f['T'])
        assert f['T'].min() >= 0.0
        assert f['T'].max() <= 1.0, f"{name}: transmission > 1"


def test_load_filters_hierarchical():
    filters = _loadFilters('GOES-ABI/GOES-16')
    assert len(filters) == 16
    for name, f in filters.items():
        assert 'wn' in f and 'T' in f


def test_load_filters_unknown_raises():
    with pytest.raises(ValueError):
        _loadFilters('NONEXISTENT')


# ---------------------------------------------------------------------------
# 4. filterConvolve — flat spectrum
# ---------------------------------------------------------------------------

def test_filterconvolve_flat_tir(aster_filters):
    wn, y = _flat_wn(833.0, 1250.0, value=100.0)   # 8–12 µm in wn
    result = filterConvolve(wn, y, 'ASTER')

    tir_bands = ['Band_10', 'Band_11', 'Band_12', 'Band_13', 'Band_14']
    for band in tir_bands:
        assert band in result
        assert abs(result[band]['value'] - 100.0) < 1.0, \
            f"{band}: expected ~100, got {result[band]['value']:.2f}"


def test_filterconvolve_vnir_nan_outside_range():
    """filterConvolve returns NaN for bands fully outside the spectrum range."""
    wn, y = _flat_wn(833.0, 1250.0, value=100.0)   # TIR only
    result = filterConvolve(wn, y, 'ASTER')
    for band in ['Band_1', 'Band_2']:                # VNIR bands > 10 000 cm-1
        assert band in result
        assert np.isnan(result[band]['value']), \
            f"{band}: expected NaN for out-of-range band"


def test_filterconvolve_summary_arrays(aster_filters):
    wn, y = _flat_wn(833.0, 1250.0, value=100.0)
    result = filterConvolve(wn, y, 'ASTER')
    assert 'wn' in result and 'wl' in result and 'spec' in result
    assert len(result['wn']) == len(aster_filters)


# ---------------------------------------------------------------------------
# 5. filterConvolve — gradient spectrum
# ---------------------------------------------------------------------------

def test_filterconvolve_gradient():
    """Band values follow the spectral gradient in wn space."""
    wn, _ = _flat_wn(833.0, 1250.0, n=1000)
    # Decreasing with wavenumber: Band_14 (low wn) > Band_10 (high wn)
    y = 150.0 - (wn - 833.0) / (1250.0 - 833.0) * 100.0

    result = filterConvolve(wn, y, 'ASTER')
    v10 = result['Band_10']['value']
    v14 = result['Band_14']['value']
    assert v10 < v14, \
        f"Band_10 ({v10:.1f}) should be < Band_14 ({v14:.1f}) for spectrum decreasing with wn"
    assert 50 < v10 < 80
    assert 120 < v14 < 150


# ---------------------------------------------------------------------------
# 6. filterConvolve — integration method
# ---------------------------------------------------------------------------

def test_filterconvolve_integ_wider_band_larger():
    """Wider passband gives larger integrated value."""
    wn, y = _flat_wn(833.0, 1250.0, value=1.0)
    result = filterConvolve(wn, y, 'ASTER', method='integ')
    v10 = result['Band_10']['value']
    v13 = result['Band_13']['value']
    assert v13 > v10, \
        f"integ: Band_13 ({v13:.4f}) should > Band_10 ({v10:.4f}) — wider passband"


def test_filterconvolve_integ_same_bands_as_avg():
    wn, y = _flat_wn(833.0, 1250.0, value=1.0)
    _summary = {'wn', 'wl', 'spec', 'filters'}
    avg_bands   = {k for k in filterConvolve(wn, y, 'ASTER', method='avg')
                   if k not in _summary}
    integ_bands = {k for k in filterConvolve(wn, y, 'ASTER', method='integ')
                   if k not in _summary}
    assert avg_bands == integ_bands


def test_filterconvolve_bad_method_raises():
    wn, y = _flat_wn(833.0, 1250.0)
    with pytest.raises(ValueError):
        filterConvolve(wn, y, 'ASTER', method='bad')


# ---------------------------------------------------------------------------
# 7. applyFilters — flat wn spectrum (no Jacobian for any output type)
# ---------------------------------------------------------------------------

def test_applyfilters_rad_shape_and_output():
    s  = _make_spectra(700, 1300, n=1000, value=50.0, output='rad')
    fo = applyFilters(s, 'ASTER')
    assert fo.data.shape == (1, len(fo.band_names))
    assert fo.output == 'rad'
    assert len(fo.band_names) > 0


def test_applyfilters_flat_wn_equal_bands():
    """Flat wn spectrum → equal band values regardless of output type (no Jacobian)."""
    s  = _make_spectra(700, 1300, n=1000, value=50.0, output='rad')
    fo = applyFilters(s, 'ASTER')
    assert np.all(np.isfinite(fo.data))
    assert np.allclose(fo.data, 50.0, atol=2.0), \
        f"Expected ~50 for all bands, got range {fo.data.min():.2f}–{fo.data.max():.2f}"


@pytest.mark.parametrize("output", ['rad', 'tra', 'opt', 'bbt'])
def test_applyfilters_flat_value_preserved(output):
    """No Jacobian applied for any output type: flat spectrum → flat output."""
    s  = _make_spectra(700, 1300, n=500, value=0.7, output=output)
    fo = applyFilters(s, 'ASTER')
    assert fo.output == output
    assert np.allclose(fo.data, 0.7, atol=0.05), \
        f"{output}: expected ~0.7, got {fo.data}"


# ---------------------------------------------------------------------------
# 8. applyFilters — overlap-based band inclusion
# ---------------------------------------------------------------------------

def test_applyfilters_overlap_inclusion():
    """
    Spectrum covers 9.0–10.5 µm (952–1111 cm⁻¹).
    Band_12 (8.82–9.40 µm ≈ 1064–1134 cm⁻¹) and
    Band_13 (10.19–11.21 µm ≈ 891–982 cm⁻¹) partially overlap → IN.
    Band_10 (8.04–8.58 µm ≈ 1166–1244 cm⁻¹) fully above spectrum → OUT.
    Band_14 (10.83–11.78 µm ≈ 849–923 cm⁻¹) fully below spectrum → OUT.
    """
    wn_min = 1e4 / 10.5
    wn_max = 1e4 / 9.0
    s = _make_spectra(wn_min, wn_max, n=300, value=80.0, output='rad')
    fo = applyFilters(s, 'ASTER')

    assert 'Band_12' in fo.band_names, "Band_12 should overlap [9.0–10.5 µm]"
    assert 'Band_13' in fo.band_names, "Band_13 should overlap [9.0–10.5 µm]"
    assert 'Band_10' not in fo.band_names, "Band_10 fully outside [9.0 µm]"
    assert 'Band_14' not in fo.band_names, "Band_14 fully outside [10.5 µm]"


# ---------------------------------------------------------------------------
# 9. applyFilters — no bands in range raises ValueError
# ---------------------------------------------------------------------------

def test_applyfilters_no_overlap_raises():
    s = _make_spectra(25000, 30000, n=100, value=1.0, output='tra')
    with pytest.raises(ValueError, match="No.*bands overlap"):
        applyFilters(s, 'ASTER')


# ---------------------------------------------------------------------------
# 10. applyFilters — batch (n_spectra > 1)
# ---------------------------------------------------------------------------

def test_applyfilters_batch_scaling():
    wn   = np.linspace(700, 1300, 500)
    data = np.vstack([np.ones(500) * v for v in [50.0, 100.0, 150.0]])
    s    = Spectra(wn=wn, data=data, labels=np.array([0.0, 30.0, 60.0]),
                   label_type='elevation', output='rad')
    fo   = applyFilters(s, 'ASTER')

    assert fo.data.shape[0] == 3
    # Flat wn spectra at 1:2:3 ratio → band values preserve ratio exactly.
    assert np.allclose(fo.data[1] / fo.data[0], 2.0, rtol=1e-6)
    assert np.allclose(fo.data[2] / fo.data[0], 3.0, rtol=1e-6)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
