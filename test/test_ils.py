"""
test_ils.py — pytest tests for makeILS and applyILS.

No RFM executable required — all tests operate on synthetic data.

Run with:
    pytest test/test_ils.py -v
"""
import numpy as np
import pytest

from rfmwrapper import makeILS, applyILS, ILSOut, Spectra
from rfmwrapper.functions import _APOD_TYPES, loadILSParams

_NB_WEAK_APOD0 = 0.964000   # Norton-Beer weak apod[0] by design (Norton & Beer 1976)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _flat_spectra(n_spectra=3):
    wn = np.linspace(2000.0, 2600.0, 6001)
    return Spectra(
        wn=wn,
        data=np.ones((n_spectra, len(wn))),
        labels=np.array([30.0, 60.0, 90.0][:n_spectra]),
        label_type='elevation',
        output='tra',
    )


def _zero_spectra():
    wn = np.linspace(2000.0, 2600.0, 6001)
    return Spectra(
        wn=wn,
        data=np.zeros((2, len(wn))),
        labels=np.array([45.0, 90.0]),
        label_type='elevation',
        output='tra',
    )


def _default_ils():
    return makeILS(fov=0.03, max_opd=1.8, nper_wn=10, wn_center=2300.0, apod_type='NB medium')


# ---------------------------------------------------------------------------
# makeILS
# ---------------------------------------------------------------------------

def test_makeils_required_keys():
    ils = _default_ils()
    for key in ('kernel', 'wn_grid', 'apod', 'L_grid', 'fov_width', 'apod_type',
                'fov', 'wn_center'):
        assert key in ils


def test_makeils_kernel_wngrid_same_length():
    ils = _default_ils()
    assert len(ils['kernel']) == len(ils['wn_grid'])


def test_makeils_metadata_preserved():
    ils = makeILS(fov=0.05, max_opd=2.0, nper_wn=15, wn_center=1500.0, apod_type='Hann')
    assert ils['fov'] == 0.05
    assert ils['wn_center'] == 1500.0
    assert ils['apod_type'] == 'Hann'


def test_makeils_l_grid_bounds():
    ils = makeILS(max_opd=2.5, nper_wn=10)
    assert ils['L_grid'][0] == 0.0
    assert np.isclose(ils['L_grid'][-1], 2.5)


def test_makeils_kernel_normalised():
    ils = _default_ils()
    assert np.isclose(ils['kernel'].sum(), 1.0, atol=1e-6)


def test_makeils_wngrid_symmetric():
    ils = _default_ils()
    wng = ils['wn_grid']
    assert np.isclose(wng[0], -wng[-1], atol=1e-6)


def test_makeils_kernel_trim():
    max_opd = 1.8
    ils = makeILS(max_opd=max_opd, nper_wn=10)
    limit = 5 / max_opd
    assert not np.any(np.abs(ils['wn_grid']) > limit + 1e-9)


def test_makeils_fov_width_formula():
    fov, wn = 0.04, 2000.0
    ils = makeILS(fov=fov, max_opd=1.8, nper_wn=10, wn_center=wn)
    assert np.isclose(ils['fov_width'], wn * (fov / 2) ** 2 / 2)


def test_makeils_zero_fov_normalised():
    ils = makeILS(fov=0.0, max_opd=1.8, nper_wn=10, wn_center=2300.0)
    assert np.isclose(ils['kernel'].sum(), 1.0, atol=1e-6)


def test_makeils_fov_broadening_has_no_subgrid_floor():
    """
    FOV self-apodization is applied analytically as sinc(fov_width * OPD), so
    it acts at any magnitude.  The previous boxcar of ceil(fov_width * nper_wn)
    grid points collapsed to a single point — the identity — whenever
    fov_width fell below one grid spacing, silently discarding the FOV effect.
    """
    grid_spacing = 1.0 / 10
    # fov_width well below one grid spacing at this wavenumber
    ils_zero = makeILS(fov=0.0, max_opd=1.8, nper_wn=10, wn_center=1000.0)
    ils_tiny = makeILS(fov=0.008, max_opd=1.8, nper_wn=10, wn_center=1000.0)
    assert ils_tiny['fov_width'] < grid_spacing
    assert not np.array_equal(ils_zero['kernel'], ils_tiny['kernel'])


def test_makeils_fov_broadens_monotonically():
    """
    Wider FOV must broaden the ILS core.  Measured as equivalent width
    (1 / peak, the kernel having unit area) — a second moment is unusable
    here because it is dominated by sinc ringing, which FOV apodization
    suppresses even as the core broadens.
    """
    widths = [1.0 / makeILS(fov=f, max_opd=1.8, nper_wn=10,
                            wn_center=2300.0)['kernel'].max()
              for f in (0.0, 0.02, 0.04, 0.06)]
    assert all(b > a for a, b in zip(widths, widths[1:])), widths


def test_makeils_apod_excludes_fov_term():
    """
    out['apod'] is the apodization function proper and stays within [0, 1];
    the FOV sinc (which goes negative past its first zero) is applied to a
    working copy only.
    """
    ils = makeILS(fov=0.06, max_opd=1.8, nper_wn=10, wn_center=2300.0)
    assert ils['fov_width'] * ils['L_grid'].max() > 1.0   # past the first zero
    assert ils['apod'].min() >= -1e-9
    assert ils['apod'].max() <= 1.0 + 1e-9


def test_makeils_invalid_apod_raises():
    with pytest.raises(ValueError):
        makeILS(apod_type='NotARealApod')


@pytest.mark.parametrize('apod_type', _APOD_TYPES)
def test_makeils_all_apod_types_normalised(apod_type):
    ils = makeILS(fov=0.03, max_opd=1.8, nper_wn=10, wn_center=2300.0, apod_type=apod_type)
    assert len(ils['kernel']) > 0
    assert np.isclose(ils['kernel'].sum(), 1.0, atol=1e-6)


@pytest.mark.parametrize('apod_type', _APOD_TYPES)
def test_makeils_apod_values_in_range(apod_type):
    ils = makeILS(fov=0.0, max_opd=1.8, nper_wn=10, wn_center=2300.0, apod_type=apod_type)
    assert ils['apod'].min() >= -1e-9
    assert ils['apod'].max() <= 1.0 + 1e-9


@pytest.mark.parametrize('apod_type', _APOD_TYPES)
def test_makeils_apod_starts_at_one(apod_type):
    ils = makeILS(fov=0.0, max_opd=1.8, nper_wn=10, wn_center=2300.0, apod_type=apod_type)
    if apod_type == 'NB weak':
        # Norton & Beer (1976) coefficients sum to 0.964 at OPD=0 by design
        assert np.isclose(ils['apod'][0], _NB_WEAK_APOD0, atol=1e-4)
    else:
        assert np.isclose(ils['apod'][0], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# applyILS
# ---------------------------------------------------------------------------

def test_applyils_returns_ilsout():
    assert isinstance(applyILS(_flat_spectra(), _default_ils()), ILSOut)


def test_applyils_shape_preserved():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    assert out.data.shape == spec.data.shape


def test_applyils_wn_unchanged():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    assert np.array_equal(out.wn, spec.wn)


def test_applyils_metadata_inherited():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    assert np.array_equal(out.labels, spec.labels)
    assert out.label_type == spec.label_type
    assert out.output == spec.output


def test_applyils_ils_stored():
    ils = _default_ils()
    out = applyILS(_flat_spectra(), ils)
    assert out.ils is ils


def test_applyils_single_spectrum():
    spec = _flat_spectra(n_spectra=1)
    out = applyILS(spec, _default_ils())
    assert out.data.shape == (1, len(spec.wn))


def test_applyils_flat_preserved():
    ils = _default_ils()
    out = applyILS(_flat_spectra(), ils)
    half = len(ils['kernel']) // 2
    assert np.abs(out.data[:, half:-half] - 1.0).max() < 1e-6


def test_applyils_zero_stays_zero():
    out = applyILS(_zero_spectra(), _default_ils())
    assert np.abs(out.data).max() < 1e-12


def test_applyils_row_independence():
    spec = _flat_spectra(n_spectra=2)
    spec.data[1, :] = 0.0
    out = applyILS(spec, _default_ils())
    half = len(_default_ils()['kernel']) // 2
    assert out.data[0, half:-half].mean() > 0.9
    assert out.data[1, half:-half].mean() < 1e-6


def test_applyils_labels_copy():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    original = spec.labels[0]
    out.labels[0] = -999.0
    assert spec.labels[0] == original


@pytest.mark.parametrize('apod_type', ['Boxcar', 'Triangular', 'NB medium', 'Hann', 'Blackman-Harris'])
def test_applyils_flat_across_apod_types(apod_type):
    spec = _flat_spectra()
    ils = makeILS(fov=0.03, max_opd=1.8, nper_wn=10, wn_center=2300.0, apod_type=apod_type)
    out = applyILS(spec, ils)
    half = len(ils['kernel']) // 2
    assert np.abs(out.data[:, half:-half] - 1.0).max() < 1e-6


# ---------------------------------------------------------------------------
# IASI FWHM accuracy and resampling
# ---------------------------------------------------------------------------

def _measure_fwhm(wn: np.ndarray, y: np.ndarray) -> float:
    """Measure FWHM of a spectral line by linear interpolation at half-maximum."""
    peak_idx = int(np.argmax(y))
    half_max = y[peak_idx] / 2.0
    above = y >= half_max
    trans = np.diff(above.astype(int))
    rise  = np.where(trans ==  1)[0]
    fall  = np.where(trans == -1)[0]
    i_l = rise[0]
    i_r = fall[-1]
    # Linear interpolation: find wn where y crosses half_max
    wn_left  = np.interp(half_max, [y[i_l],     y[i_l + 1]], [wn[i_l],     wn[i_l + 1]])
    wn_right = np.interp(half_max, [y[i_r + 1], y[i_r]],     [wn[i_r + 1], wn[i_r]])
    return float(wn_right - wn_left)


def test_iasi_fwhm_accuracy():
    """IASI ILS (Gaussian, FWHM=0.5 cm⁻¹) broadens a narrow line to ≈0.5 cm⁻¹."""
    # Fine-resolution input: 100 pts/cm⁻¹ over 100 cm⁻¹ centred at 1050 cm⁻¹
    wn = np.linspace(1000.0, 1100.0, 10001)
    sigma = 0.02   # cm⁻¹ — negligible compared to IASI FWHM (0.5 cm⁻¹)
    data = np.exp(-0.5 * ((wn - 1050.0) / sigma) ** 2)[np.newaxis, :]
    spec = Spectra(wn=wn, data=data, labels=np.array([90.0]),
                   label_type='elevation', output='tra')

    out = applyILS(spec, 'IASI')
    fwhm = _measure_fwhm(out.wn, out.data[0])

    assert abs(fwhm - 0.5) < 0.05, f"IASI FWHM: expected 0.50 cm⁻¹, got {fwhm:.3f} cm⁻¹"


def test_applyils_resample_iasi_spacing():
    """resample=True resamples the ILSOut to IASI's native 0.25 cm⁻¹ grid."""
    wn = np.linspace(1000.0, 1100.0, 10001)
    spec = Spectra(wn=wn, data=np.ones((1, len(wn))),
                   labels=np.array([90.0]), label_type='elevation', output='tra')

    out = applyILS(spec, 'IASI', resample=True)

    expected_spacing = loadILSParams('IASI')['spacing']   # 0.25 cm⁻¹
    actual_spacing   = float(np.mean(np.diff(out.wn)))
    assert abs(actual_spacing - expected_spacing) < 1e-9


def test_applyils_resample_dict_raises():
    """resample=True with a dict ILS (no spacing available) raises ValueError."""
    spec = _flat_spectra()
    with pytest.raises(ValueError, match='resample=True requires a named instrument'):
        applyILS(spec, _default_ils(), resample=True)
