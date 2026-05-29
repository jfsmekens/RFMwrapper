"""
test_ils.py — Standalone tests for makeILS and applyILS.

No RFM executable required — all tests operate on synthetic data.

Usage:
    python test/test_ils.py
"""
import numpy as np

from rfmwrapper import makeILS, applyILS, ILSOut, Spectra
from rfmwrapper.functions import _APOD_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass(msg):
    print(f"  ✓  {msg}")

def _fail(msg):
    print(f"  ✗  {msg}")
    raise AssertionError(msg)

def _check(condition, msg):
    if condition:
        _pass(msg)
    else:
        _fail(msg)


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
    return makeILS(fov=0.03, max_opd=1.8, nper_wn=10, wn=2300.0, apod_type='NB medium')


# ---------------------------------------------------------------------------
# makeILS tests
# ---------------------------------------------------------------------------

def test_makeils_required_keys():
    ils = _default_ils()
    for key in ('kernel', 'wn_grid', 'apod', 'L_grid', 'fov_width', 'apod_type', 'fov', 'wn'):
        _check(key in ils, f"key '{key}' present in output")

def test_makeils_kernel_wngrid_same_length():
    ils = _default_ils()
    _check(len(ils['kernel']) == len(ils['wn_grid']),
           f"kernel and wn_grid same length ({len(ils['kernel'])})")

def test_makeils_metadata_preserved():
    ils = makeILS(fov=0.05, max_opd=2.0, nper_wn=15, wn=1500.0, apod_type='Hann')
    _check(ils['fov'] == 0.05,    "fov stored correctly")
    _check(ils['wn'] == 1500.0,   "wn stored correctly")
    _check(ils['apod_type'] == 'Hann', "apod_type stored correctly")

def test_makeils_l_grid_bounds():
    ils = makeILS(max_opd=2.5, nper_wn=10)
    _check(ils['L_grid'][0] == 0.0, "L_grid starts at 0")
    _check(np.isclose(ils['L_grid'][-1], 2.5), "L_grid ends at max_opd")

def test_makeils_kernel_normalised():
    ils = _default_ils()
    s = ils['kernel'].sum()
    _check(np.isclose(s, 1.0, atol=1e-6), f"kernel sums to 1.0 (got {s:.8f})")

def test_makeils_wngrid_symmetric():
    ils = _default_ils()
    wng = ils['wn_grid']
    _check(np.isclose(wng[0], -wng[-1], atol=1e-6),
           f"wn_grid symmetric about 0 (min={wng[0]:.4f}, max={wng[-1]:.4f})")

def test_makeils_kernel_trim():
    max_opd = 1.8
    ils = makeILS(max_opd=max_opd, nper_wn=10)
    limit = 5 / max_opd
    exceeded = np.any(np.abs(ils['wn_grid']) > limit + 1e-9)
    _check(not exceeded, f"all |wn_grid| ≤ 5/max_opd ({limit:.3f} cm⁻¹)")

def test_makeils_fov_width_formula():
    fov, wn = 0.04, 2000.0
    ils = makeILS(fov=fov, max_opd=1.8, nper_wn=10, wn=wn)
    expected = wn * (fov / 2) ** 2 / 2
    _check(np.isclose(ils['fov_width'], expected),
           f"fov_width formula correct ({ils['fov_width']:.6f} vs {expected:.6f})")

def test_makeils_zero_fov_normalised():
    ils = makeILS(fov=0.0, max_opd=1.8, nper_wn=10, wn=2300.0)
    s = ils['kernel'].sum()
    _check(np.isclose(s, 1.0, atol=1e-6), f"zero-FOV kernel sums to 1.0 (got {s:.8f})")

def test_makeils_invalid_apod_raises():
    raised = False
    try:
        makeILS(apod_type='NotARealApod')
    except ValueError:
        raised = True
    _check(raised, "ValueError raised for invalid apod_type")

def test_makeils_all_apod_types():
    for apod_type in _APOD_TYPES:
        ils = makeILS(fov=0.03, max_opd=1.8, nper_wn=10, wn=2300.0, apod_type=apod_type)
        s = ils['kernel'].sum()
        _check(len(ils['kernel']) > 0,        f"[{apod_type}] kernel non-empty")
        _check(np.isclose(s, 1.0, atol=1e-6), f"[{apod_type}] kernel sums to 1.0 (got {s:.8f})")

def test_makeils_apod_values_in_range():
    for apod_type in _APOD_TYPES:
        ils = makeILS(fov=0.0, max_opd=1.8, nper_wn=10, wn=2300.0, apod_type=apod_type)
        apod = ils['apod']
        _check(apod.min() >= -1e-9,    f"[{apod_type}] apod ≥ 0 (min={apod.min():.6f})")
        _check(apod.max() <= 1.0+1e-9, f"[{apod_type}] apod ≤ 1 (max={apod.max():.6f})")

def test_makeils_apod_starts_at_one():
    # Norton-Beer weak is the sole exception: its published coefficients
    # (Norton & Beer 1976) sum to 0.964 at OPD=0, not 1.0.  All other
    # apodization functions are expected to be unity at the centerburst.
    _NB_WEAK_APOD0 = 0.964000
    for apod_type in _APOD_TYPES:
        ils = makeILS(fov=0.0, max_opd=1.8, nper_wn=10, wn=2300.0, apod_type=apod_type)
        if apod_type == 'NB weak':
            _check(np.isclose(ils['apod'][0], _NB_WEAK_APOD0, atol=1e-4),
                   f"[{apod_type}] apod[0] = {_NB_WEAK_APOD0} by design (got {ils['apod'][0]:.6f})")
        else:
            _check(np.isclose(ils['apod'][0], 1.0, atol=1e-6),
                   f"[{apod_type}] apod[0] = 1.0 (got {ils['apod'][0]:.6f})")


# ---------------------------------------------------------------------------
# applyILS tests
# ---------------------------------------------------------------------------

def test_applyils_returns_ilsout():
    out = applyILS(_flat_spectra(), _default_ils())
    _check(isinstance(out, ILSOut), "applyILS returns ILSOut instance")

def test_applyils_shape_preserved():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    _check(out.data.shape == spec.data.shape,
           f"output shape {out.data.shape} matches input {spec.data.shape}")

def test_applyils_wn_unchanged():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    _check(np.array_equal(out.wn, spec.wn), "wn axis unchanged")

def test_applyils_metadata_inherited():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    _check(np.array_equal(out.labels, spec.labels), "labels copied correctly")
    _check(out.label_type == spec.label_type, "label_type preserved")
    _check(out.output == spec.output, "output type preserved")

def test_applyils_ils_stored():
    ils = _default_ils()
    out = applyILS(_flat_spectra(), ils)
    _check(out.ils is ils, "ils dict stored on ILSOut")

def test_applyils_single_spectrum():
    spec = _flat_spectra(n_spectra=1)
    out = applyILS(spec, _default_ils())
    _check(out.data.shape == (1, len(spec.wn)),
           f"single-spectrum shape correct: {out.data.shape}")

def test_applyils_flat_preserved():
    ils = _default_ils()
    out = applyILS(_flat_spectra(), ils)
    half = len(ils['kernel']) // 2
    interior = out.data[:, half:-half]
    max_dev = np.abs(interior - 1.0).max()
    _check(max_dev < 1e-6, f"flat spectrum → 1.0 in interior (max deviation {max_dev:.2e})")

def test_applyils_zero_stays_zero():
    out = applyILS(_zero_spectra(), _default_ils())
    max_val = np.abs(out.data).max()
    _check(max_val < 1e-12, f"zero spectrum → 0.0 (max value {max_val:.2e})")

def test_applyils_row_independence():
    spec = _flat_spectra(n_spectra=2)
    spec.data[1, :] = 0.0
    out = applyILS(spec, _default_ils())
    half = len(_default_ils()['kernel']) // 2
    mean0 = out.data[0, half:-half].mean()
    mean1 = out.data[1, half:-half].mean()
    _check(mean0 > 0.9,   f"row 0 (ones) preserved: mean={mean0:.6f}")
    _check(mean1 < 1e-6,  f"row 1 (zeros) preserved: mean={mean1:.2e}")

def test_applyils_labels_copy():
    spec = _flat_spectra()
    out = applyILS(spec, _default_ils())
    original = spec.labels[0]
    out.labels[0] = -999.0
    _check(spec.labels[0] == original, "mutating ILSOut.labels does not affect input Spectra")

def test_applyils_flat_across_apod_types():
    spec = _flat_spectra()
    for apod_type in ['Boxcar', 'Triangular', 'NB medium', 'Hann', 'Blackman-Harris']:
        ils = makeILS(fov=0.03, max_opd=1.8, nper_wn=10, wn=2300.0, apod_type=apod_type)
        out = applyILS(spec, ils)
        half = len(ils['kernel']) // 2
        max_dev = np.abs(out.data[:, half:-half] - 1.0).max()
        _check(max_dev < 1e-6,
               f"[{apod_type}] flat spectrum preserved (max deviation {max_dev:.2e})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    # makeILS
    test_makeils_required_keys,
    test_makeils_kernel_wngrid_same_length,
    test_makeils_metadata_preserved,
    test_makeils_l_grid_bounds,
    test_makeils_kernel_normalised,
    test_makeils_wngrid_symmetric,
    test_makeils_kernel_trim,
    test_makeils_fov_width_formula,
    test_makeils_zero_fov_normalised,
    test_makeils_invalid_apod_raises,
    test_makeils_all_apod_types,
    test_makeils_apod_values_in_range,
    test_makeils_apod_starts_at_one,
    # applyILS
    test_applyils_returns_ilsout,
    test_applyils_shape_preserved,
    test_applyils_wn_unchanged,
    test_applyils_metadata_inherited,
    test_applyils_ils_stored,
    test_applyils_single_spectrum,
    test_applyils_flat_preserved,
    test_applyils_zero_stays_zero,
    test_applyils_row_independence,
    test_applyils_labels_copy,
    test_applyils_flat_across_apod_types,
]


if __name__ == '__main__':
    print(f"\n{'='*55}")
    print("  ILS tests")
    print(f"{'='*55}\n")

    passed, failed = 0, 0
    for test_fn in _TESTS:
        section = 'makeILS' if test_fn.__name__.startswith('test_makeils') else 'applyILS'
        print(f"[{section}] {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
        except Exception as e:
            print(f"  ✗  UNEXPECTED ERROR: {e}")
            failed += 1
        print()

    print(f"{'='*55}")
    print(f"  {passed} passed  |  {failed} failed")
    print(f"{'='*55}\n")
