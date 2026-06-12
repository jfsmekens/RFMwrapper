"""
demo_xsc.py — RFM integration demos for XSC-based species (SF6, aerosol).

Requires:
  - RFM executable  ($RFM_DIR/rfm)
  - Standard atmosphere  ($RFM_DIR/atm/std.atm)
  - $RFM_DIR/xsc/sf6.xsc      (run concatXSC — see test_xsc.py)
  - $RFM_DIR/xsc/aerosol.xsc  (provide externally)

Run with:
    python test/demo_xsc.py
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rfmwrapper import (
    readAtm, adjustAtm, writeAtm,
    __XSCPATH__, __ATMPATH__,
    RFM,
)
from rfmwrapper.functions import _parseXSCHeader


def _pass(msg): print(f"  ✓  {msg}")

def _section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


ATM_PATH = os.path.join(__ATMPATH__, 'std.atm')


# ---------------------------------------------------------------------------
# Demo 1: SF6 nadir run (ν₃ band ~948 cm⁻¹)
# ---------------------------------------------------------------------------

_section("Demo 1: RFM nadir run with SF6 (ν₃ band ~948 cm⁻¹)")

SF6_XSC = os.path.join(__XSCPATH__, 'sf6.xsc')

if not os.path.exists(SF6_XSC):
    print(f"  ⚠  {SF6_XSC} not found — run concatXSC first (test_xsc.py::test_concatxsc_default_output_path)")
elif not os.path.exists(ATM_PATH):
    print(f"  ⚠  {ATM_PATH} not found")
else:
    # SF6 background is ~10 pptv; use 1 ppmv for a visible signal
    atm = adjustAtm(readAtm(ATM_PATH), 'SF6', 1.0)

    m_base = RFM('nadir', 920, 960, 100, gas='H2O CO2',
                 atm_path=ATM_PATH, out='RAD OPT', silent=True)
    m_base.run()
    rad_base = m_base.getSpectra('rad').data[0]
    opt_base = m_base.getSpectra('opt').data[0]

    with tempfile.NamedTemporaryFile(suffix='.atm', mode='w', delete=False) as f:
        writeAtm(atm, f.name)
        atm_sf6_path = f.name

    try:
        m_sf6 = RFM('nadir', 920, 960, 100, gas='H2O CO2 SF6',
                    atm_path=atm_sf6_path, xsc={'SF6': SF6_XSC},
                    out='RAD OPT', silent=True)
        m_sf6.run()
        rad_sf6 = m_sf6.getSpectra('rad').data[0]
        opt_sf6 = m_sf6.getSpectra('opt').data[0]
    finally:
        os.unlink(atm_sf6_path)

    delta_opt = opt_sf6 - opt_base
    peak_wn   = m_sf6.getSpectra('opt').wn[np.argmax(delta_opt)]
    peak_dopt = delta_opt.max()

    assert peak_dopt > 0,        "SF6 should increase optical depth"
    assert 930 < peak_wn < 960,  f"SF6 peak outside expected band: {peak_wn:.1f} cm⁻¹"

    _pass(f"SF6 XSC loaded and used by RFM")
    _pass(f"Peak ΔOptical Depth = {peak_dopt:.4f} at {peak_wn:.2f} cm⁻¹")
    _pass(f"Max ΔRAD = {(rad_sf6 - rad_base).max():.4f} mW/(m²·sr·cm⁻¹)")


# ---------------------------------------------------------------------------
# Demo 2: aerosol nadir run (km⁻¹ units, dimensionless scaling ATM)
# ---------------------------------------------------------------------------

_section("Demo 2: RFM nadir run with aerosol XSC (km⁻¹, scaling ATM)")

AEROSOL_XSC = os.path.join(__XSCPATH__, 'aerosol.xsc')

if not os.path.exists(AEROSOL_XSC):
    print(f"  ⚠  {AEROSOL_XSC} not found — skipping")
elif not os.path.exists(ATM_PATH):
    print(f"  ⚠  {ATM_PATH} not found — skipping")
else:
    with open(AEROSOL_XSC) as f:
        for line in f:
            if not line.startswith('!') and line.strip():
                rec_aero = _parseXSCHeader(line)
                break
    assert rec_aero['name'] == 'aerosol', \
        f"Expected 'aerosol' in XSC header, got {rec_aero['name']!r}"
    _pass(f"aerosol.xsc: name='aerosol', "
          f"wn={rec_aero['wn_start']:.0f}–{rec_aero['wn_end']:.0f} cm⁻¹, "
          f"spacing={rec_aero['spacing']:.1f} cm⁻¹")

    # ATM profile = dimensionless scaling factor (1.0 = reference extinction)
    atm_aero = adjustAtm(readAtm(ATM_PATH), 'AEROSOL', 1.0)
    with tempfile.NamedTemporaryFile(suffix='.atm', mode='w', delete=False) as f:
        writeAtm(atm_aero, f.name)
        atm_aero_path = f.name

    try:
        m_base_aero = RFM('nadir', 800, 1200, 10, gas='H2O CO2',
                          atm_path=ATM_PATH, out='RAD OPT', silent=True)
        m_base_aero.run()
        opt_base = m_base_aero.getSpectra('opt').data[0]

        m_aero = RFM('nadir', 800, 1200, 10, gas='H2O CO2 AEROSOL',
                     atm_path=atm_aero_path, xsc={'AEROSOL': AEROSOL_XSC},
                     out='RAD OPT', silent=True)
        m_aero.run()
        opt_aero = m_aero.getSpectra('opt').data[0]
    finally:
        os.unlink(atm_aero_path)

    delta_opt = opt_aero - opt_base
    assert delta_opt.max() > 0, "Aerosol should increase optical depth"

    wn = m_aero.getSpectra('opt').wn
    _pass(f"Aerosol XSC loaded and applied by RFM")
    _pass(f"Mean ΔOptical Depth across 800–1200 cm⁻¹: {delta_opt.mean():.4f}")
    _pass(f"Max  ΔOptical Depth: {delta_opt.max():.4f} at {wn[np.argmax(delta_opt)]:.1f} cm⁻¹")
    _pass(f"Min  ΔOptical Depth: {delta_opt.min():.4f} at {wn[np.argmin(delta_opt)]:.1f} cm⁻¹")


print(f"\n{'─' * 60}")
print("  Done.")
print('─' * 60)
