"""
test_surf_emiss_satellite.py — Verify surf_emiss / TRA behaviour at true satellite altitude.

Tests whether the RFM v5.10 TRA ≈ (1−ε)·T_atm² finding holds when the observer
is well above the top of atmosphere (h_obs=800 km), ruling out any artefact from
the observer being inside the atmosphere.

Runs three nadir RFM instances:
  - e = 1.0  → TRA should equal one-way T_atm (baseline)
  - e = 0.5  → TRA should ≈ (1−0.5)·T_atm² = 0.5·T_atm²
  - e = 0.1  → TRA should ≈ (1−0.1)·T_atm² = 0.9·T_atm²

Reports per-wavenumber residuals and max absolute error for the formula fit.

Run with:
    python test/test_surf_emiss_satellite.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

import rfmwrapper as rw
from demo_config import get_output_path, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

H_OBS      = 800.0          # true satellite altitude [km] — well above TOA
WN_START   = 800.0          # atmospheric window — minimal absorption variation
WN_STOP    = 1200.0
NPERWN     = 1              # coarse grid — diagnostics only
ATM_TYPE   = 'FASMLS'
SURF_TEMP  = 300.0
EMISS_VALS = [1.0, 0.5, 0.1]

print("=" * 65)
print("surf_emiss / TRA diagnostic  —  satellite geometry (800 km)")
print("=" * 65)
print(f"  Atmosphere : {ATM_TYPE}    T_sfc = {SURF_TEMP:.0f} K")
print(f"  Spectral   : {WN_START}–{WN_STOP} cm⁻¹  ({NPERWN} pts/cm⁻¹)")
print(f"  h_obs      : {H_OBS:.0f} km  (above FASMLS TOA ~120 km)")
print()

# ---------------------------------------------------------------------------
# Run RFM for each emissivity
# ---------------------------------------------------------------------------

spectra = {}

for eps in EMISS_VALS:
    rfm = rw.RFM(
        type='nadir',
        wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
        gas='H2O CO2 O3',
        atm_type=ATM_TYPE,
        elev=-90.0,          # nadir (VZA=0)
        h_obs=H_OBS,
        surf_temp=SURF_TEMP,
        surf_emiss=eps,
        out='RAD TRA',
        silent=True,
    )
    rfm.run(silent=True)
    spectra[eps] = {
        'rad': rfm.getSpectra('rad'),
        'tra': rfm.getSpectra('tra'),
    }
    rfm.cleanup()
    print(f"  ε = {eps:.1f}  ✓")

wn = spectra[1.0]['tra'].wn

# ---------------------------------------------------------------------------
# Extract arrays — single elevation (index 0)
# ---------------------------------------------------------------------------

tra_1   = spectra[1.0]['tra'].data[0]   # T_atm  (one-way)
tra_05  = spectra[0.5]['tra'].data[0]
tra_01  = spectra[0.1]['tra'].data[0]

rad_1   = spectra[1.0]['rad'].data[0]
rad_05  = spectra[0.5]['rad'].data[0]
rad_01  = spectra[0.1]['rad'].data[0]

# ---------------------------------------------------------------------------
# Test: TRA(ε) ≈ (1−ε)·T_atm²
# formula prediction from baseline (ε=1)
# ---------------------------------------------------------------------------

print()
print("─" * 65)
print("Test 1: TRA(ε) ≈ (1−ε)·T_atm²")
print("─" * 65)

for eps, tra_obs in [(0.5, tra_05), (0.1, tra_01)]:
    tra_pred = (1.0 - eps) * tra_1 ** 2
    residual = tra_obs - tra_pred
    max_abs  = np.nanmax(np.abs(residual))
    rel_err  = np.nanmax(np.abs(residual) / (tra_pred + 1e-12))
    print(f"  ε = {eps:.1f}  max |TRA_obs − (1−ε)·T_atm²| = {max_abs:.4e}"
          f"   rel = {rel_err:.4e}")

# ---------------------------------------------------------------------------
# Test: RAD changes between ε=1 and ε<1 are consistent with physics
# ---------------------------------------------------------------------------

print()
print("─" * 65)
print("Test 2: RAD difference ε=1 → ε<1")
print("─" * 65)

for eps, rad_obs in [(0.5, rad_05), (0.1, rad_01)]:
    delta = rad_obs - rad_1
    print(f"  ε = {eps:.1f}  mean ΔRAD = {np.mean(delta):.4e}  "
          f"max |ΔRAD| = {np.nanmax(np.abs(delta)):.4e}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True,
                         sharex=True)
fig.suptitle(f"surf_emiss / TRA diagnostic — satellite geometry\n"
             f"{ATM_TYPE}  |  T_sfc = {SURF_TEMP:.0f} K  |  h_obs = {H_OBS:.0f} km",
             fontsize=11)

colors = {1.0: '#2166ac', 0.5: '#e07800', 0.1: '#d4001a'}
labels = {1.0: 'ε = 1.0  (T_atm)',
          0.5: 'ε = 0.5  (measured)',
          0.1: 'ε = 0.1  (measured)'}

# Panel 1 — TRA
ax = axes[0]
for eps in EMISS_VALS:
    ax.plot(wn, spectra[eps]['tra'].data[0], color=colors[eps],
            lw=1.2, label=labels[eps])
ax.set_ylabel('TRA')
ax.set_title('Transmittance output (TRA)')
ax.legend(fontsize=8)

# Panel 2 — TRA(ε) vs formula prediction (1−ε)·T_atm²
ax = axes[1]
for eps, tra_obs in [(0.5, tra_05), (0.1, tra_01)]:
    tra_pred = (1.0 - eps) * tra_1 ** 2
    ax.plot(wn, tra_obs,  color=colors[eps], lw=1.2, label=f'ε = {eps:.1f}  measured')
    ax.plot(wn, tra_pred, color=colors[eps], lw=0.8, ls='--',
            label=f'ε = {eps:.1f}  (1−ε)·T_atm² predicted')
ax.set_ylabel('TRA')
ax.set_title('TRA vs formula prediction  (1−ε)·T_atm²')
ax.legend(fontsize=7)

# Panel 3 — residuals
ax = axes[2]
for eps, tra_obs in [(0.5, tra_05), (0.1, tra_01)]:
    tra_pred = (1.0 - eps) * tra_1 ** 2
    ax.plot(wn, tra_obs - tra_pred, color=colors[eps], lw=1.0,
            label=f'ε = {eps:.1f}  residual')
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('TRA residual')
ax.set_xlabel('Wavenumber [cm⁻¹]')
ax.set_title('Residual:  TRA_obs − (1−ε)·T_atm²')
ax.legend(fontsize=8)

out = get_output_path('surf_emiss', 'tra_satellite_test.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print()
print(f"Figure saved: {out}")
plt.show()

print()
print("Done.")
