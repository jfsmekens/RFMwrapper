"""
demo_bug30_nad_obs.py — Test for RFM bug #30 (NAD + OBS flags, no atmospheric contribution).

Bug #30 (reported 23APR21):
  Affects : v5.00–v5.10
  Symptom : Combination of NAD and OBS flags has no atmospheric contribution.
  Reason  : Bug in setting vertical start/end levels for radiative transfer integration.
  Fix     : RFM v5.11

If the bug is active, the layer radiative transfer integration is skipped and the
sensor-reaching radiance is simply the surface Planck emission propagated through
the atmosphere with no self-emission from layers:

    RAD_bug = B(T_surf) × TRA

If the atmosphere contributes correctly, the atmospheric layers self-emit and the
sensor radiance is higher than the surface-only term in absorption bands:

    RAD_correct > B(T_surf) × TRA   (wherever atmosphere is warmer than effective BB)

Test design:
  - Nadir, elev=-90 (VZA=0), surf_emiss=1.0, FASMLS (T_sfc=294 K)
  - Spectral range 550–1400 cm⁻¹ — covers CO₂ 15 µm, O₃ 9.6 µm, H₂O continuum
  - Two h_obs values:
      * 800 km  (above TOA — observer outside atmosphere)
      * 10 km   (inside troposphere — observer inside atmosphere, most likely to trigger bug)
  - For each: compute B(T_surf)×TRA and compare to RAD
  - If |RAD - B(T_surf)×TRA| ≈ 0 everywhere  → bug active
  - If RAD > B(T_surf)×TRA in CO₂/O₃ bands → atmosphere contributing correctly

Run with:
    python test/demo_bug30_nad_obs.py
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

WN_START  = 550.0
WN_STOP   = 1400.0
NPERWN    = 5
ATM_TYPE  = 'FASMLS'     # T_sfc = 294 K
SURF_TEMP = 294.0         # match FASMLS surface temperature
H_OBS_VALS = [800.0, 10.0]   # km: above TOA, inside troposphere

print("=" * 65)
print("RFM bug #30 diagnostic  —  NAD + OBS atmospheric contribution")
print("=" * 65)
print(f"  Atmosphere : {ATM_TYPE}    T_sfc = {SURF_TEMP:.0f} K")
print(f"  Spectral   : {WN_START}–{WN_STOP} cm⁻¹  ({NPERWN} pts/cm⁻¹)")
print()

# ---------------------------------------------------------------------------
# Run nadir RAD + TRA for each h_obs
# ---------------------------------------------------------------------------

results = {}

for h_obs in H_OBS_VALS:
    rfm = rw.RFM(
        type='nadir',
        wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
        gas='H2O CO2 O3',
        atm_type=ATM_TYPE,
        elev=-90.0,
        h_obs=h_obs,
        surf_temp=SURF_TEMP,
        surf_emiss=1.0,
        out='RAD TRA',
        silent=True,
    )
    rfm.run(silent=True)
    rad = rfm.getSpectra('rad')
    tra = rfm.getSpectra('tra')
    rfm.cleanup()

    wn  = rad.wn
    rad_arr = rad.data[0]
    tra_arr = tra.data[0]

    # Surface Planck function [mW m⁻² sr⁻¹ (cm⁻¹)⁻¹]
    b_surf = rw.rad(wn, SURF_TEMP)

    # Bug #30 prediction: RAD = B(T_surf) × TRA (no atmospheric self-emission)
    rad_bug = b_surf * tra_arr

    # Residual: RAD - (B × TRA)
    # Positive values mean atmosphere is self-emitting (correct behaviour)
    residual = rad_arr - rad_bug

    max_residual  = np.nanmax(np.abs(residual))
    mean_residual = np.nanmean(residual)

    results[h_obs] = {
        'wn': wn, 'rad': rad_arr, 'tra': tra_arr,
        'b_surf': b_surf, 'rad_bug': rad_bug, 'residual': residual,
    }

    print(f"h_obs = {h_obs:.0f} km:")
    print(f"  max  |RAD − B×TRA| = {max_residual:.4e}  mW m⁻² sr⁻¹ (cm⁻¹)⁻¹")
    print(f"  mean  RAD − B×TRA  = {mean_residual:.4e}  mW m⁻² sr⁻¹ (cm⁻¹)⁻¹")
    if max_residual < 1e-3:
        print("  → FAIL: residual ≈ 0 everywhere — bug #30 likely active")
    else:
        print("  → PASS: atmosphere contributing to RAD — bug #30 not triggered")
    print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True,
                         sharex='col')
fig.suptitle("RFM bug #30 diagnostic: NAD+OBS atmospheric contribution\n"
             f"{ATM_TYPE}  |  T_sfc = {SURF_TEMP:.0f} K  |  surf_emiss = 1.0",
             fontsize=11)

col_labels = [f"h_obs = {h:.0f} km" for h in H_OBS_VALS]

for col, h_obs in enumerate(H_OBS_VALS):
    r   = results[h_obs]
    wn  = r['wn']

    # Row 0 — RAD vs B(T_surf)×TRA
    ax = axes[0, col]
    ax.plot(wn, r['rad'],     color='#2166ac', lw=1.2, label='RAD (RFM output)')
    ax.plot(wn, r['rad_bug'], color='#d4001a', lw=0.9, ls='--',
            label='B(T_surf)×TRA  (bug #30 prediction)')
    ax.set_ylabel('Radiance  [mW m⁻² sr⁻¹ (cm⁻¹)⁻¹]')
    ax.set_title(col_labels[col])
    ax.legend(fontsize=7)

    # Row 1 — TRA
    ax = axes[1, col]
    ax.plot(wn, r['tra'], color='#555555', lw=1.0)
    ax.set_ylabel('TRA')
    ax.set_ylim(-0.05, 1.05)

    # Row 2 — Residual
    ax = axes[2, col]
    ax.plot(wn, r['residual'], color='#762a83', lw=1.0)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('RAD − B×TRA')
    ax.set_xlabel('Wavenumber [cm⁻¹]')
    ax.set_title(f"Residual  (max = {np.nanmax(np.abs(r['residual'])):.3e})")

out = get_output_path('bug30', 'nad_obs_atmospheric_contribution.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Figure saved: {out}")
plt.show()

print("Done.")
