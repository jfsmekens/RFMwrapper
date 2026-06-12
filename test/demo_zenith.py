"""
demo_zenith.py — Ground-based upward-looking observation at multiple elevations.

A ground-based instrument (h_obs=0 km) looks up at several elevation angles.
The longer atmospheric path at low elevations deepens absorption features and
increases emission in atmospheric bands.

Spectral range: 650–1350 cm⁻¹ (TIR, capturing CO₂ at 667 cm⁻¹, O₃ at
1042 cm⁻¹, and the H₂O rotation band throughout).

Run with:
    python test/demo_zenith.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import rfmwrapper as rw
from rfmwrapper.plotting import plotSpectra, _add_wl_axis
from demo_config import get_output_path, timer, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START   = 650.0
WN_STOP    = 1350.0
NPERWN     = 10
ATM_TYPE   = 'FASMLS'
ELEVATIONS = [10.0, 30.0, 60.0, 90.0]   # degrees (90 = zenith)
H_OBS      = 0.0                         # ground-based observer [km]

# ---------------------------------------------------------------------------
# Run RFM
# ---------------------------------------------------------------------------

print("Setting up zenith model...")
model = rw.RFM(
    type='zenith',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas='CO2 H2O O3 N2O CH4',
    atm_type=ATM_TYPE,
    elev=ELEVATIONS,
    h_obs=H_OBS,
    out='RAD BBT TRA',
    silent=True,
)

with timer("RFM run"):
    model.run(silent=True)

rad = model.getSpectra('rad')
bbt = model.getSpectra('bbt')
tra = model.getSpectra('tra')

print(f"  Elevation angles: {rad.labels.tolist()} degrees")

# ---------------------------------------------------------------------------
# Figure: 3-panel overview
# ---------------------------------------------------------------------------

# Colour by elevation angle
norm  = mcolors.Normalize(vmin=min(ELEVATIONS), vmax=max(ELEVATIONS))
cmap  = plt.cm.plasma
colors = [cmap(norm(e)) for e in ELEVATIONS]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                         constrained_layout=True)
fig.suptitle(
    f"Zenith-looking emission — {ATM_TYPE}  |  h_obs={H_OBS:.0f} km\n"
    f"Gases: CO₂, H₂O, O₃, N₂O, CH₄"
)

# Radiance
for i, (row, c) in enumerate(zip(rad.data, colors)):
    lbl = f"elev = {ELEVATIONS[i]:.0f}°"
    axes[0].plot(rad.wn, row, color=c, lw=0.9, label=lbl)
axes[0].set_ylabel(r'Radiance [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$]')
axes[0].set_title('Upwelling Radiance')
axes[0].legend(fontsize=8)
_add_wl_axis(axes[0])

# Brightness temperature
for i, (row, c) in enumerate(zip(bbt.data, colors)):
    axes[1].plot(bbt.wn, row, color=c, lw=0.9)
axes[1].set_ylabel('Brightness Temperature [K]')
axes[1].set_title('Brightness Temperature')

# Transmission
for i, (row, c) in enumerate(zip(tra.data, colors)):
    axes[2].plot(tra.wn, row, color=c, lw=0.9)
axes[2].set_ylabel('Transmittance')
axes[2].set_title('One-Way Transmittance (observer → TOA)')
axes[2].set_ylim(-0.05, 1.05)
axes[2].set_xlabel(r'Wavenumber [cm$^{-1}$]')

# Shared colorbar
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.015, pad=0.02)
cbar.set_label('Elevation angle [°]')

out = get_output_path('zenith', 'spectra.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# Figure: airmass dependence at three diagnostic wavenumbers
# ---------------------------------------------------------------------------

# Pick wavenumbers in the CO2, H2O window, and O3 bands
diag_wns = [667.0, 900.0, 1042.0]   # cm⁻¹
diag_labels = ['CO₂ 667 cm⁻¹', 'Window 900 cm⁻¹', 'O₃ 1042 cm⁻¹']

fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
fig2.suptitle(f"Transmission vs elevation angle — {ATM_TYPE}")

for wn0, lbl in zip(diag_wns, diag_labels):
    idx = np.argmin(np.abs(tra.wn - wn0))
    t_vals = tra.data[:, idx]
    ax2.plot(ELEVATIONS, t_vals, 'o-', label=f"{lbl} ({tra.wn[idx]:.0f} cm⁻¹)")

ax2.set_xlabel('Elevation angle [°]')
ax2.set_ylabel('Transmittance')
ax2.set_xlim(0, 95)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(fontsize=9)

out2 = get_output_path('zenith', 'airmass_dependence.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")
plt.show()

model.cleanup()
print("Done.")
