"""
demo_limb.py — Limb-sounding simulation across the TIR.

Models limb emission at seven tangent heights (5–35 km) for a mid-latitude
summer atmosphere.  Shows how the radiance source shifts from tropospheric
H₂O (low tangent heights) to stratospheric O₃ (high tangent heights).

Run with:
    python test/demo_limb.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import rfmwrapper as rw
from rfmwrapper.plotting import plotLimbMap, _add_wl_axis, plotAtmGrid
from demo_config import get_output_path, timer, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START     = 550.0
WN_STOP      = 1350.0
NPERWN       = 2
ATM_TYPE     = 'FASMLS'
TAN_HEIGHTS  = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]   # km

# ---------------------------------------------------------------------------
# Run RFM
# ---------------------------------------------------------------------------

print("Setting up limb model...")
model = rw.RFM(
    type='limb',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas='CO2 H2O O3',
    atm_type=ATM_TYPE,
    tan_heights=TAN_HEIGHTS,
    out='RAD BBT',
    silent=True,
)

with timer("RFM run"):
    model.run(silent=True)

rad = model.getSpectra('rad')
bbt = model.getSpectra('bbt')

print(f"  Tangent heights: {rad.labels.tolist()} km")
print(f"  Spectral points: {rad.wn.shape[0]}")

# ---------------------------------------------------------------------------
# Figure 1 — individual limb spectra coloured by tangent height
# ---------------------------------------------------------------------------

norm   = mcolors.Normalize(vmin=TAN_HEIGHTS[0], vmax=TAN_HEIGHTS[-1])
cmap   = plt.cm.viridis
colors = [cmap(norm(h)) for h in TAN_HEIGHTS]

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                         constrained_layout=True)
fig.suptitle(f"Limb emission — {ATM_TYPE}  |  gases: CO₂, H₂O, O₃")

for i, (r_row, b_row, c) in enumerate(zip(rad.data, bbt.data, colors)):
    lbl = f"TH = {TAN_HEIGHTS[i]:.0f} km"
    axes[0].plot(rad.wn, r_row, color=c, lw=0.8, label=lbl)
    axes[1].plot(bbt.wn, b_row, color=c, lw=0.8)

axes[0].set_ylabel(r'Radiance [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$]')
axes[0].set_title('Limb Radiance')
axes[0].legend(fontsize=7, ncol=2)
_add_wl_axis(axes[0])

axes[1].set_ylabel('Brightness Temperature [K]')
axes[1].set_title('Brightness Temperature')
axes[1].set_xlabel(r'Wavenumber [cm$^{-1}$]')

sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.015, pad=0.02)
cbar.set_label('Tangent height [km]')

out = get_output_path('limb', 'limb_spectra.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# Figure 2 — 2D radiance map
# ---------------------------------------------------------------------------

try:
    import cmcrameri.cm as cmc
    limb_cmap = 'cmc.lajolla'
except ImportError:
    limb_cmap = 'hot_r'

fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)
fig2.suptitle(f"Limb radiance map — {ATM_TYPE}")

plotLimbMap(rad, ax=ax2, cmap=limb_cmap, title='Radiance [mW m⁻² sr⁻¹ (cm⁻¹)⁻¹]')

out2 = get_output_path('limb', 'limb_map.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")
plt.show()

# ---------------------------------------------------------------------------
# Figure 3 — atmospheric context (TEM + H2O + O3)
# ---------------------------------------------------------------------------

fig3, axes3 = plotAtmGrid(
    atm=model.atm,
    species=['TEM', 'PRE', 'H2O', 'O3'],
    title=f'Atmosphere — {ATM_TYPE}',
    color='k',
)

out3 = get_output_path('limb', 'atmosphere.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"  Saved: {out3}")
plt.show()

model.cleanup()
print("Done.")
