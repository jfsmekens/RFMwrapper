"""
demo_layer.py — Homogeneous layer: volcanic gas sensing scenario.

Simulates a volcanic plume as a 1 km homogeneous gas slab (T=290 K,
P=900 mb) with SO₂ concentrations spanning four orders of magnitude.
The SO₂ ν₁ band (9.6 µm, ~1062 cm⁻¹) and the weaker ν₃ band
(7.3 µm, ~1360 cm⁻¹) are both covered.

Run with:
    python test/demo_layer.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import rfmwrapper as rw
from rfmwrapper.plotting import plotILSKernel, _add_wl_axis
from demo_config import get_output_path, timer, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START   = 950.0
WN_STOP    = 1150.0
NPERWN     = 10
TEMP       = 290.0       # K
PRES       = 900.0       # mbar
PATHLENGTH = 1.0         # km
CO2_PPM    = 420.0       # background CO₂ [ppmv]
SO2_VALS   = [1, 3, 5, 10]   # ppmv

# ---------------------------------------------------------------------------
# Batch run — one model per SO₂ concentration
# ---------------------------------------------------------------------------

print("Running layer models...")
models = []
for so2 in SO2_VALS:
    m = rw.RFM(
        type='layer',
        wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
        gas='CO2 SO2',
        temp=TEMP, pres=PRES, pathlength=PATHLENGTH,
        vmr={'CO2': CO2_PPM, 'SO2': float(so2)},
        out='TRA OPT',
        silent=True,
    )
    models.append(m)

with timer("RFM runs"):
    for m in models:
        m.run(silent=True)
results = models

# ---------------------------------------------------------------------------
# Collect spectra
# ---------------------------------------------------------------------------

tra_list = [r.getSpectra('tra') for r in results]
opt_list = [r.getSpectra('opt') for r in results]

# Apply IASI ILS to all transmission spectra
for m in results:
    m.applyILS('IASI')
tra_ils_list = [r.getILSOut('tra') for r in results]

wn    = tra_list[0].wn
ilswn = tra_ils_list[0].wn

# Colour by SO₂ concentration (log scale)
norm   = mcolors.LogNorm(vmin=min(SO2_VALS), vmax=max(SO2_VALS))
cmap   = plt.cm.inferno_r
colors = [cmap(norm(v)) for v in SO2_VALS]

# ---------------------------------------------------------------------------
# Figure 1 — raw transmission + IASI-convolved
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                         constrained_layout=True)
fig.suptitle(
    f"SO₂ layer — T={TEMP:.0f} K  P={PRES:.0f} mb  "
    f"thickness={PATHLENGTH * 1e3:.0f} m\n"
    f"950–1150 cm⁻¹ (SO₂ ν₁ 1062 cm⁻¹)"
)

for tra, tra_ils, c, so2 in zip(tra_list, tra_ils_list, colors, SO2_VALS):
    lbl = f"SO₂ = {so2:,} ppmv"
    axes[0].plot(wn, tra.data[0], color=c, lw=0.9, label=lbl)
    axes[1].plot(ilswn, tra_ils.data[0], color=c, lw=0.9)

axes[0].set_ylabel('Transmittance')
axes[0].set_title('Raw RFM')
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(fontsize=8)
_add_wl_axis(axes[0])

axes[1].set_ylabel('Transmittance')
axes[1].set_title('IASI ILS convolved')
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_xlabel(r'Wavenumber [cm$^{-1}$]')

sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.015, pad=0.02)
cbar.set_label('SO₂ VMR [ppmv]')

out = get_output_path('layer', 'so2_transmittance.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# Figure 2 — absorption depth vs SO₂ at band centre
# ---------------------------------------------------------------------------

band_wn   = 1062.0   # cm⁻¹ — SO₂ ν₁ band centre
band_idx  = np.argmin(np.abs(wn - band_wn))
band_tra  = np.array([t.data[0, band_idx] for t in tra_list])

fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax2.plot(SO2_VALS, 1.0 - band_tra, 'o-', color='tab:red')
ax2.set_xscale('log')
ax2.set_xlabel('SO₂ VMR [ppmv]')
ax2.set_ylabel(f'Absorption at {wn[band_idx]:.0f} cm⁻¹')
ax2.set_title('SO₂ band-centre absorption vs concentration')
fig2.suptitle(f"Layer model — T={TEMP:.0f} K  P={PRES:.0f} mb  L={PATHLENGTH:.0f} km")

out2 = get_output_path('layer', 'so2_absorption_curve.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")
plt.show()

for m in results:
    m.cleanup()
print("Done.")
