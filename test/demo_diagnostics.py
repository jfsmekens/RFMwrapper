"""
demo_diagnostics.py — Ray-path diagnostic visualisation.

Demonstrates the ``diagnostics=True`` mode of the RFM wrapper, which enables
the PTH (ray-path) diagnostic output.  Two geometries are shown:

  1. Limb — seven tangent heights (5–35 km) at a realistic LEO satellite
     altitude.  Shows how path length and gas column loading vary with
     tangent height.

  2. Nadir — six elevation angles (−90° to −15°) from the same orbit.
     Shows how path lengths and airmass scale with viewing zenith angle.

Figures saved to ``test/demo_output/diagnostics/``.

Run with:
    python test/demo_diagnostics.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmcrameri.cm as cmc

import rfmwrapper as rw
from demo_config import get_output_path, timer, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ATM_TYPE    = 'FASMLS'            # mid-latitude summer
GAS         = 'H2O CO2 O3'
H_SAT       = 800.0               # satellite altitude [km] (LEO)
WN_START    = 700.0               # narrow spectral range — diagnostics only
WN_STOP     = 702.0
NPERWN      = 1

TAN_HEIGHTS  = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]   # km
ELEV_ANGLES  = [-90.0, -60.0, -45.0, -30.0, -20.0, -15.0]   # deg

# Colourmap for tangent heights — batlow: perceptually uniform, colourblind-safe
TH_NORM  = mcolors.Normalize(vmin=TAN_HEIGHTS[0], vmax=TAN_HEIGHTS[-1])
TH_CMAP  = cmc.batlow

# Colourmap for elevation angles — lajolla: warm tones, distinct from batlow
EL_NORM  = mcolors.Normalize(vmin=min(ELEV_ANGLES), vmax=max(ELEV_ANGLES))
EL_CMAP  = cmc.lajolla

# ---------------------------------------------------------------------------
# 1. Limb geometry
# ---------------------------------------------------------------------------

print("─" * 60)
print("LIMB geometry")
print("─" * 60)
print(f"  Atmosphere : {ATM_TYPE}")
print(f"  Gases      : {GAS}")
print(f"  Satellite  : {H_SAT:.0f} km")
print(f"  Tan heights: {TAN_HEIGHTS} km")

limb = rw.RFM(
    type='limb',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas=GAS,
    atm_type=ATM_TYPE,
    tan_heights=TAN_HEIGHTS,
    h_obs=H_SAT,
    diagnostics=True,
    out='RAD',
    silent=True,
)

with timer("Limb RFM run"):
    limb.run(silent=True)

# Parse all ray-path files
pth_raw    = limb.getRayPath()
pth_limb   = {k: rw.parseRayPath(v) for k, v in sorted(pth_raw.items())}
tan_labels = [p['header']['tan_hgt'] for p in pth_limb.values()]

print(f"  Parsed {len(pth_limb)} tangent heights: {tan_labels} km")

# ---- Figure L1: ray geometry grid (path length vs altitude) ---------------

ncols = 4
nrows = (len(TAN_HEIGHTS) + ncols - 1) // ncols
fig_lg, axes_lg = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows),
                                sharey=True, constrained_layout=True)
fig_lg.suptitle(f"Limb ray geometry — path length per layer\n"
                f"{ATM_TYPE}  |  satellite {H_SAT:.0f} km")
axes_lg_flat = axes_lg.flatten()

for i, (key, p) in enumerate(pth_limb.items()):
    ax  = axes_lg_flat[i]
    th  = p['header']['tan_hgt']
    col = TH_CMAP(TH_NORM(th))
    rw.plotRayGeometry(p, ax=ax, color=col,
                       title=f"TH = {th:.0f} km")
    if i % ncols != 0:
        ax.set_ylabel('')

# hide unused panels
for j in range(len(TAN_HEIGHTS), len(axes_lg_flat)):
    axes_lg_flat[j].set_visible(False)

out = get_output_path('diagnostics', 'limb_ray_geometry.png')
fig_lg.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

# ---- Figure L2: H2O and O3 column loading vs altitude (all TH overlay) ----

fig_lc, axes_lc = plt.subplots(1, 2, figsize=(10, 7), sharey=True,
                                constrained_layout=True)
fig_lc.suptitle(f"Limb ray path — gas column loading per layer\n"
                f"{ATM_TYPE}  |  satellite {H_SAT:.0f} km")

for key, p in pth_limb.items():
    th  = p['header']['tan_hgt']
    col = TH_CMAP(TH_NORM(th))
    lbl = f"TH = {th:.0f} km"
    for ax, gas in zip(axes_lc, ['H2O', 'O3']):
        if gas not in p['gases']:
            continue
        df = p['gases'][gas]
        ax.plot(df['amt'], df['z_low'], color=col, lw=1.1, label=lbl)
        ax.fill_betweenx(df['z_low'], 0, df['amt'], color=col, alpha=0.06)

for ax, gas in zip(axes_lc, ['H2O', 'O3']):
    ax.set_xscale('log')
    ax.set_xlabel(r'Column amount [kmol cm$^{-2}$]')
    ax.set_title(gas)

axes_lc[0].set_ylabel('Altitude [km]')

sm   = plt.cm.ScalarMappable(cmap=TH_CMAP, norm=TH_NORM)
sm.set_array([])
cbar = fig_lc.colorbar(sm, ax=axes_lc, location='right',
                       fraction=0.03, pad=0.02)
cbar.set_label('Tangent height [km]')

out = get_output_path('diagnostics', 'limb_column_loading.png')
fig_lc.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

limb.cleanup()

# ---------------------------------------------------------------------------
# 2. Nadir geometry
# ---------------------------------------------------------------------------

print()
print("─" * 60)
print("NADIR geometry")
print("─" * 60)
print(f"  Atmosphere : {ATM_TYPE}")
print(f"  Gases      : {GAS}")
print(f"  Satellite  : {H_SAT:.0f} km")
print(f"  Elevations : {ELEV_ANGLES} deg")

nadir = rw.RFM(
    type='nadir',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas=GAS,
    atm_type=ATM_TYPE,
    elev=ELEV_ANGLES,
    h_obs=H_SAT,
    diagnostics=True,
    out='RAD',
    silent=True,
)

with timer("Nadir RFM run"):
    nadir.run(silent=True)

pth_raw_n  = nadir.getRayPath()
pth_nadir  = {k: rw.parseRayPath(v) for k, v in sorted(pth_raw_n.items())}
ele_labels = [p['header']['ele_ang'] for p in pth_nadir.values()]

print(f"  Parsed {len(pth_nadir)} elevation angles: {ele_labels} deg")

# ---- Figure N1: ray geometry overlay (all elevations on one plot) ----------
# parseRayPath applies the 1/cos(VZA) slant correction for nadir geometry,
# so len is the true slant path length per layer — different for each angle.

fig_ng, ax_ng = plt.subplots(figsize=(6, 8), constrained_layout=True)
fig_ng.suptitle(f"Nadir ray geometry — slant path length per layer\n"
                f"{ATM_TYPE}  |  satellite {H_SAT:.0f} km")

for key, p in sorted(pth_nadir.items()):
    ele = p['header']['ele_ang']
    col = EL_CMAP(EL_NORM(ele))
    df  = next(iter(p['gases'].values()))    # geometry is gas-independent
    vza = 90.0 + ele
    ax_ng.plot(df['len'], df['z_low'], color=col, lw=1.2,
               label=f"elev {ele:.0f}° (VZA {vza:.0f}°)")

ax_ng.set_xlabel('Slant path length per layer [km]')
ax_ng.set_ylabel('Altitude [km]')
ax_ng.legend(fontsize=8, loc='upper right')

sm_n = plt.cm.ScalarMappable(cmap=EL_CMAP, norm=EL_NORM)
sm_n.set_array([])
cbar_n = fig_ng.colorbar(sm_n, ax=ax_ng, fraction=0.03, pad=0.02)
cbar_n.set_label('Elevation angle [deg]')

out = get_output_path('diagnostics', 'nadir_ray_geometry.png')
fig_ng.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

# ---- Figure N2: total slant column vs VZA — gas vs 1/cos(VZA) reference ---
# Since parseRayPath corrects len/amt to slant, totals now reflect the actual
# column along the ray.  All gases should track 1/cos(VZA) closely.

gases_to_plot = ['H2O', 'CO2', 'O3']

# Build arrays sorted by VZA (ascending)
items_sorted = sorted(pth_nadir.items(),
                      key=lambda kv: kv[1]['header']['ele_ang'])
vza_vals = np.array([90.0 + p['header']['ele_ang'] for _, p in items_sorted])

fig_nc, ax_nc = plt.subplots(figsize=(7, 5), constrained_layout=True)
fig_nc.suptitle(f"Nadir airmass factor vs VZA\n"
                f"{ATM_TYPE}  |  satellite {H_SAT:.0f} km")

for gas in gases_to_plot:
    col_amts = np.array([
        p['totals'][gas]['amt'] if gas in p['totals'] else np.nan
        for _, p in items_sorted
    ])
    # nadir (VZA=0) is the last entry after sorting by ele_ang ascending
    nadir_amt = col_amts[vza_vals == 0.0]
    if nadir_amt.size == 0:
        # no exact nadir — normalise to the smallest VZA
        nadir_amt = col_amts[np.argmin(vza_vals)]
    else:
        nadir_amt = nadir_amt[0]
    col = rw.get_species_color(gas)
    ax_nc.plot(vza_vals, col_amts / nadir_amt, color=col, lw=1.5,
               marker='o', ms=5, label=gas)

# theoretical 1/cos(VZA) reference
vza_ref = np.linspace(0, vza_vals.max(), 200)
ax_nc.plot(vza_ref, 1.0 / np.cos(np.radians(vza_ref)),
           'k--', lw=1, label='1/cos(VZA)')

ax_nc.set_xlabel('Viewing zenith angle [deg]')
ax_nc.set_ylabel('Airmass factor (normalised to nadir)')
ax_nc.legend(fontsize=9)

out = get_output_path('diagnostics', 'nadir_airmass.png')
fig_nc.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

nadir.cleanup()
print()
print("Done.")
