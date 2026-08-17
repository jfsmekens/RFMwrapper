"""
demo_nadir.py — Nadir-looking RFM simulation with ILS and filter convolution.

Runs a downward-looking simulation over the TIR (650–1350 cm⁻¹) using the
mid-latitude summer atmosphere, then applies IASI ILS and ASTER filter
convolution.

Run with:
    python test/demo_nadir.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

import rfmwrapper as rw
from rfmwrapper.plotting import (
    plotSpectra, plotFilterBands, plotILSKernel,
)
from demo_config import get_output_path, timer, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START   = 650.0
WN_STOP    = 1350.0
NPERWN     = 25
ATM_TYPE   = 'FASMLS'
SURF_TEMP  = 300.0
SURF_EMISS = 1.0
ILS        = 'IASI'
SENSOR     = 'ASTER'

# ---------------------------------------------------------------------------
# 1. Run RFM
# ---------------------------------------------------------------------------

print("Setting up nadir model...")
model = rw.RFM(
    type='nadir',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas='all', atm_type=ATM_TYPE,
    surf_temp=SURF_TEMP, surf_emiss=SURF_EMISS,
    silent=True,
)

with timer("RFM run"):
    model.run(silent=True)

# ---------------------------------------------------------------------------
# 2. Apply ILS
# ---------------------------------------------------------------------------

model.applyILS(ILS)
ils_params  = rw.loadILSParams(ILS)
ils_label   = ILS
fov_used    = ils_params['fov']
max_opd_used = ils_params['max_opd']
apod_used   = ils_params['apod_type']
fwhm_used   = ils_params.get('fwhm_wn', None)

wn_center = (WN_START + WN_STOP) / 2.0
ils = rw.makeILS(fov=fov_used, max_opd=max_opd_used, nper_wn=NPERWN,
                 wn_center=wn_center, apod_type=apod_used, fwhm_wn=fwhm_used)

print(f"  ILS: {ILS}  apod={apod_used}  OPD={max_opd_used} cm  "
      f"FOV={fov_used * 1e3:.2f} mrad  kernel={len(ils['kernel'])} pts")

# ---------------------------------------------------------------------------
# 3. Apply filters
# ---------------------------------------------------------------------------

model.applyFilters(SENSOR)
filt_rad = model.getFilterOut(SENSOR, 'rad')
filt_bbt = model.getFilterOut(SENSOR, 'bbt')
filt_tra = model.getFilterOut(SENSOR, 'tra')
filt_opt = model.getFilterOut(SENSOR, 'opt')
print(f"  {SENSOR} bands in range: {filt_rad.band_names}")

# ---------------------------------------------------------------------------
# 4. Build figure — 4 spectral panels
# ---------------------------------------------------------------------------

if fwhm_used is not None:
    ils_desc = f"{ils_label} ({apod_used}, OPD={max_opd_used} cm, FWHM={fwhm_used} cm⁻¹)"
else:
    ils_desc = f"{ils_label} ({apod_used}, OPD={max_opd_used} cm)"

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True,
                         constrained_layout=True)
fig.suptitle(
    f"RFM nadir — {ATM_TYPE}  |  T_sfc={SURF_TEMP:.0f} K  ε={SURF_EMISS:.2f}"
    f"\nILS: {ils_desc}"
)

rad_raw = model.getSpectra('rad')
bbt_raw = model.getSpectra('bbt')
tra_raw = model.getSpectra('tra')
opt_raw = model.getSpectra('opt')
rad_ils = model.getILSOut('rad')
bbt_ils = model.getILSOut('bbt')
tra_ils = model.getILSOut('tra')
opt_ils = model.getILSOut('opt')

# Radiance — secondary axis for band values (units are consistent since
# convolution is wn-domain, but secondary axis keeps the plot uncluttered)
ax = axes[0]
ax.plot(rad_raw.wn, rad_raw.data[0], color='0.65', lw=0.7, label='Raw RFM')
ax.plot(rad_ils.wn, rad_ils.data[0], color='tab:red', lw=1.2, label=f'ILS ({ils_label})')
plotFilterBands(filt_rad, ax, secondary=True, color='tab:orange')
ax.set_ylabel(r'Radiance [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$]')
ax.set_title('Radiance')
ax.legend(fontsize=8, loc='upper left')

# Brightness temperature
ax = axes[1]
ax.plot(bbt_raw.wn, bbt_raw.data[0], color='0.65', lw=0.7, label='Raw RFM')
ax.plot(bbt_ils.wn, bbt_ils.data[0], color='tab:blue', lw=1.2, label=f'ILS ({ils_label})')
plotFilterBands(filt_bbt, ax, secondary=False, color='tab:orange')
ax.set_ylabel('Brightness Temperature [K]')
ax.set_title('Brightness Temperature')
ax.legend(fontsize=8)

# Transmittance
ax = axes[2]
ax.plot(tra_raw.wn, tra_raw.data[0], color='0.65', lw=0.7, label='Raw RFM')
ax.plot(tra_ils.wn, tra_ils.data[0], color='tab:green', lw=1.2, label=f'ILS ({ils_label})')
plotFilterBands(filt_tra, ax, secondary=False, color='tab:orange')
ax.set_ylabel('Transmittance')
ax.set_title('Transmittance')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)

# Optical depth
ax = axes[3]
ax.plot(opt_raw.wn, opt_raw.data[0], color='0.65', lw=0.7, label='Raw RFM')
ax.plot(opt_ils.wn, opt_ils.data[0], color='tab:purple', lw=1.2, label=f'ILS ({ils_label})')
plotFilterBands(filt_opt, ax, secondary=False, color='tab:orange')
ax.set_ylabel('Optical Depth')
ax.set_yscale('log')
ax.set_title('Optical Depth')
ax.set_xlabel(r'Wavenumber [cm$^{-1}$]')
ax.legend(fontsize=8)

# Shared wavelength axis on top of radiance panel
from rfmwrapper.plotting import _add_wl_axis
_add_wl_axis(axes[0])

out = get_output_path('nadir', 'spectra.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# 5. ILS kernel figure
# ---------------------------------------------------------------------------

fig2, axes2 = plotILSKernel(ils, title=f"ILS kernel — {ils_desc}")
out2 = get_output_path('nadir', 'ils_kernel.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")
plt.show()

model.cleanup()
print("Done.")
