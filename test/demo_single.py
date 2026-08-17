"""
demo_single.py — End-to-end single RFM run with ILS and filter convolution.

Runs a nadir-looking simulation over the TIR (650–1350 cm⁻¹) using the
standard mid-latitude summer atmosphere, then applies:
  - ILS convolution (named instrument or custom parameters)
  - MMT-gasCam filter convolution

Usage:
    python test/demo_single.py
"""
import numpy as np
import matplotlib.pyplot as plt

import rfmwrapper as rw

# ---------------------------------------------------------------------------
# Global rcParams — match the demo script figure style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':                   11,
    'axes.labelsize':              9,
    'axes.titlesize':              12,
    'xtick.labelsize':             9,
    'xtick.direction':             'in',
    'ytick.labelsize':             9,
    'ytick.direction':             'in',
    'lines.linewidth':             0.9,
    'axes.grid':                   True,
    'grid.linewidth':              0.5,
    'grid.alpha':                  0.5,
    'grid.linestyle':              '--',
    'axes.axisbelow':              True,
    'figure.constrained_layout.use': True,
    'axes.formatter.limits':       (-4, 4),
    'errorbar.capsize':             2,
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START  = 650.0    # cm⁻¹
WN_STOP   = 1350.0   # cm⁻¹
NPERWN    = 25       # points per cm⁻¹
ATM_TYPE  = 'FASMLS' # mid-latitude summer

FOV      = 0.030     # radians  — used only when ILS is None
MAX_OPD  = 1.8       # cm       — used only when ILS is None
APOD     = 'NB medium'           # used only when ILS is None

ILS      = 'IASI'    # set to None to use custom FOV/MAX_OPD/APOD above

SENSOR   = 'ASTER'

SURF_TEMP  = 300.0   # K    — set to None to inherit from atmospheric profile base
SURF_EMISS = 1.0     # 0–1  — WARNING: values < 1 change the meaning of TRA/OPT output.
                     #         RFM will report (1-ε)×T_atm² (reflected path transmittance)
                     #         instead of one-way atmospheric transmittance.


# ---------------------------------------------------------------------------
# 1. Run RFM
# ---------------------------------------------------------------------------

print("Setting up RFM...")
model = rw.RFM(
    type='nadir',
    wn_start=WN_START,
    wn_stop=WN_STOP,
    nperwn=NPERWN,
    gas='all',
    atm_type=ATM_TYPE,
    surf_temp=SURF_TEMP,
    surf_emiss=SURF_EMISS,
    silent=False,
)

print("\nRunning RFM...")
model.run(silent=True)
print("  ✓ RFM complete")


# ---------------------------------------------------------------------------
# 2. Retrieve raw spectra
# ---------------------------------------------------------------------------

rad_raw = model.getSpectra('rad')
bbt_raw = model.getSpectra('bbt')
tra_raw = model.getSpectra('tra')
opt_raw = model.getSpectra('opt')

print(f"\nOutput shape: {rad_raw.data.shape}  "
      f"(n_spectra={rad_raw.data.shape[0]}, n_wn={rad_raw.data.shape[1]})")
print(f"Wavenumber range: {rad_raw.wn[0]:.1f} – {rad_raw.wn[-1]:.1f} cm⁻¹")


# ---------------------------------------------------------------------------
# 3. Apply ILS
# ---------------------------------------------------------------------------

if ILS is not None:
    model.applyILS(ILS)
    ils_params = rw.loadILSParams(ILS)
    ils_label  = ILS
else:
    model.applyILS(fov=FOV, max_opd=MAX_OPD, apod_type=APOD)
    ils_params = {'fov': FOV, 'max_opd': MAX_OPD, 'apod_type': APOD}
    ils_label  = APOD

fov_used     = ils_params['fov']
max_opd_used = ils_params['max_opd']
apod_used    = ils_params['apod_type']
fwhm_used    = ils_params.get('fwhm_wn', None)

print(f"\nILS applied: {ils_label}")
print(f"  apod_type : {apod_used}")
print(f"  max_opd   : {max_opd_used} cm")
print(f"  fov       : {fov_used * 1e3:.2f} mrad")
if fwhm_used is not None:
    print(f"  fwhm_wn   : {fwhm_used} cm⁻¹")

rad_ils = model.getILSOut('rad')
bbt_ils = model.getILSOut('bbt')
tra_ils = model.getILSOut('tra')
opt_ils = model.getILSOut('opt')

# Rebuild the kernel with the resolved parameters for kernel diagnostics/plot
wn_center = (WN_START + WN_STOP) / 2.0
ils = rw.makeILS(fov=fov_used, max_opd=max_opd_used, nper_wn=NPERWN,
                 wn_center=wn_center, apod_type=apod_used, fwhm_wn=fwhm_used)
print(f"  kernel length     : {len(ils['kernel'])} pts")
print(f"  FOV broadening    : {ils['fov_width']:.4f} cm⁻¹")
print(f"  Nominal resolution: {0.9 / max_opd_used:.3f} cm⁻¹")


# ---------------------------------------------------------------------------
# 4. Apply filters
# ---------------------------------------------------------------------------

print(f"\nApplying {SENSOR!r} filter convolution...")
model.applyFilters(sensor=SENSOR)
print("  ✓ Filters applied")

filt_rad = model.getFilterOut(SENSOR, 'rad')
filt_bbt = model.getFilterOut(SENSOR, 'bbt')
filt_tra = model.getFilterOut(SENSOR, 'tra')
filt_opt = model.getFilterOut(SENSOR, 'opt')
print(f"  Bands in range  : {filt_rad.band_names}")
print(f"  Band centres    : {np.round(filt_rad.wl, 2)} µm")
print(f"  Radiance values : {np.round(filt_rad.data[0], 4)}")


# ---------------------------------------------------------------------------
# 5. Plot: raw vs ILS-convolved spectra + filter band values
# ---------------------------------------------------------------------------

# Build figure title from resolved ILS parameters
if fwhm_used is not None:
    ils_desc = f"{ils_label} ({apod_used}, OPD={max_opd_used} cm, FWHM={fwhm_used} cm⁻¹)"
else:
    ils_desc = f"{ils_label} ({apod_used}, OPD={max_opd_used} cm)"

surf_desc_parts = []
if SURF_TEMP is not None:
    surf_desc_parts.append(f"T_sfc={SURF_TEMP:.0f} K")
if SURF_EMISS is not None:
    surf_desc_parts.append(f"ε={SURF_EMISS:.2f}")
surf_desc = f"  |  Surface: {', '.join(surf_desc_parts)}" if surf_desc_parts else ""

fig, axes = plt.subplots(4, 1, figsize=[10, 12], sharex=True, tight_layout=True)
fig.suptitle(f"RFM nadir — {ATM_TYPE}{surf_desc}  |  ILS: {ils_desc}")

wn = rad_raw.wn
wl_axis_label = r"Wavenumber [cm$^{-1}$]"

# Band centres in wavenumber space, shared across all panels
wn_bands   = 1e4 / filt_rad.wl
band_names = filt_rad.band_names


def _add_bands_simple(ax, wn_b, values, names, color='tab:orange'):
    """Overlay band-integrated values as scatter points + vertical guides (single axis)."""
    for i, (x, y, name) in enumerate(zip(wn_b, values, names)):
        ax.axvline(x, color=color, lw=0.8, ls='--', alpha=0.5)
        ax.scatter(x, y, color=color, s=60, zorder=5,
                   label=f'{SENSOR} bands' if i == 0 else None)
        ax.annotate(name, xy=(x, y), xytext=(4, 4),
                    textcoords='offset points', fontsize=7, color=color)


def _add_bands_secondary(ax, wn_b, values, names, ylabel, color='tab:orange'):
    """
    Overlay band-integrated values on a secondary right-hand y-axis.

    Vertical guide lines are drawn on the primary axis; scatter points and
    labels are placed on the secondary axis so both y-scales are visible.
    """
    ax2 = ax.twinx()
    for i, (x, y, name) in enumerate(zip(wn_b, values, names)):
        ax.axvline(x, color=color, lw=0.8, ls='--', alpha=0.5)
        ax2.scatter(x, y, color=color, s=60, zorder=5,
                    label=f'{SENSOR} bands' if i == 0 else None)
        ax2.annotate(name, xy=(x, y), xytext=(4, 4),
                     textcoords='offset points', fontsize=7, color=color)
    ax2.set_ylabel(ylabel, color=color, fontsize=9)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(fontsize=9, loc='upper right')
    return ax2


# --- Radiance ---
# Primary axis: mW/(m²·sr·cm⁻¹); band values are in mW/(m²·sr·µm) → secondary axis
ax = axes[0]
ax.plot(wn, rad_raw.data[0], color='0.6', lw=0.8, label='Raw RFM')
ax.plot(wn, rad_ils.data[0], color='tab:red', lw=1.2, label=f'ILS ({ils_label})')
_add_bands_secondary(ax, wn_bands, filt_rad.data[0], band_names,
                     ylabel=r'Radiance [mW/(m²·sr·µm)]')
ax.set(ylabel=r'Radiance [mW/(m²·sr·cm⁻¹)]', title='Radiance')
ax.legend(fontsize=9, loc='upper left')

# --- BBT ---
ax = axes[1]
ax.plot(wn, bbt_raw.data[0], color='0.6', lw=0.8, label='Raw RFM')
ax.plot(wn, bbt_ils.data[0], color='tab:blue', lw=1.2, label=f'ILS ({ils_label})')
_add_bands_simple(ax, wn_bands, filt_bbt.data[0], band_names)
ax.set(ylabel='Brightness Temperature [K]', title='Brightness Temperature')
ax.legend(fontsize=9)

# --- Transmission ---
ax = axes[2]
ax.plot(wn, tra_raw.data[0], color='0.6', lw=0.8, label='Raw RFM')
ax.plot(wn, tra_ils.data[0], color='tab:green', lw=1.2, label=f'ILS ({ils_label})')
_add_bands_simple(ax, wn_bands, filt_tra.data[0], band_names)
ax.set(ylabel='Transmission', title='Transmission')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9)

# --- Optical depth ---
ax = axes[3]
ax.plot(wn, opt_raw.data[0], color='0.6', lw=0.8, label='Raw RFM')
ax.plot(wn, opt_ils.data[0], color='tab:purple', lw=1.2, label=f'ILS ({ils_label})')
_add_bands_simple(ax, wn_bands, filt_opt.data[0], band_names)
ax.set(xlabel=wl_axis_label, ylabel='Optical Depth', title='Optical Depth', yscale='log')
ax.legend(fontsize=9)

# Secondary wavelength axis on top of the radiance panel.
# np.errstate suppresses the divide-by-zero warning that matplotlib triggers
# when it probes the conversion function at x=0 during axis setup.
def _wn2wl(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1e4 / x, np.inf)

axes[0].secondary_xaxis('top', functions=(_wn2wl, _wn2wl)).set_xlabel(r'Wavelength [µm]')

plt.show()


# ---------------------------------------------------------------------------
# 6. Plot: ILS kernel shape
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 2, figsize=[10, 3.5], tight_layout=True)
fig2.suptitle(f"ILS kernel — {ils_desc}")

axes2[0].plot(ils['L_grid'], ils['apod'], color='tab:blue')
axes2[0].set(xlabel='OPD [cm]', ylabel='Amplitude', title='Apodization function')

axes2[1].plot(ils['wn_grid'], ils['kernel'], color='tab:red')
axes2[1].axhline(0, color='k', lw=0.5, ls='--')
axes2[1].set(xlabel=r'Wavenumber offset [cm⁻¹]', ylabel='Weight', title='ILS kernel')

plt.show()

# Cleanup temp files
model.cleanup()
print("\nDone.")
