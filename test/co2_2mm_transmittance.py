"""
co2_2mm_transmittance.py — CO2 transmittance over short homogeneous paths.

Ambient CO2 (420 ppmv) in 1, 2 and 5 mm gas layers at lab STP
(298 K, 1013 mb), 2200-2500 cm-1 at 10 points per cm-1 (0.1 cm-1 spacing).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

import rfmwrapper as rw

WN_START   = 2200.0
WN_STOP    = 2500.0
NPERWN     = 10
TEMP       = 298.0        # K
PRES       = 1013.0       # mbar
CO2_PPM    = 420.0        # ppmv
PATH_MM    = [1.0, 2.0, 5.0]     # path lengths [mm]

OUT_DIR = os.path.join(os.path.dirname(__file__), 'demo_output', 'co2_2mm')
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
color_of = {1.0: 'C0', 2.0: 'C3', 5.0: 'C2'}

# Draw longest path first so the shorter (shallower) lines end up on top.
for mm in sorted(PATH_MM, reverse=True):
    c = color_of[mm]
    m = rw.RFM(
        type='layer',
        wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
        gas='CO2',
        temp=TEMP, pres=PRES, pathlength=mm * 1e-6,   # mm -> km
        vmr={'CO2': CO2_PPM},
        out='TRA',
        silent=True,
    )
    m.run(silent=True)

    tra = m.getSpectra('tra')
    wn, t = tra.wn, tra.data[0]

    data_path = os.path.join(OUT_DIR, f'co2_{mm:g}mm_transmittance.txt')
    np.savetxt(
        data_path,
        np.column_stack([wn, t]),
        header=f"CO2 {CO2_PPM:.0f} ppmv | {mm:g} mm | {TEMP:.0f} K | {PRES:.0f} mb\n"
               f"wn[cm-1]    transmittance",
    )
    print(f"{mm:g} mm: T_min = {t.min():.6f} at {wn[np.argmin(t)]:.2f} cm-1  ->  {data_path}")

    ax.plot(wn, t, lw=0.6, color=c, label=f"{mm:g} mm")

ax.set_xlabel(r'Wavenumber [cm$^{-1}$]')
ax.set_ylabel('Transmittance')
ax.set_title(f"CO$_2$ {CO2_PPM:.0f} ppmv — {TEMP:.0f} K, {PRES:.0f} mb")
ax.set_xlim(WN_START, WN_STOP)
# Order legend by ascending path length regardless of draw order.
handles, labels = ax.get_legend_handles_labels()
order = np.argsort([float(l.split()[0]) for l in labels])
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          title='Path length')

fig_path = os.path.join(OUT_DIR, 'co2_pathlength_transmittance.png')
fig.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
