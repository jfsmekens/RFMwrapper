#!/usr/bin/env python3
"""
demo_filters.py — Spectral response functions for all bundled imager instruments.

One subplot per instrument; channels coloured by centre wavelength using a
global rainbow colormap (violet = short, red/dark = long).  For multi-platform
instruments the most representative platform is shown; the number of available
platform variants is noted in the title.

Run with:
    python test/demo_filters.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path

from rfmwrapper.functions import _loadFilters

# ---------------------------------------------------------------------------
# Layout: 3 rows × 5 columns = 15 panels
# (instrument_dir, canonical_platform_or_None, subplot_title, n_platforms)
# ---------------------------------------------------------------------------
INSTRUMENTS = [
    # Row 1 — geostationary imagers
    ('GOES-ABI',         'GOES-16',     'GOES-ABI',         4),
    ('HIMAWARI-AHI',     'Himawari-8',  'Himawari AHI',     2),
    ('MSG-SEVIRI',       'MSG-4',       'MSG SEVIRI',        4),
    ('MTG-FCI',          None,          'MTG FCI',           1),
    ('GOES-Sounder',     'GOES-15',     'GOES Sounder',      8),
    # Row 2 — polar-orbit multispectral
    ('MODIS',            'Terra',       'MODIS',             2),
    ('VIIRS',            'SNPP',        'VIIRS',             3),
    ('AVHRR',            'NOAA-19',     'AVHRR',             8),
    ('Sentinel3-SLSTR',  'Sentinel-3A', 'Sentinel-3 SLSTR',  4),
    ('GOES-Imager',      'GOES-15',     'GOES Imager',       8),
    # Row 3 — specialised / reference
    ('ASTER',            None,          'ASTER',             1),
    ('Landsat-TM',       'Landsat-7',   'Landsat ETM+',      3),
    ('Landsat-TIRS',     'Landsat-8',   'Landsat TIRS',      2),
    ('ECOSTRESS',        None,          'ECOSTRESS',         1),
    ('MMT-gasCam',       None,          'MMT gasCam',        1),
]

# Global wavelength → colour mapping (0.4 – 15 µm)
WL_MIN, WL_MAX = 0.4, 15.0
_norm  = mcolors.Normalize(vmin=WL_MIN, vmax=WL_MAX)
_cmap  = plt.cm.rainbow


def _channel_color(center_wl: float):
    return _cmap(_norm(center_wl))


def _center_wl(filt: dict) -> float:
    """Centre wavelength of a filter: median of the >1 % passband."""
    wl_f, T_f = filt['wl'], filt['T']
    passband = wl_f[T_f > 0.01]
    return float(np.median(passband)) if len(passband) else float(np.mean(wl_f))


def main() -> None:
    n_rows, n_cols = 3, 5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(20, 10),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, (inst, platform, title, n_plat) in zip(axes_flat, INSTRUMENTS):
        sensor = f"{inst}/{platform}" if platform else inst
        filters = _loadFilters(sensor)

        for name, filt in filters.items():
            # wl from _loadFilters is descending (sorted by ascending wn)
            wl = filt['wl'][::-1]
            T  = filt['T'][::-1]
            c  = _channel_color(_center_wl(filt))
            ax.plot(wl, T, color=c, lw=0.9, alpha=0.85)

        # Axis decoration
        plat_tag = f"\n({platform})" if platform else ""
        plat_note = f"  [{n_plat} platforms]" if n_plat > 1 else ""
        ax.set_title(f"{title}{plat_tag}{plat_note}", fontsize=8, pad=3)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel("Wavelength [µm]", fontsize=7)
        ax.set_ylabel("Response", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.text(0.97, 0.96, f"{len(filters)} ch",
                transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='0.45')

    # Shared colorbar — wavelength axis
    sm  = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat, location='bottom',
                        fraction=0.015, pad=0.06, aspect=60)
    cbar.set_label("Channel centre wavelength [µm]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Spectral Response Functions — rfmwrapper bundled sensors  "
        "(channels coloured by centre wavelength)",
        fontsize=11,
    )

    out_path = Path(__file__).parent / 'demo_filters.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
