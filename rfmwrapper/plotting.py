"""
rfmwrapper/plotting.py — Standalone figure-building functions.

All public functions accept an optional ``ax``/``axes`` argument and return
``(fig, ax[es])`` without calling ``plt.show()`` — the caller decides when to
display or save.  :meth:`RFM.plotOut` and :meth:`RFM.plotAtm` are thin
wrappers that call these functions and then show the figure.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

from .constants import (rfm_gases, plot_colors, plot_cmaps, atm_fields,
                        atm_units, pretty_names, _FALLBACK_COLOR,
                        _FALLBACK_CMAP)


# ============================================================================
# Style
# ============================================================================

STYLE: dict = {
    'font.size': 11,
    'axes.labelsize': 9,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'xtick.direction': 'in',
    'xtick.top': True,
    'ytick.labelsize': 9,
    'ytick.direction': 'in',
    'ytick.right': True,
    'lines.linewidth': 0.9,
    'axes.grid': True,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'figure.constrained_layout.use': True,
    'axes.formatter.limits': (-4, 4),
}


def applyStyle() -> None:
    """Apply the standard rfmwrapper figure style to rcParams."""
    plt.rcParams.update(STYLE)


# ============================================================================
# Internal helpers
# ============================================================================

def _wn2wl(x: np.ndarray) -> np.ndarray:
    """Wavenumber [cm⁻¹] → wavelength [µm], suppressing divide-by-zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1e4 / x, np.inf)


def _add_wl_axis(ax: plt.Axes, wl_unit: str = 'um') -> plt.Axes:
    """
    Add a secondary wavelength axis on top of a wavenumber axis.

    Parameters
    ----------
    ax : plt.Axes
        Host axes (x-axis must be wavenumber [cm⁻¹]).
    wl_unit : str
        ``'um'`` for µm (default, TIR), ``'nm'`` for nm (VIS/NIR).

    Returns
    -------
    plt.Axes
        The secondary axes object.
    """
    if wl_unit == 'nm':
        def _fw(x: np.ndarray) -> np.ndarray:
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(x != 0, 1e7 / x, np.inf)
        label = r'Wavelength [nm]'
    else:
        _fw = _wn2wl
        label = r'Wavelength [µm]'

    wlax = ax.secondary_xaxis('top', functions=(_fw, _fw))
    wlax.set_xlabel(label)
    wlax.minorticks_on()
    return wlax


_YLABELS: dict[str, str] = {
    'rad':    r'Radiance [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$]',
    'rad_wl': r'Radiance [mW m$^{-2}$ sr$^{-1}$ µm$^{-1}$]',
    'bbt':    'Brightness Temperature [K]',
    'tra':    'Transmittance',
    'opt':    'Optical Depth',
    'jac':    r'Jacobian [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$ K$^{-1}$]',
}

_LABEL_UNITS: dict[str, str] = {
    'elevation': '°',
    'altitude':  ' km',
    'tangent':   ' km',
    'layer':     ' km',
    'column':    '',
}

_OUTPUT_TITLES: dict[str, str] = {
    'rad': 'Radiance',
    'bbt': 'Brightness Temperature',
    'tra': 'Transmittance',
    'opt': 'Optical Depth',
    'jac': 'Jacobian',
}


# ============================================================================
# Spectral output plots
# ============================================================================

def plotSpectra(
    spectra,
    ax: plt.Axes | None = None,
    color: str | None = None,
    cmap=None,
    species: str | None = None,
    wl_axis: bool = True,
    rad_wl: bool = False,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more spectra from a :class:`Spectra` or :class:`ILSOut` container.

    Parameters
    ----------
    spectra : Spectra or ILSOut
        Input container as returned by :meth:`RFM.getSpectra` or
        :meth:`RFM.getILSOut`.
    ax : plt.Axes, optional
        Target axes.  A new figure is created when ``None``.
    color : str, optional
        Single colour for all rows.  Ignored when ``cmap`` is given.
    cmap : str or Colormap, optional
        Matplotlib colourmap name or object; colours ``n_spectra`` rows from
        the map.  Overrides ``color`` and ``species``.
    species : str, optional
        Gas species key (e.g. ``'SO2'``).  When neither ``color`` nor ``cmap``
        is given, the species-canonical colour (n=1) or colourmap (n>1) is
        applied automatically.
    wl_axis : bool
        Add a secondary wavelength axis on top.
    rad_wl : bool
        If True, convert radiance from cm⁻¹ to µm domain before plotting.
    title : str, optional
        Axes title.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    n = spectra.data.shape[0]

    if cmap is not None:
        _cm = _resolve_cmap(cmap) if isinstance(cmap, str) else cmap
        colors = [_cm(i / max(n - 1, 1)) for i in range(n)]
    elif color is not None:
        colors = [color] * n
    elif species is not None:
        if n == 1:
            colors = [get_species_color(species)]
        else:
            _cm = get_species_cmap(species)
            colors = [_cm(i / max(n - 1, 1)) for i in range(n)]
    else:
        prop = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [prop[i % len(prop)] for i in range(n)]

    lu = _LABEL_UNITS.get(spectra.label_type, '')
    is_rad_wl = (spectra.output == 'rad') and rad_wl

    for i, row in enumerate(spectra.data):
        lbl = f"{spectra.labels[i]:.2f}{lu}"
        if is_rad_wl:
            xdata = 1e4 / spectra.wn
            ydata = row * (spectra.wn ** 2) / 1e4
        else:
            xdata = spectra.wn
            ydata = row
        ax.plot(xdata, ydata, color=colors[i], label=lbl)

    ylabel_key = 'rad_wl' if is_rad_wl else spectra.output
    ax.set_ylabel(_YLABELS.get(ylabel_key, ''))
    ax.set_xlabel(r'Wavelength [µm]' if is_rad_wl
                  else r'Wavenumber [cm$^{-1}$]')

    if (spectra.output == 'opt'
            and spectra.data.max() / max(float(spectra.data.min()), 1e-30) > 1e3):
        ax.set_yscale('log')

    if wl_axis and not is_rad_wl:
        _add_wl_axis(ax)

    if n > 1:
        ax.legend(fontsize=8)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def plotSpectraGrid(
    spectra_dict: dict,
    figsize: tuple | None = None,
    wl_axis: bool = True,
    titles: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot multiple :class:`Spectra` containers in a subplot grid.

    Parameters
    ----------
    spectra_dict : dict of str → Spectra
        Mapping from output name to container (e.g. ``{'rad': ..., 'tra': ...}``).
    figsize : tuple, optional
        Figure size; auto-computed when ``None``.
    wl_axis : bool
        Add secondary wavelength axis to each panel.
    titles : dict, optional
        Per-panel titles keyed by the same keys as ``spectra_dict``.
    **kwargs
        Forwarded to :func:`plotSpectra` (``color``, ``cmap``, ``rad_wl``).

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
    """
    keys = list(spectra_dict)
    n = len(keys)
    ncols = min(n, 2)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (ncols * 5, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes_flat = np.asarray(axes).flatten()

    for i, key in enumerate(keys):
        t = (titles or {}).get(key, _OUTPUT_TITLES.get(key, key.upper()))
        plotSpectra(spectra_dict[key], ax=axes_flat[i],
                    wl_axis=wl_axis, title=t, **kwargs)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    return fig, axes_flat[:n]


# ============================================================================
# Filter band overlay
# ============================================================================

def plotFilterBands(
    filter_out,
    ax: plt.Axes,
    secondary: bool = False,
    color: str = 'tab:orange',
    sensor_label: str | None = None,
    spectrum_index: int = 0,
) -> plt.Axes | None:
    """
    Overlay band-integrated filter values on an existing spectral axes.

    Parameters
    ----------
    filter_out : FilterOut
        Container as returned by :meth:`RFM.getFilterOut` or :func:`applyFilters`.
    ax : plt.Axes
        Axes to annotate (primary).
    secondary : bool
        If True, draw band points on a twin y-axis.
    color : str
        Colour for scatter points, vertical guides, and band labels.
    sensor_label : str, optional
        Legend entry.  Defaults to ``filter_out.sensor``.
    spectrum_index : int
        Row index in ``filter_out.data`` to visualise.

    Returns
    -------
    ax2 : plt.Axes or None
        Twin y-axes when ``secondary=True``; ``None`` otherwise.
    """
    wn_bands = filter_out.wn
    values   = filter_out.data[spectrum_index]
    names    = filter_out.band_names
    label    = sensor_label or filter_out.sensor

    draw_ax = ax.twinx() if secondary else ax

    for i, (x, y, name) in enumerate(zip(wn_bands, values, names)):
        ax.axvline(x, color=color, lw=0.8, ls='--', alpha=0.5)
        draw_ax.scatter(x, y, color=color, s=60, zorder=5,
                        label=f'{label} bands' if i == 0 else None)
        draw_ax.annotate(name, xy=(x, y), xytext=(4, 4),
                         textcoords='offset points', fontsize=7, color=color)

    if secondary:
        draw_ax.set_ylabel(_YLABELS.get(filter_out.output, ''), color=color)
        draw_ax.tick_params(axis='y', labelcolor=color)
        draw_ax.legend(fontsize=9, loc='upper right')
        return draw_ax

    return None


# ============================================================================
# ILS kernel
# ============================================================================

def plotILSKernel(
    ils: dict,
    axes: np.ndarray | None = None,
    figsize: tuple = (10, 3.5),
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Show the apodization function and ILS kernel for one instrument.

    Parameters
    ----------
    ils : dict
        ILS kernel dict as returned by :func:`makeILS`.
    axes : array of two Axes, optional
        Target axes.  A new 1×2 figure is created when ``None``.
    figsize : tuple
        Figure size when ``axes`` is ``None``.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    fig, axes : Figure, ndarray (shape (2,))
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    else:
        fig = axes[0].get_figure()

    axes[0].plot(ils['L_grid'], ils['apod'], color='tab:blue')
    axes[0].set(xlabel='OPD [cm]', ylabel='Amplitude',
                title='Apodization function')

    axes[1].plot(ils['wn_grid'], ils['kernel'], color='tab:red')
    axes[1].axhline(0, color='k', lw=0.5, ls='--')
    axes[1].set(xlabel=r'Wavenumber offset [cm$^{-1}$]', ylabel='Weight',
                title='ILS kernel')

    if title is not None:
        fig.suptitle(title)

    return fig, axes


# ============================================================================
# Atmosphere profile plots
# ============================================================================

def plotAtmProfile(
    atm: dict,
    species: str,
    ax: plt.Axes | None = None,
    hgt_obs: float | None = None,
    hgt_surf: float | None = None,
    zoom: tuple | None = None,
    color: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one atmospheric field vs altitude.

    Parameters
    ----------
    atm : dict
        Atmosphere dict (keys: ``'HGT'``, ``'TEM'``, ``'PRE'``, gas names).
    species : str
        Field key to plot.
    ax : plt.Axes, optional
    hgt_obs : float, optional
        Observer altitude [km]; draws a dashed blue reference line.
    hgt_surf : float, optional
        Surface altitude [km]; draws a dashed red reference line.
    zoom : tuple of (h_lo, h_hi), optional
        Altitude range for cubic-spline zoom interpolation.
    color : str, optional
        Profile color.  ``None`` uses the species-canonical colour from
        :data:`plot_colors`, or ``'k'`` for non-gas fields (TEM, PRE).

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    _color = color if color is not None else plot_colors.get(species, 'k')

    hgt = atm['HGT']
    y   = atm[species]

    if zoom is not None:
        cs   = CubicSpline(hgt, y)
        hgtz = np.arange(zoom[0], zoom[1])
        ax.plot(cs(hgtz), hgtz, color=_color)
    else:
        ax.plot(y, hgt, color=_color)

    if species in atm_fields:
        ax.set(title=atm_fields[species],
               xlabel=atm_units[species],
               ylabel='Height [km]')
    else:
        ax.set(title=pretty_names.get(species, species),
               xlabel='[ppm]',
               ylabel='Height [km]')

    legend_handles = []
    if hgt_obs is not None:
        ax.axhline(hgt_obs, color='tab:blue', ls='--', lw=1)
        legend_handles.append(
            Line2D([0], [0], color='tab:blue', ls='--', lw=1,
                   label=f'Observer {hgt_obs:.1f} km')
        )
    if hgt_surf is not None:
        ax.axhline(hgt_surf, color='tab:red', ls='--', lw=1)
        legend_handles.append(
            Line2D([0], [0], color='tab:red', ls='--', lw=1,
                   label=f'Surface {hgt_surf:.2f} km')
        )
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc='best')

    return fig, ax


def plotAtmGrid(
    atm: dict,
    species: list | None = None,
    figsize: tuple | None = None,
    hgt_obs: float | None = None,
    hgt_surf: float | None = None,
    zoom: tuple | None = None,
    color: str | None = 'k',
    title: str = 'RFM atmosphere',
) -> tuple[plt.Figure, np.ndarray]:
    """
    Multi-panel plot of atmospheric profiles.

    Parameters
    ----------
    atm : dict
        Atmosphere dict.
    species : list of str, optional
        Field keys to plot.  Defaults to ``TEM``, ``PRE``, and all RFM gases
        present in ``atm``.
    figsize : tuple, optional
        Figure size; auto-computed when ``None``.
    hgt_obs : float, optional
        Observer altitude [km]; reference line drawn on all panels, legend on
        the first.
    hgt_surf : float, optional
        Surface altitude [km]; reference line drawn on all panels.
    zoom : tuple, optional
        Altitude zoom range ``(h_lo, h_hi)`` for spline interpolation.
    color : str, optional
        Profile color.  ``None`` or ``'all'`` applies species-specific colours
        from :func:`get_species_color`; any other string is used directly.
    title : str
        Figure suptitle.

    Returns
    -------
    fig, axes_flat : Figure, ndarray of active Axes
    """
    if species is None:
        species = [k for k in atm
                   if k not in ('header',)
                   and k in rfm_gases + ['PRE', 'TEM']]

    n     = len(species)
    ncols = max(int(np.ceil(n / 2)), 1)
    nrows = 2
    if figsize is None:
        figsize = (ncols * 2, 8.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             squeeze=False, constrained_layout=True)
    axes_flat = axes.flatten()
    fig.suptitle(title)

    for i, key in enumerate(species):
        if i >= len(axes_flat):
            break
        c = None if (color == 'all' or color is None) else color
        plotAtmProfile(atm, key, ax=axes_flat[i], zoom=zoom, color=c)

    # Reference lines on all active panels
    for i in range(min(n, len(axes_flat))):
        if hgt_obs is not None:
            axes_flat[i].axhline(hgt_obs, color='tab:blue', ls='--', lw=1)
        if hgt_surf is not None:
            axes_flat[i].axhline(hgt_surf, color='tab:red', ls='--', lw=1)

    # Shared legend on first panel
    if n > 0:
        legend_handles = []
        if hgt_obs is not None:
            legend_handles.append(
                Line2D([0], [0], color='tab:blue', ls='--', lw=1,
                       label=f'Observer {hgt_obs:.1f} km')
            )
        if hgt_surf is not None:
            legend_handles.append(
                Line2D([0], [0], color='tab:red', ls='--', lw=1,
                       label=f'Surface {hgt_surf:.2f} km')
            )
        if legend_handles:
            axes_flat[0].legend(handles=legend_handles, fontsize=8, loc='best')

    for j in range(n, len(axes_flat)):
        axes_flat[j].remove()

    return fig, axes_flat[:n]


# ============================================================================
# Limb radiance 2-D map
# ============================================================================

def plotLimbMap(
    spectra,
    ax: plt.Axes | None = None,
    cmap: str = 'cmc.lajolla',
    vmin: float | None = None,
    vmax: float | None = None,
    wl_axis: bool = True,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    2-D colour map of limb radiance (tangent height × wavenumber).

    Parameters
    ----------
    spectra : Spectra
        Limb radiance, ``label_type='tangent'``, shape ``(n_tan, n_wn)``.
    ax : plt.Axes, optional
    cmap : str
        Matplotlib or cmcrameri colourmap name (e.g. ``'cmc.lajolla'``).
    vmin, vmax : float, optional
        Colour scale limits; symmetric auto-scale when both are ``None``.
    wl_axis : bool
        Add a secondary wavelength axis.
    title : str, optional

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    _cmap = _resolve_cmap(cmap)

    tan_heights = spectra.labels
    wn          = spectra.wn
    data        = spectra.data  # (n_tan, n_wn)

    im = ax.pcolormesh(wn, tan_heights, data, cmap=_cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(_YLABELS.get(spectra.output, ''))

    ax.set_xlabel(r'Wavenumber [cm$^{-1}$]')
    ax.set_ylabel('Tangent height [km]')

    if wl_axis:
        _add_wl_axis(ax)
    if title is not None:
        ax.set_title(title)

    return fig, ax


# ============================================================================
# Jacobian 2-D map
# ============================================================================

def plotJacobians(
    jac_spectra,
    ax: plt.Axes | None = None,
    cmap: str = 'cmc.vik',
    vmax: float | None = None,
    title: str | None = None,
    wl_axis: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    2-D colour map of Jacobian sensitivity (altitude × wavenumber).

    Parameters
    ----------
    jac_spectra : Spectra
        Jacobian container, ``label_type='altitude'``, ``output='jac'``,
        shape ``(n_levels, n_wn)``.
    ax : plt.Axes, optional
    cmap : str
        Diverging colourmap name (default ``'cmc.vik'``).
    vmax : float, optional
        Symmetric colour limit ±vmax; auto when ``None``.
    title : str, optional
    wl_axis : bool
        Add a secondary wavelength axis.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    _cmap = _resolve_cmap(cmap)

    altitudes = jac_spectra.labels
    wn        = jac_spectra.wn
    data      = jac_spectra.data  # (n_levels, n_wn)

    if vmax is None:
        vmax = float(np.abs(data).max())
    vmin = -vmax

    im = ax.pcolormesh(wn, altitudes, data, cmap=_cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(_YLABELS.get('jac', ''))

    ax.set_xlabel(r'Wavenumber [cm$^{-1}$]')
    ax.set_ylabel('Altitude [km]')

    if wl_axis:
        _add_wl_axis(ax)
    if title is not None:
        ax.set_title(title)

    return fig, ax


# ============================================================================
# Retrieval profile
# ============================================================================

def plotRetrievalProfile(
    altitudes: np.ndarray,
    x_prior: np.ndarray,
    x_ret: np.ndarray,
    x_true: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    xlabel: str = 'Temperature [K]',
    title: str = 'Retrieved profile',
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot prior, retrieved, and (optionally) true profiles vs altitude.

    Parameters
    ----------
    altitudes : np.ndarray
        Altitude levels [km].
    x_prior : np.ndarray
        Prior state vector.
    x_ret : np.ndarray
        Retrieved state vector.
    x_true : np.ndarray, optional
        True state vector for validation.
    ax : plt.Axes, optional
    xlabel : str
        X-axis label.
    title : str

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(x_prior, altitudes, color='tab:blue', ls='--', lw=1.2, label='Prior')
    ax.plot(x_ret,   altitudes, color='tab:red',  ls='-',  lw=1.5, label='Retrieved')
    if x_true is not None:
        ax.plot(x_true, altitudes, color='k', ls='-', lw=1.0, label='True')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Altitude [km]')
    ax.set_title(title)
    ax.legend(fontsize=9)

    return fig, ax


# ============================================================================
# Ray-path diagnostics
# ============================================================================

def plotRayGeometry(
    pth_parsed: dict,
    ax: plt.Axes | None = None,
    color: str = 'tab:blue',
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualise the geometric structure of one RFM ray path as horizontal bars.

    Each atmospheric layer is drawn as a horizontal bar whose width equals the
    ray path length through that layer [km].  The bar is positioned at the
    layer's bottom altitude.  A dashed line marks the tangent height for limb
    geometries.

    Parameters
    ----------
    pth_parsed : dict
        Output of :func:`~rfmwrapper.functions.parseRayPath` for a single
        elevation / tangent height.
    ax : plt.Axes, optional
    color : str
        Bar face colour (default ``'tab:blue'``).
    title : str, optional

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 7))
    else:
        fig = ax.get_figure()

    header = pth_parsed['header']
    gases  = pth_parsed['gases']

    if not gases:
        ax.text(0.5, 0.5, 'No ray-path data', transform=ax.transAxes,
                ha='center', va='center')
        return fig, ax

    # Use path lengths from the first gas (geometry is the same for all)
    first_gas = next(iter(gases.values()))
    z   = first_gas['z_low'].values
    ln  = first_gas['len'].values

    ax.barh(z, ln, height=np.diff(z, append=z[-1] + (z[-1] - z[-2])) * 0.8,
            left=0, align='edge', color=color, alpha=0.7, edgecolor='none')

    if header['geometry'] == 'limb' and header['tan_hgt'] is not None:
        ax.axhline(header['tan_hgt'], color='tab:red', ls='--', lw=1,
                   label=f"Tangent hgt {header['tan_hgt']:.1f} km")
        ax.legend(fontsize=8)

    ax.set_xlabel('Path length [km]')
    ax.set_ylabel('Altitude [km]')

    if title is not None:
        ax.set_title(title)
    elif header['tan_hgt'] is not None:
        ax.set_title(f"Ray geometry — TH = {header['tan_hgt']:.1f} km")
    elif header['ele_ang'] is not None:
        ax.set_title(f"Ray geometry — elev = {header['ele_ang']:.1f}°")

    return fig, ax


def plotRayPath(
    pth_parsed: dict,
    gas: str,
    ax: plt.Axes | None = None,
    color: str | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the column amount per layer along a ray path for one gas species.

    Shows where absorption is concentrated along the ray — the vertical
    distribution of the effective gas loading at each altitude layer.

    Parameters
    ----------
    pth_parsed : dict
        Output of :func:`~rfmwrapper.functions.parseRayPath`.
    gas : str
        Gas species key (case-insensitive, e.g. ``'H2O'``, ``'CO2'``).
    ax : plt.Axes, optional
    color : str, optional
        Profile colour.  ``None`` uses the species-canonical colour.
    title : str, optional

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 7))
    else:
        fig = ax.get_figure()

    header = pth_parsed['header']
    gases  = pth_parsed['gases']
    key    = gas.upper()

    if key not in gases:
        available = list(gases.keys())
        raise KeyError(f"Gas '{gas}' not found in ray-path data. Available: {available}")

    df    = gases[key]
    z     = df['z_low'].values
    amt   = df['amt'].values

    _color = color if color is not None else plot_colors.get(key, _FALLBACK_COLOR)
    ax.plot(amt, z, color=_color, lw=1.2)
    ax.fill_betweenx(z, 0, amt, color=_color, alpha=0.2)

    if header['geometry'] == 'limb' and header['tan_hgt'] is not None:
        ax.axhline(header['tan_hgt'], color='tab:red', ls='--', lw=1,
                   label=f"Tangent hgt {header['tan_hgt']:.1f} km")
        ax.legend(fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel(r'Column amount [kmol cm$^{-2}$]')
    ax.set_ylabel('Altitude [km]')

    _name = key
    if title is not None:
        ax.set_title(title)
    elif header['tan_hgt'] is not None:
        ax.set_title(f"{_name} column — TH = {header['tan_hgt']:.1f} km")
    elif header['ele_ang'] is not None:
        ax.set_title(f"{_name} column — elev = {header['ele_ang']:.1f}°")

    return fig, ax


# ============================================================================
# Species colour/cmap helpers
# ============================================================================

def get_species_color(species: str) -> str:
    """
    Return the canonical single colour for a gas species.

    Falls back to :data:`_FALLBACK_COLOR` for unknown species.

    Parameters
    ----------
    species : str
        Gas/species key (e.g. ``'SO2'``, ``'H2O'``).

    Returns
    -------
    str
        Matplotlib colour string or hex code.
    """
    return plot_colors.get(species, _FALLBACK_COLOR)


def get_species_cmap(species: str):
    """
    Return the canonical sequential colourmap for a gas species.

    Falls back to :data:`_FALLBACK_CMAP` for unknown species.

    Parameters
    ----------
    species : str
        Gas/species key (e.g. ``'SO2'``, ``'H2O'``).

    Returns
    -------
    matplotlib Colormap
    """
    return plot_cmaps.get(species, _FALLBACK_CMAP)


# ============================================================================
# Internal utility
# ============================================================================

def _resolve_cmap(name: str):
    """
    Return a colourmap object from a name, supporting the ``'cmc.xxx'`` prefix
    for cmcrameri maps.
    """
    if name.startswith('cmc.'):
        try:
            import cmcrameri.cm as cmc
            return getattr(cmc, name[4:])
        except (ImportError, AttributeError):
            pass
    return plt.get_cmap(name)
