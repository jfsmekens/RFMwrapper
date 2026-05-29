"""
Written by @jfsmekens - Last updated 15/02/2024

Wrapper for the RFM radiative transfer code.

Dudhia, A (2017): The Reference Forward Model (RFM), Journal of Quantitative
Spectroscopy and Radiative Transfer 186, pp. 243-253,
doi:10.1016/j.jqsrt.2016.06.018
"""
import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from subprocess import run, DEVNULL

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import pandas as pd

from .config import (get_rfm_dir, get_hitran_path,
                     get_exec_path, get_atm_dir, get_aria_dir)
from .functions import (readStandard, listSensors, _loadFilters, makeILS,
                        loadILSParams,
                        readAtm as _readAtm, writeAtm as _writeAtm,
                        adjustAtm as _adjustAtm,
                        insertPlume as _insertPlume,
                        sliceAtm as _sliceAtm)
from .constants import (rfm_gases, plot_colors, atm_fields, atm_units, pretty_names)

logger = logging.getLogger(__name__)

# ===========================================================================
# ========================= Module-level path config ========================
# ===========================================================================

__RFMDIR__    = get_rfm_dir()
__EXECPATH__  = get_exec_path()
__HITRANPATH__ = get_hitran_path()
__ATMPATH__   = get_atm_dir()
__ARIAPATH__  = get_aria_dir()

if os.path.exists(__EXECPATH__):
    logger.info('RFM executable found: %s', __EXECPATH__)
else:
    logger.warning('RFM executable not found at %s. Set __EXECPATH__ or pass exec_path= to RFM().', __EXECPATH__)

if __HITRANPATH__ and os.path.exists(__HITRANPATH__):
    logger.info('HITRAN file found: %s', __HITRANPATH__)
else:
    logger.warning('HITRAN file not found. Run: python -m rfmwrapper configure')

if os.path.exists(__ATMPATH__):
    logger.info('Atmosphere directory: %s', __ATMPATH__)
else:
    logger.warning('Atmosphere directory not found: %s.', __ATMPATH__)

if __ARIAPATH__ and os.path.exists(__ARIAPATH__):
    logger.info('ARIA directory: %s', __ARIAPATH__)

_possible_types = ['layer', 'zenith', 'nadir']

_std_atmospheres = {
    'FASSTD':   os.path.join(__ATMPATH__, 'std.atm'),
    'FASMLS':   os.path.join(__ATMPATH__, 'mls.atm'),
    'FASMLW':   os.path.join(__ATMPATH__, 'mlw.atm'),
    'FASSAS':   os.path.join(__ATMPATH__, 'sas.atm'),
    'FASSAW':   os.path.join(__ATMPATH__, 'saw.atm'),
    'FASTRO':   os.path.join(__ATMPATH__, 'tro.atm'),
    'FASMIN':   os.path.join(__ATMPATH__, 'minor.atm'),
    'MIPDAY':   os.path.join(__ATMPATH__, 'day.atm'),
    'MIPNGT':   os.path.join(__ATMPATH__, 'ngt.atm'),
    'MIPWIN':   os.path.join(__ATMPATH__, 'win.atm'),
    'MIPSUM':   os.path.join(__ATMPATH__, 'sum.atm'),
    'MIPEQU':   os.path.join(__ATMPATH__, 'equ.atm'),
    'MIPDAY_X': os.path.join(__ATMPATH__, 'day_200km.atm'),
    'MIPNGT_X': os.path.join(__ATMPATH__, 'ngt_200km.atm'),
    'MIPWIN_X': os.path.join(__ATMPATH__, 'win_200km.atm'),
    'MIPSU_XM': os.path.join(__ATMPATH__, 'sum_200km.atm'),
    'MIPEQU_X': os.path.join(__ATMPATH__, 'equ_200km.atm'),
    'HGT':      os.path.join(__ATMPATH__, 'hgt_std.atm'),
}

# Keep the old public name for backwards compatibility
std_atmospheres = _std_atmospheres


# ===========================================================================
# ========================= Output container ================================
# ===========================================================================

@dataclass
class Spectra:
    """
    Container for one type of RFM spectral output.

    Attributes
    ----------
    wn : np.ndarray, shape (n_wn,)
        Wavenumber axis [cm^-1].
    data : np.ndarray, shape (n_spectra, n_wn)
        Spectral data — always 2-D, even for a single spectrum.
    labels : np.ndarray, shape (n_spectra,)
        Numeric label for each spectrum row.  Interpretation depends on
        ``label_type``:

        * ``'elevation'`` — elevation angle [degrees]
        * ``'altitude'``  — output level altitude [km]  (LEV output)
        * ``'layer'``     — path length [km]
    label_type : str
        One of ``'elevation'``, ``'altitude'``, or ``'layer'``.
    output : str
        Output quantity: ``'tra'``, ``'rad'``, ``'bbt'``, or ``'opt'``.
    xunit : str
        Spectral axis unit: ``'wn'`` for wavenumber [cm⁻¹] (default) or
        ``'wl'`` for wavelength [µm].  The ``wn`` field always stores the
        spectral axis in ascending order regardless of unit.
    yunit : str
        Data unit string, e.g. ``'mW/(m²·sr·cm⁻¹)'``, ``'mW/(m²·sr·µm)'``,
        ``'K'``, or ``''`` for dimensionless quantities.
    """
    wn: np.ndarray
    data: np.ndarray
    labels: np.ndarray
    label_type: str
    output: str
    xunit: str = 'wn'
    yunit: str = ''

    def toWavelength(self) -> 'Spectra':
        """
        Convert this :class:`Spectra` from wavenumber to wavelength space.

        The output axis is always monotonically increasing (the data columns
        are flipped accordingly).  For ``output='rad'`` the Jacobian
        ``L_wl = L_wn · wn² / 1e4`` is applied so that units change from
        ``mW/(m²·sr·cm⁻¹)`` to ``mW/(m²·sr·µm)``; all other output types
        have their axis converted only.

        Returns
        -------
        Spectra
            New instance with ``xunit='wl'``.  The ``wn`` field contains
            wavelength values [µm] in ascending order.

        Raises
        ------
        ValueError
            If the Spectra is already in wavelength space (``xunit='wl'``).
        """
        if self.xunit == 'wl':
            raise ValueError("Spectra is already in wavelength space (xunit='wl').")

        wl = (1e4 / self.wn)[::-1]              # ascending [µm]
        data = self.data[:, ::-1].copy()         # flip columns to match wl order

        if self.output == 'rad':
            # Jacobian: L_wl = L_wn · wn² / 1e4
            # wn_flip[j] is the wavenumber corresponding to wl[j] after flipping
            wn_flip = self.wn[::-1]
            data = data * (wn_flip ** 2 / 1e4)
            yunit = 'mW/(m²·sr·µm)'
        else:
            yunit = self.yunit

        return Spectra(wn=wl, data=data, labels=self.labels.copy(),
                       label_type=self.label_type, output=self.output,
                       xunit='wl', yunit=yunit)

    def toWavenumber(self) -> 'Spectra':
        """
        Convert this :class:`Spectra` from wavelength to wavenumber space.

        The output axis is always monotonically increasing (the data columns
        are flipped accordingly).  For ``output='rad'`` the inverse Jacobian
        ``L_wn = L_wl · wl² / 1e4`` is applied so that units change from
        ``mW/(m²·sr·µm)`` to ``mW/(m²·sr·cm⁻¹)``; all other output types
        have their axis converted only.

        Returns
        -------
        Spectra
            New instance with ``xunit='wn'``.  The ``wn`` field contains
            wavenumber values [cm⁻¹] in ascending order.

        Raises
        ------
        ValueError
            If the Spectra is already in wavenumber space (``xunit='wn'``).
        """
        if self.xunit == 'wn':
            raise ValueError("Spectra is already in wavenumber space (xunit='wn').")

        wn = (1e4 / self.wn)[::-1]              # ascending [cm⁻¹]
        data = self.data[:, ::-1].copy()         # flip columns to match wn order

        if self.output == 'rad':
            # Inverse Jacobian: L_wn = L_wl · wl² / 1e4
            # wl_flip[j] is the wavelength corresponding to wn[j] after flipping
            wl_flip = self.wn[::-1]
            data = data * (wl_flip ** 2 / 1e4)
            yunit = 'mW/(m²·sr·cm⁻¹)'
        else:
            yunit = self.yunit

        return Spectra(wn=wn, data=data, labels=self.labels.copy(),
                       label_type=self.label_type, output=self.output,
                       xunit='wn', yunit=yunit)

    def resample(self, dx: float = None, x_out: np.ndarray = None) -> 'Spectra':
        """
        Resample this :class:`Spectra` onto a new axis.

        Thin wrapper around :func:`resampleSpectra`.  Exactly one of ``dx``
        or ``x_out`` must be supplied.

        Parameters
        ----------
        dx : float, optional
            Uniform grid spacing in the same units as ``self.wn``.
        x_out : np.ndarray, optional
            Explicit output axis; must lie within ``[self.wn[0], self.wn[-1]]``.

        Returns
        -------
        Spectra
            New instance on the resampled axis.
        """
        from .functions import resampleSpectra
        return resampleSpectra(self, dx=dx, x_out=x_out)


@dataclass
class FilterOut:
    """
    Container for band-integrated filter convolution results.

    Attributes
    ----------
    wl : np.ndarray, shape (n_bands,)
        Band centre wavelengths [µm].
    wn : np.ndarray, shape (n_bands,)
        Band centre wavenumbers [cm^-1].
    data : np.ndarray, shape (n_spectra, n_bands)
        Band-integrated values — always 2-D, even for a single spectrum.
    labels : np.ndarray, shape (n_spectra,)
        Inherited from the input :class:`Spectra` (elevation, altitude, or
        path length depending on ``label_type``).
    label_type : str
        Inherited from the input :class:`Spectra`.
    output : str
        RFM output quantity: ``'tra'``, ``'rad'``, ``'bbt'``, or ``'opt'``.
    sensor : str
        Name of the sensor used for convolution.
    band_names : list of str
        Band identifiers in column order (length ``n_bands``).
    """
    wl: np.ndarray
    wn: np.ndarray
    data: np.ndarray
    labels: np.ndarray
    label_type: str
    output: str
    sensor: str
    band_names: list


@dataclass
class ILSOut:
    """
    Container for ILS-convolved spectral output.

    Attributes
    ----------
    wn : np.ndarray, shape (n_wn,)
        Spectral axis — identical to the input :class:`Spectra`.  Units
        and direction are determined by ``xunit``.
    data : np.ndarray, shape (n_spectra, n_wn)
        ILS-convolved spectra — always 2-D, even for a single spectrum.
    labels : np.ndarray, shape (n_spectra,)
        Inherited from the input :class:`Spectra`.
    label_type : str
        Inherited from the input :class:`Spectra`.
    output : str
        RFM output quantity: ``'tra'``, ``'rad'``, ``'bbt'``, or ``'opt'``.
    ils : dict
        ILS kernel dictionary as returned by :func:`makeILS`.  Carries all
        instrument parameters (``fov``, ``max_opd``, ``apod_type``, etc.).
    xunit : str
        Inherited from the input :class:`Spectra` — ``'wn'`` or ``'wl'``.
    yunit : str
        Inherited from the input :class:`Spectra`.
    """
    wn: np.ndarray
    data: np.ndarray
    labels: np.ndarray
    label_type: str
    output: str
    ils: dict
    xunit: str = 'wn'
    yunit: str = ''

    def resample(self, dx: float = None, x_out: np.ndarray = None) -> 'ILSOut':
        """
        Resample this :class:`ILSOut` onto a new axis.

        Thin wrapper around :func:`resampleSpectra`.  Exactly one of ``dx``
        or ``x_out`` must be supplied.

        Parameters
        ----------
        dx : float, optional
            Uniform grid spacing in the same units as ``self.wn``.
        x_out : np.ndarray, optional
            Explicit output axis; must lie within ``[self.wn[0], self.wn[-1]]``.

        Returns
        -------
        ILSOut
            New instance on the resampled axis.
        """
        from .functions import resampleSpectra
        return resampleSpectra(self, dx=dx, x_out=x_out)


# ===========================================================================
# =================== Standalone filter convolution =========================
# ===========================================================================

def applyFilters(spectra: Spectra, sensor: str | dict,
                 method: str = 'avg') -> FilterOut:
    """
    Convolve a :class:`Spectra` object with imager filter response functions.

    This is a pure function: it does not modify ``spectra`` and has no side
    effects.  Radiance spectra are automatically converted to wavelength-space
    units [mW/(m²·sr·µm)] before convolution; all other output types use the
    original values with only the axis converted.

    Only bands whose centre wavelength falls within the spectral range of
    ``spectra`` are included in the output.

    Parameters
    ----------
    spectra : Spectra
        Input spectral data as returned by :meth:`RFM.getSpectra`.
    sensor : str or dict
        Name of a bundled sensor (e.g. ``'ASTER'``, ``'MMT-gasCam'``), or a
        dict mapping band name → ``{'wl': np.ndarray, 'T': np.ndarray}``.
        Use :func:`listSensors` to see available built-in sensors.
    method : str
        ``'avg'`` for transmission-weighted average (default),
        ``'integ'`` for trapezoid integration over the passband.

    Returns
    -------
    FilterOut
        ``data`` is shape ``(n_spectra, n_bands)``.  ``wl`` and ``band_names``
        describe the band axis; ``labels`` and ``label_type`` are carried over
        from the input ``spectra``.
    """
    filters = _loadFilters(sensor) if isinstance(sensor, str) else sensor
    sensor_name = sensor if isinstance(sensor, str) else 'custom'

    if method not in ('avg', 'integ'):
        raise ValueError(f"method must be 'avg' or 'integ', got {method!r}")

    # Work entirely in wavelength space — convert if needed.
    # toWavelength() owns the axis flip and Jacobian for radiance.
    s = spectra if spectra.xunit == 'wl' else spectra.toWavelength()
    wl_axis = s.wn          # ascending [µm]
    spec_wl = s.data        # (n_spectra, n_wn), rad already in mW/(m²·sr·µm)

    # Determine which bands overlap the spectral range
    wl_min, wl_max = wl_axis.min(), wl_axis.max()
    active_bands = []
    for band_name, filt in filters.items():
        T_f = filt['T']
        center = float(np.median(filt['wl'][T_f > 0.01])) if np.any(T_f > 0.01) else float(np.mean(filt['wl']))
        if wl_min < center < wl_max:
            active_bands.append((band_name, filt, center))

    if not active_bands:
        raise ValueError(
            f"No {sensor_name!r} bands overlap the spectral range "
            f"{wl_min:.3f}–{wl_max:.3f} µm."
        )

    band_names   = [b[0] for b in active_bands]
    band_centers = np.array([b[2] for b in active_bands])

    # Pre-interpolate all band transmissions onto wl_axis — shape (n_bands, n_wn)
    T_interp = np.stack([
        np.interp(wl_axis, filt['wl'], filt['T'], left=0.0, right=0.0)
        for _, filt, _ in active_bands
    ])

    # Convolve: (n_spectra, n_bands)
    if method == 'avg':
        T_sum = T_interp.sum(axis=1)                       # (n_bands,)
        data = (spec_wl @ T_interp.T) / np.where(T_sum > 0, T_sum, np.nan)
    else:  # 'integ'
        # Vectorised trapz: for each band, integrate only over passband
        data = np.full((len(spectra.data), len(active_bands)), np.nan)
        for j, (T_band,) in enumerate(zip(T_interp)):
            mask = T_band > 0
            if mask.any():
                data[:, j] = np.trapz(spec_wl[:, mask] * T_band[mask], wl_axis[mask], axis=1)

    return FilterOut(
        wl=band_centers,
        wn=1e4 / band_centers,
        data=data,
        labels=spectra.labels.copy(),
        label_type=spectra.label_type,
        output=spectra.output,
        sensor=sensor_name,
        band_names=band_names,
    )


# ===========================================================================
# =================== Standalone ILS convolution ============================
# ===========================================================================

def applyILS(spectra: Spectra, ils: dict | str) -> ILSOut:
    """
    Convolve a :class:`Spectra` object with an ILS kernel.

    This is a pure function: it does not modify ``spectra`` and has no side
    effects.  The ILS kernel is assumed to have been computed with the same
    ``nper_wn`` as the spectrum (i.e. the kernel and spectrum share a common
    wavenumber spacing), so the convolution is applied directly in sample space
    without resampling.

    Each spectrum row is convolved independently; all rows are processed in a
    single vectorised call via :func:`scipy.ndimage.convolve1d`.

    Parameters
    ----------
    spectra : Spectra
        Input spectral data as returned by :meth:`RFM.getSpectra`.
    ils : dict or str
        Either an ILS kernel dictionary as returned by :func:`makeILS`, or the
        name of a bundled instrument (e.g. ``'Midac'``).  When a string is
        given the ILS is built automatically using the instrument's stored
        parameters; ``nper_wn`` is inferred from the spectrum's wavenumber
        spacing and ``wn`` from its spectral midpoint.  Use
        :func:`listInstruments` to see available names.

    Returns
    -------
    ILSOut
        ``data`` is shape ``(n_spectra, n_wn)``, same as ``spectra.data``.
        The wavenumber axis is identical to ``spectra.wn`` — no axis shift is
        applied; the kernel is assumed to be centred at zero.
    """
    from scipy.ndimage import convolve1d

    if isinstance(ils, str):
        params   = loadILSParams(ils)
        nper_wn  = 1.0 / float(np.mean(np.diff(spectra.wn)))
        wn_mid   = float(spectra.wn[len(spectra.wn) // 2])
        ils = makeILS(fov=params['fov'], max_opd=params['max_opd'],
                      nper_wn=nper_wn, wn=wn_mid,
                      apod_type=params['apod_type'],
                      fwhm_wn=params.get('fwhm_wn', None))

    kernel = np.asarray(ils['kernel'], dtype=np.float64)
    if kernel.sum() != 0:
        kernel = kernel / kernel.sum()   # guarantee unit-sum normalisation

    # convolve1d applies the kernel along axis 1 (the spectral axis) for
    # every row simultaneously — no Python loop over spectra.
    convolved = convolve1d(spectra.data.astype(np.float64), kernel,
                           axis=1, mode='nearest')

    return ILSOut(
        wn=spectra.wn.copy(),
        data=convolved,
        labels=spectra.labels.copy(),
        label_type=spectra.label_type,
        output=spectra.output,
        ils=ils,
        xunit=spectra.xunit,
        yunit=spectra.yunit,
    )


# ===========================================================================
# ============================= RFM class ===================================
# ===========================================================================

class RFM:
    """
    Manages a single RFM forward model simulation.

    Workflow::

        model = RFM(type='zenith', ...)
        model.insertPlume(...)       # optional atmosphere modification
        model.run()                  # write driver, execute RFM, read output
        model.plotOut()              # inspect results

    Parameters
    ----------
    type : str
        Geometry: ``'layer'``, ``'zenith'``, or ``'nadir'``.
    wn_start, wn_stop : float
        Spectral range [cm^-1].
    nperwn : float
        Points per wavenumber (spectral oversampling factor).
    gas : str or ``'all'``
        Space-separated gas names, or ``'all'`` to use every species in the
        atmosphere file.
    elev : float or list of float, optional
        Elevation angle(s) [degrees].  Positive for zenith, negative for nadir.
    h_obs : float or list of float, optional
        Observer altitude(s) [km].
    levels : list of float, optional
        Additional output altitudes [km] (RFM *LEV section).
    h_surf : float, optional
        Surface altitude [km].  Defaults to 0.01 km for nadir geometry.
    surf_emiss : float, optional
        Surface emissivity [0–1].
    surf_temp : float, optional
        Surface skin temperature [K].
    pathlength : float
        Layer path length [km]. Only used for ``type='layer'``.
    pres : float
        Layer pressure [mbar].  Only used for ``type='layer'``.
    temp : float
        Layer temperature [K].  Only used for ``type='layer'``.
    vmr : dict, optional
        Gas VMRs [ppmv].  Only used / seeded for ``type='layer'``.
    add_vmr : bool
        If True and ``gas='all'``, append vmr keys to the atmosphere gas list.
    atm_path : str, optional
        Path to a custom atmosphere file.
    atm_type : str, optional
        Key into the built-in standard atmosphere table (e.g. ``'FASSTD'``).
    adjust : dict, optional
        VMR adjustments to apply after loading the atmosphere, e.g.
        ``{'CO2': 420}``.
    exec_path : str, optional
        Path to the RFM executable.
    hitran_path : str, optional
        Path to the HITRAN binary database.
    out_path : str, optional
        Directory for RFM output files.
    drv_path : str, optional
        Directory where the driver file is written.
    atm_dir : str, optional
        Directory containing standard atmosphere files.
    aria_dir : str, optional
        Directory containing ARIA aerosol data.
    silent : bool
        Suppress the summary printout at construction time.
    """

    def __init__(self, type: str, wn_start: float, wn_stop: float, nperwn: float,
                 gas='all', elev=None, h_obs=None, levels=None,
                 h_surf=None, surf_emiss=None, surf_temp=None,
                 pathlength: float = 1.0, pres: float = 1013, temp: float = 298,
                 vmr: dict = None, add_vmr: bool = False,
                 atm_path: str = None, atm_type: str = None, adjust: dict = None,
                 exec_path: str = None, hitran_path: str = None,
                 out_path: str = None, drv_path: str = None,
                 atm_dir: str = None, aria_dir: str = None,
                 silent: bool = False) -> None:

        self.silent = silent
        self.atm_type = atm_type

        if type.lower() not in _possible_types:
            raise ValueError(f"RFM type must be one of {_possible_types}")
        self.type = type.lower()

        self.wn_start = float(wn_start)
        self.wn_stop = float(wn_stop)
        self.nperwn = float(nperwn)
        self.wn_spacing = 1.0 / self.nperwn

        # ----- RFM executable -----
        if exec_path is None:
            if not os.path.exists(__EXECPATH__):
                raise ValueError('RFM executable not found. Pass exec_path= or set rfmwrapper.__EXECPATH__.')
            self.exec_path = __EXECPATH__
        elif os.path.exists(exec_path):
            self.exec_path = exec_path
        else:
            raise ValueError(f"RFM executable not found at: {exec_path}")

        # ----- HITRAN database -----
        if hitran_path is None:
            self.hitran_path = __HITRANPATH__
        elif os.path.exists(hitran_path):
            self.hitran_path = os.path.abspath(hitran_path)
        else:
            raise ValueError(f"HITRAN file not found at: {hitran_path}")

        # ----- Work directory (owns temp dirs when paths are not explicit) -----
        # A single temp directory is created when drv_path or out_path is left
        # as None.  Parallel instances each get their own tree so they never
        # write to the same files simultaneously.
        if drv_path is None or out_path is None:
            self._work_dir = tempfile.mkdtemp(prefix='rfm_')
            self._owns_work_dir = True
        else:
            self._work_dir = None
            self._owns_work_dir = False

        # ----- Driver file directory -----
        if drv_path is None:
            self.drv_path = os.path.join(self._work_dir, 'drv')
            os.makedirs(self.drv_path)
        else:
            os.makedirs(drv_path, exist_ok=True)
            self.drv_path = os.path.abspath(drv_path)

        # ----- Output directory -----
        if out_path is None:
            self.out_path = os.path.join(self._work_dir, 'out')
            os.makedirs(self.out_path)
        else:
            os.makedirs(out_path, exist_ok=True)
            self.out_path = os.path.abspath(out_path)

        # ----- Atmosphere directory -----
        if atm_dir is None:
            if os.path.exists(__ATMPATH__):
                self.atm_dir = __ATMPATH__
            else:
                self.atm_dir = os.path.join(__RFMDIR__, 'atm')
                os.makedirs(self.atm_dir, exist_ok=True)
                logger.warning('Atmosphere directory created but empty: %s', self.atm_dir)
        elif os.path.exists(atm_dir):
            self.atm_dir = atm_dir
        else:
            raise ValueError(f"Atmosphere directory not found: {atm_dir}")

        # ----- ARIA directory (optional) -----
        self.aria_dir = None
        if aria_dir is None:
            if os.path.exists(__ARIAPATH__):
                self.aria_dir = __ARIAPATH__
        elif os.path.exists(aria_dir):
            self.aria_dir = aria_dir
        elif aria_dir is not None:
            raise ValueError(f"ARIA directory not found: {aria_dir}")

        # ----- Viewing geometry -----
        if h_obs is None:
            self.h_obs = None
        elif isinstance(h_obs, list):
            self.h_obs = [float(h) for h in h_obs]
        else:
            self.h_obs = float(h_obs)

        if elev is None:
            self.elev = None
        elif isinstance(elev, list):
            self.elev = [float(e) for e in elev]
        else:
            self.elev = float(elev)

        # ----- Surface properties -----
        self.h_surf = 0.01 if (h_surf is None and self.type == 'nadir') else (
            float(h_surf) if h_surf is not None else None)
        if surf_emiss is not None:
            val = float(surf_emiss)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"surf_emiss must be in [0, 1], got {val:.4f}")
            self.surf_emiss = val
        else:
            self.surf_emiss = None
        self.surf_temp = float(surf_temp) if surf_temp is not None else None

        # ----- Levels -----
        self.levels = levels

        # ----- Atmosphere -----
        if atm_type is not None:
            if atm_type not in _std_atmospheres:
                raise ValueError(f"Unknown atm_type {atm_type!r}. Choose from: {list(_std_atmospheres)}")
            self.atm_path = _std_atmospheres[atm_type]
            if not os.path.exists(self.atm_path):
                raise ValueError(f"Missing atmosphere file for {atm_type}: {self.atm_path}")
        elif atm_path is None:
            self.atm_path = _std_atmospheres['FASSTD']
        else:
            self.atm_path = os.path.abspath(atm_path)

        if not silent:
            logger.info('Atmosphere path: %s', self.atm_path)
        self.readAtm(self.atm_path)
        self._base_atm_path = self.atm_path  # original path, never mutated
        self.atm_top = self.atm['HGT'][-1]

        if adjust is not None:
            for key, value in adjust.items():
                self.adjustAtm(key, float(value))

        # ----- Gas list -----
        atm_gases = [k for k in self.atm if k not in ('HGT', 'TEM', 'PRE', 'header')]
        if gas == 'all':
            if self.type == 'layer' and vmr is not None and not add_vmr:
                self.gas = ' '.join(vmr.keys())
            elif self.type == 'layer' and vmr is not None and add_vmr:
                self.gas = ' '.join(atm_gases + [k for k in vmr if k not in atm_gases])
            else:
                self.gas = ' '.join(atm_gases)
        else:
            self.gas = gas

        # ----- Layer parameters -----
        self.pathlength = float(pathlength)
        self.pres = float(pres)
        self.temp = float(temp)
        if vmr is None or add_vmr:
            if vmr is None:
                vmr = {}
            std = readStandard()
            for item in self.gas.split():
                if item not in vmr:
                    if item in self.atm:
                        vmr[item] = self.atm[item][0]
                    elif item in std:
                        vmr[item] = std[item]['vmr']
        self.vmr = vmr

        # ----- Summary -----
        if not silent:
            self._printSummary()

        self.ran = False

    # -----------------------------------------------------------------------
    def _printSummary(self) -> None:
        if self.type == 'layer':
            print(f'===== RFM object: "{self.type}" =====\n'
                  f'{self.pathlength * 1e3:.0f} m | {self.temp:.0f} K | {self.pres:.0f} mb\n'
                  f'wn_start = {self.wn_start:.0f} cm^-1\n'
                  f'wn_stop  = {self.wn_stop:.0f} cm^-1\n'
                  f'nperwn   = {self.nperwn:.0f}\n'
                  f'Gases: {self.gas}\n'
                  '==============================')

        elif self.type == 'zenith':
            if self.elev is None:
                print_elev = "90 deg [vertical]"
            elif isinstance(self.elev, list):
                print_elev = "[" + ",".join(f"{x:.1f}" for x in self.elev) + "] deg"
            else:
                print_elev = f"{self.elev:.1f} deg"
            if self.h_obs is None:
                print_hobs = "0.0 km"
            elif isinstance(self.h_obs, list):
                print_hobs = "[" + ",".join(f"{x:.1f}" for x in self.h_obs) + "] km"
            else:
                print_hobs = f"{self.h_obs:.1f} km"
            print(f'===== RFM object: "{self.type}" =====\n'
                  f'Elevation angle: {print_elev}\n'
                  f'Observer height: {print_hobs}\n'
                  f'wn_start = {self.wn_start:.0f} cm^-1\n'
                  f'wn_stop  = {self.wn_stop:.0f} cm^-1\n'
                  f'nperwn   = {self.nperwn:.0f}\n'
                  f'Atmosphere: {self.atm_path}\n'
                  f'Gases: {self.gas}\n'
                  '=============================')

        elif self.type == 'nadir':
            if self.elev is None:
                print_elev = "-90 deg [vertical]"
            elif isinstance(self.elev, list):
                print_elev = "[" + ",".join(f"{x:.1f}" for x in self.elev) + "] deg"
            else:
                print_elev = f"{self.elev:.1f} deg"
            if self.h_obs is None:
                print_hobs = "Top of Atmosphere"
            elif isinstance(self.h_obs, list):
                print_hobs = "[" + ",".join(f"{x:.1f}" for x in self.h_obs) + "] km"
            else:
                print_hobs = f"{self.h_obs:.1f} km"
            print(f'===== RFM object: "{self.type}" =====\n'
                  f'Viewing angle:   {print_elev}\n'
                  f'Observer height: {print_hobs}\n'
                  f'wn_start = {self.wn_start:.0f} cm^-1\n'
                  f'wn_stop  = {self.wn_stop:.0f} cm^-1\n'
                  f'nperwn   = {self.nperwn:.0f}\n'
                  f'Atmosphere: {self.atm_path}\n'
                  f'Gases: {self.gas}\n'
                  '=============================')

    # -----------------------------------------------------------------------
    def readAtm(self, fname: str) -> None:
        """Read an RFM atmosphere file and store it in ``self.atm``."""
        self.atm = _readAtm(fname)

    # -----------------------------------------------------------------------
    def writeAtm(self, outname: str, header: str = "") -> None:
        """Write ``self.atm`` to disk in RFM atmosphere format."""
        _writeAtm(self.atm, outname, header=header)

    # -----------------------------------------------------------------------
    def adjustAtm(self, gas: str, value: float) -> None:
        """
        Scale or insert a gas profile so its surface-level VMR equals ``value``.

        Delegates to :func:`rfmwrapper.adjustAtm`, then persists the result to
        the instance work directory and updates ``self.atm`` and
        ``self.atm_path``.
        """
        self.atm = _adjustAtm(self.atm, gas, value)
        atm_fname = os.path.basename(self._base_atm_path).replace('.atm', '_adjusted.atm')
        atm_dir = self._work_dir if self._work_dir else os.path.dirname(self._base_atm_path)
        self.atm_path = os.path.join(atm_dir, atm_fname)
        self.writeAtm(self.atm_path)
        logger.info('Atmosphere adjusted: %s → %.1f ppm. Written to %s', gas, value, self.atm_path)

    # -----------------------------------------------------------------------
    def insertPlume(self, h: float, th: float, plume: dict = None,
                    gradient_flag: int = 1) -> None:
        """
        Insert a gas plume layer into the loaded atmosphere.

        Delegates to :func:`rfmwrapper.insertPlume`, then persists the result
        and updates ``self.atm``, ``self.atm_path``, and ``self.gas``.

        Parameters
        ----------
        h : float
            Centre altitude of the plume [km].
        th : float
            Vertical thickness [km].
        plume : dict, optional
            Additive VMR enhancements by species [ppmv].
        gradient_flag : int
            Concentration profile shape: 0=uniform, 1=triangular, 2=Gaussian.
        """
        if plume is None:
            plume = {}
        # Add any new plume species to the gas list before calling the function
        for key in plume:
            if key not in self.gas.split():
                self.gas = f"{self.gas} {key}"
        self.atm = _insertPlume(self.atm, h, th, plume=plume, gradient_flag=gradient_flag)
        atm_fname = os.path.basename(self._base_atm_path).replace('.atm', '_plume.atm')
        atm_dir = self._work_dir if self._work_dir else os.path.dirname(self._base_atm_path)
        self.atm_path = os.path.join(atm_dir, atm_fname)
        self.writeAtm(self.atm_path)

    # -----------------------------------------------------------------------
    def sliceAtm(self, h1: float, h2: float) -> None:
        """
        Trim the loaded atmosphere to the altitude range ``[h1, h2]``.

        Delegates to :func:`rfmwrapper.sliceAtm`, then persists the result
        and updates ``self.atm`` and ``self.atm_path``.

        Parameters
        ----------
        h1 : float
            Bottom altitude of the slice [km].
        h2 : float
            Top altitude of the slice [km].
        """
        self.atm = _sliceAtm(self.atm, h1, h2)
        atm_fname = os.path.basename(self._base_atm_path).replace('.atm', '_slice.atm')
        atm_dir = self._work_dir if self._work_dir else os.path.dirname(self._base_atm_path)
        self.atm_path = os.path.join(atm_dir, atm_fname)
        self.writeAtm(self.atm_path)

    # -----------------------------------------------------------------------
    def plotAtm(self, title: str = 'RFM atmosphere', color='k', zoom=None) -> None:
        """
        Visualise the atmosphere associated with this RFM instance.

        For ``type='layer'`` a text table is printed; for zenith/nadir a
        multi-panel profile plot is shown.
        """
        if self.type == 'layer':
            header = "{:^10} | {:^15}".format("GAS", "VMR [ppmv]")
            sep = "-" * len(header)
            print(sep)
            print("{:^28}".format('Homogeneous Layer'))
            print("{:^28}".format(f"Thickness: {self.pathlength * 1000:.0f} m"))
            print("{:^28}".format(f"Temperature: {self.temp:.1f} K"))
            print("{:^28}".format(f"Pressure: {self.pres:.0f} mb"))
            print(sep)
            print(header)
            print(sep)
            for gas in self.gas.split():
                print("{:<10} | {:<15}".format(gas, str(self.vmr[gas])))
            print(sep)
            return

        atm = dict(self.atm)
        hgt = atm.pop('HGT')
        atm.pop('header', None)
        for key in [k for k in list(atm) if k not in rfm_gases + ['PRE', 'TEM']]:
            atm.pop(key)

        n = max(len(atm) // 2, 1)
        fig, axes = plt.subplots(2, n, figsize=[n * 2, 8.0], squeeze=False, tight_layout=True)
        axes = axes.flatten()
        fig.suptitle(title)

        c = 'k'
        for i, key in enumerate(atm):
            if i >= len(axes):
                break
            if color == 'all':
                c = plot_colors.get(key, 'k')
            elif isinstance(color, str):
                c = color
            ax = axes[i]
            if zoom is not None:
                cs = CubicSpline(hgt, atm[key])
                ax.plot(cs(np.arange(zoom[0], zoom[1])), np.arange(zoom[0], zoom[1]), c=c)
            else:
                ax.plot(atm[key], hgt, c=c)
            if key in atm_fields:
                ax.set(title=atm_fields[key], xlabel=atm_units[key], ylabel='Height [km]')
            else:
                ax.set(title=pretty_names.get(key, key), xlabel='[ppm]', ylabel='Height [km]')

        if self.type in ('zenith', 'nadir'):
            if self.type == 'nadir':
                h_obs_eff = min(self.h_obs, self.atm_top) if self.h_obs is not None else self.atm_top
            else:
                h_obs_eff = self.h_obs if self.h_obs is not None else 0.0

            legend_handles = [
                Line2D([0], [0], color='tab:blue', linestyle='--', lw=1,
                       label=f"Observer {h_obs_eff:.1f} km")
            ]
            for ax in axes[:len(atm)]:
                ax.axhline(h_obs_eff, color='tab:blue', linestyle='--', linewidth=1)
                if self.type == 'nadir' and self.h_surf is not None:
                    ax.axhline(self.h_surf, color='tab:red', linestyle='--', linewidth=1)
            if self.type == 'nadir' and self.h_surf is not None:
                legend_handles.append(
                    Line2D([0], [0], color='tab:red', linestyle='--', lw=1,
                           label=f"Surface {self.h_surf:.2f} km")
                )
            axes[0].legend(handles=legend_handles, fontsize=8, loc='best')

        for i in range(len(atm), len(axes)):
            axes[i].remove()

        plt.show(block=False)

    # -----------------------------------------------------------------------
    def writeDrv(self, out: str = "RAD BBT OPT TRA", CTM: bool = True,
                 MIX: bool = True, PRF: bool = False) -> None:
        """
        Write the RFM driver file to ``self.drv_path/rfm.drv``.

        Parameters
        ----------
        out : str
            Space-separated list of output flags passed to RFM (e.g. ``"RAD BBT OPT TRA"``).
        CTM : bool
            Enable continuum model (H2O CTM flag).  Applied only when H2O is in the gas list.
        MIX : bool
            Enable line mixing (CO2 MIX flag).  Applied only when CO2 is in the gas list.
        PRF : bool
            If True, write a profile output file (``prf.atm``) in addition to spectral output.
        """
        # When surf_emiss < 1, RFM redefines TRA and OPT as the reflected-path
        # transmittance (1-ε)·T_atm², not one-way atmospheric transmittance.
        if (self.type == 'nadir'
                and self.surf_emiss is not None
                and self.surf_emiss < 1.0):
            requested = set(out.upper().split())
            affected  = requested & {'TRA', 'OPT'}
            if affected:
                logger.warning(
                    "surf_emiss=%.4f < 1: RFM will output the reflected-path "
                    "transmittance (1-ε)·T_atm² for %s, not one-way atmospheric "
                    "transmittance. Set surf_emiss=1.0 to recover T_atm.",
                    self.surf_emiss, ', '.join(sorted(affected))
                )

        fname = os.path.join(self.drv_path, 'rfm.drv')
        with open(fname, 'w') as drv:
            drv.write("! RFM driver table for radiance calculation\n\n")

            drv.write("*HDR\n")
            labels = {'layer': 'Single homogeneous layer',
                      'zenith': 'Upward-looking observation',
                      'nadir': 'Downward-looking observation'}
            drv.write(labels[self.type] + "\n\n")

            # *FLG
            drv.write("*FLG\n")
            flags = {'layer': 'HOM', 'zenith': 'ZEN OBS', 'nadir': 'NAD OBS'}[self.type]
            if self.type == 'nadir':
                flags += ' SFC'
            if 'H2O' in self.gas.split() and CTM:
                flags += ' CTM'
            if 'CO2' in self.gas.split() and MIX:
                flags += ' MIX'
            flags += ' ' + out
            if PRF:
                flags += ' PRF'
            if self.levels is not None and self.type != 'layer':
                flags += ' LEV'
            flags += ' SHH'
            drv.write(flags + "\n\n")

            # *SPC
            drv.write(f"*SPC\n{self.wn_start:f}\t{self.wn_stop:f}\t{self.nperwn:f}\n\n")

            # *GAS
            drv.write(f"*GAS\n{self.gas}\n\n")

            # *ATM
            drv.write("*ATM\n")
            if self.type == 'layer':
                drv.write(f"TEM={self.temp:f}\nPRE={self.pres:f}\n")
                for item in self.gas.split():
                    drv.write(f"{item}={self.vmr[item]:f}\n")
            else:
                if self.atm_type is not None and 'MIP' in self.atm_type:
                    drv.write(f"{_std_atmospheres['HGT']}\n")
                drv.write(f"{self.atm_path}")
            drv.write("\n\n")

            if self.type == 'layer':
                drv.write(f"*LEN\n{self.pathlength:f}\n\n")

            elif self.type == 'zenith':
                if self.elev is not None:
                    drv.write("*ELE\n")
                    if isinstance(self.elev, list):
                        if not all(0.1 <= x <= 90 for x in self.elev):
                            raise ValueError("Zenith elevation angles must be in [0.1, 90] deg")
                        drv.write(', '.join(f"{x:f}" for x in self.elev))
                    else:
                        if not 0.1 <= self.elev <= 90:
                            raise ValueError(f"Zenith elevation must be in [0.1, 90] deg, got {self.elev:.2f}")
                        drv.write(f"{self.elev:f}")
                    drv.write("\n\n")
                else:
                    drv.write("*TAN\n1\n\n")

                hgtobs_zen = self.h_obs if self.h_obs is not None else 0.0
                drv.write(f"*OBS\nHGTOBS={hgtobs_zen:f}\n\n")

            elif self.type == 'nadir':
                drv.write("*ELE\n")
                if self.elev is not None:
                    if isinstance(self.elev, list):
                        if not all(-90 <= x <= -0.1 for x in self.elev):
                            raise ValueError("Nadir elevation angles must be in [-90, -0.1] deg")
                        drv.write(', '.join(f"{x:f}" for x in self.elev))
                    else:
                        if not -90 <= self.elev <= -0.1:
                            raise ValueError(f"Nadir elevation must be in [-90, -0.1] deg, got {self.elev:.2f}")
                        drv.write(f"{self.elev:f}")
                else:
                    drv.write('-90.000000')
                drv.write("\n\n")

                hgtobs = self.h_obs if self.h_obs is not None else self.atm_top
                hgtobs = min(hgtobs, self.atm_top)
                drv.write(f"*OBS\nHGTOBS={hgtobs:f}\n\n")

                hgtsfc = self.h_surf if self.h_surf is not None else 0.01
                drv.write(f"*SFC\nHGTSFC={hgtsfc:f}\n")
                if self.surf_temp is not None:
                    drv.write(f"TEMSFC={self.surf_temp:f}\n")
                if self.surf_emiss is not None:
                    drv.write(f"EMSSFC={self.surf_emiss:f}\n")
                drv.write("\n")

            # *HIT
            drv.write(f"*HIT\n{self.hitran_path}\n\n")

            # *LEV
            if self.levels is not None:
                levels_str = ' '.join(str(x) for x in self.levels)
                drv.write(f"*LEV\n{levels_str}  ! Additional levels for output\n\n")

            # *OUT
            out_dir = self.out_path if self.out_path.endswith(os.sep) else self.out_path + os.sep
            drv.write(f"*OUT\nOUTDIR = {out_dir}\n")
            for item in out.split():
                drv.write(f"{item}FIL = {item.lower()}*.out\n")
            if PRF:
                drv.write("PRFFIL = prf.atm\n")
            drv.write("\n")

            drv.write("*END\r")

    # -----------------------------------------------------------------------
    def run(self, silent: bool = False) -> None:
        """
        Write the driver file, execute RFM, and read the output.

        Parameters
        ----------
        silent : bool
            If True, suppress RFM terminal output for this call.
        """
        if silent:
            self.silent = True

        for f in glob(os.path.join(self.out_path, '*.out')):
            os.remove(f)

        self.writeDrv()

        kwargs = {'cwd': self.drv_path, 'stdout': DEVNULL} if self.silent else {'cwd': self.drv_path}
        result = run(self.exec_path, **kwargs)

        if result.returncode != 0:
            raise RuntimeError(
                f"RFM exited with code {result.returncode}. Check {self.drv_path}/rfm.log for details."
            )

        self.ran = True
        self._readOut()

    # -----------------------------------------------------------------------
    def _readOut(self) -> None:
        """
        Parse all RFM output files from ``self.out_path`` into ``self._out``.

        Internal — called automatically by :meth:`run`.  Each entry in
        ``self._out`` is a list of raw file dicts; public access is via
        :meth:`getSpectra` and :meth:`getSpectrum`.
        """
        def _parse_one(fname):
            """Parse a single RFM .out file into a raw dict."""
            with open(fname, 'r') as f:
                lines = f.readlines()

            rec = {'fname': os.path.basename(fname), 'header': lines[0:3]}
            grid = lines[3].split()
            rec['wn_start']   = float(grid[1])
            rec['wn_stop']    = float(grid[3])
            rec['resolution'] = float(grid[2])
            _type_map = {'Optical': 'opt', 'Radiance': 'rad',
                         'Bright.': 'bbt', 'Transmission': 'tra'}
            rec['output'] = _type_map[grid[4][1:]]

            # Decode label value(s) from filename
            # RFM encodes angle/level as integers × 1000 in the stem, e.g.:
            #   tra90000.out        → elev 90.0 deg  (base height)
            #   tra90000_02500up.out → elev 90.0, level 2.5 km
            #   tra00000.out        → layer (pathlength encoded as 0)
            stem = rec['fname'].split('.')[0]
            is_layer = 'layer' in ''.join(l.strip() for l in rec['header'])
            if '_' in stem:
                left, right = stem.split('_', 1)
                n1 = float(left.replace(rec['output'], ''))
                n2 = float(right.replace('up', ''))
                rec['label_type']  = 'altitude'
                rec['label_value'] = n2 / 1e3        # km
                rec['elev']        = n1 / 1e3        # deg (context)
            elif is_layer:
                n1 = float(stem.replace(rec['output'], ''))
                rec['label_type']  = 'layer'
                rec['label_value'] = abs(n1) / 1e3   # km pathlength
                rec['elev']        = None
            else:
                n1 = float(stem.replace(rec['output'], ''))
                rec['label_type']  = 'elevation'
                rec['label_value'] = n1 / 1e3        # deg
                rec['elev']        = n1 / 1e3

            data = np.asarray([float(x) for x in lines[4].split()])
            rec['wn'] = np.linspace(rec['wn_start'], rec['wn_stop'], len(data))
            # RFM writes radiance in units × 100; normalise to mW/(m²·sr·cm⁻¹)
            if rec['output'] == 'rad':
                data = data / 100.0
            rec['data'] = data
            return rec

        if not self.ran:
            raise RuntimeError('RFM has not been run yet.')
        if not glob(os.path.join(self.out_path, '*.out')):
            raise RuntimeError(f"No output files found in: {self.out_path}")

        self._out = {}
        for prefix in ('opt', 'rad', 'bbt', 'tra'):
            matches = sorted(glob(os.path.join(self.out_path, '%s*.out' % prefix)))
            if matches:
                self._out[prefix] = [_parse_one(f) for f in matches]

    # -----------------------------------------------------------------------
    def getSpectra(self, output: str) -> 'Spectra':
        """
        Return all spectra of one output type as a :class:`Spectra` container.

        Parameters
        ----------
        output : str
            Output quantity: ``'tra'``, ``'rad'``, ``'bbt'``, or ``'opt'``.

        Returns
        -------
        Spectra
            ``data`` is always shape ``(n_spectra, n_wn)``, even for a single
            spectrum.  ``labels`` and ``label_type`` describe the second axis
            (elevation angles, altitudes, or path length).
        """
        if not self.ran:
            raise RuntimeError('RFM has not been run yet.')
        key = output.lower()
        if key not in self._out:
            raise KeyError(f"Output {key!r} not available. Run with: {list(self._out)}")
        recs = self._out[key]
        wn     = recs[0]['wn']
        data   = np.vstack([r['data'] for r in recs])   # (n_spectra, n_wn)
        labels = np.array([r['label_value'] for r in recs])
        label_type = recs[0]['label_type']
        _yunit = {'rad': 'mW/(m²·sr·cm⁻¹)', 'bbt': 'K', 'tra': '', 'opt': ''}.get(key, '')
        return Spectra(wn=wn, data=data, labels=labels,
                       label_type=label_type, output=key,
                       xunit='wn', yunit=_yunit)

    # -----------------------------------------------------------------------
    def getSpectrum(self, output: str, index: int = 0) -> tuple:
        """
        Return a single spectrum as a ``(wn, data)`` tuple.

        Convenience wrapper around :meth:`getSpectra` for the common
        single-spectrum case.

        Parameters
        ----------
        output : str
            Output quantity: ``'tra'``, ``'rad'``, ``'bbt'``, or ``'opt'``.
        index : int
            Index into the spectra array (default 0).

        Returns
        -------
        wn : np.ndarray, shape (n_wn,)
        data : np.ndarray, shape (n_wn,)
        """
        s = self.getSpectra(output)
        return s.wn, s.data[index]

    # -----------------------------------------------------------------------
    def plotOut(self, which: str = 'all', reverse: bool = False,
                color=None, rad_wl: bool = False) -> None:
        """
        Plot RFM output spectra.

        Parameters
        ----------
        which : str
            Output quantity to plot (``'tra'``, ``'rad'``, ``'bbt'``, ``'opt'``),
            or ``'all'`` to show every available output in a grid.
        reverse : bool
            If True, invert the x-axis (useful for wavelength-ordered plots).
        color : str or None
            Matplotlib color string.  ``None`` cycles through the default palette.
        rad_wl : bool
            If True, convert radiance output to wavelength space [mW/(m²·sr·µm)]
            and invert the x-axis.
        """
        ylabels = {
            'opt': 'Optical Depth',
            'rad': r'Radiance [$mW\,(cm^{2}\cdot sr \cdot cm^{-1})^{-1}$]',
            'rad_wl': r'Radiance [$mW\,(m^{2}\cdot sr \cdot µm)^{-1}$]',
            'tra': 'Transmission',
            'bbt': 'Brightness Temperature [K]',
        }
        label_units = {'elevation': 'deg', 'altitude': 'km', 'layer': 'km'}

        def _plot_spectra(s, ax, reverse=False, color=None):
            for i, row in enumerate(s.data):
                lbl = f"{s.labels[i]:.2f} {label_units.get(s.label_type, '')}"
                if s.output == 'rad' and rad_wl:
                    wl = 1e4 / s.wn
                    y  = row * s.wn ** 2 / 1e4
                    ax.plot(wl, y, label=lbl, color=color)
                    ylabel = ylabels['rad_wl']
                else:
                    ax.plot(s.wn, row, label=lbl, color=color)
                    ylabel = ylabels[s.output]

            if s.wn.max() < 1e4:
                fw = lambda x: 1e4 / x
                wlax = ax.secondary_xaxis('top', functions=(fw, fw))
                wlax.set_xlabel(r'Wavelength [$\mu$m]')
            else:
                fw = lambda x: 1e7 / x
                wlax = ax.secondary_xaxis('top', functions=(fw, fw))
                wlax.set_xlabel(r'Wavelength [$nm$]')
            wlax.minorticks_on()

            if reverse or (s.output == 'rad' and rad_wl):
                ax.invert_xaxis()
            if (s.output == 'opt'
                    and s.data.max() / max(float(s.data.min()), 1e-30) > 1e3):
                ax.set_yscale('log')
            ax.set(xlabel=r'Wavenumber [$cm^{-1}$]', ylabel=ylabel)
            ax.legend(fontsize=8)

        if not self.ran:
            raise RuntimeError('RFM has not been run yet.')

        if which.lower() == 'all':
            keys = list(self._out)
            nplots = len(keys)
            if max(len(self._out[k]) for k in keys) > 1:
                color = None
            ncols = int(np.ceil(np.sqrt(nplots)))
            nrows = int(np.ceil(nplots / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=[ncols * 5, nrows * 4])
            for i, key in enumerate(keys):
                _plot_spectra(self.getSpectra(key), np.asarray(axes).flatten()[i],
                              reverse=reverse, color=color)
            fig.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots()
            _plot_spectra(self.getSpectra(which.lower()), ax, reverse=reverse, color=color)
            fig.tight_layout()
            plt.show()

    # -----------------------------------------------------------------------
    def applyFilters(self, sensor: str = 'all', method: str = 'avg') -> None:
        """
        Convolve all RFM output spectra with imager filter response functions.

        Delegates to the standalone :func:`applyFilters` for each output type
        and each sensor.  Results are stored internally and accessed via
        :meth:`getFilterOut`.

        Parameters
        ----------
        sensor : str
            Name of a bundled sensor (e.g. ``'ASTER'``, ``'MMT-gasCam'``), or
            ``'all'`` to apply every available sensor.  Use
            :func:`rfmwrapper.listSensors` to see what is available.
        method : str
            ``'avg'`` (transmission-weighted average) or
            ``'integ'`` (trapezoid integral over the passband).
        """
        if not self.ran:
            raise RuntimeError("RFM has not been run yet.")

        sensors = listSensors() if sensor == 'all' else [sensor]
        self._filterout = {}

        for sensor_name in sensors:
            self._filterout[sensor_name] = {}
            for key in self._out:
                self._filterout[sensor_name][key] = applyFilters(
                    self.getSpectra(key), sensor_name, method=method
                )

    # -----------------------------------------------------------------------
    def getFilterOut(self, sensor: str, output: str) -> FilterOut:
        """
        Return filter convolution results for one sensor and output type.

        Parameters
        ----------
        sensor : str
            Sensor name used when calling :meth:`applyFilters`.
        output : str
            Output quantity: ``'tra'``, ``'rad'``, ``'bbt'``, or ``'opt'``.

        Returns
        -------
        FilterOut
            ``data`` is shape ``(n_spectra, n_bands)``.
        """
        if not hasattr(self, '_filterout'):
            raise RuntimeError("applyFilters() has not been called yet.")
        if sensor not in self._filterout:
            raise KeyError(f"Sensor {sensor!r} not found. Available: {list(self._filterout)}")
        key = output.lower()
        if key not in self._filterout[sensor]:
            raise KeyError(f"Output {key!r} not found for sensor {sensor!r}. Available: {list(self._filterout[sensor])}")
        return self._filterout[sensor][key]

    # -----------------------------------------------------------------------
    def applyILS(self, instrument: str = None, fov: float = 0.03,
                 max_opd: float = 1.8, apod_type: str = 'NB medium',
                 fwhm_wn: float = None) -> None:
        """
        Convolve all RFM output spectra with an Instrument Line Shape kernel.

        The ILS kernel is computed via :func:`makeILS` using the centre
        wavenumber of this instance's spectral range and the ``nperwn``
        sampling density set at construction.  Results are stored internally
        and accessed via :meth:`getILSOut`.

        Parameters
        ----------
        instrument : str, optional
            Name of a bundled instrument (e.g. ``'Midac'``).  When given,
            ``fov``, ``max_opd``, ``apod_type``, and ``fwhm_wn`` are loaded
            from the instrument library and the explicit keyword arguments are
            ignored.  Use :func:`listInstruments` to see available names.
        fov : float
            Full-angle field of view [radians].  Ignored when ``instrument``
            is specified.
        max_opd : float
            Maximum optical path difference [cm].  Ignored when ``instrument``
            is specified.
        apod_type : str
            Apodization function name.  Ignored when ``instrument`` is
            specified.  See :func:`makeILS` for options.
        fwhm_wn : float, optional
            Target spectral FWHM in cm⁻¹; only used for Gaussian apodization.
            Ignored when ``instrument`` is specified.
        """
        if not self.ran:
            raise RuntimeError("RFM has not been run yet.")

        if instrument is not None:
            params    = loadILSParams(instrument)
            fov       = params['fov']
            max_opd   = params['max_opd']
            apod_type = params['apod_type']
            fwhm_wn   = params.get('fwhm_wn', None)

        wn_centre = (self.wn_start + self.wn_stop) / 2.0
        ils = makeILS(fov=fov, max_opd=max_opd, nper_wn=self.nperwn,
                      wn=wn_centre, apod_type=apod_type, fwhm_wn=fwhm_wn)

        self._ilsout = {}
        for key in self._out:
            self._ilsout[key] = applyILS(self.getSpectra(key), ils)

    # -----------------------------------------------------------------------
    def getILSOut(self, output: str) -> ILSOut:
        """
        Return ILS-convolved spectra for one output type.

        Parameters
        ----------
        output : str
            Output quantity: ``'tra'``, ``'rad'``, ``'bbt'``, or ``'opt'``.

        Returns
        -------
        ILSOut
            ``data`` is shape ``(n_spectra, n_wn)``.
        """
        if not hasattr(self, '_ilsout'):
            raise RuntimeError("applyILS() has not been called yet.")
        key = output.lower()
        if key not in self._ilsout:
            raise KeyError(f"Output {key!r} not found. Available: {list(self._ilsout)}")
        return self._ilsout[key]

    # -----------------------------------------------------------------------
    def cleanup(self) -> None:
        """
        Remove the temporary work directory created for this instance.

        Only has an effect when ``drv_path`` and/or ``out_path`` were not
        supplied at construction time (i.e. the instance owns its temp dir).
        Safe to call multiple times.
        """
        if self._owns_work_dir and self._work_dir and os.path.exists(self._work_dir):
            shutil.rmtree(self._work_dir)
            self._work_dir = None
            self._owns_work_dir = False

    def __enter__(self) -> 'RFM':
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

    # -----------------------------------------------------------------------
    def getParameters(self) -> dict:
        """Return the instance attribute dictionary."""
        return self.__dict__


# ===========================================================================
# ========================= Parallel batch runner ===========================
# ===========================================================================

def _run_one(rfm: RFM) -> RFM:
    """Worker function: run a single RFM instance and return it."""
    rfm.run(silent=True)
    return rfm


def runBatch(rfm_list: list, max_workers: int = None, cleanup: bool = True) -> list:
    """
    Run a list of RFM instances in parallel using separate processes.

    Every instance must have a unique ``drv_path`` and ``out_path`` (which is
    guaranteed automatically when neither path is passed at construction time).

    Parameters
    ----------
    rfm_list : list of RFM
        Instances to run.  Each must be fully configured (atmosphere loaded,
        plumes inserted, etc.) before calling this function.
    max_workers : int, optional
        Maximum number of parallel worker processes.  Defaults to
        ``os.cpu_count()``.
    cleanup : bool
        If True (default), call ``cleanup()`` on each instance that owns a
        temporary work directory after its output has been read back into
        ``rfm.out``.  Set to False to inspect the temp files after the run.

    Returns
    -------
    list of RFM
        The same instances, in the same order, each with ``rfm.out`` populated.

    Raises
    ------
    RuntimeError
        If any worker raises an exception; the message includes which index
        failed and the original traceback.

    Examples
    --------
    >>> jobs = [
    ...     RFM('layer', 2000, 2600, 10, gas='SO2', vmr={'SO2': scd})
    ...     for scd in np.linspace(10, 1000, 16)
    ... ]
    >>> results = runBatch(jobs, max_workers=8)
    >>> spectra = [r.out['tra'][0]['data'] for r in results]
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    # Map from future → original list index so results come back in order
    index_map = {}
    results = [None] * len(rfm_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, rfm in enumerate(rfm_list):
            fut = executor.submit(_run_one, rfm)
            index_map[fut] = i

        for fut in as_completed(index_map):
            i = index_map[fut]
            try:
                completed = fut.result()
            except Exception as exc:
                raise RuntimeError(f"RFM worker {i} failed: {exc}") from exc
            results[i] = completed
            if cleanup:
                completed.cleanup()

    return results
