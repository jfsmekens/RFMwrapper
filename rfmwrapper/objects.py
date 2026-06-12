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
import pandas as pd

from .config import (get_rfm_dir, get_hitran_path, get_xsc_dir,
                     get_exec_path, get_atm_dir, get_aria_dir)
from .functions import (listSensors, _loadFilters, makeILS,
                        loadILSParams,
                        readAtm as _readAtm, writeAtm as _writeAtm,
                        adjustAtm as _adjustAtm,
                        insertPlume as _insertPlume,
                        sliceAtm as _sliceAtm)

logger = logging.getLogger(__name__)

# ===========================================================================
# ========================= Module-level path config ========================
# ===========================================================================

__RFMDIR__    = get_rfm_dir()
__EXECPATH__  = get_exec_path()
__HITRANPATH__ = get_hitran_path()
__ATMPATH__   = get_atm_dir()
__ARIAPATH__  = get_aria_dir()
__XSCPATH__   = get_xsc_dir()

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

_possible_types = ['layer', 'zenith', 'nadir', 'limb']

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

    Convolution is performed in wavenumber space: ``∫L(ν)·T(ν)dν / ∫T(ν)dν``.
    No Jacobian correction is applied — this is the physically correct
    radiometric integral for all output types.

    Only bands whose passband overlaps the spectral range of ``spectra`` are
    included in the output.

    Parameters
    ----------
    spectra : Spectra
        Input spectral data as returned by :meth:`RFM.getSpectra`.
    sensor : str or dict
        Name of a bundled sensor/platform (e.g. ``'ASTER'``,
        ``'GOES-ABI/GOES-16'``), or a dict mapping band name →
        ``{'wn': np.ndarray, 'T': np.ndarray}``.
        Use :func:`listSensors` to see available built-in sensors.
    method : str
        ``'avg'`` for transmission-weighted average (default),
        ``'integ'`` for trapezoid integration over the passband.

    Returns
    -------
    FilterOut
        ``data`` is shape ``(n_spectra, n_bands)``.  ``wn`` and ``wl``
        describe the band centre axis; ``band_names`` lists band labels;
        ``labels`` and ``label_type`` are carried over from ``spectra``.
    """
    filters = _loadFilters(sensor) if isinstance(sensor, str) else sensor
    sensor_name = sensor if isinstance(sensor, str) else 'custom'

    if method not in ('avg', 'integ'):
        raise ValueError(f"method must be 'avg' or 'integ', got {method!r}")

    wn_axis = spectra.wn      # [cm⁻¹], ascending
    spec_wn = spectra.data    # (n_spectra, n_wn)

    # Determine which bands overlap the spectral range
    wn_min, wn_max = wn_axis.min(), wn_axis.max()
    active_bands = []
    for band_name, filt in filters.items():
        wn_f = filt['wn']
        T_f  = filt['T']
        passband = wn_f[T_f > 0.01]
        center_wn = float(np.median(passband)) if len(passband) else float(np.mean(wn_f))
        band_min = passband.min() if len(passband) else wn_f.min()
        band_max = passband.max() if len(passband) else wn_f.max()
        if band_min < wn_max and band_max > wn_min:
            active_bands.append((band_name, filt, center_wn))

    if not active_bands:
        raise ValueError(
            f"No {sensor_name!r} bands overlap the spectral range "
            f"{wn_min:.1f}–{wn_max:.1f} cm⁻¹."
        )

    band_names      = [b[0] for b in active_bands]
    band_centers_wn = np.array([b[2] for b in active_bands])

    # Pre-interpolate all band transmissions onto wn_axis — shape (n_bands, n_wn)
    T_interp = np.stack([
        np.interp(wn_axis, filt['wn'], filt['T'], left=0.0, right=0.0)
        for _, filt, _ in active_bands
    ])

    # Convolve: (n_spectra, n_bands)
    if method == 'avg':
        T_sum = T_interp.sum(axis=1)                       # (n_bands,)
        data = (spec_wn @ T_interp.T) / np.where(T_sum > 0, T_sum, np.nan)
    else:  # 'integ'
        data = np.full((len(spectra.data), len(active_bands)), np.nan)
        for j, T_band in enumerate(T_interp):
            mask = T_band > 0
            if mask.any():
                data[:, j] = np.trapezoid(spec_wn[:, mask] * T_band[mask],
                                           wn_axis[mask], axis=1)

    return FilterOut(
        wl=1e4 / band_centers_wn,
        wn=band_centers_wn,
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

def applyILS(spectra: Spectra, ils: dict | str,
             resample: bool = False) -> ILSOut:
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
    resample : bool
        If ``True``, resample the convolved output onto the instrument's native
        output grid (``spacing`` from the instrument library).  Requires
        ``ils`` to be a named instrument string; raises ``ValueError`` if a
        dict is passed.

    Returns
    -------
    ILSOut
        ``data`` is shape ``(n_spectra, n_wn)``, same as ``spectra.data``
        (or ``(n_spectra, n_wn_resampled)`` when ``resample=True``).
        The wavenumber axis is identical to ``spectra.wn`` unless
        ``resample=True``; the kernel is assumed to be centred at zero.

    Raises
    ------
    ValueError
        If ``resample=True`` and ``ils`` is a dict (no spacing available), or
        if the named instrument has no ``spacing`` field defined.
    """
    from scipy.ndimage import convolve1d

    params = None
    if isinstance(ils, str):
        params  = loadILSParams(ils)
        nper_wn = 1.0 / float(np.mean(np.diff(spectra.wn)))
        wn_mid  = float(spectra.wn[len(spectra.wn) // 2])
        ils = makeILS(fov=params['fov'], max_opd=params['max_opd'],
                      nper_wn=nper_wn, wn=wn_mid,
                      apod_type=params['apod_type'],
                      fwhm_wn=params.get('fwhm_wn', None))
    elif resample:
        raise ValueError(
            "resample=True requires a named instrument string so that the "
            "native output spacing can be looked up from instruments.json. "
            "Pass resample=False and call .resample(dx=...) manually."
        )

    kernel = np.asarray(ils['kernel'], dtype=np.float64)
    if kernel.sum() != 0:
        kernel = kernel / kernel.sum()   # guarantee unit-sum normalisation

    # convolve1d applies the kernel along axis 1 (the spectral axis) for
    # every row simultaneously — no Python loop over spectra.
    convolved = convolve1d(spectra.data.astype(np.float64), kernel,
                           axis=1, mode='nearest')

    ilsout = ILSOut(
        wn=spectra.wn.copy(),
        data=convolved,
        labels=spectra.labels.copy(),
        label_type=spectra.label_type,
        output=spectra.output,
        ils=ils,
        xunit=spectra.xunit,
        yunit=spectra.yunit,
    )

    if resample:
        if 'spacing' not in params:
            raise ValueError(
                f"Instrument has no 'spacing' defined in instruments.json; "
                "cannot resample automatically."
            )
        return ilsout.resample(dx=params['spacing'])
    return ilsout


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
        Surface altitude [km], must be > 0.  Defaults to 0.01 km for nadir
        geometry.  RFM requires the surface to be strictly above the lowest
        atmosphere grid level (typically 0.0 km); ``h_surf=0.0`` will raise
        a ``ValueError``.
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
    tan_heights : list of float, optional
        Tangent heights [km] for ``type='limb'``.  Required when ``type='limb'``.
    out : str
        Space-separated RFM output flags written to *FLG (default
        ``'RAD BBT OPT TRA'``).  Stored on the instance and used by
        :meth:`run`.  Modify via ``model.out = 'RAD TRA'`` before calling
        :meth:`run`.
    physics : dict, optional
        Override physics flags.  Recognised keys (all bool):

        * ``'ctm'`` — H₂O/CO₂ continuum (default True, applied only when the
          relevant gas is in the gas list)
        * ``'mix'`` — CO₂ line mixing (default True, applied only when CO₂ is
          present)
        * ``'rex'`` — Rayleigh extinction (default False)
        * ``'chi'`` — CO₂ chi-factor (default False, applied only when CO₂
          is present)
        * ``'nte'`` — Non-LTE calculations (default False)

        Any key not supplied falls back to the default.
    binary : bool
        If True, add the BIN flag so RFM writes binary output files.  Binary
        reading is not yet implemented; parsing will raise
        ``NotImplementedError``.
    diagnostics : bool
        If True, add PRF and PTH flags so RFM writes the internal atmospheric
        profile and ray-path diagnostics.  Access results via
        :meth:`getProfile` and :meth:`getRayPath`.
    jacobians : dict, optional
        Specify Jacobian elements for the *JAC section.  Keys are RFM
        parameter names (e.g. ``'H2O'``, ``'TEM'``, ``'SFCTEM'``); values
        are either ``None`` (scalar/column perturbation) or a list of
        altitudes [km] for layer-by-layer perturbations.  Example::

            jacobians = {'TEM': [0, 5, 10, 20, 40], 'H2O': None}

        Access results via :meth:`getJacobians`.
    xsc : dict, optional
        Explicit cross-section file paths, keyed by gas name exactly as it
        appears in ``gas`` (including any ``x`` suffix if required).  Example::

            xsc = {'CFC11': '/data/cfc11.xsc', 'AEROSOL': '/data/aero.xsc'}

        Writes one ``FILXSC`` entry per key in the ``*XSC`` section.
        Gas names are written to ``*GAS`` unchanged — include the ``x`` suffix
        yourself for molecules that default to line data (e.g. ``'C2H6x'``).
        Pure XSC / aerosol species (RFM molecule IDs ≥ 100) do not need the
        suffix.
    xsc_dir : str, optional
        Directory containing ``.xsc`` files named by lowercase gas name
        (e.g. ``cfc11.xsc``).  Writes a single ``DEFXSC`` wildcard entry;
        RFM expands ``*`` to the lowercase molecule name for each XSC species
        not already covered by an explicit ``FILXSC`` entry.
    gra_profiles : dict of float → str or dict, optional
        Enables the RFM GRA flag (2D horizontally inhomogeneous atmosphere).
        Only valid with ``type='limb'``.  Keys are horizontal PSI angles [deg]
        (in the range ``[-90, 90]``); values are either a path to an ``.atm``
        file (str) or an atmosphere dict as returned by :func:`readAtm`.  Dict
        values are written to temporary files inside ``drv_path`` at
        :meth:`writeDrv` time.

        If PSI=0.0 is absent from the keys, ``self.atm_path`` (the reference
        atmosphere) is automatically inserted at PSI=0.0.  At least two
        profiles (i.e. two distinct PSI angles) must be provided in total.

        With GRA active, :meth:`parseRayPath` returns separate downward
        (``'gases'``, ``'totals'``) and upward (``'gases_up'``, ``'totals_up'``)
        segment DataFrames, because the two ray legs traverse different
        horizontal columns.
    gra_obs_psi : float, optional
        Horizontal PSI angle [deg] of the observer in the 2D atmosphere grid,
        written as ``PSIOBS`` in the ``*OBS`` section.  Required when
        ``gra_profiles`` is set; defaults to ``0.0`` (observer at the reference
        profile position).  Must be within the range of PSI angles supplied in
        ``gra_profiles``.
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
                 tan_heights: list = None,
                 out: str = 'RAD BBT OPT TRA',
                 physics: dict = None,
                 binary: bool = False,
                 dbl: bool = False,
                 diagnostics: bool = False,
                 jacobians: dict = None,
                 xsc: dict = None,
                 xsc_dir: str = None,
                 gra_profiles: dict = None,
                 gra_obs_psi: float = 0.0,
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
        if self.h_surf is not None and self.h_surf <= 0.0:
            raise ValueError(
                f"h_surf must be > 0.0 km (got {self.h_surf:.4f}). "
                "RFM requires the surface to be strictly above the lowest "
                "atmosphere grid level. Use h_surf=0.01 (default) for a "
                "sea-level surface."
            )
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

        # ----- Limb tangent heights -----
        if self.type == 'limb':
            if tan_heights is None:
                raise ValueError("type='limb' requires tan_heights=[...] parameter")
            self.tan_heights = [float(h) for h in tan_heights]
        else:
            self.tan_heights = None

        # ----- Spectral output selection -----
        self.out = out

        # ----- Physics flags with smart defaults -----
        _physics_defaults = {'ctm': True, 'mix': True, 'rex': False,
                              'chi': False, 'nte': False}
        if physics is None:
            self.physics = _physics_defaults.copy()
        else:
            self.physics = {**_physics_defaults,
                            **{k.lower(): v for k, v in physics.items()}}

        # ----- Output format -----
        self.binary = binary
        self.dbl = dbl

        # ----- Diagnostic outputs -----
        self.diagnostics = diagnostics

        # ----- Jacobians -----
        self.jacobians = jacobians

        # ----- Cross-section data -----
        self.xsc     = xsc      # dict: gas_name → file path
        self.xsc_dir = xsc_dir  # str: directory for DEFXSC wildcard

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
        if vmr is None:
            vmr = {}
        # For any gas not already in vmr, attempt to look it up from the
        # reference atmosphere at the specified pressure level — unless the
        # caller explicitly opted out by passing add_vmr=False with a partial
        # vmr dict, in which case we still fill missing gases (partial dicts
        # are never valid for layer runs) but suppress filling of gases that
        # are already present.
        # The same FASSTD default used by nadir/zenith applies here.
        _ref_atm = None   # loaded lazily — only if at least one gas needs it
        for item in self.gas.split():
            if item not in vmr:
                if _ref_atm is None:
                    from .functions import readAtm as _readAtm
                    _ref_atm_path = self.atm_path if self.atm_path else _std_atmospheres['FASSTD']
                    _ref_atm = _readAtm(_ref_atm_path)
                if item not in _ref_atm:
                    raise KeyError(
                        f"Gas '{item}' not found in the reference atmosphere "
                        f"({os.path.basename(_ref_atm_path)}). "
                        f"Pass an explicit value via vmr={{'{item}': value}}."
                    )
                # Interpolate VMR at the specified pressure level.
                # ATM pressure decreases with altitude, so reverse for interp.
                vmr[item] = float(np.interp(
                    self.pres,
                    _ref_atm['PRE'][::-1],
                    _ref_atm[item][::-1]
                ))
                logger.info(
                    "Layer VMR for '%s' set from reference atmosphere at "
                    "%.1f mbar: %.3g ppmv", item, self.pres, vmr[item]
                )
        self.vmr = vmr

        # ----- GRA profiles (2D atmosphere) -----
        if gra_profiles is not None:
            if self.type != 'limb':
                raise ValueError(
                    f"gra_profiles requires type='limb' (got type='{self.type}'). "
                    "The RFM GRA flag is only valid for limb-viewing geometry."
                )
            # Normalise keys to float
            gra = {float(psi): v for psi, v in gra_profiles.items()}
            if any(abs(psi) > 90.0 for psi in gra):
                bad = [psi for psi in gra if abs(psi) > 90.0]
                raise ValueError(
                    f"GRA PSI angles must be in [-90, 90] deg; got {bad}."
                )
            # Auto-insert reference profile at PSI=0 if absent
            if 0.0 not in gra:
                logger.info(
                    "GRA: PSI=0.0 not in gra_profiles — inserting self.atm_path "
                    "as the reference profile."
                )
                gra[0.0] = self.atm_path
            if len(gra) < 2:
                raise ValueError(
                    "gra_profiles must contain at least 2 profiles at distinct "
                    "PSI angles (including PSI=0.0)."
                )
            self.gra_profiles = gra
            self.gra_obs_psi  = float(gra_obs_psi)
        else:
            self.gra_profiles = None
            self.gra_obs_psi  = None

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

        elif self.type == 'limb':
            hlist = self.tan_heights
            if len(hlist) <= 4:
                hstr = ', '.join(f"{h:.1f}" for h in hlist) + ' km'
            else:
                hstr = ', '.join(f"{h:.1f}" for h in hlist[:3])
                hstr += f', ... {hlist[-1]:.1f} km ({len(hlist)} levels)'
            if self.h_obs is None:
                print_hobs = "Top of Atmosphere"
            else:
                print_hobs = f"{self.h_obs:.1f} km"
            print(f'===== RFM object: "{self.type}" =====\n'
                  f'Tangent heights: {hstr}\n'
                  f'Observer height: {print_hobs}\n'
                  f'wn_start = {self.wn_start:.0f} cm^-1\n'
                  f'wn_stop  = {self.wn_stop:.0f} cm^-1\n'
                  f'nperwn   = {self.nperwn:.0f}\n'
                  f'Atmosphere: {self.atm_path}\n'
                  f'Gases: {self.gas}\n'
                  '=================================')

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

        For ``type='layer'`` a text table is printed; for all other geometry
        types a multi-panel profile plot is produced via
        :func:`rfmwrapper.plotting.plotAtmGrid`.
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

        from .plotting import plotAtmGrid

        if self.type == 'nadir':
            hgt_obs  = (min(self.h_obs, self.atm_top)
                        if self.h_obs is not None else self.atm_top)
            hgt_surf = self.h_surf
        elif self.type in ('zenith', 'limb'):
            hgt_obs  = self.h_obs if self.h_obs is not None else 0.0
            hgt_surf = None
        else:
            hgt_obs  = None
            hgt_surf = None

        plotAtmGrid(
            atm=self.atm,
            hgt_obs=hgt_obs,
            hgt_surf=hgt_surf,
            zoom=zoom,
            color=color,
            title=title,
        )
        plt.show(block=False)

    # -----------------------------------------------------------------------
    def writeDrv(self, out: str = None,
                 CTM: bool = None, MIX: bool = None, PRF: bool = None) -> None:
        """
        Write the RFM driver file to ``self.drv_path/rfm.drv``.

        All configuration is normally set at construction time and stored on
        the instance.  The optional keyword arguments exist only for backward
        compatibility; prefer setting ``model.out``, ``model.physics``, and
        ``model.diagnostics`` directly.

        Parameters
        ----------
        out : str, optional
            Override ``self.out`` for this call only.
        CTM : bool, optional
            Deprecated.  Set ``physics={'ctm': ...}`` at construction instead.
        MIX : bool, optional
            Deprecated.  Set ``physics={'mix': ...}`` at construction instead.
        PRF : bool, optional
            Deprecated.  Set ``diagnostics=True`` at construction instead.
        """
        # ---- Backward-compatibility overrides ----
        if CTM is not None:
            logger.warning("CTM kwarg on writeDrv() is deprecated; "
                           "pass physics={'ctm': %s} to the constructor.", CTM)
            self.physics['ctm'] = bool(CTM)
        if MIX is not None:
            logger.warning("MIX kwarg on writeDrv() is deprecated; "
                           "pass physics={'mix': %s} to the constructor.", MIX)
            self.physics['mix'] = bool(MIX)
        if PRF is not None:
            logger.warning("PRF kwarg on writeDrv() is deprecated; "
                           "pass diagnostics=True to the constructor.")
            if PRF:
                self.diagnostics = True

        active_out = (out if out is not None else self.out).upper()

        fname = os.path.join(self.drv_path, 'rfm.drv')
        with open(fname, 'w') as drv:
            drv.write("! RFM driver table\n\n")

            # ---- *HDR ----
            _hdr = {
                'layer':  'Single homogeneous layer',
                'zenith': 'Upward-looking observation',
                'nadir':  'Downward-looking observation',
                'limb':   'Limb-viewing observation',
            }
            drv.write(f"*HDR\n{_hdr[self.type]}\n\n")

            # ---- *FLG ----
            flags = []

            # Geometry
            if self.type == 'layer':
                flags.append('HOM')
            elif self.type == 'zenith':
                flags += ['ZEN', 'OBS']
            elif self.type == 'nadir':
                flags += ['NAD', 'OBS', 'SFC']
            elif self.type == 'limb':
                if self.h_obs is not None:
                    flags.append('OBS')

            # Physics
            p = self.physics
            gas_list = self.gas.split()
            if p.get('ctm') and 'H2O' in gas_list:
                flags.append('CTM')
            if p.get('mix') and 'CO2' in gas_list:
                flags.append('MIX')
            if p.get('rex'):
                flags.append('REX')
            if p.get('chi') and 'CO2' in gas_list:
                flags.append('CHI')
            if p.get('nte'):
                flags.append('NTE')

            # Spectral outputs
            flags += active_out.split()

            # Note: no XSC flag — RFM activates cross-section handling automatically
            # when XSC species appear in *GAS; the *XSC section provides file paths.

            # Guard: RFM has exactly 2 slots for user-defined XSC molecules (X109, X110).
            # A third causes a silent crash.  Count user-defined species as anything in
            # xsc dict that is not 'aerosol'.
            if self.xsc:
                n_user = sum(1 for k in self.xsc if k.lower() != 'aerosol')
                if n_user > 2:
                    raise ValueError(
                        f"RFM supports at most 2 user-defined XSC species (X109, X110); "
                        f"{n_user} were provided: "
                        f"{[k for k in self.xsc if k.lower() != 'aerosol']}. "
                        "Combine species extinctions into a single XSC file instead."
                    )

            # Jacobians
            if self.jacobians is not None:
                flags.append('JAC')

            # LEV
            if self.levels is not None and self.type != 'layer':
                flags.append('LEV')

            # Binary / double precision
            if self.binary:
                flags.append('BIN')
            if self.dbl:
                flags.append('DBL')

            # 2D atmosphere
            if self.gra_profiles is not None:
                flags.append('GRA')

            # Diagnostics
            if self.diagnostics:
                flags += ['PRF', 'PTH']

            flags.append('SHH')
            drv.write("*FLG\n" + ' '.join(flags) + "\n\n")

            # ---- *SPC ----
            drv.write(f"*SPC\n{self.wn_start:f}\t{self.wn_stop:f}\t{self.nperwn:f}\n\n")

            # ---- *GAS ----
            drv.write(f"*GAS\n{self.gas}\n\n")

            # ---- *ATM ----
            drv.write("*ATM\n")
            if self.type == 'layer':
                drv.write(f"TEM={self.temp:f}\nPRE={self.pres:f}\n")
                for item in gas_list:
                    drv.write(f"{item}={self.vmr[item]:f}\n")
            elif self.gra_profiles is not None:
                # GRA mode: one .atm file per horizontal PSI angle,
                # each listed as "filepath(psi)" in the *ATM section.
                # Dict-valued entries are written to temp files first.
                for psi in sorted(self.gra_profiles):
                    val = self.gra_profiles[psi]
                    if isinstance(val, dict):
                        # Write the atm dict to a temp file in drv_path
                        safe = f"{psi:+.1f}".replace('.', 'p').replace('+', 'p').replace('-', 'm')
                        tmp_path = os.path.join(self.drv_path, f"gra_psi{safe}.atm")
                        _writeAtm(val, tmp_path)
                        atm_entry = tmp_path
                    else:
                        atm_entry = os.path.abspath(str(val))
                    drv.write(f"{atm_entry}({psi:g})\n")
            else:
                if self.atm_type is not None and 'MIP' in self.atm_type:
                    drv.write(f"{_std_atmospheres['HGT']}\n")
                drv.write(self.atm_path)
            drv.write("\n\n")

            # ---- Geometry-specific secondary sections ----
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
                            raise ValueError(
                                f"Zenith elevation must be in [0.1, 90] deg, got {self.elev:.2f}")
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
                            raise ValueError(
                                f"Nadir elevation must be in [-90, -0.1] deg, got {self.elev:.2f}")
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

            elif self.type == 'limb':
                drv.write("*TAN\n")
                drv.write(', '.join(f"{h:f}" for h in self.tan_heights))
                drv.write("\n\n")
                if self.h_obs is not None:
                    drv.write(f"*OBS\nHGTOBS={self.h_obs:f}\n")
                    if self.gra_profiles is not None:
                        drv.write(f"PSIOBS={self.gra_obs_psi:g}\n")
                    drv.write("\n")

            # ---- *HIT ----
            drv.write(f"*HIT\n{self.hitran_path}\n\n")

            # ---- *XSC ----
            if self.xsc or self.xsc_dir:
                drv.write("*XSC\n")
                if self.xsc:
                    # Explicit file per molecule — one bare path per line
                    for fpath in self.xsc.values():
                        drv.write(f"{fpath}\n")
                if self.xsc_dir:
                    # Wildcard template — RFM expands * to lowercase gas name
                    xsc_dir = self.xsc_dir
                    if not xsc_dir.endswith(os.sep):
                        xsc_dir = xsc_dir + os.sep
                    drv.write(f"{xsc_dir}*.xsc\n")
                drv.write("\n")

            # ---- *LEV ----
            if self.levels is not None and self.type != 'layer':
                levels_str = ' '.join(str(x) for x in self.levels)
                drv.write(f"*LEV\n{levels_str}  ! Additional levels for output\n\n")

            # ---- *JAC ----
            if self.jacobians is not None:
                drv.write("*JAC\n")
                for param, alts in self.jacobians.items():
                    if alts is None:
                        # Scalar / column-integrated perturbation
                        drv.write(f"{param}\n")
                    else:
                        # Layer-by-layer perturbation at specified altitudes [km]
                        alts_str = ' '.join(f"{a:g}" for a in alts)
                        drv.write(f"{param} {alts_str}\n")
                drv.write("\n")

            # ---- *OUT ----
            # OUTDIR sets the base directory; per-type FIL entries set filename
            # templates relative to OUTDIR.  The wildcard * is replaced by RFM
            # with the spectral-range and tangent-height labels.
            out_dir = self.out_path
            if not out_dir.endswith(os.sep):
                out_dir = out_dir + os.sep
            drv.write(f"*OUT\nOUTDIR = {out_dir}\n")
            for item in active_out.split():
                drv.write(f"{item}FIL = {item.lower()}*.out\n")
            # Note: JACFIL is not a recognised *OUT key in RFM — JAC output
            # files use RFM's default naming convention (jac_[spc][prm][tan])
            # and are read by globbing for jac*.asc in the output directory.
            if self.diagnostics:
                drv.write("PRFFIL = prf*.out\n")
                drv.write("PTHFIL = pth*.out\n")
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

        Supports both ASCII (default) and Fortran sequential unformatted binary
        (``binary=True`` / BIN flag) output.

        Binary format (gfortran little-endian, confirmed from RFM v5.10):
        - Records 1–3: char\\*80 header text (Fortran record markers on each side)
        - Record 4: int32 (RFM-internal, ignore) + float64 WNO1 + float64 WNOD
          + float64 WNO2 + char\\*15 LABSPC (includes surrounding apostrophes)
        - Record 5: float32 × NPNT spectral values
          (NPNT inferred from record-5 byte length; DBL flag would give float64)
        """
        if not self.ran:
            raise RuntimeError('RFM has not been run yet.')
        if not glob(os.path.join(self.out_path, '*.out')):
            raise RuntimeError(f"No output files found in: {self.out_path}")

        _type_map = {
            'Optical':      'opt',
            'Radiance':     'rad',
            'Bright.':      'bbt',
            'Transmission': 'tra',
            'Jacobian':     'jac',
        }

        # ---- *LEV output-suffix decoder (shared by the splitters below) ----
        def _decode_lev_suffix(suffix: str):
            """
            Decode a *LEV output-filename suffix → ``(altitude_km, direction)``.

            RFM (v5.21) writes the direction *before* the zero-padded
            altitude×1000, e.g. ``up01500`` → ``(1.5, 'up')``; earlier builds
            used the ``01500up`` ordering.  Both orderings and the downward
            marker (``dn`` / ``down``) are accepted.  Returns ``None`` when the
            suffix is not a LEV marker — e.g. a Jacobian parameter such as
            ``tem05000`` — so callers can tell LEV and Jacobian outputs apart.
            """
            import re as _re
            m = (_re.match(r'^(up|dn|down)(\d+)$', suffix)
                 or _re.match(r'^(\d+)(up|dn|down)$', suffix))
            if m is None:
                return None
            a, b = m.groups()
            digits, direction = (b, a) if a in ('up', 'dn', 'down') else (a, b)
            return float(digits) / 1e3, direction

        # ---- Label-decoding helper (shared by ASCII and binary parsers) ----
        def _apply_labels(rec: dict) -> dict:
            """
            Decode viewing geometry from the filename stem and attach
            ``label_type``, ``label_value``, and ``elev`` fields.

            RFM encodes angle/height as integers × 1000 in the file stem, e.g.:
            - ``tra90000.out``         → elevation 90.0°
            - ``tra90000_02500up.out`` → elevation 90.0°, level 2.5 km (LEV)
            - ``tra00000.out``         → layer geometry
            - ``rad05000.out``         → limb tangent height 5.0 km
            - ``rad-90000.out``        → nadir (elevation −90°)
            - ``rad-90000_tem05000.out`` → Jacobian file (handled separately)
            """
            stem = rec['fname'].split('.')[0]
            out_pfx = rec['output']
            if '_' in stem:
                left, right = stem.split('_', 1)
                lev = _decode_lev_suffix(right)
                if lev is not None:
                    # LEV output — integrated column from this altitude towards
                    # the remote side; elevation kept from the left stem.
                    rec['label_type']  = 'altitude'
                    rec['label_value'] = lev[0]
                    rec['elev']        = float(left.replace(out_pfx, '')) / 1e3
                    rec['lev_dir']     = lev[1]
                else:
                    # Jacobian file — labels will be set by _parse_jac_labels
                    rec['label_type']  = 'jacobian'
                    rec['label_value'] = None
                    rec['elev']        = None
            elif self.type == 'layer':
                n1 = float(stem.replace(out_pfx, ''))
                rec['label_type']  = 'layer'
                rec['label_value'] = abs(n1) / 1e3
                rec['elev']        = None
            elif self.type == 'limb':
                n1 = float(stem.replace(out_pfx, ''))
                rec['label_type']  = 'tangent'
                rec['label_value'] = n1 / 1e3
                rec['elev']        = None
            else:
                n1 = float(stem.replace(out_pfx, ''))
                if self.levels is not None and not isinstance(self.h_obs, list):
                    # LEV run: the suffix-less output is the column at the
                    # observer altitude, so it joins the altitude axis (giving
                    # getSpectra a coherent [h_obs, level1, …] second axis).
                    rec['label_type']  = 'altitude'
                    rec['label_value'] = float(self.h_obs)
                    rec['elev']        = n1 / 1e3
                else:
                    rec['label_type']  = 'elevation'
                    rec['label_value'] = n1 / 1e3
                    rec['elev']        = n1 / 1e3
            return rec

        def _parse_jac_labels(rec: dict) -> dict:
            """
            Extract Jacobian metadata from the file and attach ``param``,
            ``label_type``, ``label_value``, and ``elev``.

            RFM header line 2 (index 2) always contains one of:
            - ``! Jacobian spectrum for tem perturbed at altitude =5.00 [km]``
            - ``! Jacobian spectrum for h2o``

            The elevation is recovered from the left part of the filename stem.
            """
            import re as _re
            hdr2 = rec['header'][2] if len(rec['header']) > 2 else ''
            m = _re.search(
                r'Jacobian spectrum for (\S+)'
                r'(?:\s+perturbed at altitude\s*=\s*([\d.]+))?',
                hdr2, _re.IGNORECASE
            )
            if m:
                rec['param'] = m.group(1).upper()
                level_str = m.group(2)
                if level_str is not None:
                    rec['label_type']  = 'altitude'
                    rec['label_value'] = float(level_str)   # km
                else:
                    rec['label_type']  = 'column'
                    rec['label_value'] = None
            else:
                rec['param'] = 'unknown'
                rec['label_type']  = 'column'
                rec['label_value'] = None

            # Recover the viewing angle from the filename left-hand stem
            stem = rec['fname'].split('.')[0]
            left = stem.split('_', 1)[0]
            try:
                n1 = float(left.replace(rec['output'], ''))
                rec['elev'] = n1 / 1e3
            except ValueError:
                rec['elev'] = None
            return rec

        # ---- ASCII parser ----
        def _parse_ascii(fname: str) -> dict:
            """Parse one RFM ASCII spectral file."""
            with open(fname, 'r') as f:
                lines = f.readlines()
            rec = {'fname': os.path.basename(fname), 'header': lines[0:3]}
            grid = lines[3].split()
            rec['wn_start']   = float(grid[1])
            rec['wn_stop']    = float(grid[3])
            rec['resolution'] = float(grid[2])
            rec['output'] = _type_map.get(grid[4][1:], 'unknown')
            data = np.asarray([float(x) for x in lines[4].split()])
            rec['wn'] = np.linspace(rec['wn_start'], rec['wn_stop'], len(data))
            if rec['output'] == 'rad':
                data = data / 100.0
            rec['data'] = data
            return _apply_labels(rec)

        # ---- Binary parser ----
        def _parse_binary(fname: str) -> dict:
            """
            Parse one RFM Fortran-sequential-unformatted binary spectral file.

            Record layout (little-endian, gfortran default):
            - Records 1–3: char*80 (header text, 80-byte records)
            - Record 4   : int32 (ignore) + float64 WNO1 + float64 WNOD
                           + float64 WNO2 + char*15 LABSPC  [total 43 bytes]
            - Record 5   : float32 × NPNT  (NPNT = record-5 bytes / 4)
            """
            import struct as _struct

            def _read_record(f):
                hdr = f.read(4)
                if len(hdr) < 4:
                    raise EOFError(f"Unexpected EOF in {fname}")
                n = _struct.unpack('<i', hdr)[0]
                payload = f.read(n)
                f.read(4)   # trailing marker (same value as leading)
                return payload

            with open(fname, 'rb') as f:
                # Records 1–3: header text
                headers = [_read_record(f).decode('ascii', errors='replace').rstrip()
                           for _ in range(3)]

                # Record 4: metadata
                meta = _read_record(f)
                # Layout: int32(4) + float64(8) + float64(8) + float64(8)
                #         + char*15(15) = 43 bytes total
                wno1   = _struct.unpack_from('<d', meta, 4)[0]
                wnod   = _struct.unpack_from('<d', meta, 12)[0]
                wno2   = _struct.unpack_from('<d', meta, 20)[0]
                labspc = meta[28:43].decode('ascii', errors='replace').strip("' ")

                # Record 5: spectral data — infer NPNT from record size
                spc_raw       = _read_record(f)
                # DBL flag → float64 (8 bytes/value); default → float32 (4)
                bytes_per_val = 8 if self.dbl else 4
                dtype         = '<f8' if self.dbl else '<f4'
                npnt          = len(spc_raw) // bytes_per_val
                spc           = np.frombuffer(spc_raw, dtype=dtype).astype(np.float64)

            # Determine output type from LABSPC (first word)
            lbl_key = labspc.split()[0] if labspc.strip() else ''
            # Handle truncated keys (e.g. 'Bright.' matches 'Brightness...')
            out_type = _type_map.get(lbl_key,
                       next((v for k, v in _type_map.items()
                             if lbl_key.startswith(k)), 'unknown'))

            rec = {
                'fname':      os.path.basename(fname),
                'header':     headers,
                'wn_start':   wno1,
                'wn_stop':    wno2,
                'resolution': wnod,
                'output':     out_type,
                'wn':         np.linspace(wno1, wno2, npnt),
            }
            if out_type == 'rad':
                spc = spc / 100.0
            rec['data'] = spc
            return _apply_labels(rec)

        # ---- Dispatch ----
        _parse = _parse_binary if self.binary else _parse_ascii

        import re as _re

        def _is_jac_stem(stem: str) -> bool:
            """True when the filename stem belongs to a Jacobian file.

            Jacobian files follow the pattern ``{out}{elev}_{param}[{level}]``
            where the suffix after ``_`` is a parameter name (e.g. ``tem05000``).
            *LEV output uses a direction-marked altitude suffix (``up01500``);
            ``_decode_lev_suffix`` distinguishes the two.
            """
            if '_' not in stem:
                return False
            suffix = stem.split('_', 1)[1]
            return _decode_lev_suffix(suffix) is None

        # Standard spectral outputs and Jacobian files share the same glob
        # pattern ({prefix}*.out).  Split them at collection time.
        self._out = {}
        self._jac_raw = []
        for prefix in ('opt', 'rad', 'bbt', 'tra'):
            matches = sorted(glob(os.path.join(self.out_path, f'{prefix}*.out')))
            if not matches:
                continue
            normal, jac = [], []
            for fpath in matches:
                stem = os.path.splitext(os.path.basename(fpath))[0]
                (jac if _is_jac_stem(stem) else normal).append(fpath)
            if normal:
                self._out[prefix] = [_parse(f) for f in normal]
            for fpath in jac:
                rec = _parse(fpath)
                _parse_jac_labels(rec)
                self._jac_raw.append(rec)

        # Diagnostic outputs — RFM writes prf/pth relative to its CWD
        # (drv_path), not to OUTDIR, so search drv_path first then out_path.
        if self.diagnostics:
            for search_dir in (self.drv_path, self.out_path):
                prf_candidate = os.path.join(search_dir, 'prf.out')
                if os.path.exists(prf_candidate):
                    from .functions import readAtm as _readAtmFn
                    try:
                        self._prf = _readAtmFn(prf_candidate)
                    except Exception:
                        with open(prf_candidate, 'r') as f:
                            self._prf = f.readlines()
                    break

            for search_dir in (self.drv_path, self.out_path):
                pth_files = sorted(glob(os.path.join(search_dir, 'pth*.out')))
                if pth_files:
                    self._pth = {}
                    for pth_file in pth_files:
                        key = os.path.splitext(os.path.basename(pth_file))[0]
                        with open(pth_file, 'r') as f:
                            self._pth[key] = f.readlines()
                    break

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

        Delegates to :func:`rfmwrapper.plotting.plotSpectra` (single output)
        or :func:`rfmwrapper.plotting.plotSpectraGrid` (``which='all'``).

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
        if not self.ran:
            raise RuntimeError('RFM has not been run yet.')

        from .plotting import plotSpectra, plotSpectraGrid

        if which.lower() == 'all':
            keys = list(self._out)
            # Don't force a single colour when multiple spectra are present
            c = None if max(len(self._out[k]) for k in keys) > 1 else color
            spectra_dict = {k: self.getSpectra(k) for k in keys}
            plotSpectraGrid(spectra_dict, color=c, rad_wl=rad_wl)
        else:
            fig, ax = plotSpectra(self.getSpectra(which.lower()),
                                  color=color, rad_wl=rad_wl)
            if reverse:
                ax.invert_xaxis()

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
                 fwhm_wn: float = None, resample: bool = False) -> None:
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
        resample : bool
            If ``True``, resample the convolved output onto the instrument's
            native output grid after convolution.  Requires ``instrument`` to
            be specified and the instrument entry to have a ``'spacing'``
            field; raises ``ValueError`` otherwise.
        """
        if not self.ran:
            raise RuntimeError("RFM has not been run yet.")

        params = None
        if instrument is not None:
            params    = loadILSParams(instrument)
            fov       = params['fov']
            max_opd   = params['max_opd']
            apod_type = params['apod_type']
            fwhm_wn   = params.get('fwhm_wn', None)
        elif resample:
            raise ValueError(
                "resample=True requires instrument= to be specified so that "
                "the native output spacing can be looked up."
            )

        wn_centre = (self.wn_start + self.wn_stop) / 2.0
        ils = makeILS(fov=fov, max_opd=max_opd, nper_wn=self.nperwn,
                      wn=wn_centre, apod_type=apod_type, fwhm_wn=fwhm_wn)

        self._ilsout = {}
        for key in self._out:
            self._ilsout[key] = applyILS(self.getSpectra(key), ils)

        if resample:
            if 'spacing' not in params:
                raise ValueError(
                    f"Instrument {instrument!r} has no 'spacing' defined in "
                    "instruments.json; cannot resample automatically."
                )
            spacing = params['spacing']
            self._ilsout = {key: out.resample(dx=spacing)
                            for key, out in self._ilsout.items()}

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
    def getJacobians(self, param: str = None) -> 'Spectra | dict':
        """
        Return Jacobian spectra from a JAC-flag run.

        Each Jacobian file represents the sensitivity of the output spectrum
        to a perturbation of one atmospheric parameter at one viewing geometry.
        Multiple files (one per tangent height / elevation) are stacked into a
        :class:`Spectra` object with ``label_type='tangent'`` (limb) or
        ``'elevation'`` (nadir/zenith).

        Parameters
        ----------
        param : str, optional
            Jacobian parameter name (e.g. ``'H2O'``, ``'TEM'``, ``'SFCTEM'``).
            Case-insensitive.  If ``None``, returns a dict of all available
            parameters mapped to their :class:`Spectra` objects.

        Returns
        -------
        Spectra or dict of str → Spectra
            If ``param`` is given, a single :class:`Spectra` whose rows
            correspond to tangent heights / elevation angles.  If ``param`` is
            ``None``, a dict keyed by parameter name.

        Raises
        ------
        RuntimeError
            If ``jacobians`` was not set at construction or RFM has not run.
        KeyError
            If the requested ``param`` was not found in the Jacobian output.
        """
        if not self.ran:
            raise RuntimeError("RFM has not been run yet.")
        if not hasattr(self, '_jac_raw') or not self._jac_raw:
            raise RuntimeError(
                "No Jacobian output found. Set jacobians={...} at construction "
                "and re-run. Remember: N altitude values produce N-2 output "
                "spectra (boundary altitudes are interpolation anchors)."
            )

        # Group records by parameter name
        groups: dict = {}
        for rec in self._jac_raw:
            p = rec.get('param', 'unknown')
            groups.setdefault(p, []).append(rec)

        def _to_spectra(recs: list) -> 'Spectra':
            # Sort by altitude (layer Jacobians) or keep single row (column)
            has_level = any(r['label_value'] is not None for r in recs)
            if has_level:
                recs = sorted(recs, key=lambda r: (r['label_value'] or 0.0))
            wn     = recs[0]['wn']
            data   = np.vstack([r['data'] for r in recs])
            if has_level:
                labels = np.array([r['label_value'] for r in recs])
                lt     = 'altitude'
            else:
                labels = np.array([0.0])
                lt     = 'column'
            return Spectra(wn=wn, data=data, labels=labels,
                           label_type=lt, output='jac',
                           xunit='wn', yunit='')

        if param is None:
            return {p: _to_spectra(recs) for p, recs in groups.items()}

        key = param.upper()
        if key not in groups:
            available = sorted(groups)
            raise KeyError(
                f"Jacobian parameter {param!r} not found. "
                f"Available: {available}"
            )
        return _to_spectra(groups[key])

    # -----------------------------------------------------------------------
    def getProfile(self) -> dict:
        """
        Return the RFM internal atmospheric profile written by the PRF flag.

        The profile is the atmosphere as interpolated onto RFM's internal
        height grid — useful for verifying that the input profiles were read
        and processed correctly.

        Returns
        -------
        dict
            Atmosphere dictionary (same structure as :func:`readAtm`), keyed by
            field name (``'HGT'``, ``'TEM'``, ``'PRE'``, gas names, …).

        Raises
        ------
        RuntimeError
            If ``diagnostics=False`` at construction, or if RFM has not run.
        """
        if not self.ran:
            raise RuntimeError("RFM has not been run yet.")
        if not self.diagnostics:
            raise RuntimeError(
                "Profile output not available. Set diagnostics=True at "
                "construction and re-run."
            )
        if not hasattr(self, '_prf'):
            raise RuntimeError(
                "Profile file (prf.out) was not found in the output directory."
            )
        return self._prf

    # -----------------------------------------------------------------------
    def getRayPath(self) -> dict:
        """
        Return the RFM ray-path diagnostic text written by the PTH flag.

        The raw content is returned as a dict of filename stem → list of text
        lines (one entry per tangent height / elevation).  Parsing the
        structured content is left to the caller.

        Returns
        -------
        dict of str → list of str
            Keyed by file stem (e.g. ``'pth90000'`` for 90° elevation).

        Raises
        ------
        RuntimeError
            If ``diagnostics=False`` at construction, or if RFM has not run.
        """
        if not self.ran:
            raise RuntimeError("RFM has not been run yet.")
        if not self.diagnostics:
            raise RuntimeError(
                "Ray-path diagnostics not available. Set diagnostics=True at "
                "construction and re-run."
            )
        if not hasattr(self, '_pth'):
            raise RuntimeError(
                "No ray-path files (pth*.out) were found in the output directory."
            )
        return self._pth

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
