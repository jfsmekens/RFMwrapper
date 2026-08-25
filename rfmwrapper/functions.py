"""
Written by @jfsmekens - Last updated 15/02/2024

Utility functions for the RFM wrapper: atmosphere I/O, Planck / unit conversions,
filter convolution, and plotting helpers.
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import CubicSpline
import pandas as pd

from .constants import weights, AEROSOL_SPECIES
from .config import get_xsc_dir

# Trapezoidal integration: ``np.trapezoid`` (NumPy ≥ 2.0) replaced the now-
# deprecated ``np.trapz``.  Resolve to whichever the installed NumPy provides
# so the package works on both old and new versions.
_trapz = getattr(np, 'trapezoid', None) or np.trapz

logger = logging.getLogger(__name__)

# Path to bundled data file — works regardless of working directory
_PACKAGE_DIR = Path(__file__).parent

# Matplotlib defaults
plt.rcParams.update({'font.size': 11,
                     'lines.linewidth': 1.0,
                     'axes.prop_cycle': plt.cycler(color=plt.cm.Dark2.colors),
                     'xtick.direction': 'in',
                     'xtick.top': True,
                     'ytick.direction': 'in',
                     'xtick.labelsize': 10,
                     'ytick.right': True,
                     'ytick.labelsize': 10,
                     'axes.grid': True,
                     'axes.axisbelow': True,
                     'axes.labelsize': 12,
                     'axes.titlesize': 12,
                     'grid.linestyle': '--',
                     'axes.formatter.limits': (-4, 4),
                     'errorbar.capsize': 2})


# ===========================================================================
# ========================= Atmosphere I/O ==================================
# ===========================================================================

def readAtm(fname: str) -> dict:
    """
    Read a standard RFM atmospheric profile from disk.

    Parameters
    ----------
    fname : str
        Path to the atmosphere file.

    Returns
    -------
    dict
        Keys: 'header', 'HGT' [km], 'PRE' [mb], 'TEM' [K], and one key per
        gas species [ppmv].
    """
    with open(fname, 'r') as f:
        lines = f.readlines()

    sections = [lines[i].split()[0].replace('*', '').upper()
                for i in range(len(lines)) if '*' in lines[i] and '!' not in lines[i]]
    indices = [i for i in range(len(lines)) if '*' in lines[i] and '!' not in lines[i]]

    out = {'header': [x.strip('\n') for x in lines[0:indices[0] - 1]]}

    for idx, field in enumerate(sections[:-1]):
        start = indices[idx] + 1
        end = indices[idx + 1]
        chunks = [np.fromstring(lines[i].strip().replace(',', ' '), sep=' ')
                  for i in range(start, end)]
        out[field] = np.concatenate(chunks)

    return out


def writeAtm(atm: dict, outname: str, header: str = "") -> None:
    """
    Write an RFM atmospheric profile to disk.

    Parameters
    ----------
    atm : dict
        Atmosphere dictionary as returned by :func:`readAtm`.
    outname : str
        Destination file path.
    header : str or list of str, optional
        Custom header lines. If empty, any existing 'header' key is used.
    """
    with open(outname, 'w') as f:
        if header == "":
            if 'header' in atm:
                for x in atm['header']:
                    f.write(f"{x}\n")
            else:
                f.write("! RFM atmospheric profile\n")
        else:
            if isinstance(header, list):
                for x in header:
                    f.write(f"{x}\n")
            else:
                f.write(f"{header}\n")

        f.write(f"{len(atm['HGT'])}")
        f.write("  ! No.Levels in profiles\n\n")

        for key in [k for k in atm if k != 'header']:
            f.write(f"*{key}\n")
            f.write(', '.join([f"{x:.5f}" for x in atm[key]]))
            f.write("\n\n")

        f.write("*END")


def adjustAtm(atm: dict, gas: str, value: float) -> dict:
    """
    Scale or insert a gas profile so its surface-level VMR equals ``value``.

    If ``gas`` is absent from the atmosphere, a uniform profile at ``value``
    ppmv is added across all levels.  If present, the entire profile is scaled
    so that ``atm[gas][0] == value`` while preserving the vertical shape.

    Parameters
    ----------
    atm : dict
        Atmosphere dictionary as returned by :func:`readAtm`.
    gas : str
        Gas species name (e.g. ``'CO2'``).
    value : float
        Target surface VMR [ppmv].

    Returns
    -------
    dict
        New atmosphere dictionary with the adjusted gas profile.
    """
    if gas.upper() in AEROSOL_SPECIES:
        logger.info(
            "adjustAtm: '%s' is an aerosol species — profile values are "
            "dimensionless scaling factors (1.0 = reference extinction), not VMR [ppmv].",
            gas
        )
    atm = dict(atm)
    if gas not in atm:
        atm[gas] = np.ones(len(atm['HGT'])) * value
    else:
        old = atm[gas]
        atm[gas] = old / old[0] * value
    return atm


def insertPlume(atm: dict, h: float, th: float, plume: dict = None,
                gradient_flag: int = 1) -> dict:
    """
    Insert a gas plume layer into an atmosphere dictionary.

    Parameters
    ----------
    atm : dict
        Atmosphere dictionary as returned by :func:`readAtm`.
    h : float
        Centre altitude of the plume [km].
    th : float
        Vertical thickness of the plume [km].
    plume : dict, optional
        Gas concentrations within the plume, keyed by species name.

        * For species **already present** in ``atm``: values are additive
          enhancements [ppmv] above the existing background profile.
        * For species **absent** from ``atm``: values are the **peak total
          concentration** [ppmv] in the plume (background is implicitly zero).
          This holds for every ``gradient_flag`` — the shape is peak-normalised
          in all three cases.
          To use a non-zero background for an absent species, add it first
          with :func:`adjustAtm` before calling this function.

        Aerosol species (see :data:`AEROSOL_SPECIES`) follow the same
        convention but values are dimensionless scaling factors, not ppmv.
    gradient_flag : int
        Concentration profile shape within the plume layer:

        * 0 — uniform (step function)
        * 1 — triangular, peak at centre (default)
        * 2 — Gaussian, truncated at ±2.5σ (σ = th/5)

        All three are peak-normalised and confined to ``[h - th/2, h + th/2]``
        by zero guard levels, so the plume never extends beyond the requested
        thickness.  Column per unit peak VMR differs by shape: ``th`` for
        uniform, ``th/2`` for triangular, ``0.495·th`` for Gaussian (the
        truncation drops 1.24% of the nominal integral and is not
        renormalised).

    Returns
    -------
    dict
        New atmosphere dictionary with the plume layer inserted.
    """
    if plume is None:
        plume = {}

    for key in plume:
        if key.upper() in AEROSOL_SPECIES:
            logger.info(
                "insertPlume: '%s' is an aerosol species — plume values are "
                "dimensionless scaling factor enhancements (additive on top of "
                "the background scaling), not VMR [ppmv].",
                key
            )

    h1 = h - th / 2
    h2 = h + th / 2

    atm = dict(atm)
    for key in plume:
        if key not in atm:
            # Species absent from the background ATM — seed with 0 ppmv so the
            # plume enhancement is added on top of a clean background.
            # Pass a pre-seeded ATM profile (via adjustAtm) if a non-zero
            # background is required.
            atm[key] = np.zeros(len(atm['HGT']))

    if gradient_flag == 0:
        xx = np.asarray([h1 - 0.001, h1, h2, h2 + 0.001])
        yy = np.asarray([0.0, 1.0, 1.0, 0.0])
    elif gradient_flag == 1:
        xx = np.asarray([h1, h, h2])
        yy = np.asarray([0.0, 1.0, 0.0])
    elif gradient_flag == 2:
        sig = (h - h1) / 2.5
        # An ODD number of samples so the centre is on the grid and the peak is
        # actually attained: arange(h1, h2+th/9, th/9) gave 10 points straddling
        # h, so the sampled peak was 0.962 of the requested VMR.
        xx = np.linspace(h1, h2, 11)
        # Peak-normalised, so `plume` means peak VMR exactly as it does for
        # gradient_flag 0 and 1.  This previously used the Gaussian *density*
        # 1/(sqrt(2 pi) sigma), which for a 0.4 km plume made the peak 4.8x the
        # requested value while flags 0 and 1 gave 1.0x.
        yy = np.exp(-np.power((xx - h) / sig, 2.0) / 2)
        # Guard levels, as flag 0 has.  The Gaussian is truncated at +/-2.5
        # sigma, where it is still 0.219 of peak; with no zeros immediately
        # outside the band, RFM interpolates from that edge value down to the
        # nearest *background* level.  On a 1 km background grid that placed
        # 15.1% of the inserted column outside the nominal plume, spread over
        # the km below and above it.
        xx = np.concatenate([[h1 - 0.001], xx, [h2 + 0.001]])
        yy = np.concatenate([[0.0], yy, [0.0]])
    else:
        raise ValueError('gradient_flag must be 0, 1, or 2')

    below = np.where(atm['HGT'] < h1)[0]
    above = np.where(atm['HGT'] > h2)[0]

    plume_layer = {}
    for key in [k for k in atm if k != 'header']:
        plume_layer[key] = np.interp(xx, atm['HGT'], atm[key])
        if key in plume:
            plume_layer[key] = plume_layer[key] + plume[key] * yy

    new_atm = {}
    for key in [k for k in atm if k != 'header']:
        new_atm[key] = np.concatenate([atm[key][below], plume_layer[key], atm[key][above]])

    new_atm['header'] = [
        "! Standard profile modified with plume layer for input to the RFM forward model",
        f"! Plume layer: {h1:.2f} - {h2:.2f} km",
        *[f"! {k} enhancement: {v:.1f} ppmv" for k, v in plume.items()],
    ]
    return new_atm


def sliceAtm(atm: dict, h1: float, h2: float) -> dict:
    """
    Extract a vertical slice of an atmosphere between two altitudes.

    The output height grid retains all original levels that fall within
    ``[h1, h2]``, bracketed by interpolated values at exactly ``h1`` and
    ``h2`` if they are not already present.

    Parameters
    ----------
    atm : dict
        Atmosphere dictionary as returned by :func:`readAtm`.
    h1 : float
        Bottom altitude of the slice [km].
    h2 : float
        Top altitude of the slice [km].

    Returns
    -------
    dict
        New atmosphere dictionary covering only the requested altitude range.
    """
    hgt = [x for x in atm['HGT'] if h1 <= x <= h2]
    if len(hgt) == 0:
        hgt = [h1, h2]
    else:
        if hgt[0] > h1:
            hgt.insert(0, np.round(h1, 3))
        if hgt[-1] < h2:
            hgt.append(np.round(h2, 3))

    out = {
        'header': [
            "! Modified atmospheric profile for RFM input",
            f"! Slice between altitudes: {h1:.3f} - {h2:.3f} km",
        ]
    }
    for key in [k for k in atm if k != 'header']:
        out[key] = np.interp(hgt, atm['HGT'], atm[key])
    return out


# ===========================================================================
# ========================= Planck functions =================================
# ===========================================================================

def rad(wn: np.ndarray, temp: float, wl: bool = False, norm: bool = False) -> np.ndarray:
    """
    Compute blackbody radiance using the Planck function.

    Parameters
    ----------
    wn : array_like
        Wavenumber [cm^-1] or wavelength [µm] axis.
    temp : float
        Blackbody temperature [K].
    wl : bool
        If True, treat ``wn`` as wavelength [µm] and return [mW/m²·sr·µm].
    norm : bool
        If True, return a normalized curve.

    Returns
    -------
    np.ndarray
        Radiance [mW/m²·sr·cm^-1] (wavenumber) or [mW/m²·sr·µm] (wavelength).
    """
    if wl:
        c1 = 1.191042e8   # W/m²·sr·µm^-4
        c2 = 1.4387752e4  # K·µm
        L = c1 / (wn ** 5 * (np.exp(c2 / (wn * temp)) - 1))
    else:
        c1 = 1.191042e-5  # mW/m²·sr·cm^-4
        c2 = 1.4387752    # K·cm
        L = c1 * wn ** 3 / (np.exp(c2 * wn / temp) - 1)

    return L / L.max() if norm else L


def bbt(wn: np.ndarray, rad_in: np.ndarray, wl: bool = False) -> np.ndarray:
    """
    Convert spectral radiance to brightness temperature via the Planck inverse.

    Parameters
    ----------
    wn : array_like
        Wavenumber [cm^-1] or wavelength [µm].
    rad_in : array_like
        Radiance [mW/m²·sr·cm^-1].
    wl : bool
        If True, convert ``wn`` from wavelength [µm] to wavenumber first.

    Returns
    -------
    np.ndarray
        Brightness temperature [K].
    """
    if wl:
        wn = 1e4 / wn
    c1 = 1.191042e-5
    c2 = 1.4387752
    return c2 * wn / np.log(c1 * wn ** 3 / rad_in + 1)


def wn2wl(wn: np.ndarray, L1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spectral radiance from wavenumber to wavelength space.

    Parameters
    ----------
    wn : array_like
        Wavenumber axis [cm^-1].
    L1 : array_like
        Spectral radiance [mW/m²·sr·cm^-1].

    Returns
    -------
    wl : np.ndarray
        Wavelength axis [µm].
    L2 : np.ndarray
        Spectral radiance [mW/m²·sr·µm].
    """
    wl = 1e4 / wn
    L2 = L1 * wn ** 2 / 1e4
    return wl, L2


def wl2wn(wl: np.ndarray, L2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spectral radiance from wavelength to wavenumber space.

    Parameters
    ----------
    wl : array_like
        Wavelength axis [µm].
    L2 : array_like
        Spectral radiance [mW/m²·sr·µm].

    Returns
    -------
    wn : np.ndarray
        Wavenumber axis [cm⁻¹].
    L1 : np.ndarray
        Spectral radiance [mW/m²·sr·cm⁻¹].
    """
    wn = 1e4 / wl
    L1 = L2 * wl ** 2 / 1e4
    return wn, L1


# ===========================================================================
# ========================= Unit conversions =================================
# ===========================================================================

def ppm2scd(ppm: float, pathlength, temp: float = 298, pres: float = 1013,
            elev: float = 90, output: str = 'molec/cm2') -> float:
    """
    Convert VMR to slant column density for a homogeneous layer.

    Parameters
    ----------
    ppm : float or array_like
        Gas VMR [ppmv], or vertical profile of VMRs.
    pathlength : float or array_like
        Path length [km], or altitude array for profile integration.
    temp : float
        Temperature [K].
    pres : float
        Pressure [mbar].
    elev : float
        Elevation angle [degrees] for slant path correction.
    output : str
        Output unit: ``'molec/cm2'``, ``'ppmm'``.

    Returns
    -------
    float
        Slant column density in the requested unit.
    """
    avo = 6.02214e23
    R = 8.31446261815324e-2  # m³·mbar·K⁻¹·mol⁻¹

    if output == 'ppmm':
        return ppm * pathlength

    n_tot = pres / R / temp
    n_gas = ppm * 1e-6 * n_tot
    if isinstance(pathlength, (list, np.ndarray)):
        scd = _trapz(n_gas, x=np.asarray(pathlength) * 1e3) / np.sin(np.deg2rad(elev))
    else:
        scd = n_gas * pathlength * 1e3
    if output == 'molec/cm2':
        scd = scd * avo * 1e-4

    return scd


def scd2ppm(scd: float, pathlength: float, temp: float = 298, pres: float = 1013,
            input: str = 'molec/cm2', species: str = None) -> float:
    """
    Convert slant column density to VMR for a homogeneous layer.

    Parameters
    ----------
    scd : float
        Column density (units given by ``input``).
    pathlength : float
        Path length [km].
    temp : float
        Temperature [K].
    pres : float
        Pressure [mbar].
    input : str
        Input unit: ``'molec/cm2'``, ``'ppmm'``, ``'g/m2'``.
    species : str, optional
        Species name — required when ``input='g/m2'``.

    Returns
    -------
    float
        Volume mixing ratio [ppmv].
    """
    avo = 6.02214e23
    R = 8.31446261815324e-2

    if input == 'ppmm':
        return scd / (pathlength * 1e3)
    if input == 'g/m2':
        scd = gm2mcm2(scd, species=species)
        input = 'molec/cm2'
    # molec/cm2
    n_gas = scd * 1e4 / (pathlength * 1e3)
    n_tot = pres / R / temp * avo
    return n_gas / n_tot * 1e6


def ppmm2mcm(ppmm: float, pres: float = 1013, temp: float = 298) -> float:
    """Convert SCD from [ppm·m] to [molec/cm²]."""
    avo = 6.02214e23
    R = 8.31446261815324e-2
    return ppmm * 1e-6 * pres / R / temp * avo * 1e-4


def mcm2ppmm(mcm: float, pres: float = 1013, temp: float = 298) -> float:
    """Convert SCD from [molec/cm²] to [ppm·m]."""
    avo = 6.02214e23
    R = 8.31446261815324e-2
    return mcm * 1e4 / avo * R * temp / pres * 1e6


def mcm2gm2(mcm: float, species: str) -> float:
    """Convert SCD from [molec/cm²] to [g/m²]."""
    avo = 6.02214e23
    return mcm * 1e4 / avo * weights[species]


def gm2mcm2(gm2: float, species: str) -> float:
    """Convert SCD from [g/m²] to [molec/cm²]."""
    avo = 6.02214e23
    return gm2 / weights[species] * avo / 1e4


def ppm2mm3(ppm: float, temp: float = 298, pres: float = 1013) -> float:
    """Convert VMR [ppmv] to molar concentration [mol/m³]."""
    R = 8.31446261815324e-2
    n_tot = pres / R / temp
    return ppm * 1e-6 * n_tot


def mm32ppm(mm3: float, temp: float = 298, pres: float = 1013) -> float:
    """Convert molar concentration [mol/m³] to VMR [ppmv]."""
    R = 8.31446261815324e-2
    n_tot = pres / R / temp
    return mm3 / n_tot * 1e6


def _psat(temp: float) -> float:
    """Saturation vapour pressure [hPa] over water (T>273.16 K) or ice."""
    if temp > 273.16:
        Tc, Pc = 647.096, 220640.0
        t = 1 - temp / Tc
        poly = Tc / temp * (-7.85951783 * t + 1.84408259 * t ** 1.5
                            - 11.7866497 * t ** 3 + 22.6807411 * t ** 3.5
                            - 15.9618719 * t ** 4 + 1.80122502 * t ** 7.5)
        return np.exp(poly) * Pc
    else:
        Pn, Tn = 6.11657, 273.16
        t = temp / Tn
        poly = -13.928169 * (1 - t ** -1.5) + 34.707823 * (1 - t ** -1.25)
        return np.exp(poly) * Pn


# Keep the public name as well
Psat = _psat


def RH2ppm(RH: float, pres: float = 1013, temp: float = 298) -> float:
    """Convert relative humidity [%] to water vapour VMR [ppmv]."""
    return RH / 100 * _psat(temp) / pres * 1e6


def ppm2RH(ppm: float, pres: float = 1013, temp: float = 298) -> float:
    """Convert water vapour VMR [ppmv] to relative humidity [%]."""
    return ppm * pres / _psat(temp) * 100 / 1e6


def RH2mm3(RH: float, temp: float = 298, pres: float = 1013) -> float:
    """Convert relative humidity [%] to molar concentration [mol/m³]."""
    return ppm2mm3(RH2ppm(RH, temp=temp, pres=pres), temp=temp, pres=pres)


def mm32RH(mm3: float, temp: float = 298, pres: float = 1013) -> float:
    """Convert molar concentration [mol/m³] to relative humidity [%]."""
    return ppm2RH(mm32ppm(mm3, temp=temp, pres=pres), temp=temp, pres=pres)


def RH2gm3(RH: float, temp: float = 298, pres: float = 1013) -> float:
    """Convert relative humidity [%] to absolute humidity [g/m³]."""
    return RH2mm3(RH, temp=temp, pres=pres) * weights['H2O']


def gm32RH(gm3: float, temp: float = 298, pres: float = 1013) -> float:
    """Convert absolute humidity [g/m³] to relative humidity [%]."""
    return mm32RH(gm3 / weights['H2O'], temp=temp, pres=pres)


def c2k(deg: float) -> float:
    """Celsius to Kelvin."""
    return deg + 273.15

def k2c(kelv: float) -> float:
    """Kelvin to Celsius."""
    return kelv - 273.15

def f2c(far: float) -> float:
    """Fahrenheit to Celsius."""
    return (far - 32) / 1.8

def c2f(deg: float) -> float:
    """Celsius to Fahrenheit."""
    return deg * 1.8 + 32

def f2k(far: float) -> float:
    """Fahrenheit to Kelvin."""
    return (far - 32) / 1.8 + 273.15


# ===========================================================================
# ========================= RFM spectral file I/O ===========================
# ===========================================================================

def writeRFMspc(wn: np.ndarray, spc: np.ndarray, outname: str,
                header: str = "") -> None:
    """
    Write a spectral file in RFM format (e.g. for spectrally variable emissivity).

    Parameters
    ----------
    wn : array_like
        Wavenumber axis [cm^-1].
    spc : array_like
        Spectral values.
    outname : str
        Output file path.
    header : str or list of str, optional
        Header text.
    """
    if len(wn) != len(spc):
        raise ValueError(f"wn and spc must have the same length ({len(wn)} vs {len(spc)})")

    npts = len(wn)
    wn_start, wn_stop = wn[0], wn[-1]
    diffs = np.diff(wn)
    if np.all(np.isclose(diffs, diffs[0])):
        regular = True
        wn_spacing = float(np.mean(diffs))
    else:
        regular = False
        wn_spacing = 0

    spc_type = "'Spectrum   '"

    with open(outname, 'w') as f:
        if header == "":
            f.write("! RFM spectral file\n")
        elif isinstance(header, list):
            for x in header:
                f.write(f"{x}\n")
        else:
            for s in header.split('\n'):
                f.write(f"! {s}\n")

        if regular:
            f.write("! No.Pts  Lower_Wavenumber   Delta_Wavenumber  Upper_Wavenumber  Label\n")
            f.write(f"{npts}\t{wn_start:.3f}\t{wn_spacing:.3f}\t{wn_stop:.3f}\t{spc_type}\n")
            f.write('\t'.join([f"{x:.5f}" for x in spc]))
            f.write("\n")
        else:
            f.write("! No.Pts  Lower_Wavenumber   Delta_Wavenumber  Upper_Wavenumber  Label\n")
            f.write(f"{npts}\t{wn_start:.3f}\t{wn_spacing}\t{wn_stop:.3f}\t{spc_type}\n")
            for i in range(npts):
                f.write(f"{wn[i]:.5f}\t{spc[i]:.5f}\n")


# ===========================================================================
# ========================= XSC file utilities ==============================
# ===========================================================================

def _parseXSCHeader(line: str) -> dict:
    """
    Parse one HITRAN XSC header line into a dict.

    HITRAN XSC fixed-width format (1-based columns):
      1–20   molecule name
     21–30   WN_start [cm⁻¹]
     31–40   WN_end   [cm⁻¹]
     41–47   NPTS
     48–54   T [K]
     55–60   p [Torr]
     61+     max_k, resolution, broadener, reference (not parsed here)
    """
    return {
        'name':    line[0:20].strip(),
        'wn_start': float(line[20:30]),
        'wn_end':   float(line[30:40]),
        'npts':     int(line[40:47]),
        'T':        float(line[47:54]),
        'p_torr':   float(line[54:60]),
        'p_mbar':   float(line[54:60]) * 1.33322,   # Torr → mbar
        'spacing':  (float(line[30:40]) - float(line[20:30])) / (int(line[40:47]) - 1),
    }


def readXSCHeaders(files: list) -> 'pd.DataFrame':
    """
    Read the header line from each XSC file and return a summary table.

    Useful for inspecting a dataset before concatenating — shows the (T, p)
    grid, spectral range, and spectral spacing of every tabulation block.

    Parameters
    ----------
    files : list of str or Path
        XSC file paths.

    Returns
    -------
    pd.DataFrame
        One row per file with columns: ``file``, ``name``, ``wn_start``,
        ``wn_end``, ``npts``, ``T``, ``p_torr``, ``p_mbar``, ``spacing``.
    """
    rows = []
    for fpath in sorted(files):
        with open(fpath, 'r') as f:
            hdr = f.readline()
        rec = _parseXSCHeader(hdr)
        rec['file'] = str(fpath)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(['T', 'p_torr']).reset_index(drop=True)


def concatXSC(files: list, output: str = None,
              T_range: tuple = None,
              p_range: tuple = None,
              verbose: bool = True) -> 'pd.DataFrame':
    """
    Concatenate HITRAN XSC tabulation files into a single RFM-ready file.

    RFM requires one file per molecule containing all (T, p) tabulations
    concatenated end-to-end.  Each tabulation is one header line followed by
    cross-section data lines; concatenation is the literal joining of all
    selected files.

    The output is written to the package XSC directory (``__XSCPATH__``,
    i.e. ``$RFM_DIR/xsc/``) by default, keeping all cross-section files in
    one place alongside the HITRAN and atmosphere files.  The directory is
    created if it does not already exist.

    Parameters
    ----------
    files : list of str or Path
        XSC file paths to include.  Pass ``sorted(glob.glob('*.xsc'))`` or
        similar.  The output file is automatically excluded if it appears in
        this list (safe to glob the XSC directory itself on subsequent calls).
    output : str, optional
        Destination file path.  Defaults to
        ``$RFM_DIR/xsc/{molecule_name}.xsc`` where ``molecule_name`` is the
        lowercase species name read from the first input file header (e.g.
        ``sf6.xsc``).
    T_range : tuple of (float, float), optional
        ``(T_min, T_max)`` in K.  Tabulations outside this range are excluded.
    p_range : tuple of (float, float), optional
        ``(p_min, p_max)`` in **mbar**.  Tabulations outside this range are
        excluded.  Note that HITRAN stores pressure in Torr; the conversion
        (1 Torr = 1.33322 mbar) is applied internally.
    verbose : bool
        If True (default), print a summary table of included tabulations and
        warn about any inconsistencies.

    Returns
    -------
    pd.DataFrame
        Summary table of the included tabulations (same as
        :func:`readXSCHeaders` output).

    Raises
    ------
    ValueError
        If no files remain after filtering, or if the selected files span
        more than one spectral range (which would confuse RFM's interpolation).

    Notes
    -----
    **Spectral grid consistency**: files from the same measurement campaign
    typically share the same WN_start/WN_end but differ in NPTS (the spectral
    resolution varies with pressure — lower pressure → narrower lines →
    finer grid needed).  This is normal and RFM handles it correctly.

    **Mixing spectral ranges**: including files with different WN extents
    (e.g. 925–955 cm⁻¹ alongside 780–1100 cm⁻¹) will trigger a warning.
    RFM can technically handle this, but the (T, p) triangulation will be
    inconsistent across wavenumber — some (T, p) nodes will only exist for
    part of the spectral range.  Prefer a single consistent dataset.
    """
    # ---- Parse all headers (output path not known yet — skip after resolving) ----
    rows = []
    for fpath in sorted(str(f) for f in files):
        with open(fpath, 'r') as f:
            hdr = f.readline()
        rec = _parseXSCHeader(hdr)
        rec['file'] = fpath
        rows.append(rec)

    if not rows:
        raise ValueError("No XSC files provided.")

    # ---- Resolve output path ----
    if output is None:
        mol_name = rows[0]['name'].lower()
        xsc_dir = Path(get_xsc_dir())
        xsc_dir.mkdir(parents=True, exist_ok=True)
        output = str(xsc_dir / f'{mol_name}.xsc')

    output_abs = str(Path(output).resolve())

    # ---- Drop the output file if it appeared in the input list ----
    n_before = len(rows)
    rows = [r for r in rows if str(Path(r['file']).resolve()) != output_abs]
    if len(rows) < n_before:
        logger.warning("concatXSC: skipping output file found in input list: %s", output)

    df = pd.DataFrame(rows).sort_values(['T', 'p_torr']).reset_index(drop=True)

    # ---- Apply filters ----
    if T_range is not None:
        mask = (df['T'] >= T_range[0]) & (df['T'] <= T_range[1])
        n_dropped = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        if verbose and n_dropped:
            logger.info("concatXSC: dropped %d tabulation(s) outside T_range %s K",
                        n_dropped, T_range)

    if p_range is not None:
        mask = (df['p_mbar'] >= p_range[0]) & (df['p_mbar'] <= p_range[1])
        n_dropped = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        if verbose and n_dropped:
            logger.info("concatXSC: dropped %d tabulation(s) outside p_range %s mbar",
                        n_dropped, p_range)

    if df.empty:
        raise ValueError("No XSC tabulations remain after filtering.")

    # ---- Warn about mixed spectral ranges ----
    wn_combos = df[['wn_start', 'wn_end']].drop_duplicates()
    if len(wn_combos) > 1:
        import logging as _logging
        logger.warning(
            "concatXSC: files span %d different spectral ranges:\n%s\n"
            "RFM can read this but the (T,p) triangulation will be "
            "inconsistent across wavenumber. Consider using a single "
            "spectral dataset.",
            len(wn_combos),
            wn_combos.to_string(index=False)
        )

    # ---- Warn about (T, p) duplicates in the source files ----
    # Round to 0.5 K / 0.5 Torr to catch near-identical measurements
    df['_T_r'] = (df['T'] / 0.5).round() * 0.5
    df['_p_r'] = (df['p_torr'] / 0.5).round() * 0.5
    dups = df[df.duplicated(subset=['_T_r', '_p_r'], keep=False)]
    if not dups.empty:
        logger.warning(
            "concatXSC: %d tabulation(s) share nearly identical (T, p) values "
            "— keeping all, but RFM may behave unexpectedly:\n%s",
            len(dups),
            dups[['file', 'T', 'p_torr']].to_string(index=False)
        )
    df = df.drop(columns=['_T_r', '_p_r'])

    # ---- Print summary ----
    if verbose:
        print(f"concatXSC: including {len(df)} tabulation(s) → {output}")
        print(df[['T', 'p_torr', 'p_mbar', 'wn_start', 'wn_end', 'npts', 'spacing']]
              .rename(columns={'p_torr': 'p[Torr]', 'p_mbar': 'p[mbar]',
                               'wn_start': 'WN1', 'wn_end': 'WN2',
                               'spacing': 'Δν[cm⁻¹]'})
              .to_string(index=False))
        print()

    # ---- Write concatenated file (create parent dir if needed) ----
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as out:
        for fpath in df['file']:
            with open(fpath, 'r') as inp:
                out.write(inp.read())
            # Ensure a newline between blocks (some files may omit trailing newline)
            out.write('\n')

    if verbose:
        import os as _os
        size_mb = _os.path.getsize(output) / 1e6
        print(f"✓ Written: {output}  ({size_mb:.1f} MB, {len(df)} blocks)")

    return df


# ===========================================================================
# ========================= Filter convolution ==============================
# ===========================================================================

_FILTERS_DIR = _PACKAGE_DIR / 'filters'


def listSensors() -> list[str]:
    """
    List all available filter sensor/platform paths for :func:`filterConvolve`.

    For single-platform instruments (e.g. ``'ASTER'``) the name is the folder
    itself.  For multi-platform instruments the name includes the platform
    subfolder (e.g. ``'GOES-ABI/GOES-16'``).

    Returns
    -------
    list of str
        Relative paths from the package ``filters/`` directory to every leaf
        folder that contains at least one CSV file.
    """
    return sorted({
        str(p.parent.relative_to(_FILTERS_DIR))
        for p in _FILTERS_DIR.rglob('*.csv')
    })


def listPlatforms(instrument: str) -> list[str]:
    """
    List available platform variants for a multi-platform instrument.

    Parameters
    ----------
    instrument : str
        Instrument folder name (e.g. ``'GOES-ABI'``, ``'MSG-SEVIRI'``).

    Returns
    -------
    list of str
        Platform sub-folder names, sorted alphabetically.

    Raises
    ------
    ValueError
        If ``instrument`` is not found or has no platform sub-directories
        (i.e. it is a single-platform instrument — use :func:`listSensors`
        directly in that case).
    """
    inst_dir = _FILTERS_DIR / instrument
    if not inst_dir.is_dir():
        raise ValueError(
            f"Unknown instrument {instrument!r}. Use listSensors() to see all available."
        )
    platforms = sorted(d.name for d in inst_dir.iterdir() if d.is_dir())
    if not platforms:
        raise ValueError(
            f"{instrument!r} has no platform sub-directories. "
            "Use listSensors() to reference it directly."
        )
    return platforms


def readRTTOVsrf(filepath: str | Path) -> dict:
    """
    Parse a single RTTOV NWP-SAF spectral response function file.

    The RTTOV ``.txt`` format consists of a 4-line header followed by two
    columns of data:

    .. code-block:: text

        <channel_num>, <label> ; <source>
        Number of data points:
        <npts>
        Wavenumber (cm-1)   Filter response
        <wn_1>   <T_1>
        ...

    Parameters
    ----------
    filepath : str or Path
        Path to the RTTOV ``.txt`` SRF file.

    Returns
    -------
    dict
        Keys:

        * ``'wn'``    — wavenumber axis [cm⁻¹], ascending, shape ``(npts,)``
        * ``'wl'``    — wavelength axis [µm], derived as ``10000 / wn``, shape ``(npts,)``
        * ``'T'``     — filter response [0–1], shape ``(npts,)``
        * ``'label'`` — band label string extracted from header line 1
        * ``'npts'``  — number of data points declared in the header
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        line1 = f.readline()
        f.readline()               # "Number of data points:"
        npts  = int(f.readline())
        f.readline()               # "Wavenumber (cm-1)   Filter response"
        data  = np.loadtxt(f)

    # Extract band label: everything between the first comma and the semicolon
    # (or end of line if no semicolon).
    after_comma = line1.split(',', 1)[1] if ',' in line1 else line1
    label = after_comma.split(';')[0].strip()

    wn = data[:, 0]
    T  = data[:, 1]

    # Ensure ascending wavenumber order (some source files are descending)
    if len(wn) > 1 and wn[0] > wn[-1]:
        wn = wn[::-1]
        T  = T[::-1]

    return {
        'label': label,
        'wn':    wn,
        'wl':    1e4 / wn,
        'T':     T,
        'npts':  npts,
    }


def _loadFilters(sensor: str) -> dict:
    """
    Load all filter CSVs for a named sensor/platform from the bundled directory.

    Each CSV must have a ``wn`` [cm⁻¹] column and a ``T`` [0–1] column.  A
    ``wl`` [µm] column is optional — if absent it is derived as ``10000/wn``.
    The filename stem (without ``.csv``) is used as the band name.

    The ``sensor`` string may include a platform sub-path
    (e.g. ``'GOES-ABI/GOES-16'`` or simply ``'ASTER'``).

    Parameters
    ----------
    sensor : str
        Sensor path relative to the package ``filters/`` directory.

    Returns
    -------
    dict
        Mapping band name → ``{'wn': ndarray, 'wl': ndarray, 'T': ndarray}``,
        with both axes sorted ascending by wavenumber.
    """
    sensor_dir = _FILTERS_DIR / sensor
    if not sensor_dir.is_dir():
        available = listSensors()
        raise ValueError(f"Unknown sensor {sensor!r}. Available: {available}")
    filters = {}
    for csv in sorted(sensor_dir.glob('*.csv')):
        df = pd.read_csv(csv, comment='#')
        wn = df['wn'].values.astype(float)
        wl = df['wl'].values.astype(float) if 'wl' in df.columns else 1e4 / wn
        T  = df['T'].values.astype(float)
        # Guarantee ascending wavenumber order for np.interp
        order = np.argsort(wn)
        filters[csv.stem] = {
            'wn': wn[order],
            'wl': wl[order],
            'T':  T[order],
        }
    if not filters:
        raise ValueError(f"No CSV files found in filter directory: {sensor_dir}")
    return filters


def filterConvolve(x: np.ndarray, y: np.ndarray,
                   sensor: str | dict, method: str = 'avg') -> dict:
    """
    Convolve a spectrum with imager filter response functions.

    Convolution is performed in wavenumber space: ``x`` must be a wavenumber
    axis [cm⁻¹].  No Jacobian correction is applied — the output value is the
    transmission-weighted average (or integral) of ``y`` in wavenumber space,
    which is the physically correct radiometric integral.

    Parameters
    ----------
    x : array_like
        Wavenumber axis [cm⁻¹], monotonically increasing.
    y : array_like
        Spectral values aligned with ``x``.
    sensor : str or dict
        Name of a bundled sensor (e.g. ``'ASTER'``, ``'GOES-ABI/GOES-16'``) or
        a dict mapping band name → ``{'wn': ndarray, 'T': ndarray}``.
        Use :func:`listSensors` to see available built-in sensors.
    method : str
        ``'avg'`` — transmission-weighted average (default).
        ``'integ'`` — trapezoid integral over the passband.

    Returns
    -------
    dict
        Per-band results keyed by band name, each containing:

        * ``'center_wn'`` — band centre wavenumber [cm⁻¹]
        * ``'center_wl'`` — band centre wavelength [µm]
        * ``'value'``     — convolved spectral value
        * ``'T'``         — filter transmission interpolated onto ``x``
        * ``'through'``   — ``y * T``

        Plus summary arrays ``'wn'``, ``'wl'``, ``'spec'`` (one entry per band)
        and ``'filters'`` (the raw filter dict).
    """
    filters = _loadFilters(sensor) if isinstance(sensor, str) else sensor

    out = {}
    wn_centers, wl_centers, spec_values = [], [], []

    for name, filt in filters.items():
        wn_f, T_f = filt['wn'], filt['T']
        T = np.interp(x, wn_f, T_f, left=0.0, right=0.0)
        passband = wn_f[T_f > 0.01]
        center_wn = float(np.median(passband)) if len(passband) else float(np.mean(wn_f))
        center_wl = float(1e4 / center_wn)

        if method == 'avg':
            value = float(np.average(y, weights=T)) if T.sum() > 0 else np.nan
        elif method == 'integ':
            mask = T > 0
            value = float(_trapz(y[mask] * T[mask], x[mask])) if mask.any() else np.nan
        else:
            raise ValueError(f"method must be 'avg' or 'integ', got {method!r}")

        out[name] = {
            'center_wn': center_wn, 'center_wl': center_wl,
            'value': value, 'T': T, 'through': y * T,
        }
        wn_centers.append(center_wn)
        wl_centers.append(center_wl)
        spec_values.append(value)

    out['wn']      = np.array(wn_centers)
    out['wl']      = np.array(wl_centers)
    out['spec']    = np.array(spec_values)
    out['filters'] = filters
    return out


# ===========================================================================
# ========================= Instrument library ================================
# ===========================================================================

_INSTRUMENTS_FILE = _PACKAGE_DIR / 'instruments.json'


def listInstruments() -> list[str]:
    """
    List the instrument names available in the bundled instrument library.

    Returns
    -------
    list of str
        Instrument names as stored in ``instruments.json``.
    """
    import json
    with open(_INSTRUMENTS_FILE) as f:
        data = json.load(f)
    return sorted(data.keys())


def loadILSParams(instrument: str) -> dict:
    """
    Load ILS parameters for a named instrument from the bundled library.

    Parameters
    ----------
    instrument : str
        Instrument name (case-sensitive).  Use :func:`listInstruments` to
        see available names.

    Returns
    -------
    dict
        Keys: ``'max_opd'`` [cm], ``'fov'`` [rad], ``'apod_type'`` [str],
        ``'description'`` [str].  Instruments that define a native output
        grid also include ``'spacing'`` [cm⁻¹]; pass ``resample=True`` to
        :func:`applyILS` to resample the convolved spectrum to this grid.

    Raises
    ------
    ValueError
        If ``instrument`` is not found in the library.
    """
    import json
    with open(_INSTRUMENTS_FILE) as f:
        data = json.load(f)
    if instrument not in data:
        available = sorted(data.keys())
        raise ValueError(f"Unknown instrument {instrument!r}. Available: {available}")
    return dict(data[instrument])


# ===========================================================================
# ========================= ILS kernel =======================================
# ===========================================================================

_APOD_TYPES = [
    'Boxcar', 'Uniform', 'Triangular', 'Hann', 'Hanning', 'Hamming', 'Happ-Genzel', 'Blackman-Harris',
    'Nuttall', 'Kaiser', 'Tukey', 'Cosine', 'Lorenz', 'Gaussian', 'NB weak', 'NB medium', 'NB strong',
]



def makeILS(fov: float = 0.03, max_opd: float = 1.8, nper_wn: float = 25,
            wn_center: float = 1000.0, apod_type: str = 'NB medium',
            fwhm_wn: float = None) -> dict:
    """
    Compute an Instrument Line Shape (ILS) kernel for an FT‑IR spectrometer.

    Parameters
    ----------
    fov : float
        Full-angle field of view in radians.
    max_opd : float
        Maximum optical path difference in cm.
    nper_wn : float
        Number of kernel grid points per cm⁻¹ (typically 1 / spectral spacing).
    wn_center : float
        Centre wavenumber of the window in cm⁻¹, setting the FOV width via
        ``fov_width = wn_center * (fov/2)**2 / 2``.  This is the window
        *centre*, not its width — passing half the window width (as plumeIR v2
        did) shrinks ``fov_width`` by ~40x and effectively disables the FOV
        effect.  Renamed from ``wn`` to make that misuse harder.
    apod_type : str
        Apodization type. One of: 'Boxcar', 'Triangular', 'Hann', 'Hamming',
        'Happ-Genzel', 'Blackman-Harris', 'Nuttall', 'Kaiser', 'Tukey',
        'Cosine', 'Lorenz', 'Gaussian', 'NB weak', 'NB medium', 'NB strong'.
    fwhm_wn : float, optional
        Target spectral FWHM in cm⁻¹, used **only** for the Gaussian apodization
        to derive the OPD-domain width from the relation
        ``α = π · max_opd · fwhm_wn / (2·√(ln 2))``.
        If *None* and apod_type is 'Gaussian', the default ``α = 2.24`` is used
        (width ≈ 0.9 / max_opd in cm⁻¹).

    Returns
    -------
    dict
        Keys: 'wn_grid', 'kernel', 'L_grid', 'apod', 'apod_type', 'fov',
        'fov_width', 'wn_center'.  'apod' is the apodization window *before*
        FOV self-apodization, so it remains bounded by [0, 1].
    """

    out = {'apod_type': apod_type, 'fov': fov, 'wn_center': wn_center}
    apod_key = apod_type.lower()

    L = max_opd
    _kernel_opd = nper_wn

    n = int(L * nper_wn)
    if n % 2 == 0:
        n += 1
    L_grid = np.linspace(0, L, n)

    n_tot  = int(_kernel_opd * nper_wn)
    filler = np.zeros(n_tot - n)

    # ============================================================
    #                     APO D I Z A T I O N S
    # ============================================================

    # --- Boxcar / Uniform ---
    if 'boxcar' in apod_key or 'uniform' in apod_key:
        apod = np.ones(n)

    # --- Triangular ---
    elif 'triang' in apod_key:
        apod = 1 - np.abs(L_grid) / L

    # --- Norton–Beer ---
    elif 'nb' in apod_key or 'norton' in apod_key or 'beer' in apod_key:
        # Coefficients from Norton & Beer (1976), J. Opt. Soc. Am. 66, 259.
        # Note: NB weak coefficients sum to 0.964 at OPD=0, not 1.0 — this
        # is a known property of the published values, not an error.
        nb_coeffs = {
            'weak':   [0.348093, -0.087577,  0.703484, 0.0],
            'strong': [0.045335,  0.0,        0.554883, 0.399782],
            'medium': [0.152442, -0.136176,  0.983734, 0.0],
        }
        key = 'strong' if 'strong' in apod_key else ('weak' if 'weak' in apod_key else 'medium')
        c = nb_coeffs[key]
        t = (L_grid / L) ** 2
        apod = sum(c[i] * (1 - t) ** i for i in range(4))

    # --- Hann (Hanning) ---
    elif 'hann' in apod_key or 'hanning' in apod_key:
        x = L_grid / L
        apod = 0.5 + 0.5 * np.cos(np.pi * x)

    # --- Hamming ---
    elif 'hamming' in apod_key:
        x = L_grid / L
        apod = 0.54 + 0.46 * np.cos(np.pi * x)

    # --- Happ–Genzel (Hann–Poisson) ---
    elif 'happ' in apod_key or 'genzel' in apod_key:
        alpha = 2.5   # typical spectroscopy value
        x = L_grid / L
        apod = np.exp(-alpha * np.abs(x)) * (0.5 + 0.5 * np.cos(np.pi * x))

    # --- Blackman–Harris (3‑term) ---
    elif 'blackman' in apod_key or 'harris' in apod_key:
        x = L_grid / L
        apod = (0.42
                + 0.5 * np.cos(np.pi * x)
                + 0.08 * np.cos(2 * np.pi * x))

    # --- Nuttall (4‑term continuous) ---
    elif 'nuttall' in apod_key:
        x = L_grid / L
        apod = (0.355768
                + 0.487396 * np.cos(np.pi * x)
                + 0.144232 * np.cos(2 * np.pi * x)
                + 0.012604 * np.cos(3 * np.pi * x))

    # --- Kaiser ---
    elif 'kaiser' in apod_key:
        beta = 8.6  # typical for −90 dB sidelobes
        x = L_grid / L
        apod = np.i0(beta * np.sqrt(1 - x**2)) / np.i0(beta)

    # --- Tukey (tapered cosine) ---
    # One-sided OPD taper: flat at 1.0 from x=0 to x=1-r, then cosine ramp
    # down to 0 over the final fraction r.  This preserves apod[0]=1 at the
    # centerburst and tapers only the far end of the interferogram.
    elif 'tukey' in apod_key:
        r = 0.5   # fraction of the OPD range that is tapered
        x = L_grid / L
        apod = np.ones_like(x)
        mask = x > 1 - r
        apod[mask] = 0.5 * (1 - np.cos(np.pi * (1 - x[mask]) / r))

    # --- Cosine ---
    elif 'cos' in apod_key:
        apod = np.cos(np.pi * L_grid / (2 * L))

    # --- Lorentz ---
    elif 'lorenz' in apod_key:
        apod = np.exp(-np.abs(L_grid) / L)

    # --- Gaussian ---
    # α sets the width of the OPD-domain Gaussian relative to max_opd.
    # When fwhm_wn is given, α is derived so that the spectral FWHM of the
    # resulting ILS matches the target:  α = π·L·FWHM / (2·√(ln 2))
    # The default α=2.24 gives FWHM ≈ 0.9/L cm⁻¹ (reasonable generic default).
    elif 'gauss' in apod_key:
        if fwhm_wn is not None:
            alpha = np.pi * L * fwhm_wn / (2.0 * np.sqrt(np.log(2.0)))
        else:
            alpha = 2.24
        apod = np.exp(-(alpha * L_grid / L) ** 2)

    else:
        raise ValueError(f"Invalid apod_type {apod_type!r}.")

    # ============================================================
    #                     FFT → ILS KERNEL
    # ============================================================

    # ============================================================
    #                     FOV SELF-APODIZATION
    # ============================================================
    # Broadening by a rectangle of width fov_width in wavenumber is exactly
    # multiplication by sinc(fov_width * OPD) in the interferogram domain
    # (np.sinc is the normalised sin(pi z)/(pi z)).
    #
    # This replaces a boxcar convolution of ceil(fov_width * nper_wn) grid
    # points.  That integer count over-broadened the FOV term by up to a full
    # grid cell — 60% at small fov / low wavenumber, erratic elsewhere — and
    # silently dropped the FOV effect altogether whenever fov_width fell below
    # one grid spacing (n_box forced to 1 = identity).  The sinc form is exact
    # rather than discretized, and continuous in fov at any magnitude.
    #
    # Applied to a copy: out['apod'] keeps the apodization function proper,
    # which stays within [0, 1]; the sinc goes negative past its first zero
    # (fov_width * max_opd > 1).

    fov_width = wn_center * (fov / 2) ** 2 / 2
    apod_fov = apod * np.sinc(fov_width * L_grid)

    # fftshift both halves of the transform pair so the kernel's DC component
    # and wn_grid's zero coincide *by construction*.
    #
    # The previous form rolled by ``mid = n_tot // 2`` but built the grid with
    # ``linspace(..., n_tot)``, which places zero at index ``(n_tot - 1) / 2``.
    # Those differ by half a sample for even ``n_tot`` and a full sample for
    # odd, so the kernel sat off its own grid: measured with ``fov = 0`` (no
    # intentional shift, peak must be at zero) the offset was exactly 0.500 and
    # 1.000 samples respectively.  For a convolution-only consumer that is a
    # small spectral shift; plumeIR, which *fits* a wavenumber shift, was
    # silently absorbing it into ``nu_shift``, where it varied with nper_wn.
    dx      = float(L_grid[1] - L_grid[0])
    kernel  = np.fft.fftshift(np.fft.fft(np.concatenate((apod_fov, filler))).real)
    wn_grid = np.fft.fftshift(np.fft.fftfreq(n_tot, d=dx))
    kernel  = kernel - kernel[:n_tot // 4].mean()

    # A finite FOV displaces line centres as well as broadening them.
    shift = fov_width / 2
    kernel = np.interp(wn_grid, wn_grid - shift, kernel)

    # ============================================================
    #                     TRIM TO ±5 SINC LIMBS
    # ============================================================

    idx = np.abs(wn_grid) <= 5 / L
    wn_grid = wn_grid[idx]
    kernel  = kernel[idx]

    min_len = min(len(wn_grid), len(kernel))
    wn_grid = wn_grid[:min_len]
    kernel  = kernel[:min_len]

    # Normalise after trimming so the kernel always sums to 1.0.
    # Normalising before the trim causes drift because sidelobe energy
    # outside ±5/L is discarded.
    kernel = kernel / kernel.sum()

    out.update({
        'wn_grid':   wn_grid,
        'kernel':    kernel,
        'L_grid':    L_grid,
        'apod':      apod,
        'fov_width': fov_width,
    })
    return out



# ===========================================================================
# ========================= Spectral resampling ==============================
# ===========================================================================

def resampleSpectra(spectra, dx: float = None,
                    x_out: np.ndarray = None):
    """
    Resample a :class:`Spectra` or :class:`ILSOut` onto a new axis.

    Exactly one of ``dx`` or ``x_out`` must be supplied.  Interpolation is
    linear (``np.interp``), which is appropriate after ILS convolution has
    removed spectral content above the Nyquist of the target grid.

    Parameters
    ----------
    spectra : Spectra or ILSOut
        Input spectral data.  The axis is ``spectra.wn`` regardless of whether
        ``xunit`` is ``'wn'`` or ``'wl'``; the ``dx`` / ``x_out`` units must
        match accordingly.
    dx : float, optional
        Uniform grid spacing in the same units as ``spectra.wn``.  The output
        axis spans ``[spectra.wn[0], spectra.wn[-1]]`` at this interval.
    x_out : np.ndarray, optional
        Explicit output axis.  Must lie entirely within
        ``[spectra.wn[0], spectra.wn[-1]]``; raises ``ValueError`` otherwise.

    Returns
    -------
    Spectra or ILSOut
        New instance of the same type with the resampled axis and data.  All
        metadata fields (``labels``, ``label_type``, ``output``, ``xunit``,
        ``yunit``) are preserved unchanged.

    Raises
    ------
    ValueError
        If neither or both of ``dx`` / ``x_out`` are supplied, or if
        ``x_out`` extends outside the input axis range.
    """
    if (dx is None) == (x_out is None):
        raise ValueError("Supply exactly one of 'dx' or 'x_out'.")

    x_in = spectra.wn

    if dx is not None:
        # np.arange can miss the last point by a tiny floating-point margin;
        # adding dx/2 guarantees inclusion without overshooting.
        x_out = np.arange(x_in[0], x_in[-1] + dx / 2.0, dx)

    else:
        x_out = np.asarray(x_out, dtype=np.float64)
        if x_out[0] < x_in[0] or x_out[-1] > x_in[-1]:
            raise ValueError(
                f"x_out [{x_out[0]:.4g}, {x_out[-1]:.4g}] extends outside "
                f"the input axis range [{x_in[0]:.4g}, {x_in[-1]:.4g}]."
            )

    # Interpolate each spectrum row independently; np.interp is vectorised
    # over evaluation points but not over rows, so we broadcast manually.
    # Shape: (n_spectra, len(x_out))
    data_out = np.stack(
        [np.interp(x_out, x_in, row) for row in spectra.data],
        axis=0,
    )

    # Import here to avoid a circular import at module level; objects imports
    # resampleSpectra from functions, so we must not import at the top.
    from .objects import Spectra as _Spectra, ILSOut as _ILSOut

    if isinstance(spectra, _Spectra):
        return _Spectra(
            wn=x_out,
            data=data_out,
            labels=spectra.labels.copy(),
            label_type=spectra.label_type,
            output=spectra.output,
            xunit=spectra.xunit,
            yunit=spectra.yunit,
        )
    elif isinstance(spectra, _ILSOut):
        return _ILSOut(
            wn=x_out,
            data=data_out,
            labels=spectra.labels.copy(),
            label_type=spectra.label_type,
            output=spectra.output,
            ils=spectra.ils,
            xunit=spectra.xunit,
            yunit=spectra.yunit,
        )
    else:
        raise TypeError(
            f"resampleSpectra expects a Spectra or ILSOut, got {type(spectra).__name__!r}."
        )


# ===========================================================================
# ====================== Ray-path diagnostics ================================
# ===========================================================================

def parseRayPath(lines: list[str]) -> dict:
    """
    Parse the raw text lines of one RFM ``pth*.out`` file into structured data.

    Each ``pth*.out`` file holds the ray-path diagnostic for one elevation angle
    (nadir/zenith) or one tangent height (limb).

    **Limb geometry**: ``Len`` and ``Amt`` are true slant quantities — RFM
    traces the curved ray and reports the actual slant path per layer.

    **Nadir/zenith geometry (RFM < v5.21)**: RFM reports the *vertical* layer
    thickness and vertical column amount (``Zen = 0`` for every layer).  This
    function applies the ``1 / cos(VZA)`` slant-path correction automatically
    so that ``len`` and ``amt`` are always slant quantities.  The correction
    factor is stored in ``header['slant_factor']`` for transparency.

    **Nadir/zenith geometry (RFM v5.21+)**: RFM applies the ``SECFAC =
    1 / cos(VZA)`` correction internally in ``pthwrt_sub.f90`` (Bug#35 fix)
    before writing the file.  ``Len``, ``Amt``, and the ``Zen`` column already
    carry slant quantities; no further correction is applied here.

    **GRA flag — 2D atmosphere (NSeg2 > 0)**: When RFM is run with the GRA
    flag, the ray traverses two different horizontal columns on the downward
    (observer→tangent) and upward (tangent→observer) legs.  The PTH file
    contains two data blocks per gas, each followed by its own ``Total:`` line.
    The returned dict includes the additional keys ``'gases_up'`` and
    ``'totals_up'`` for the upward segment.  The primary ``'gases'`` and
    ``'totals'`` keys always carry the downward (first) segment, ensuring
    backward compatibility with non-GRA code.  The ``Zen`` column is named
    ``Psi`` in the raw file for GRA mode (horizontal angle in 2D atmosphere)
    but is mapped to the same ``zen`` column in the DataFrame.

    Parameters
    ----------
    lines : list of str
        Raw text lines as returned by :meth:`RFM.getRayPath`.

    Returns
    -------
    dict with keys:

    ``'header'`` : dict
        Scalar geometry info extracted from the file header:

        - ``'geometry'``    : ``'limb'`` or ``'nadir'`` (str)
        - ``'tan_hgt'``     : tangent/refracted height [km] (float or None)
        - ``'ele_ang'``     : elevation angle [deg] (float or None)
        - ``'rfm_version'`` : version string from header, e.g. ``'5.21_18DEC'``
          (str or None)
        - ``'slant_factor'``: ``1/cos(VZA)`` factor — applied by this parser
          (RFM < v5.21) or recorded from RFM output (v5.21+). Always 1.0 for
          limb and GRA (no SECFAC in those modes).
        - ``'one_way'``     : bool — for standard limb with ``NSeg2 = 0``,
          ``True`` means only the observer→tangent half is stored; full limb
          column = ``2 × totals[gas]['amt']``.
        - ``'gra'``         : bool — ``True`` when ``NSeg2 > 0`` (GRA flag
          active, 2D atmosphere).
        - ``'n_gas'``       : number of gases (int)
        - ``'n_seg'``       : NSeg1, number of downward segments (int)
        - ``'n_seg2'``      : NSeg2 (int); 0 unless GRA flag active

    ``'gases'`` : dict of str → :class:`pandas.DataFrame`
        Downward-segment (or only segment for non-GRA) data per gas.
        Columns: ``lev`` (int), ``z_low`` (float, km), ``zen`` (float, deg),
        ``temp`` (float, K), ``press`` (float, mb), ``vmr`` (float, ppv),
        ``amt`` (float, slant kmol cm⁻²), ``len`` (float, slant km).
        ``temp`` and ``press`` are Curtis-Godson weighted means.

    ``'totals'`` : dict of str → dict
        Downward-segment totals: ``{'amt': float, 'len': float}``.
        For standard limb with ``one_way=True``, multiply by 2 for the full
        round-trip optical path.

    ``'gases_up'`` : dict of str → :class:`pandas.DataFrame`
        Upward-segment data per gas.  **Present only when** ``header['gra']``
        is ``True``; absent otherwise.

    ``'totals_up'`` : dict of str → dict
        Upward-segment totals.  **Present only when** ``header['gra']`` is
        ``True``; absent otherwise.
    """
    import math
    import pandas as pd
    import re

    header = {
        'geometry':    None,
        'tan_hgt':     None,
        'ele_ang':     None,
        'rfm_version': None,
        'slant_factor': 1.0,
        'one_way':     False,
        'gra':         False,
        'n_gas':       None,
        'n_seg':       None,
        'n_seg2':      None,
    }

    # ---- parse file header -------------------------------------------------
    for line in lines[:10]:
        s = line.strip()
        if 'Limb-viewing' in s:
            header['geometry'] = 'limb'
        elif ('Downward-looking' in s or 'Upward-looking' in s
              or 'Nadir' in s or 'Zenith' in s):
            header['geometry'] = 'nadir'
        m = re.search(r'by RFM v\.([\w.]+)', s)
        if m:
            header['rfm_version'] = m.group(1)
        m = re.search(r'Tan\s+Hgt\s*=\s*([\d.]+)', s)
        if m:
            header['tan_hgt'] = float(m.group(1))
            header['geometry'] = 'limb'
        m = re.search(r'Ele\.Ang\s*=\s*([-\d.]+)', s)
        if m:
            header['ele_ang'] = float(m.group(1))
        m = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s*=\s*NGas', s)
        if m:
            header['n_gas']  = int(m.group(1))
            header['n_seg']  = int(m.group(2))
            header['n_seg2'] = int(m.group(3))
            n_seg2 = int(m.group(3))
            if n_seg2 > 0:
                header['gra'] = True
            elif header['geometry'] == 'limb':
                header['one_way'] = True

    # Determine whether slant correction has already been applied by RFM.
    # Bug#35 (v5.21) introduced SECFAC in pthwrt_sub for NAD/ZEN/HOM paths.
    # GRA paths do not use SECFAC regardless of version.
    rfm_ver = header['rfm_version']
    try:
        ver_num = float(rfm_ver.split('_')[0]) if rfm_ver else 0.0
    except (ValueError, AttributeError):
        ver_num = 0.0
    rfm_applies_secfac = (ver_num >= 5.21) and not header['gra']

    # ---- parse per-gas tables ----------------------------------------------
    # For each gas track two row buffers (dn, up) and two total slots.
    # Non-GRA files only ever fill the dn buffer.
    data_cols = ['lev', 'z_low', 'zen', 'temp', 'press', 'vmr', 'amt', 'len']

    # buf[gas] = [dn_rows, up_rows]
    buf: dict[str, list[list]] = {}
    # tot[gas] = [dn_total_or_None, up_total_or_None]
    tot: dict[str, list] = {}

    current_gas = None

    for line in lines:
        s = line.strip()
        if not s or s.startswith('!'):
            continue

        # gas header: bare word on its own line
        if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*\s*$', s):
            current_gas = s.strip().upper()
            if current_gas not in buf:
                buf[current_gas] = [[], []]   # [dn_rows, up_rows]
                tot[current_gas] = [None, None]
            continue

        # skip column-header line (Lev ... Zen[dg] or Psi[dg] ...)
        if s.startswith('Lev'):
            continue

        # NGas/NSeg summary line — already parsed
        if re.match(r'\s*\d+\s+\d+\s+\d+\s*=\s*NGas', s):
            continue

        # Total line — closes the current segment buffer for this gas
        m = re.match(r'.*Total:\s+([\dE+\-\.]+)\s+([\dE+\-\.]+)', s)
        if m and current_gas is not None:
            entry = {'amt': float(m.group(1)), 'len': float(m.group(2))}
            if tot[current_gas][0] is None:
                tot[current_gas][0] = entry   # first Total → downward
            else:
                tot[current_gas][1] = entry   # second Total → upward
            continue

        # data row: starts with an integer level index
        if current_gas is not None:
            m = re.match(
                r'\s*(\d+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)'
                r'\s+([\d.E+\-]+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)\s+([\d.E+\-]+)',
                s
            )
            if m:
                row = [int(m.group(1))] + [float(m.group(i)) for i in range(2, 9)]
                # Rows belong to the upward segment once the downward Total
                # has been recorded.
                seg = 1 if tot[current_gas][0] is not None else 0
                buf[current_gas][seg].append(row)

    def _to_df(rows: list) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=data_cols)

    gases_dn = {g: _to_df(b[0]) for g, b in buf.items() if b[0]}
    gases_up = {g: _to_df(b[1]) for g, b in buf.items() if b[1]}
    totals_dn = {g: t[0] for g, t in tot.items() if t[0] is not None}
    totals_up = {g: t[1] for g, t in tot.items() if t[1] is not None}

    # For nadir/zenith, ensure len and amt are slant quantities.
    # RFM < v5.21: file contains vertical values — apply 1/cos(VZA) here.
    # RFM v5.21+:  SECFAC already applied inside pthwrt_sub.f90.
    # GRA mode:    two distinct horizontal columns; no SECFAC in either version.
    if header['geometry'] == 'nadir' and header['ele_ang'] is not None:
        vza_rad = math.radians(90.0 + header['ele_ang'])
        slant = 1.0 / math.cos(vza_rad)
        header['slant_factor'] = slant
        if not rfm_applies_secfac:
            for df in gases_dn.values():
                df['len'] = df['len'] * slant
                df['amt'] = df['amt'] * slant
            for g in totals_dn:
                totals_dn[g]['len'] *= slant
                totals_dn[g]['amt'] *= slant
            # GRA upward segment (not expected for nadir, but guard anyway)
            for df in gases_up.values():
                df['len'] = df['len'] * slant
                df['amt'] = df['amt'] * slant
            for g in totals_up:
                totals_up[g]['len'] *= slant
                totals_up[g]['amt'] *= slant

    result = {
        'header':  header,
        'gases':   gases_dn,
        'totals':  totals_dn,
    }
    if header['gra']:
        result['gases_up']  = gases_up
        result['totals_up'] = totals_up

    return result


# ===========================================================================
# ========================= Plotting helpers =================================
# ===========================================================================

def plotGrid(N: int, ncols: int = None, nrows: int = None,
              sharex: bool = True, sharey: bool = False,
              cbar: bool = True) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a tidy subplot grid for N panels with an optional colorbar row.

    Parameters
    ----------
    N : int
        Number of data panels needed.
    ncols, nrows : int, optional
        Override automatic layout.
    sharex, sharey : bool
        Axis sharing flags.
    cbar : bool
        If True, append a thin colorbar row at the bottom.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes_flat : np.ndarray of Axes
        Flattened array; last element is the colorbar Axes when ``cbar=True``.
    """
    if ncols is not None and nrows is not None:
        pass
    elif ncols is not None:
        nrows = int(np.ceil(N / ncols))
    elif nrows is not None:
        ncols = int(np.ceil(N / nrows))
    else:
        if N < 4:
            nrows, ncols = 1, N
        elif N < 7:
            nrows, ncols = 2, 3
        elif N < 10:
            nrows, ncols = 3, 3
        else:
            nrows = 3
            ncols = int(np.ceil(N / nrows))

    scaling = 1 if N < 10 else 10 / N
    figsize = [ncols * 3 * scaling, nrows * 3 * scaling]
    width_ratios = [10] * ncols
    height_ratios = [10] * nrows

    if cbar:
        figsize[1] += figsize[1] / 10
        nrows += 1
        height_ratios.append(1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             width_ratios=width_ratios,
                             height_ratios=height_ratios,
                             sharex=sharex, sharey=sharey)
    axes_flat = axes.flatten()

    for j in range(N, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if cbar:
        cax = fig.add_subplot(axes[0, 0].get_gridspec()[nrows - 1, :])
        axes_flat = np.append(axes_flat, [cax])

    fig.tight_layout()
    plt.show(block=False)

    return fig, axes_flat
