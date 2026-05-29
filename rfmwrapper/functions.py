"""
Written by @jfsmekens - Last updated 15/02/2024

Utility functions for the RFM wrapper: atmosphere I/O, Planck / unit conversions,
filter convolution, and plotting helpers.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import CubicSpline
import pandas as pd

from .constants import weights

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

    Species absent from ``atm`` are seeded with their standard background VMR
    before the plume enhancement is applied.

    Parameters
    ----------
    atm : dict
        Atmosphere dictionary as returned by :func:`readAtm`.
    h : float
        Centre altitude of the plume [km].
    th : float
        Vertical thickness of the plume [km].
    plume : dict, optional
        Additive VMR enhancements keyed by gas name [ppmv]. Default is ``{}``.
    gradient_flag : int
        Concentration profile shape within the plume layer:

        * 0 — uniform (step function)
        * 1 — triangular (default)
        * 2 — Gaussian

    Returns
    -------
    dict
        New atmosphere dictionary with the plume layer inserted.
    """
    if plume is None:
        plume = {}

    h1 = h - th / 2
    h2 = h + th / 2

    atm = dict(atm)
    std = readStandard()
    for key in plume:
        if key not in atm:
            atm[key] = np.ones(len(atm['HGT'])) * std[key]['vmr']

    if gradient_flag == 0:
        xx = np.asarray([h1 - 0.001, h1, h2, h2 + 0.001])
        yy = np.asarray([0.0, 1.0, 1.0, 0.0])
    elif gradient_flag == 1:
        xx = np.asarray([h1, h, h2])
        yy = np.asarray([0.0, 1.0, 0.0])
    elif gradient_flag == 2:
        step = th / 9
        xx = np.arange(h1, h2 + step, step)
        sig = (h - h1) / 2.5
        yy = (1.0 / (np.sqrt(2.0 * np.pi) * sig)
              * np.exp(-np.power((xx - h) / sig, 2.0) / 2))
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
# ========================= Standards file ==================================
# ===========================================================================

def readStandard() -> dict:
    """
    Read default VMR and aerosol properties from the bundled ``standards.txt``.

    Returns
    -------
    dict
        Keyed by species name; each value is a dict of property name → value.
    """
    with open(_PACKAGE_DIR / 'standards.txt') as f:
        lines = f.readlines()

    line = [l for l in lines if l.startswith('*')][0]
    keys = [k.strip() for k in line.strip('*').split('#')[0].split(':')]

    lines = [l for l in lines if not l.startswith('\n')]
    lines = [l for l in lines if not l.startswith('*')]
    lines = [l for l in lines if not l.startswith('#')]

    std = {}
    for line in lines:
        line = line.split('#')[0].split('(')[0]
        values = line.split(':')
        for i, v in enumerate(values):
            v = v.strip()
            if v == 'None':
                v = None
            elif v == '':
                v = None
            elif v == 'True':
                v = True
            elif v == 'False':
                v = False
            elif len(v.split(',')) > 1:
                try:
                    v = [float(x) for x in v.split(',')]
                except ValueError:
                    v = [x.strip() for x in v.split(',')]
            else:
                try:
                    v = float(v)
                except ValueError:
                    pass
            values[i] = v
        for i, key in enumerate(keys):
            if i == 0:
                name = values[i]
                std[name] = {}
            else:
                std[name][key] = values[i]

    return std


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
        scd = np.trapz(n_gas, x=np.asarray(pathlength) * 1e3) / np.sin(np.deg2rad(elev))
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
# ========================= Filter convolution ==============================
# ===========================================================================

_FILTERS_DIR = _PACKAGE_DIR / 'filters'


def listSensors() -> list[str]:
    """
    List the bundled filter sensor names available for :func:`filterConvolve`.

    Returns
    -------
    list of str
        Names of sensor folders found in the package ``filters/`` directory.
    """
    return sorted(p.name for p in _FILTERS_DIR.iterdir() if p.is_dir())


def _loadFilters(sensor: str) -> dict:
    """
    Load all filter CSVs for a named sensor from the bundled filters directory.

    Each CSV must have ``wl`` [µm] and ``T`` [0–1] columns.  The filename
    stem (without ``.csv``) is used as the band name.

    Parameters
    ----------
    sensor : str
        Sensor folder name (e.g. ``'ASTER'``, ``'MMT-gasCam'``).

    Returns
    -------
    dict
        Mapping band name → ``{'wl': np.ndarray, 'T': np.ndarray}``.
    """
    sensor_dir = _FILTERS_DIR / sensor
    if not sensor_dir.is_dir():
        available = listSensors()
        raise ValueError(f"Unknown sensor {sensor!r}. Available: {available}")
    filters = {}
    for csv in sorted(sensor_dir.glob('*.csv')):
        df = pd.read_csv(csv, comment='#')
        filters[csv.stem] = {
            'wl': df['wl'].values,
            'T':  df['T'].values,
        }
    if not filters:
        raise ValueError(f"No CSV files found in filter directory: {sensor_dir}")
    return filters


def filterConvolve(x: np.ndarray, y: np.ndarray,
                   sensor: str | dict, method: str = 'avg') -> dict:
    """
    Convolve a spectrum with imager filter response functions.

    Parameters
    ----------
    x : array_like
        Wavelength axis [µm].
    y : array_like
        Spectral values.
    sensor : str or dict
        Name of a bundled sensor (e.g. ``'ASTER'``, ``'MMT-gasCam'``) or a
        dict mapping band name → ``{'wl': np.ndarray, 'T': np.ndarray}``.
        Use :func:`listSensors` to see available built-in sensors.
    method : str
        ``'avg'`` for transmission-weighted average,
        ``'integ'`` for trapezoid integration over the passband.

    Returns
    -------
    dict
        Per-band results keyed by band name, each containing:

        * ``'center_wl'`` — band centre wavelength [µm]
        * ``'value'``     — convolved spectral value
        * ``'T'``         — transmission interpolated onto ``x``
        * ``'through'``   — ``y * T``

        Plus summary arrays ``'wl'`` and ``'spec'`` (one entry per band).
    """
    filters = _loadFilters(sensor) if isinstance(sensor, str) else sensor

    out = {}
    wl_centers, spec_values = [], []

    for name, filt in filters.items():
        wl_f, T_f = filt['wl'], filt['T']
        T = np.interp(x, wl_f, T_f, left=0.0, right=0.0)
        center = float(np.median(wl_f[T_f > 0.01])) if np.any(T_f > 0.01) else float(np.mean(wl_f))

        if method == 'avg':
            value = float(np.average(y, weights=T)) if T.sum() > 0 else np.nan
        elif method == 'integ':
            mask = T > 0
            value = float(np.trapz(y[mask] * T[mask], x[mask])) if mask.any() else np.nan
        else:
            raise ValueError(f"method must be 'avg' or 'integ', got {method!r}")

        out[name] = {'center_wl': center, 'value': value, 'T': T, 'through': y * T}
        wl_centers.append(center)
        spec_values.append(value)

    out['wl']     = np.array(wl_centers)
    out['spec']   = np.array(spec_values)
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
        ``'description'`` [str].

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
            wn: float = 1000.0, apod_type: str = 'NB medium',
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
    wn : float
        Centre wavenumber in cm⁻¹; used to compute the FOV boxcar half-width.
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
        'fov_width', 'wn'.
    """

    out = {'apod_type': apod_type, 'fov': fov, 'wn': wn}
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

    kernel = np.fft.fft(np.concatenate((apod, filler))).real
    mid    = n_tot // 2
    kernel = np.concatenate((kernel[mid:], kernel[:mid]))
    kernel = kernel - kernel[:mid // 2].mean()

    wn_grid = np.linspace(-_kernel_opd / 2, _kernel_opd / 2, n_tot)

    # ============================================================
    #                     FOV BROADENING
    # ============================================================

    fov_width = wn * (fov / 2) ** 2 / 2
    n_box = int(np.ceil(fov_width * nper_wn))
    if n_box < 1 or n_box > mid:
        n_box = 1
    boxcar = np.ones(n_box) / n_box
    kernel = np.convolve(kernel, boxcar, 'same')

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
