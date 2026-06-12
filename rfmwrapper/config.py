"""
Path configuration for rfmwrapper.

All RFM-related paths are resolved through this module.  Resolution priority
for each path (highest to lowest):

  1. Runtime assignment   e.g. rfmwrapper.__EXECPATH__ = '/path/to/rfm'
  2. Environment variable e.g. export RFM_DIR=/path/to/RFM
  3. User config file     ~/.config/rfmwrapper/config.ini  (set via python -m rfmwrapper configure)
  4. Derived default      path constructed from rfm_dir

Paths that are individually overridable (via config file or env var):

  rfm_dir     — root of the RFM installation tree (all other defaults derive from this)
  hitran_path — full path to a HITRAN binary (.bin); often lives outside the RFM tree

The remaining subdirectories (atm, ARIA, output, source/rfm) always derive
from rfm_dir and are not individually configurable, as they follow the standard
RFM installation layout.

Filter files are bundled inside the package (rfmwrapper/filters/) and require
no path configuration.
"""
import configparser
import os
from glob import glob
from pathlib import Path

_CONFIG_DIR  = Path.home() / '.config' / 'rfmwrapper'
_CONFIG_FILE = _CONFIG_DIR / 'config.ini'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_config() -> configparser.ConfigParser:
    """Return a ConfigParser loaded from the config file, or empty if absent."""
    cfg = configparser.ConfigParser()
    if _CONFIG_FILE.exists():
        cfg.read(_CONFIG_FILE)
    return cfg


def _cfg_get(key: str) -> str | None:
    """Return a value from the [paths] section of the config file, or None."""
    return _read_config().get('paths', key, fallback=None)


def _expand(path: str | None) -> str | None:
    return os.path.expanduser(path) if path else None


# ---------------------------------------------------------------------------
# Public path resolvers
# ---------------------------------------------------------------------------

def get_rfm_dir() -> str:
    """
    Resolve the RFM installation root directory.

    Priority: RFM_DIR env var → config file rfm_dir → ~/RFM fallback.

    Returns
    -------
    str
        Absolute path to the RFM installation directory.
    """
    env = _expand(os.environ.get('RFM_DIR'))
    if env:
        return env
    cfg = _expand(_cfg_get('rfm_dir'))
    if cfg:
        return cfg
    return os.path.expanduser('~/RFM')


def get_hitran_path() -> str | None:
    """
    Resolve the HITRAN binary database path.

    Priority: HITRAN_PATH env var → config file hitran_path →
    glob for hitran*.bin under $rfm_dir/Hitran/.

    Returns
    -------
    str or None
        Path to the HITRAN .bin file, or None if not found.
    """
    env = _expand(os.environ.get('HITRAN_PATH'))
    if env:
        return env
    cfg = _expand(_cfg_get('hitran_path'))
    if cfg:
        return cfg
    # Derived default: search under rfm_dir/Hitran/
    hitran_dir = os.path.join(get_rfm_dir(), 'Hitran')
    hits = sorted(glob(os.path.join(hitran_dir, 'hitran*.bin')))
    return hits[-1] if hits else None


# ---------------------------------------------------------------------------
# Derived paths (always relative to rfm_dir, not individually configurable)
# ---------------------------------------------------------------------------

def get_exec_path() -> str:
    """Return the expected RFM executable path ($rfm_dir/source/rfm)."""
    return os.path.join(get_rfm_dir(), 'source', 'rfm')


def get_atm_dir() -> str:
    """Return the standard atmosphere directory ($rfm_dir/atm/)."""
    return os.path.join(get_rfm_dir(), 'atm')


def get_aria_dir() -> str:
    """Return the ARIA aerosol data directory ($rfm_dir/ARIA/)."""
    return os.path.join(get_rfm_dir(), 'ARIA')


def get_xsc_dir() -> str:
    """Return the default XSC data directory ($rfm_dir/xsc/).

    Store concatenated ``.xsc`` files here so they can be referenced by name
    rather than by full path (e.g. ``xsc_dir=__XSCDIR__``).
    """
    return os.path.join(get_rfm_dir(), 'xsc')


def get_out_dir() -> str:
    """Return the default output directory ($rfm_dir/output/)."""
    return os.path.join(get_rfm_dir(), 'output')


# ---------------------------------------------------------------------------
# Config file I/O
# ---------------------------------------------------------------------------

def write_config(rfm_dir: str = None, hitran_path: str = None) -> None:
    """
    Write path settings to the user config file, creating it if needed.

    Only non-None arguments are written; existing values for other keys are
    left unchanged.

    Parameters
    ----------
    rfm_dir : str, optional
        Path to the RFM installation directory.
    hitran_path : str, optional
        Path to the HITRAN .bin file.
    """
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _read_config()
    if 'paths' not in cfg:
        cfg['paths'] = {}
    if rfm_dir is not None:
        cfg['paths']['rfm_dir'] = rfm_dir
    if hitran_path is not None:
        cfg['paths']['hitran_path'] = hitran_path
    with open(_CONFIG_FILE, 'w') as f:
        cfg.write(f)
