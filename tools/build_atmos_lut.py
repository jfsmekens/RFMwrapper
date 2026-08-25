"""
Atmospheric-correction LUT factory — Stage 1: build the master OPT spectra.

This is the expensive, filter-agnostic stage. It runs RFM line-by-line (with the
MT_CKD continuum) over a grid of ambient conditions and caches the resulting
**optical-depth spectra** OPT(ν) — one spectrum per node, wavenumber axis intact,
no SRF, no k-distribution. Stage 2 (separate) reads this master, applies an
instrument's spectral response, and collapses it into the per-instrument
k-distribution written as the portable ``<instrument>_atmos_lut.npz``.

Physics / conventions (validated interactively before this was written):
  * Homogeneous ``layer`` geometry; OPT scales linearly with path, so the stored
    OPT at ``L0 = 1 km`` is the absorption coefficient k(ν) [km⁻¹] directly.
  * Water vapour (H₂O) varies across the grid, derived from (temp, pres, rh) via
    ``RH2ppm``. CO₂ is overridden to a current 420 ppm; O₃/N₂O/CH₄ are auto-filled
    by RFM from the standard profile at each node's pressure (so O₃ rises with
    altitude). See the ``GASES`` note below.
  * Range 700–1450 cm⁻¹ is absorber-bounded below (CO₂ ν₂ at 667 makes the
    atmosphere opaque by 700) and instrument-bounded above (broadband LWIR and the
    MMT-gasCam tails top out ~1408–1428; the atmosphere stays open past 1450 only
    in dry air, where no detector responds).
  * ``nperwn = 25`` validated against 100 (band transmittance agreed to 0.01 %pts).

Build policy: **atomic, no resume.** The master is written only if every node
succeeds; interrupted runs leave no partial file and are never merged (avoids
combining nodes computed under different settings). The full configuration is
stamped into the master for provenance and for Stage 2 to verify compatibility.

macOS note: ``runBatch`` uses ``ProcessPoolExecutor`` (spawn), so this module is
guarded by ``if __name__ == '__main__'``.

Usage
-----
    python tools/build_atmos_lut.py            # full grid -> tools/lut_cache/
    python tools/build_atmos_lut.py --smoke    # tiny 2x2x2 grid, end-to-end check
    python tools/build_atmos_lut.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rfmwrapper import config
from rfmwrapper.functions import RH2ppm
from rfmwrapper.objects import RFM, runBatch

logger = logging.getLogger('build_atmos_lut')

# ======================================================================
# CONFIG  (edit here; a change means a full rebuild — see build policy)
# ======================================================================
MASTER_VERSION = '1'

# Per-window line-by-line config: spectral range + gas set + resolution. The ambient
# grid, L0, and the CO₂ override are shared across windows. Select with --window
# (default LWIR); adding a window is just another entry here. WINDOW names the output
# (master_<WINDOW>.npz) and is recorded in the stamp.
WINDOWS = {
    'LWIR': dict(wn_start=700.0,  wn_stop=1450.0, nperwn=25.0, gases='H2O CO2 O3 N2O CH4'),
    # MWIR: 1800–3700 cm⁻¹ (2.70–5.56 µm) contains the InSb SRFs; adds CO (4.6 µm) and
    # keeps CH4 (3.3 µm ν3) / N2O (4.5 µm) / the opaque CO2 4.3 µm band.
    'MWIR': dict(wn_start=1800.0, wn_stop=3700.0, nperwn=25.0, gases='H2O CO2 O3 N2O CH4 CO'),
}
WINDOW = 'LWIR'                                      # default; overridden by --window
WN_START = WINDOWS[WINDOW]['wn_start']               # cm⁻¹
WN_STOP = WINDOWS[WINDOW]['wn_stop']                 # cm⁻¹
NPERWN = WINDOWS[WINDOW]['nperwn']                   # samples per cm⁻¹
GASES = WINDOWS[WINDOW]['gases']

L0_KM = 1.0                                          # reference path; OPT/L0 = k [km⁻¹]
OUT = 'OPT'                                          # RFM output: optical depth
CO2_PPM = 420.0                                      # CO₂ override (both windows; std profiles are dated)

# Ambient grid axes (shared across windows). H₂O varies per node; CO₂ overridden;
# other gases auto-filled from the standard profile at each node's pressure.
TEMP_GRID = np.arange(233.0, 333.1, 5.0)            # 233–333 K, 5 K step  -> 21
PRES_GRID = np.linspace(470.0, 1030.0, 8)           # ~6000 m ↔ sea level  -> 8
RH_GRID = np.linspace(5.0, 99.0, 15)                # 5–99 %               -> 15

# Output.
CACHE_DIR = Path(__file__).resolve().parent / 'lut_cache'
MASTER_PATH = CACHE_DIR / f'master_{WINDOW}.npz'


def _applyWindow(window: str) -> None:
    """Point the module-level config at ``window`` (used by ``--window``)."""
    global WINDOW, WN_START, WN_STOP, NPERWN, GASES, MASTER_PATH
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {list(WINDOWS)}, got {window!r}")
    cfg = WINDOWS[window]
    WINDOW = window
    WN_START, WN_STOP = cfg['wn_start'], cfg['wn_stop']
    NPERWN, GASES = cfg['nperwn'], cfg['gases']
    MASTER_PATH = CACHE_DIR / f'master_{WINDOW}.npz'


def _makeJob(temp: float, pres: float, rh: float) -> RFM:
    """
    Build a single homogeneous-layer RFM job for one ambient node.

    Parameters
    ----------
    temp : float
        Air temperature [K].
    pres : float
        Pressure [mbar].
    rh : float
        Relative humidity [%].

    Returns
    -------
    RFM
        Configured, un-run RFM instance (unique drv/out paths auto-assigned).
    """
    vmr_h2o = RH2ppm(rh, pres=pres, temp=temp)
    return RFM(
        type='layer', gas=GASES,
        wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
        temp=temp, pres=pres, pathlength=L0_KM,
        vmr={'H2O': vmr_h2o, 'CO2': CO2_PPM},
        out=OUT, silent=True,
    )


def _settingsStamp(temp_grid: np.ndarray, pres_grid: np.ndarray,
                   rh_grid: np.ndarray, n_wn: int) -> dict:
    """Assemble the provenance / compatibility stamp embedded in the master."""
    return {
        'master_version': MASTER_VERSION,
        'window': WINDOW,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'wn_start': WN_START, 'wn_stop': WN_STOP, 'nperwn': NPERWN, 'n_wn': n_wn,
        'L0_km': L0_KM, 'out': OUT,
        'gases': GASES, 'co2_ppm': CO2_PPM,
        'temp_grid': temp_grid.tolist(),
        'pres_grid': pres_grid.tolist(),
        'rh_grid': rh_grid.tolist(),
        'rfm_dir': config.get_rfm_dir(),
        'hitran_path': config.get_hitran_path(),
    }


def _writeMaster(path: Path, opt: np.ndarray, wn: np.ndarray,
                 temp_grid: np.ndarray, pres_grid: np.ndarray,
                 rh_grid: np.ndarray, stamp: dict) -> None:
    """
    Write the master atomically (temp file + rename) so a crash mid-write can
    never leave a corrupt or partial master in place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix='.npz', dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(
            tmp,
            opt=opt.astype(np.float32),
            wn=wn.astype(np.float32),
            temp=temp_grid.astype(np.float64),
            pres=pres_grid.astype(np.float64),
            rh=rh_grid.astype(np.float64),
            stamp=json.dumps(stamp),
        )
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def buildMaster(temp_grid: np.ndarray = TEMP_GRID,
                pres_grid: np.ndarray = PRES_GRID,
                rh_grid: np.ndarray = RH_GRID,
                path: Path = MASTER_PATH,
                max_workers: int | None = None) -> Path:
    """
    Run the full ambient grid and write the master OPT-spectra cache.

    Iterates pressure-by-pressure: for each pressure level the ``(temp × rh)``
    sub-grid is run in one parallel batch, then its OPT spectra are copied into the
    preallocated master array and the RFM instances discarded. This bounds peak
    memory to one pressure slice while the master (~190 MB float32) accumulates.

    The build is atomic: the master is written only after all nodes succeed. Any
    failure raises and leaves no file.

    Parameters
    ----------
    temp_grid, pres_grid, rh_grid : numpy.ndarray
        Ambient axis vectors [K], [mbar], [%].
    path : pathlib.Path
        Destination ``.npz``.
    max_workers : int, optional
        Parallel worker processes (defaults to ``os.cpu_count()``).

    Returns
    -------
    pathlib.Path
        The master path written.
    """
    n_t, n_p, n_r = temp_grid.size, pres_grid.size, rh_grid.size
    n_nodes = n_t * n_p * n_r
    logger.info('Master build: %d nodes (%d temp × %d pres × %d rh), %s',
                n_nodes, n_t, n_p, n_r, f'{WN_START:.0f}-{WN_STOP:.0f} cm⁻¹')

    # Probe one node to fix the shared wavenumber grid, then preallocate.
    probe = _makeJob(temp_grid[n_t // 2], pres_grid[n_p // 2], rh_grid[n_r // 2])
    probe.run(silent=True)
    wn = np.asarray(probe.getSpectra(OUT).wn, dtype=np.float64)
    probe.cleanup()
    n_wn = wn.size
    logger.info('Wavenumber grid: %d points (%.3f cm⁻¹ spacing)',
                n_wn, (WN_STOP - WN_START) / (n_wn - 1))

    opt = np.empty((n_t, n_p, n_r, n_wn), dtype=np.float32)

    for ip, pres in enumerate(pres_grid):
        jobs, idx = [], []
        for it, temp in enumerate(temp_grid):
            for ir, rh in enumerate(rh_grid):
                jobs.append(_makeJob(temp, pres, rh))
                idx.append((it, ir))
        results = runBatch(jobs, max_workers=max_workers)
        for (it, ir), r in zip(idx, results):
            spec = r.getSpectra(OUT)
            wn_i = np.asarray(spec.wn)
            if wn_i.size != n_wn or not np.allclose(wn_i[[0, -1]], wn[[0, -1]]):
                raise RuntimeError(
                    f'Wavenumber grid mismatch at temp={temp_grid[it]:.0f} K, '
                    f'pres={pres:.0f} mbar, rh={rh_grid[ir]:.0f}%: '
                    f'got {wn_i.size} pts [{wn_i[0]:.2f},{wn_i[-1]:.2f}] '
                    f'vs expected {n_wn} [{wn[0]:.2f},{wn[-1]:.2f}].'
                )
            opt[it, ip, ir, :] = spec.data[0]
        logger.info('  pressure %d/%d (%.0f mbar) complete', ip + 1, n_p, pres)

    stamp = _settingsStamp(temp_grid, pres_grid, rh_grid, n_wn)
    _writeMaster(path, opt, wn, temp_grid, pres_grid, rh_grid, stamp)
    logger.info('Master written: %s (%.1f MB)', path,
                path.stat().st_size / 1024 / 1024)
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    ap = argparse.ArgumentParser(description='Build the master OPT-spectra LUT (Stage 1).')
    ap.add_argument('--window', default='LWIR', choices=list(WINDOWS),
                    help='Spectral window / master to build (default LWIR).')
    ap.add_argument('--smoke', action='store_true',
                    help='Tiny 2×2×2 grid to a scratch file, for an end-to-end check.')
    ap.add_argument('--workers', type=int, default=None,
                    help='Parallel worker processes (default: cpu_count).')
    args = ap.parse_args()
    _applyWindow(args.window)
    logger.info('Window %s: %s cm⁻¹, gases %s', WINDOW,
                f'{WN_START:.0f}-{WN_STOP:.0f}', GASES)

    if args.smoke:
        logger.info('SMOKE TEST — tiny grid, not the production master.')
        buildMaster(
            temp_grid=np.array([253.0, 313.0]),
            pres_grid=np.array([600.0, 1013.0]),
            rh_grid=np.array([20.0, 90.0]),
            path=CACHE_DIR / f'master_{WINDOW}_smoke.npz',
            max_workers=args.workers,
        )
    else:
        buildMaster(path=MASTER_PATH, max_workers=args.workers)


if __name__ == '__main__':
    main()
