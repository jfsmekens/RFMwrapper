"""
demo_batch.py — Multithreaded batch run vs internal RFM multi-angle treatment.

Compares two approaches for computing spectra at multiple elevation angles:

  Method A — Single RFM instance with a list of elevation angles.
              RFM handles the angles internally in a single Fortran process
              (sequential internal loop).

  Method B — One RFM instance per angle, run in parallel via runBatch()
              (one Fortran process per angle, dispatched across CPU cores).

The timing comparison tests whether the parallel overhead of spawning N
processes is offset by genuine parallelism. The science output (spectra
coloured by elevation angle) is identical between the two methods.

NOTE: The if __name__ == '__main__' guard is required. On macOS and Windows,
multiprocessing uses the 'spawn' start method, which re-imports the main
script in every worker process. Without the guard, each worker would
re-enter runBatch recursively, crashing the pool.

Usage:
    python test/demo_batch.py
"""
import os
import sys
import time
import threading
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import rfmwrapper as rw


# ---------------------------------------------------------------------------
# Terminal timer
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timer(label: str):
    """
    Context manager that prints a live elapsed-time counter to stdout while
    the body executes, then prints the final time on exit.

    Usage::

        with timer("Running RFM"):
            model.run()
    """
    stop_event = threading.Event()

    def _tick():
        t0 = time.perf_counter()
        while not stop_event.is_set():
            elapsed = time.perf_counter() - t0
            sys.stdout.write(f"\r  {label} ... {elapsed:6.1f} s")
            sys.stdout.flush()
            time.sleep(0.1)
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r  {label} ... {elapsed:6.2f} s  ✓\n")
        sys.stdout.flush()

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START    = 650.0    # cm⁻¹
WN_STOP     = 1350.0   # cm⁻¹
NPERWN      = 25       # points per cm⁻¹
ATM_TYPE    = 'FASMLS'
ILS         = 'IASI'

# Elevation angles: logarithmically spaced from near-horizon to zenith.
# Low angles (long slant paths) are the most interesting scientifically.
ELEVATIONS  = np.round(np.geomspace(3, 90, 16), 1).tolist()
MAX_WORKERS = os.cpu_count()

_rfm_kwargs = dict(
    type='zenith',
    wn_start=WN_START,
    wn_stop=WN_STOP,
    nperwn=NPERWN,
    gas='all',
    atm_type=ATM_TYPE,
    silent=True,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # -----------------------------------------------------------------------
    # Method A: single RFM instance, all angles in one Fortran call
    # -----------------------------------------------------------------------

    print(f"Method A — single RFM, {len(ELEVATIONS)} elevation angles (internal loop)")
    t0 = time.perf_counter()
    with timer("Running"):
        model_a = rw.RFM(elev=ELEVATIONS, **_rfm_kwargs)
        model_a.run(silent=True)
    t_a = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Method B: one RFM instance per angle, parallel via runBatch
    # -----------------------------------------------------------------------

    print(f"\nMethod B — runBatch, {len(ELEVATIONS)} workers, max_workers={MAX_WORKERS}")
    jobs = [rw.RFM(elev=e, **_rfm_kwargs) for e in ELEVATIONS]
    t0 = time.perf_counter()
    with timer("Running"):
        results = rw.runBatch(jobs, max_workers=MAX_WORKERS, cleanup=True)
    t_b = time.perf_counter() - t0

    speedup = t_a / t_b
    print(f"\nSpeedup (A / B): {speedup:.2f}×  "
          f"({'batch faster' if speedup > 1 else 'internal loop faster'})")

    # -----------------------------------------------------------------------
    # Apply ILS to Method A results (used for science plot)
    # -----------------------------------------------------------------------

    model_a.applyILS(ILS)
    ils_params   = rw.loadILSParams(ILS)
    max_opd_used = ils_params['max_opd']
    apod_used    = ils_params['apod_type']
    fwhm_used    = ils_params.get('fwhm_wn', None)

    if fwhm_used is not None:
        ils_desc = f"{ILS} ({apod_used}, OPD={max_opd_used} cm, FWHM={fwhm_used} cm⁻¹)"
    else:
        ils_desc = f"{ILS} ({apod_used}, OPD={max_opd_used} cm)"

    rad_ils = model_a.getILSOut('rad')
    bbt_ils = model_a.getILSOut('bbt')
    tra_ils = model_a.getILSOut('tra')
    opt_ils = model_a.getILSOut('opt')

    elevations_out = rad_ils.labels   # shape (n_elev,)

    # -----------------------------------------------------------------------
    # Verify Method B output matches Method A (spot-check first spectrum)
    # -----------------------------------------------------------------------

    rad_b0   = results[0].getSpectra('rad').data[0]
    rad_a0   = model_a.getSpectra('rad').data[0]
    max_diff = np.max(np.abs(rad_b0 - rad_a0))
    print(f"\nOutput check — max |A[0] - B[0]|: {max_diff:.2e}  "
          f"({'✓ match' if max_diff < 1e-6 else '✗ mismatch'})")

    # -----------------------------------------------------------------------
    # Science plot: spectra coloured by elevation angle
    # -----------------------------------------------------------------------

    wn    = rad_ils.wn
    norm  = mcolors.Normalize(vmin=ELEVATIONS[0], vmax=ELEVATIONS[-1])
    cmap  = cm.plasma_r

    fig, axes = plt.subplots(4, 1, figsize=[11, 13], sharex=True, layout='constrained')
    fig.suptitle(
        f"Zenith geometry — {ATM_TYPE}  |  ILS: {ils_desc}\n"
        f"Method A: {t_a:.1f} s (internal)  vs  "
        f"Method B: {t_b:.1f} s (batch, {MAX_WORKERS} workers)  →  {speedup:.2f}× speedup"
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    panel_cfg = [
        (axes[0], rad_ils, r'Radiance [mW/(m²·sr·cm⁻¹)]', 'Radiance'),
        (axes[1], bbt_ils, 'Brightness Temperature [K]',    'Brightness Temperature'),
        (axes[2], tra_ils, 'Transmission',                   'Transmission'),
        (axes[3], opt_ils, 'Optical Depth',                  'Optical Depth'),
    ]

    for ax, spectra, ylabel, title in panel_cfg:
        for i, elev in enumerate(elevations_out):
            color = cmap(norm(elev))
            ax.plot(wn, spectra.data[i], color=color, lw=0.8, alpha=0.85)
        ax.set(ylabel=ylabel, title=title)

    axes[2].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel(r'Wavenumber [cm$^{-1}$]')

    def _wn2wl(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x != 0, 1e4 / x, np.inf)

    axes[0].secondary_xaxis('top', functions=(_wn2wl, _wn2wl)).set_xlabel(r'Wavelength [µm]')

    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Elevation angle [°]', fontsize=11)

    plt.show()
    model_a.cleanup()
    print("\nDone.")


if __name__ == '__main__':
    main()
