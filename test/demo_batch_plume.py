"""
demo_batch_plume.py — Batch speedup demo: SO2 plume with increasing VMR.

A nadir-looking simulation with a 0.5 km thick SO2 plume at 2.5 km altitude.
The SO2 VMR is swept logarithmically across N values. Each VMR requires a
distinct atmosphere file (plume inserted at that concentration), so RFM has
no internal shortcut — this is the regime where runBatch should provide a
genuine speedup.

Methods compared
----------------
  Method A — Sequential loop: one RFM.run() call per VMR, single process.
  Method B — runBatch: all jobs dispatched in parallel across CPU cores.

Usage:
    python test/demo_batch_plume.py
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
from rfmwrapper.constants import plot_cmaps


# ---------------------------------------------------------------------------
# Terminal timer
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timer(label: str):
    """Live elapsed-time counter printed to stdout via carriage return."""
    stop_event = threading.Event()

    def _tick():
        t0 = time.perf_counter()
        while not stop_event.is_set():
            sys.stdout.write(f"\r  {label} ... {time.perf_counter() - t0:6.1f} s")
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

WN_START    = 1050.0   # cm⁻¹  — SO2 ν1 band ~1150 cm⁻¹
WN_STOP     = 1250.0   # cm⁻¹
NPERWN      = 25
ATM_TYPE    = 'FASMLS'
ILS         = 'IASI'

PLUME_ALT   = 2.5      # km  — plume centre altitude
PLUME_TH    = 0.5      # km  — plume thickness

# SO2 VMR sweep: 10 → 50 000 ppmv (log-spaced)
N_JOBS      = 16
SO2_VMRS    = np.round(np.geomspace(10, 50_000, N_JOBS), 1).tolist()

MAX_WORKERS = os.cpu_count()

_rfm_kwargs = dict(
    type='nadir',
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

def _make_jobs():
    """Construct and configure one RFM instance per SO2 VMR."""
    jobs = []
    for vmr in SO2_VMRS:
        m = rw.RFM(**_rfm_kwargs)
        m.insertPlume(h=PLUME_ALT, th=PLUME_TH, plume={'SO2': vmr})
        jobs.append(m)
    return jobs


def main():

    # -----------------------------------------------------------------------
    # Method A: sequential loop
    # -----------------------------------------------------------------------

    print(f"Method A — sequential loop, {N_JOBS} SO2 VMR steps")
    jobs_a = _make_jobs()
    t0 = time.perf_counter()
    with timer("Running"):
        for m in jobs_a:
            m.run(silent=True)
    t_a = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Method B: parallel batch
    # -----------------------------------------------------------------------

    print(f"\nMethod B — runBatch, {N_JOBS} jobs, max_workers={MAX_WORKERS}")
    jobs_b = _make_jobs()
    t0 = time.perf_counter()
    with timer("Running"):
        results_b = rw.runBatch(jobs_b, max_workers=MAX_WORKERS, cleanup=True)
    t_b = time.perf_counter() - t0

    speedup = t_a / t_b
    print(f"\nSpeedup (A / B): {speedup:.2f}×  "
          f"({'batch faster' if speedup > 1 else 'sequential faster'})")

    # -----------------------------------------------------------------------
    # Verify outputs match (spot-check first and last job)
    # -----------------------------------------------------------------------

    for idx in [0, -1]:
        tra_a = jobs_a[idx].getSpectra('tra').data[0]
        tra_b = results_b[idx].getSpectra('tra').data[0]
        diff  = np.max(np.abs(tra_a - tra_b))
        label = 'first' if idx == 0 else 'last'
        print(f"  Output check ({label}) max |A - B|: {diff:.2e}  "
              f"({'✓' if diff < 1e-6 else '✗'})")

    # -----------------------------------------------------------------------
    # Apply ILS to Method A results (sequential; used for science plot)
    # -----------------------------------------------------------------------

    ils_params   = rw.loadILSParams(ILS)
    max_opd_used = ils_params['max_opd']
    apod_used    = ils_params['apod_type']
    fwhm_used    = ils_params.get('fwhm_wn', None)

    if fwhm_used is not None:
        ils_desc = f"{ILS} ({apod_used}, OPD={max_opd_used} cm, FWHM={fwhm_used} cm⁻¹)"
    else:
        ils_desc = f"{ILS} ({apod_used}, OPD={max_opd_used} cm)"

    for m in jobs_a:
        m.applyILS(ILS)

    # Stack ILS-convolved spectra across jobs; each job has a single spectrum
    wn      = jobs_a[0].getILSOut('rad').wn
    rad_all = np.vstack([m.getILSOut('rad').data[0] for m in jobs_a])
    tra_all = np.vstack([m.getILSOut('tra').data[0] for m in jobs_a])
    opt_all = np.vstack([m.getILSOut('opt').data[0] for m in jobs_a])
    bbt_all = np.vstack([m.getILSOut('bbt').data[0] for m in jobs_a])

    # -----------------------------------------------------------------------
    # Science plot
    # -----------------------------------------------------------------------

    norm = mcolors.LogNorm(vmin=SO2_VMRS[0], vmax=SO2_VMRS[-1])
    cmap = plot_cmaps['SO2']

    fig, axes = plt.subplots(4, 1, figsize=[11, 13], sharex=True, layout='constrained')
    fig.suptitle(
        f"Nadir — {ATM_TYPE}  |  SO2 plume at {PLUME_ALT} km, Δz={PLUME_TH} km  |  ILS: {ils_desc}\n"
        f"Method A: {t_a:.1f} s (sequential)  vs  "
        f"Method B: {t_b:.1f} s (batch, {MAX_WORKERS} workers)  →  {speedup:.2f}× speedup"
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    panel_cfg = [
        (axes[0], rad_all, r'Radiance [mW/(m²·sr·cm⁻¹)]', 'Radiance'),
        (axes[1], bbt_all, 'Brightness Temperature [K]',    'Brightness Temperature'),
        (axes[2], tra_all, 'Transmission',                   'Transmission'),
        (axes[3], opt_all, 'Optical Depth',                  'Optical Depth'),
    ]

    for ax, data, ylabel, title in panel_cfg:
        for i, vmr in enumerate(SO2_VMRS):
            ax.plot(wn, data[i], color=cmap(norm(vmr)), lw=0.8, alpha=0.85)
        ax.set(ylabel=ylabel, title=title)

    axes[2].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel(r'Wavenumber [cm$^{-1}$]')

    def _wn2wl(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x != 0, 1e4 / x, np.inf)

    axes[0].secondary_xaxis('top', functions=(_wn2wl, _wn2wl)).set_xlabel(r'Wavelength [µm]')

    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('SO₂ VMR [ppmv]', fontsize=11)

    plt.show()

    for m in jobs_a:
        m.cleanup()
    print("\nDone.")


if __name__ == '__main__':
    main()
