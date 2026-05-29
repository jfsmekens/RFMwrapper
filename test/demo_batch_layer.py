"""
demo_batch_layer.py — Batch speedup demo: layer model with varying humidity.

A single atmospheric layer (1 km path at sea level) containing H2O, CO2,
CH4, and N2O. The water vapour content is swept from dry to saturated by
varying relative humidity. Each RH value requires an independent RFM run,
so runBatch should provide a genuine speedup over a sequential loop.

Methods compared
----------------
  Method A — Sequential loop: one RFM.run() call per RH value, single process.
  Method B — runBatch: all jobs dispatched in parallel across CPU cores.

Usage:
    python test/demo_batch_layer.py
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

WN_START    = 650.0    # cm⁻¹  — covers H2O rotational, CO2, CH4, N2O bands
WN_STOP     = 1350.0   # cm⁻¹
NPERWN      = 25
ILS         = 'Midac'

PATHLENGTH  = 1.22      # km
PRES        = 1013.0   # mbar — sea level
TEMP        = 298.0    # K    — ~25 °C

GASES       = ['H2O', 'CO2', 'CH4', 'N2O']

# Background VMRs for the dry gases (approximate tropospheric values)
CO2_VMR     = 420.0    # ppmv
CH4_VMR     = 1.9      # ppmv
N2O_VMR     = 0.33     # ppmv

# Relative humidity sweep: 1 % → 95 % (linear)
N_JOBS      = 16
RH_VALUES   = np.linspace(1, 95, N_JOBS).tolist()

MAX_WORKERS = os.cpu_count()


def _make_jobs():
    """Construct one layer RFM instance per RH value."""
    jobs = []
    for rh in RH_VALUES:
        h2o_vmr = rw.RH2ppm(rh, pres=PRES, temp=TEMP)
        m = rw.RFM(
            type='layer',
            wn_start=WN_START,
            wn_stop=WN_STOP,
            nperwn=NPERWN,
            gas=' '.join(GASES),
            pathlength=PATHLENGTH,
            pres=PRES,
            temp=TEMP,
            vmr={'H2O': h2o_vmr, 'CO2': CO2_VMR, 'CH4': CH4_VMR, 'N2O': N2O_VMR},
            silent=True,
        )
        jobs.append(m)
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # -----------------------------------------------------------------------
    # Method A: sequential loop
    # -----------------------------------------------------------------------

    print(f"Method A — sequential loop, {N_JOBS} RH steps  "
          f"(RH = {RH_VALUES[0]:.0f} – {RH_VALUES[-1]:.0f} %)")
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
    # Verify outputs match
    # -----------------------------------------------------------------------

    for idx in [0, -1]:
        tra_a = jobs_a[idx].getSpectra('tra').data[0]
        tra_b = results_b[idx].getSpectra('tra').data[0]
        diff  = np.max(np.abs(tra_a - tra_b))
        label = 'first' if idx == 0 else 'last'
        print(f"  Output check ({label}) max |A - B|: {diff:.2e}  "
              f"({'✓' if diff < 1e-6 else '✗'})")

    # -----------------------------------------------------------------------
    # Apply ILS and stack results
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

    wn      = jobs_a[0].getILSOut('tra').wn
    tra_all = np.vstack([m.getILSOut('tra').data[0] for m in jobs_a])
    opt_all = np.vstack([m.getILSOut('opt').data[0] for m in jobs_a])

    # H2O VMR for each job (for annotation)
    h2o_vmrs = [rw.RH2ppm(rh, pres=PRES, temp=TEMP) for rh in RH_VALUES]

    # -----------------------------------------------------------------------
    # Science plot
    # -----------------------------------------------------------------------

    norm = mcolors.Normalize(vmin=RH_VALUES[0], vmax=RH_VALUES[-1])
    cmap = plot_cmaps['H2O']

    fig, axes = plt.subplots(2, 1, figsize=[11, 7], sharex=True, layout='constrained')
    fig.suptitle(
        f"Layer model — {PATHLENGTH} km, {PRES:.0f} mbar, {TEMP:.0f} K  |  "
        f"gases: {', '.join(GASES)}  |  ILS: {ils_desc}\n"
        f"Method A: {t_a:.1f} s (sequential)  vs  "
        f"Method B: {t_b:.1f} s (batch, {MAX_WORKERS} workers)  →  {speedup:.2f}× speedup"
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for i, rh in enumerate(RH_VALUES):
        color = cmap(norm(rh))
        axes[0].plot(wn, tra_all[i], color=color, lw=0.8, alpha=0.85)
        axes[1].plot(wn, opt_all[i], color=color, lw=0.8, alpha=0.85)

    axes[0].set(ylabel='Transmission', title='Transmission')
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set(ylabel='Optical Depth', title='Optical Depth',
                xlabel=r'Wavenumber [cm$^{-1}$]')

    def _wn2wl(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x != 0, 1e4 / x, np.inf)

    axes[0].secondary_xaxis('top', functions=(_wn2wl, _wn2wl)).set_xlabel(r'Wavelength [µm]')

    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('RH [%]  /  Absolute humidity [g/m³]', fontsize=11)

    # Two-line tick labels: RH on the first line, absolute humidity on the second
    rh_ticks = np.linspace(RH_VALUES[0], RH_VALUES[-1], 6)
    cbar.set_ticks(rh_ticks)
    cbar.set_ticklabels(
        [f"{rh:.0f}%\n{rw.RH2gm3(rh, temp=TEMP, pres=PRES):.1f} g/m³"
         for rh in rh_ticks],
        fontsize=9,
    )

    plt.show()

    for m in jobs_a:
        m.cleanup()
    print("\nDone.")


if __name__ == '__main__':
    main()
