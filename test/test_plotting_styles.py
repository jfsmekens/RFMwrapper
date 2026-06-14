"""
test_plotting_styles.py — Unit tests for the Dark-cycle colouring and the
multi-spectrum filter-band overlay added for the RFMViewer plots.

Pure matplotlib (Agg) tests; no RFM execution required.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rfmwrapper.objects import Spectra, FilterOut, ILSOut
from rfmwrapper.plotting import (
    plotSpectra, plotFilterBands, DARK_CYCLE, dark_colors, light_colors,
    sweep_colors, plotSpectraSweep, plotFilterSweep, plotILS, plotILSSweep,
    plotJacobianSweep,
)


def _spectra(n_spectra: int, n_wn: int = 50, output: str = "tra") -> Spectra:
    wn = np.linspace(800.0, 1000.0, n_wn)
    data = np.tile(np.linspace(0.2, 0.9, n_wn), (n_spectra, 1))
    labels = np.arange(n_spectra, dtype=float) * 5.0
    return Spectra(wn=wn, data=data, labels=labels, label_type="altitude",
                   output=output, xunit="wn", yunit="")


def _filterout(n_spectra: int, n_bands: int = 4) -> FilterOut:
    wn = np.linspace(820.0, 980.0, n_bands)
    data = np.tile(np.linspace(0.3, 0.7, n_bands), (n_spectra, 1))
    labels = np.arange(n_spectra, dtype=float) * 5.0
    return FilterOut(wl=1e4 / wn, wn=wn, data=data, labels=labels,
                     label_type="altitude", output="tra", sensor="TEST",
                     band_names=[f"B{i}" for i in range(n_bands)])


def _ilsout(n_wn: int = 50) -> ILSOut:
    wn = np.linspace(800.0, 1000.0, n_wn)
    return ILSOut(wn=wn, data=np.linspace(0.2, 0.9, n_wn)[None, :],
                  labels=np.array([0.0]), label_type="elevation",
                  output="tra", ils={})


# ---------------------------------------------------------------------------
# Reusable sweep-overlay functions (library parity with the viewer)
# ---------------------------------------------------------------------------

def test_sweep_colors_length_and_endpoints():
    vals = [280.0, 300.0, 320.0]
    cols = sweep_colors(vals, "cmc.batlow")
    assert len(cols) == 3
    # Endpoints differ (distinct ends of the colourmap).
    assert cols[0] != cols[-1]


def test_plot_spectra_sweep_overlays_members():
    specs = [_spectra(1) for _ in range(4)]
    fig, ax = plotSpectraSweep(specs, [1, 2, 3, 4], label="surf_temp")
    assert len(ax.get_lines()) == 4
    assert "sweep over surf_temp" in ax.get_title()
    plt.close(fig)


def test_plot_filter_sweep_one_bandset_per_member():
    fl = [_filterout(1) for _ in range(3)]
    sl = [_spectra(1) for _ in range(3)]
    fig, ax = plotFilterSweep(fl, [10, 20, 30], label="elev", spectra_list=sl)
    # One scatter band-set per member.
    assert len(ax.collections) == 3
    assert "over elev" in ax.get_title()
    plt.close(fig)


def test_plot_ils_single_with_raw_background():
    fig, ax = plotILS(_ilsout(), raw=_spectra(1), instrument="Midac")
    # 1 background + 1 convolved line.
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_plot_ils_sweep_overlays_members():
    il = [_ilsout() for _ in range(3)]
    rw = [_spectra(1) for _ in range(3)]
    fig, ax = plotILSSweep(il, [1, 2, 3], label="elev", raw_list=rw)
    assert len(ax.get_lines()) == 6  # 3 background + 3 convolved
    plt.close(fig)


def test_plot_jacobian_sweep_column_overlay():
    jl = [_spectra(1, output="jac") for _ in range(3)]
    fig, ax = plotJacobianSweep(jl, [280, 300, 320], label="surf_temp")
    assert len(ax.get_lines()) == 3
    plt.close(fig)


def test_plot_jacobian_sweep_rejects_layer():
    # A layer Jacobian (>1 row) cannot be overlaid.
    layer = _spectra(3, output="jac")
    with pytest.raises(ValueError):
        plotJacobianSweep([layer], [280.0], label="x")


# ---------------------------------------------------------------------------
# Dark cycle helpers
# ---------------------------------------------------------------------------

def test_dark_cycle_matches_dark2():
    expected = [mcolors.to_hex(plt.get_cmap("Dark2")(i)) for i in range(8)]
    assert DARK_CYCLE == expected


def test_dark_colors_cycles():
    cols = dark_colors(10)
    assert len(cols) == 10
    assert cols[8] == DARK_CYCLE[0] and cols[9] == DARK_CYCLE[1]


def test_light_colors_are_lighter():
    # Each lightened colour is closer to white than its dark source.
    for dark, light in zip(dark_colors(8), light_colors(8)):
        assert sum(mcolors.to_rgb(light)) > sum(mcolors.to_rgb(dark))


# ---------------------------------------------------------------------------
# plotSpectra colouring
# ---------------------------------------------------------------------------

def test_multispectrum_uses_explicit_dark_cycle():
    # Multi-spectrum plots use the explicit Dark2 cycle even when the ambient
    # prop_cycle has been changed — the colouring does not depend on rcParams.
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=["#111111", "#222222"])}):
        fig, ax = plt.subplots()
        plotSpectra(_spectra(3), ax=ax, wl_axis=False)
        cols = [mcolors.to_hex(ln.get_color()) for ln in ax.get_lines()]
        assert cols == DARK_CYCLE[:3]
        plt.close(fig)


def test_single_spectrum_follows_prop_cycle():
    # A single spectrum uses the ambient first prop-cycle colour (no forced cycle).
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=["#abcdef", "#222222"])}):
        fig, ax = plt.subplots()
        plotSpectra(_spectra(1), ax=ax, wl_axis=False)
        col = mcolors.to_hex(ax.get_lines()[0].get_color())
        assert col == "#abcdef"
        plt.close(fig)


def test_explicit_colors_override():
    fig, ax = plt.subplots()
    plotSpectra(_spectra(2), ax=ax, colors=["#000000", "#ffffff"], wl_axis=False)
    cols = [mcolors.to_hex(ln.get_color()) for ln in ax.get_lines()]
    assert cols == ["#000000", "#ffffff"]
    plt.close(fig)


# ---------------------------------------------------------------------------
# plotFilterBands multi-spectrum
# ---------------------------------------------------------------------------

def test_filterbands_draws_all_spectra():
    fig, ax = plt.subplots()
    fout = _filterout(n_spectra=4)
    plotFilterBands(fout, ax)
    # One scatter collection per spectrum.
    assert len(ax.collections) == 4
    plt.close(fig)


def test_filterbands_small_markers_and_dark_colors():
    fig, ax = plt.subplots()
    fout = _filterout(n_spectra=3)
    plotFilterBands(fout, ax)
    sizes = {float(c.get_sizes()[0]) for c in ax.collections}
    assert sizes == {18.0}
    # Each band-set carries a connecting line in a Dark-cycle colour.
    line_cols = {mcolors.to_hex(ln.get_color()) for ln in ax.get_lines()}
    assert line_cols & set(DARK_CYCLE[:3])
    plt.close(fig)


def test_filterbands_single_index_still_works():
    fig, ax = plt.subplots()
    fout = _filterout(n_spectra=4)
    plotFilterBands(fout, ax, spectrum_index=1)
    assert len(ax.collections) == 1
    plt.close(fig)
