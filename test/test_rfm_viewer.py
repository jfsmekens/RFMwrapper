"""
Headless smoke tests for the RFMViewer GUI.

These construct the Tk widget tree against a withdrawn root and exercise the
pure-logic paths (config round-trip, kwargs translation, sweep resolution,
parsers).  They do *not* run RFM, so they need no RFM binary and are CI-safe.
Skipped automatically when no display / Tk is available.
"""

import pytest

import tkinter as tk

try:
    _root = tk.Tk()
    _root.withdraw()
    _HAVE_TK = True
except Exception:
    _HAVE_TK = False

pytestmark = pytest.mark.skipif(not _HAVE_TK, reason="no Tk display available")

from rfmwrapper.rfm_viewer import RFMViewer


@pytest.fixture
def viewer():
    """A constructed RFMViewer against the shared withdrawn root."""
    v = RFMViewer(_root)
    yield v
    # Detach log handlers / close figures without tearing down the shared root.
    for h in (getattr(v, "gui_handler", None), getattr(v, "file_handler", None)):
        if h is not None:
            v.logger.removeHandler(h)
    v.destroy()


def test_constructs(viewer):
    assert viewer.title().startswith("RFMViewer")


# ---------------------------------------------------------------------------
# Sweep-aware filter / ILS overlay (no RFM execution — synthetic members)
# ---------------------------------------------------------------------------

class _FakeRFM:
    """Minimal stand-in exposing the filter/ILS interface the viewer uses."""

    def __init__(self, scale: float):
        import numpy as np
        from rfmwrapper.objects import Spectra, FilterOut, ILSOut
        self._np = np
        wn = np.linspace(800.0, 1000.0, 40)
        self._spec = Spectra(wn=wn, data=(np.linspace(0.2, 0.8, 40) * scale)[None, :],
                             labels=np.array([0.0]), label_type="elevation",
                             output="bbt", xunit="wn", yunit="K")
        bands = np.linspace(820.0, 980.0, 3)
        self._fout = FilterOut(wl=1e4 / bands, wn=bands,
                               data=(np.array([0.5, 0.6, 0.7]) * scale)[None, :],
                               labels=np.array([0.0]), label_type="elevation",
                               output="bbt", sensor="ASTER",
                               band_names=["B1", "B2", "B3"])
        self._ils = ILSOut(wn=wn, data=(np.linspace(0.2, 0.8, 40) * scale)[None, :],
                           labels=np.array([0.0]), label_type="elevation",
                           output="bbt", ils={})

    def applyFilters(self, sensor=None):  # noqa: N802 (mirror RFM API)
        pass

    def getFilterOut(self, sensor, output):  # noqa: N802
        return self._fout

    def getSpectra(self, output):  # noqa: N802
        return self._spec

    def applyILS(self, instrument=None):  # noqa: N802
        pass

    def getILSOut(self, output):  # noqa: N802
        return self._ils


def _seed_batch(viewer, n=3):
    members = [_FakeRFM(scale=1.0 + i) for i in range(n)]
    viewer.batch = members
    viewer.batch_values = [float(-80 + 20 * i) for i in range(n)]
    viewer.batch_label = "elev"
    viewer.rfm = members[0]
    return members


def test_apply_filter_covers_whole_sweep(viewer):
    _seed_batch(viewer, n=4)
    viewer.conv_sensor_var.set("ASTER")
    viewer.conv_which_var.set("bbt")
    viewer._on_apply_filter()
    ax = viewer.conv_fig.axes[0]
    # One scatter band-set per sweep member (not just the first).
    assert len(ax.collections) == 4
    assert "over elev" in ax.get_title()


def test_apply_ils_covers_whole_sweep(viewer):
    _seed_batch(viewer, n=4)
    viewer.conv_instr_var.set("Midac")
    viewer.conv_which_var.set("bbt")
    viewer._on_apply_ils()
    ax = viewer.conv_fig.axes[0]
    # 4 background + 4 convolved curves.
    assert len(ax.get_lines()) == 8
    assert "over elev" in ax.get_title()


# ---------------------------------------------------------------------------
# Jacobian plot dispatch (column vs layer) — regression for pcolormesh crash
# ---------------------------------------------------------------------------

class _JacRFM:
    """Stand-in returning a Jacobian Spectra of a chosen label_type."""

    def __init__(self, label_type: str, n_rows: int, n_wn: int = 30):
        import numpy as np
        from rfmwrapper.objects import Spectra
        wn = np.linspace(800.0, 820.0, n_wn)
        data = np.outer(np.arange(1, n_rows + 1), np.linspace(-1.0, 1.0, n_wn))
        self._jac = Spectra(wn=wn, data=data, labels=np.arange(n_rows, dtype=float),
                            label_type=label_type, output="jac",
                            xunit="wn", yunit="")

    _jac_raw = [{"output": "rad"}]  # viewer reads this for the output selector

    def getJacobians(self, param=None, output=None):  # noqa: N802 (mirror RFM API)
        if param is None:
            return {"TEM": self._jac}
        return self._jac


def test_column_jacobian_plots_as_lines(viewer):
    # A multi-row column/elevation Jacobian must NOT go through pcolormesh.
    viewer.rfm = _JacRFM("elevation", n_rows=4)
    viewer.jac_param_var.set("TEM")
    viewer._plot_jac()
    ax = viewer.jac_fig.axes[0]
    assert len(ax.get_lines()) == 4
    assert len(ax.collections) == 0


def test_layer_jacobian_plots_as_map(viewer):
    viewer.rfm = _JacRFM("altitude", n_rows=4)
    viewer.jac_param_var.set("TEM")
    viewer._plot_jac()
    ax = viewer.jac_fig.axes[0]
    assert len(ax.collections) >= 1  # pcolormesh QuadMesh


# ---------------------------------------------------------------------------
# Structured Jacobian editor + validation
# ---------------------------------------------------------------------------

def test_jacobian_editor_collect_and_dict(viewer):
    viewer._add_jac_row("TEM", "0, 5, 10, 20, 40, 60")
    viewer._add_jac_row("H2O", "")  # blank levels → column
    rows = viewer._collect_jacobians()
    assert rows == [{"target": "TEM", "levels": "0, 5, 10, 20, 40, 60"},
                    {"target": "H2O", "levels": ""}]
    assert viewer._jacobians_dict(rows) == {
        "TEM": [0, 5, 10, 20, 40, 60], "H2O": None}


def test_jacobian_editor_config_roundtrip(viewer):
    viewer._add_jac_row("TEM", "0, 10, 20")
    viewer._add_jac_row("SFCTEM", "")
    cfg = viewer._collect_config()
    import json
    json.loads(json.dumps(cfg))  # JSON-serializable
    viewer._apply_jacobians([])  # clear
    assert viewer._collect_jacobians() == []
    viewer._apply_config(cfg)  # restore
    assert viewer._collect_jacobians() == cfg["jacobians"]


def test_jacobian_legacy_string_loads(viewer):
    viewer._apply_jacobians("TEM:0,5,10")  # old single-field format
    rows = viewer._collect_jacobians()
    assert rows == [{"target": "TEM", "levels": "0, 5, 10"}]


def test_jacobian_validation_surface_needs_nadir(viewer):
    viewer.type_var.set("zenith")
    viewer._on_geometry_changed()
    with pytest.raises(ValueError):
        viewer._validate_jacobians({"SFCTEM": None}, viewer._collect_config())
    # Nadir is fine.
    viewer.type_var.set("nadir")
    viewer._on_geometry_changed()
    viewer._validate_jacobians({"SFCTEM": None}, viewer._collect_config())


def test_jacobian_validation_aerosol_needs_species(viewer):
    cfg = viewer._collect_config()
    with pytest.raises(ValueError):
        viewer._validate_jacobians({"AEROSOL": None}, dict(cfg, gas="H2O CO2"))
    # Present as a gas → OK.
    viewer._validate_jacobians({"AEROSOL": None}, dict(cfg, gas="H2O AEROSOL"))


def test_compose_gas_vmr_checked_list():
    gas, vmr = RFMViewer._compose_gas_vmr(
        False, ["H2O", "CO2", "SO2"], [], {"SO2": "500", "H2O": ""})
    assert gas == "H2O CO2 SO2"
    assert vmr == "SO2=500"


def test_compose_gas_vmr_all_preserves_extras():
    gas, vmr = RFMViewer._compose_gas_vmr(True, [], ["aerosol"], {"CO2": "400"})
    assert gas == "all aerosol"
    assert vmr == "CO2=400"


def test_compose_gas_vmr_ignores_unticked_vmr():
    gas, vmr = RFMViewer._compose_gas_vmr(False, ["H2O"], [], {"SO2": "500"})
    assert gas == "H2O"
    assert vmr == ""   # SO2 not ticked → its VMR is dropped


def test_compose_gas_vmr_empty_defaults_all():
    assert RFMViewer._compose_gas_vmr(False, [], [], {}) == ("all", "")


def test_compose_gas_vmr_bad_value_raises():
    with pytest.raises(ValueError):
        RFMViewer._compose_gas_vmr(False, ["SO2"], [], {"SO2": "abc"})


def test_gas_picker_opens_and_closes(viewer):
    viewer.gas_var.set("H2O CO2 aerosol")
    viewer.vmr_var.set("CO2=400")
    viewer._open_gas_picker()
    tops = [w for w in viewer.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(tops) == 1
    tops[0].destroy()


def test_jac_tab_sweep_member_selector(viewer):
    members = [_JacRFM("column", n_rows=1) for _ in range(3)]
    viewer.batch = members
    viewer.batch_values = [280.0, 300.0, 320.0]
    viewer.batch_label = "surf_temp"
    viewer.rfm = members[0]
    viewer._refresh_jac_params()
    assert viewer.jac_member_combo.cget("values") == (
        "surf_temp=280", "surf_temp=300", "surf_temp=320")
    # Selecting a member targets that batch element.
    viewer.jac_member_combo.current(2)
    assert viewer._jac_rfm() is members[2]


def test_config_roundtrip(viewer):
    cfg = viewer._collect_config()
    # Must be JSON-serializable (Save/Load contract).
    import json
    json.loads(json.dumps(cfg))
    viewer._apply_config(cfg)
    assert viewer._collect_config() == cfg


def test_rfm_kwargs_defaults(viewer):
    cfg = viewer._collect_config()
    kw = viewer._rfm_kwargs(cfg)
    assert kw["type"] == "nadir"
    assert kw["out"] == "RAD BBT OPT TRA"
    assert kw["physics"] == {"ctm": True, "mix": True, "rex": False,
                             "chi": False, "nte": False}
    # Paths are resolved fresh from config, not stale module globals.
    assert "exec_path" in kw and "hitran_path" in kw


def test_geometry_switch_layer(viewer):
    viewer.type_var.set("layer")
    viewer._on_geometry_changed()
    cfg = viewer._collect_config()
    kw = viewer._rfm_kwargs(cfg)
    assert kw["type"] == "layer"
    assert kw["pres"] == 1013.0 and kw["temp"] == 298.0


def test_limb_defaults_and_requires_tan_heights(viewer):
    viewer.type_var.set("limb")
    viewer._on_geometry_changed()
    # Limb ships a default tangent-height ladder.
    assert viewer._geo_rows["tan_heights"][2].get().strip() != ""
    # Clearing them makes kwargs translation raise.
    viewer._geo_rows["tan_heights"][2].set("")
    with pytest.raises(ValueError):
        viewer._rfm_kwargs(viewer._collect_config())
    # Restoring resolves it (tab-separated, to exercise the parser).
    viewer._geo_rows["tan_heights"][2].set("10\t20\t30")
    kw = viewer._rfm_kwargs(viewer._collect_config())
    assert kw["tan_heights"] == [10.0, 20.0, 30.0]


def test_geometry_defaults_applied(viewer):
    # Default geometry is nadir → standard nadir values populated.
    geo = viewer._collect_config()["geo"]
    assert geo["elev"] == "-90" and geo["surf_emiss"] == "1.0"
    assert geo["h_obs"] == "100" and geo["h_surf"] == "0.01"
    # Switching to zenith refreshes the un-customised fields.
    viewer.type_var.set("zenith")
    viewer._on_geometry_changed()
    geo = viewer._collect_config()["geo"]
    assert geo["elev"] == "90" and geo["h_obs"] == "0.0"


def test_user_edit_survives_geometry_switch(viewer):
    # A customised surface emissivity must not be clobbered by a switch.
    viewer._geo_rows["surf_emiss"][2].set("0.92")
    viewer.type_var.set("zenith")
    viewer._on_geometry_changed()
    assert viewer._collect_config()["geo"]["surf_emiss"] == "0.92"


def test_sweep_values_range(viewer):
    sw = {"mode": "range", "start": "270", "stop": "310", "n": "5",
          "list": "", "param": "surf_temp"}
    assert viewer._sweep_values(sw) == [270.0, 280.0, 290.0, 300.0, 310.0]


def test_sweep_values_list(viewer):
    sw = {"mode": "list", "list": "270, 290, 310", "start": "", "stop": "",
          "n": "", "param": "surf_temp"}
    assert viewer._sweep_values(sw) == [270.0, 290.0, 310.0]


def test_parse_helpers(viewer):
    assert viewer._parse_scalar_or_list("5") == 5.0
    assert viewer._parse_scalar_or_list("0, 5, 10") == [0.0, 5.0, 10.0]
    assert viewer._parse_scalar_or_list("") is None
    assert viewer._parse_kv("CO2=420, SO2=5") == {"CO2": 420.0, "SO2": 5.0}
    assert viewer._parse_kv_str("CFC11=/p/a.xsc") == {"CFC11": "/p/a.xsc"}
    assert viewer._parse_jacobians("TEM, H2O") == {"TEM": None, "H2O": None}
    assert viewer._parse_jacobians("TEM:0,5,10") == {"TEM": [0.0, 5.0, 10.0]}


def test_vmr_sweep_param_listed(viewer):
    """Gases in the gas field appear as VMR:<GAS> sweep targets."""
    viewer.gas_var.set("H2O SO2")
    viewer._refresh_sweep_params()
    params = list(viewer.sweep_param_combo["values"])
    assert "VMR:H2O" in params and "VMR:SO2" in params
    # 'all' is never offered as a VMR sweep target.
    viewer.gas_var.set("all")
    viewer._refresh_sweep_params()
    assert not any(p.startswith("VMR:") for p in viewer.sweep_param_combo["values"])
