"""
RFMViewer — Interactive GUI for the Reference Forward Model (RFM).

A desktop interface over :class:`rfmwrapper.objects.RFM`:

- Configure a single run or a single-parameter batch sweep (geometry, spectral
  range, atmosphere, gases, output flags, physics, Jacobians, XSC).
- Execute RFM off the Tk thread, with a progress bar (indeterminate for a
  single run, determinate for batch).
- Visualize results (spectra, atmosphere profile, Jacobians, filter/ILS
  convolution, ray-path diagnostics) in embedded matplotlib canvases.
- View the wrapper log console and the raw RFM ``rfm.log``.
- Configure RFM paths, export spectra (RFM .spc / CSV / NPZ), and save/load the
  GUI configuration as JSON.

Launch via::

    python -m rfmwrapper.rfm_viewer
    rfm-viewer   (console script)

Design mirrors the sibling mieEODG ``mie_viewer.py`` (threaded compute, a
logging→Text handler, a scrollable left control panel, and matplotlib canvases
that delegate to the package's own plotting functions).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt

# Benign matplotlib notice emitted whenever tight_layout()/constrained_layout
# is applied to an embedded figure — suppress so the console stays clean.
warnings.filterwarnings(
    "ignore",
    message="The figure layout has changed to tight",
    category=UserWarning,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import from submodules (not the package top) to keep the GUI independent of
# rfmwrapper/__init__.py import order, mirroring mie_viewer's rationale.
from rfmwrapper.objects import RFM, runBatch, _run_one, _std_atmospheres
from rfmwrapper.functions import (
    listSensors, listInstruments, writeRFMspc, readAtm,
)
from rfmwrapper.config import (
    get_rfm_dir, get_hitran_path, get_exec_path, write_config, _CONFIG_FILE,
)
from rfmwrapper.constants import rfm_gases
from rfmwrapper.plotting import (
    plotSpectra, plotJacobians, plotFilterBands, plotAtmProfile,
    plotSpectraSweep, plotFilterSweep, plotILS, plotILSSweep,
    light_colors, _resolve_cmap,
)

# Colour maps offered for the batch-sweep overlay (cmcrameri + matplotlib).
_SWEEP_CMAPS = (
    "cmc.batlow", "cmc.hawaii", "cmc.lajolla", "cmc.roma", "cmc.oslo",
    "cmc.bamako", "cmc.lipari", "viridis", "plasma", "cividis", "turbo",
)

# Spectral outputs RFM can emit, in display order.
_OUTPUT_FLAGS = ("RAD", "BBT", "OPT", "TRA")
# Physics flags with their RFM keys and human labels.
_PHYSICS_FLAGS = (
    ("ctm", "Continuum (H₂O/CO₂)"),
    ("mix", "CO₂ line mixing"),
    ("rex", "Rayleigh extinction"),
    ("chi", "CO₂ chi-factor"),
    ("nte", "Non-LTE"),
)
# Scalar RFM constructor kwargs the batch sweep can vary directly.
_SWEEP_SCALARS = (
    "surf_temp", "surf_emiss", "h_surf", "elev", "h_obs",
    "pres", "temp", "pathlength", "nperwn",
)

# Suggested Jacobian targets for the structured editor's (editable) dropdown.
# Surface and aerosol targets first, then common gases; the field stays
# editable so isotope (e.g. H2O(1)) or vibrational (CO2(1)(2)) targets work too.
_JAC_TARGETS = (
    "TEM", "PRE", "SFCTEM", "SFCEMS", "AEROSOL",
    "H2O", "CO2", "O3", "CH4", "CO", "N2O", "SO2", "NO2", "NH3", "HNO3",
)
# Targets that require nadir geometry (RFM only emits the SFC flag for nadir).
_JAC_SURFACE_TARGETS = ("SFCTEM", "SFCEMS")

# Standard starting values for the geometry fields, per geometry type.  Applied
# by _on_geometry_changed() to fields the user hasn't customised:
#   nadir  — looking straight down from top of atmosphere onto an ε=1 surface
#   zenith — looking straight up from the ground (observer at 0 km)
#   limb   — observer above the atmosphere, a default tangent-height ladder
#   layer  — a 1 km homogeneous path at sea-level pressure/temperature
_GEO_DEFAULTS = {
    "nadir":  {"elev": "-90", "h_obs": "100", "h_surf": "0.01", "surf_emiss": "1.0"},
    "zenith": {"elev": "90", "h_obs": "0.0", "surf_emiss": "1.0"},
    "limb":   {"h_obs": "100", "h_surf": "0.01", "surf_emiss": "1.0",
               "tan_heights": "10, 20, 30"},
    "layer":  {"pathlength": "1.0", "pres": "1013", "temp": "298"},
}

# Geometry input fields: (key, label, kind, spin_kwargs, hint).
# kind 'spin' → ttk.Spinbox with steppers; 'entry' → free-text (list-valued).
_GEO_SPECS = (
    ("elev",        "Elevation [°]:",      "spin",
     dict(from_=-90, to=90, increment=5), None),
    ("h_obs",       "Observer alt [km]:",  "spin",
     dict(from_=0, to=1000, increment=10), None),
    ("levels",      "Output levels [km]:", "entry", None,
     "altitudes, comma / space / tab separated — e.g. 5, 10, 20"),
    ("h_surf",      "Surface alt [km]:",   "spin",
     dict(from_=0.0, to=100, increment=0.1, format="%.2f"), None),
    ("surf_emiss",  "Surface emiss:",      "spin",
     dict(from_=0.0, to=1.0, increment=0.05, format="%.2f"), None),
    ("surf_temp",   "Surface temp [K]:",   "spin",
     dict(from_=0, to=400, increment=1), None),
    ("tan_heights", "Tangent hgts [km]:",  "entry", None,
     "altitudes, comma / space / tab separated — e.g. 10, 20, 30"),
    ("pathlength",  "Path length [km]:",   "spin",
     dict(from_=0, to=1000, increment=1), None),
    ("pres",        "Layer pres [mbar]:",  "spin",
     dict(from_=0, to=2000, increment=10), None),
    ("temp",        "Layer temp [K]:",     "spin",
     dict(from_=0, to=1000, increment=5), None),
)


# ============================================================================
# Logging handler
# ============================================================================

class _GuiLogHandler(logging.Handler):
    """Log handler that writes to a Tkinter Text widget with level coloring."""

    _LEVEL_TAG = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, text_widget: tk.Text) -> None:
        super().__init__()
        self._text = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tag = self._LEVEL_TAG.get(record.levelno, "info")
            self._text.after(0, self._append, msg, tag)
        except Exception:
            self.handleError(record)

    def _append(self, msg: str, tag: str) -> None:
        try:
            self._text.config(state=tk.NORMAL)
            self._text.insert(tk.END, msg + "\n", tag)
            self._text.see(tk.END)
            self._text.config(state=tk.DISABLED)
        except tk.TclError:
            pass


# ============================================================================
# Collapsible section
# ============================================================================

class _CollapsibleSection(ttk.Frame):
    """A titled section whose body can be toggled open/closed by its header.

    Build controls into :attr:`body`.  Clicking the header row collapses or
    expands the body; ``toggle_callback`` (if given) fires after each toggle so
    the host can refresh an enclosing scroll region.
    """

    def __init__(self, parent: tk.Widget, title: str, expanded: bool = True,
                 toggle_callback=None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = bool(expanded)
        self._cb = toggle_callback

        self._header = ttk.Label(
            self, cursor="hand2", font=("TkDefaultFont", 10, "bold"),
            text=self._header_text())
        self._header.pack(fill=tk.X, padx=2, pady=(2, 0))
        self._header.bind("<Button-1>", lambda _e: self.toggle())

        self.body = ttk.Frame(self, padding=(8, 4))
        if self._expanded:
            self.body.pack(fill=tk.X, expand=True)

    def _header_text(self) -> str:
        return f"{'▾' if self._expanded else '▸'}  {self._title}"

    def toggle(self) -> None:
        """Flip the body between shown and hidden."""
        self._expanded = not self._expanded
        self._header.config(text=self._header_text())
        if self._expanded:
            self.body.pack(fill=tk.X, expand=True)
        else:
            self.body.pack_forget()
        if self._cb is not None:
            self._cb()


# ============================================================================
# RFMViewer main class
# ============================================================================

class RFMViewer(tk.Toplevel):
    """Interactive GUI for configuring, running, and viewing RFM simulations."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        try:
            from rfmwrapper import __version__ as _pkg_version
        except Exception:
            _pkg_version = ""
        _ver = f" v{_pkg_version}" if _pkg_version else ""
        self.title(f"RFMViewer{_ver} — Reference Forward Model")

        self.parent = parent

        # Screen-proportional sizing.
        self.update_idletasks()
        self._sw = self.winfo_screenwidth()
        self._sh = self.winfo_screenheight()
        w = max(1300, int(self._sw * 0.85))
        h = max(750, int(self._sh * 0.85))
        x = (self._sw - w) // 2
        y = (self._sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1200, 720)

        # Run state.
        self.rfm: RFM | None = None              # last single run
        self.batch: list[RFM] = []               # completed batch jobs
        self.batch_values: list[float] = []      # swept value per batch job
        self.batch_label: str = ""               # swept parameter name
        self._op_message: str = ""

        self._setup_logging()
        self._build_toolbar()
        self._build_layout()

        self.logger.info("RFMViewer initialized")
        # If RFM is not configured, prompt for paths up front (non-blocking).
        self.after(200, self._check_paths_on_startup)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        """Configure logging to the GUI console and a rotating file."""
        log_dir = Path.home() / ".rfmwrapper"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "rfmviewer.log"

        # Capture both the GUI-dedicated logger and the package logger so RFM
        # wrapper messages surface in the console.
        self.logger = logging.getLogger("rfmwrapper.gui")
        self.logger.setLevel(logging.DEBUG)
        for _h in list(self.logger.handlers):
            self.logger.removeHandler(_h)

        self.file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=2 * 1024 * 1024, backupCount=3
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
        )
        self.logger.addHandler(self.file_handler)

        self.gui_handler = None
        self.gui_fmt = logging.Formatter(
            "%(asctime)s  [%(levelname)-7s]  %(message)s", datefmt="%H:%M:%S"
        )

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        """Top row: Load/Save/Paths packed left, RUN packed right, progress fills."""
        outer = ttk.Frame(self)
        outer.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        row1 = ttk.Frame(outer)
        row1.pack(side=tk.TOP, fill=tk.X)

        # Left cluster: config I/O + path configuration.
        ttk.Button(row1, text="Load…", command=self._on_load_config).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Save…", command=self._on_save_config).pack(
            side=tk.LEFT, padx=2)
        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(row1, text="Paths…", command=self._on_configure_paths).pack(
            side=tk.LEFT, padx=2)

        # Right: the prominent RUN button.
        run_style = ttk.Style()
        run_style.configure("Run.TButton", font=("TkDefaultFont", 11, "bold"))
        self.run_btn = ttk.Button(
            row1, text="▶  RUN", style="Run.TButton", command=self._on_run)
        self.run_btn.pack(side=tk.RIGHT, padx=2)

        # Second row: status text + progress bar.
        row2 = ttk.Frame(outer)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(row2, textvariable=self._status_var, foreground="gray").pack(
            side=tk.LEFT, padx=(14, 8))
        self.progress_bar = ttk.Progressbar(row2, mode="indeterminate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        """PanedWindow: fixed-width left (config + log), expanding right (tabs)."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_width = int(self._sw * 0.27)
        left_frame = ttk.Frame(paned, width=left_width)
        left_frame.pack_propagate(False)
        paned.add(left_frame, weight=0)
        self._build_left_panel(left_frame)

        right_container = ttk.Frame(paned)
        paned.add(right_container, weight=1)
        self._build_right_panel(right_container)

    def _build_left_panel(self, parent: tk.Widget) -> None:
        """Scrollable config frames on top, log viewer pinned at the bottom."""
        # Log viewer pinned at the bottom (reserved, non-expanding).
        self._build_log_viewer(parent)

        # Scrollable container for the config frames.
        theme_bg = ttk.Style().lookup("TFrame", "background") or None
        canvas = tk.Canvas(parent, highlightthickness=0, bg=theme_bg)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        # Keep the inner frame as wide as the canvas so the controls fill the
        # column (and don't get clipped on the right).
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        # Collapsible sections build into scroll_frame via self._collapsible().
        self._sections: dict[str, _CollapsibleSection] = {}
        self._build_geometry_frame(scroll_frame)
        self._build_spectral_frame(scroll_frame)
        self._build_gases_frame(scroll_frame)
        self._build_atmosphere_frame(scroll_frame)
        self._build_output_frame(scroll_frame)
        self._build_physics_frame(scroll_frame)
        self._build_advanced_frame(scroll_frame)
        self._build_sweep_frame(scroll_frame)

        # Pack the scrollbar first (right edge), then let the canvas fill the
        # rest.  Packing the canvas first with expand=True would consume the
        # whole column and leave the scrollbar zero width.
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._cfg_canvas = canvas
        self._cfg_scrollbar = scrollbar

        # Mouse-wheel / trackpad scrolling while the pointer is over the panel.
        self._bind_mousewheel(canvas, scroll_frame)
        self._on_geometry_changed()  # set initial enable/disable + defaults

    @staticmethod
    def _bind_mousewheel(canvas: tk.Canvas, area: tk.Widget) -> None:
        """Scroll *canvas* with the wheel while the pointer is over *area*.

        Uses bind_all on enter / unbind on leave so the wheel works over the
        child widgets too.  Handles both the macOS/Windows ``<MouseWheel>``
        event (with ``delta``) and the X11 ``<Button-4/5>`` events.
        """
        def _on_wheel(event):
            if getattr(event, "delta", 0):
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _enter(_e):
            canvas.bind_all("<MouseWheel>", _on_wheel)
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        def _leave(_e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        area.bind("<Enter>", _enter)
        area.bind("<Leave>", _leave)

    def _collapsible(self, parent: tk.Widget, title: str,
                     expanded: bool = True) -> _CollapsibleSection:
        """Create, pack, and register a collapsible section; return it.

        Section builders should place their controls into the returned
        section's ``.body`` frame.
        """
        sec = _CollapsibleSection(
            parent, title, expanded=expanded,
            toggle_callback=self._update_scroll_region)
        sec.pack(fill=tk.X, padx=4, pady=(2, 2))
        self._sections[title] = sec
        return sec

    def _update_scroll_region(self) -> None:
        """Recompute the config canvas scroll region (after a section toggle)."""
        canvas = getattr(self, "_cfg_canvas", None)
        if canvas is not None:
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

    # ------------------------------------------------------------------
    # Config frames
    # ------------------------------------------------------------------
    def _build_geometry_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Geometry").body

        ttk.Label(lf, text="Type:").grid(row=0, column=0, sticky="w")
        self.type_var = tk.StringVar(value="nadir")
        type_combo = ttk.Combobox(
            lf, textvariable=self.type_var, state="readonly", width=12,
            values=["nadir", "zenith", "layer", "limb"])
        type_combo.grid(row=0, column=1, sticky="w")
        type_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_geometry_changed())

        # Viewing-geometry fields.  Numeric scalars use ttk.Spinbox (steppers +
        # free typing); list-valued fields (levels, tan_heights) stay plain
        # entries with a hint.  Fields are seeded with geometry-appropriate
        # standard values by _on_geometry_changed().
        # Each row: key → (label, widget, var, hint-label-or-None).
        self._geo_rows: dict[str, tuple] = {}
        # Tracks the value we last auto-filled per field, so a geometry switch
        # only overwrites values the user hasn't customised.
        self._geo_autofill: dict[str, str] = {}
        row = 1
        for key, label, kind, spin, hint in _GEO_SPECS:
            lbl = ttk.Label(lf, text=label)
            lbl.grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value="")
            if kind == "spin":
                ent = ttk.Spinbox(lf, textvariable=var, width=14, **spin)
            else:
                ent = ttk.Entry(lf, textvariable=var, width=16)
            ent.grid(row=row, column=1, sticky="w")
            row += 1
            hint_lbl = None
            if hint:
                hint_lbl = ttk.Label(lf, text=hint, foreground="gray",
                                     font=("TkDefaultFont", 8), wraplength=210,
                                     justify=tk.LEFT)
                hint_lbl.grid(row=row, column=1, sticky="w")
                row += 1
            self._geo_rows[key] = (lbl, ent, var, hint_lbl)

        lf.columnconfigure(1, weight=1)

    def _on_geometry_changed(self) -> None:
        """Enable only the fields relevant to the selected geometry."""
        gtype = self.type_var.get()
        # field key → set of geometries for which it applies
        applies = {
            "elev": {"nadir", "zenith"},
            "h_obs": {"nadir", "zenith", "limb"},
            "levels": {"nadir", "zenith", "limb"},
            "h_surf": {"nadir", "zenith", "limb"},
            "surf_emiss": {"nadir", "zenith", "limb"},
            "surf_temp": {"nadir", "zenith", "limb"},
            "tan_heights": {"limb"},
            "pathlength": {"layer"},
            "pres": {"layer"},
            "temp": {"layer"},
        }
        defaults = _GEO_DEFAULTS.get(gtype, {})
        for key, (lbl, ent, var, hint_lbl) in self._geo_rows.items():
            on = gtype in applies[key]
            ent.config(state="normal" if on else "disabled")
            lbl.config(foreground="black" if on else "gray")
            if hint_lbl is not None:
                hint_lbl.config(foreground="gray" if on else "#bbbbbb")
            # Seed a standard value when the field is empty or still holds the
            # value we last auto-filled (i.e. the user hasn't customised it).
            if on and key in defaults:
                cur = var.get()
                if cur == "" or cur == self._geo_autofill.get(key):
                    var.set(defaults[key])
                    self._geo_autofill[key] = defaults[key]
        # Keep the sweep parameter list consistent with the active geometry.
        if hasattr(self, "_refresh_sweep_params"):
            self._refresh_sweep_params()

    def _build_spectral_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Spectral Range").body

        ttk.Label(lf, text="WN range [cm⁻¹]:").grid(row=0, column=0, sticky="w")
        rng = ttk.Frame(lf)
        rng.grid(row=0, column=1, sticky="w")
        self.wn_start_var = tk.StringVar(value="2000")
        self.wn_stop_var = tk.StringVar(value="2600")
        ttk.Spinbox(rng, textvariable=self.wn_start_var, width=8,
                    from_=0, to=20000, increment=50).pack(side=tk.LEFT)
        ttk.Label(rng, text="–").pack(side=tk.LEFT, padx=3)
        ttk.Spinbox(rng, textvariable=self.wn_stop_var, width=8,
                    from_=0, to=20000, increment=50).pack(side=tk.LEFT)

        ttk.Label(lf, text="Points / cm⁻¹:").grid(row=1, column=0, sticky="w")
        self.nperwn_var = tk.StringVar(value="10")
        ttk.Spinbox(lf, textvariable=self.nperwn_var, width=8,
                    from_=1, to=1000, increment=1).grid(
            row=1, column=1, sticky="w")
        lf.columnconfigure(1, weight=1)

    def _build_atmosphere_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Atmosphere").body

        ttk.Label(lf, text="Standard:").grid(row=0, column=0, sticky="w")
        self.atm_type_var = tk.StringVar(value="FASSTD")
        atm_combo = ttk.Combobox(
            lf, textvariable=self.atm_type_var, state="readonly", width=14,
            values=list(_std_atmospheres))
        atm_combo.grid(row=0, column=1, sticky="ew")
        atm_combo.bind("<<ComboboxSelected>>",
                       lambda _e: self.atm_path_var.set(""))

        ttk.Label(lf, text="Custom .atm:").grid(row=1, column=0, sticky="w")
        self.atm_path_var = tk.StringVar(value="")
        ttk.Entry(lf, textvariable=self.atm_path_var, width=18).grid(
            row=1, column=1, sticky="ew")
        ttk.Button(lf, text="Browse…", command=self._on_browse_atm).grid(
            row=2, column=1, sticky="w", pady=(2, 0))

        ttk.Label(lf, text="Adjust VMR:").grid(row=3, column=0, sticky="nw", pady=(6, 0))
        self.adjust_var = tk.StringVar(value="")
        ent = ttk.Entry(lf, textvariable=self.adjust_var, width=18)
        ent.grid(row=3, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(lf, text="e.g. CO2=420, SO2=5", foreground="gray",
                  font=("TkDefaultFont", 8)).grid(
            row=4, column=1, sticky="w")
        lf.columnconfigure(1, weight=1)

    def _on_browse_atm(self) -> None:
        path = filedialog.askopenfilename(
            title="Select atmosphere file",
            filetypes=[("Atmosphere", "*.atm"), ("All files", "*.*")])
        if path:
            self.atm_path_var.set(path)

    def _build_gases_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Gases").body

        ttk.Label(lf, text="Species:").grid(row=0, column=0, sticky="w")
        self.gas_var = tk.StringVar(value="all")
        ttk.Entry(lf, textvariable=self.gas_var, width=14).grid(
            row=0, column=1, sticky="ew")
        ttk.Button(lf, text="Select…", width=8,
                   command=self._open_gas_picker).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(lf, text="'all' or e.g. H2O CO2 SO2", foreground="gray",
                  font=("TkDefaultFont", 8)).grid(row=1, column=1, columnspan=2,
                                                  sticky="w")

        self.add_vmr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="add_vmr (append VMR keys to gas list)",
                        variable=self.add_vmr_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Label(lf, text="Layer VMR:").grid(row=3, column=0, sticky="nw", pady=(4, 0))
        self.vmr_var = tk.StringVar(value="")
        ttk.Entry(lf, textvariable=self.vmr_var, width=18).grid(
            row=3, column=1, sticky="ew", pady=(4, 0))
        ttk.Label(lf, text="layer geom., e.g. SO2=500", foreground="gray",
                  font=("TkDefaultFont", 8)).grid(row=4, column=1, sticky="w")
        lf.columnconfigure(1, weight=1)

    @staticmethod
    def _compose_gas_vmr(use_all: bool, checked: list, extras: list,
                         vmr_entries: dict) -> tuple[str, str]:
        """Build ``(gas_str, vmr_str)`` from gas-picker selections.

        Parameters
        ----------
        use_all : bool
            ``gas = 'all'`` (all atmosphere line species).
        checked : list of str
            Standard gases ticked (applied only when not ``use_all``).
        extras : list of str
            Non-standard tokens (XSC species, isotopes, ``aerosol``) to keep.
        vmr_entries : dict of str → str
            Raw per-gas VMR text; applied for ticked gases (or any gas when
            ``use_all``).  Order is preserved in the output string.

        Returns
        -------
        gas_str, vmr_str : str, str
            ``gas`` field and comma-joined ``key=value`` VMR field.

        Raises
        ------
        ValueError
            If a non-empty VMR entry is not a number.
        """
        if use_all:
            parts = ["all"] + list(extras)
        else:
            parts = list(extras) + list(checked)
        gas_str = " ".join(parts).strip() or "all"

        allowed = None if use_all else set(checked)
        vmr_parts = []
        for gas, raw in vmr_entries.items():
            raw = (raw or "").strip()
            if not raw:
                continue
            if allowed is not None and gas not in allowed:
                continue
            try:
                val = float(raw)
            except ValueError:
                raise ValueError(f"VMR for {gas} is not a number: {raw!r}")
            vmr_parts.append(f"{gas}={val:g}")
        return gas_str, ", ".join(vmr_parts)

    def _open_gas_picker(self) -> None:
        """Popup to tick species and enter per-gas layer VMRs → gas/VMR fields.

        Pre-populates from the current ``gas``/``Layer VMR`` fields, lists every
        RFM line-data gas with a checkbox and a VMR entry, and writes the result
        back on OK.  Tokens not in the standard gas list (XSC species, isotopes,
        ``aerosol``) are preserved and re-appended so the picker is non-lossy.
        """
        # Parse the current state.
        cur_gas = self.gas_var.get().strip()
        use_all = cur_gas.lower() == "all"
        cur_tokens = [] if use_all else cur_gas.split()
        known = {g.upper(): g for g in rfm_gases}
        checked = {known[t.upper()] for t in cur_tokens if t.upper() in known}
        extras = [t for t in cur_tokens if t.upper() not in known]  # preserved as-is
        try:
            cur_vmr = self._parse_kv(self.vmr_var.get())
        except ValueError:
            cur_vmr = {}
        cur_vmr_up = {k.upper(): v for k, v in cur_vmr.items()}

        dlg = tk.Toplevel(self)
        dlg.title("Select gases & VMRs")
        dlg.transient(self)
        dlg.grab_set()
        outer = ttk.Frame(dlg, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        all_var = tk.BooleanVar(value=use_all)
        check_vars: dict[str, tk.BooleanVar] = {}
        vmr_vars: dict[str, tk.StringVar] = {}

        ttk.Checkbutton(
            outer, text="All atmosphere species (gas = 'all')", variable=all_var,
            command=lambda: _sync_all()).grid(row=0, column=0, columnspan=6,
                                              sticky="w", pady=(0, 6))
        ttk.Label(outer, text="Species", font=("TkDefaultFont", 9, "bold")).grid(
            row=1, column=1, sticky="w")
        ttk.Label(outer, text="VMR [ppmv] (layer)",
                  font=("TkDefaultFont", 9, "bold")).grid(row=1, column=2, sticky="w")
        ttk.Label(outer, text="Species", font=("TkDefaultFont", 9, "bold")).grid(
            row=1, column=4, sticky="w")
        ttk.Label(outer, text="VMR [ppmv] (layer)",
                  font=("TkDefaultFont", 9, "bold")).grid(row=1, column=5, sticky="w")

        rows: list[tuple] = []   # (checkbutton, entry) for enable/disable
        half = (len(rfm_gases) + 1) // 2
        for i, gas in enumerate(rfm_gases):
            colbase = 0 if i < half else 3
            r = (i % half) + 2
            cvar = tk.BooleanVar(value=gas in checked)
            svar = tk.StringVar(value=(f"{cur_vmr_up[gas.upper()]:g}"
                                       if gas.upper() in cur_vmr_up else ""))
            check_vars[gas] = cvar
            vmr_vars[gas] = svar
            cb = ttk.Checkbutton(outer, text=gas, variable=cvar, width=8)
            cb.grid(row=r, column=colbase + 1, sticky="w")
            ent = ttk.Entry(outer, textvariable=svar, width=8)
            ent.grid(row=r, column=colbase + 2, sticky="w", padx=(0, 10))
            rows.append((cb, ent))

        def _sync_all() -> None:
            # When 'all' is on, grey the per-gas checkboxes (VMR entries stay
            # usable for layer overrides via add_vmr).
            state = "disabled" if all_var.get() else "normal"
            for cb, _ent in rows:
                cb.config(state=state)
        _sync_all()

        if extras:
            ttk.Label(outer, text=f"Kept as-is: {' '.join(extras)}",
                      foreground="gray", font=("TkDefaultFont", 8)).grid(
                row=half + 2, column=0, columnspan=6, sticky="w", pady=(6, 0))

        def _ok() -> None:
            checked = [g for g in rfm_gases if check_vars[g].get()]
            vmr_entries = {g: vmr_vars[g].get() for g in rfm_gases}
            try:
                gas_str, vmr_str = self._compose_gas_vmr(
                    all_var.get(), checked, extras, vmr_entries)
            except ValueError as e:
                messagebox.showerror("Invalid VMR", str(e), parent=dlg)
                return
            self.gas_var.set(gas_str)
            self.vmr_var.set(vmr_str)
            # VMRs while gas='all' only take effect with add_vmr.
            if all_var.get() and vmr_str:
                self.add_vmr_var.set(True)
            dlg.destroy()

        btns = ttk.Frame(outer)
        btns.grid(row=half + 3, column=0, columnspan=6, pady=(10, 0))
        ttk.Button(btns, text="OK", command=_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

    def _build_output_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Output").body
        self.out_vars: dict[str, tk.BooleanVar] = {}
        for i, flag in enumerate(_OUTPUT_FLAGS):
            var = tk.BooleanVar(value=True)
            self.out_vars[flag] = var
            ttk.Checkbutton(lf, text=flag, variable=var).grid(
                row=0, column=i, sticky="w", padx=(0, 8))

    def _build_physics_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Physics").body
        # Defaults follow the RFM constructor: ctm/mix on, rest off.
        defaults = {"ctm": True, "mix": True, "rex": False, "chi": False, "nte": False}
        self.physics_vars: dict[str, tk.BooleanVar] = {}
        for i, (key, label) in enumerate(_PHYSICS_FLAGS):
            var = tk.BooleanVar(value=defaults[key])
            self.physics_vars[key] = var
            ttk.Checkbutton(lf, text=label, variable=var).grid(
                row=i, column=0, sticky="w")

    def _build_advanced_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Advanced").body

        # --- Structured Jacobian editor: one row per target ---
        jac_lf = ttk.LabelFrame(lf, text="Jacobians", padding=4)
        jac_lf.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        jac_lf.columnconfigure(0, weight=1)
        hdr = ttk.Frame(jac_lf)
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Target", width=12,
                  font=("TkDefaultFont", 8, "bold")).pack(side=tk.LEFT)
        ttk.Label(hdr, text="Levels [km] (blank = column)",
                  font=("TkDefaultFont", 8, "bold")).pack(side=tk.LEFT, padx=(4, 0))
        self._jac_rows_frame = ttk.Frame(jac_lf)
        self._jac_rows_frame.grid(row=1, column=0, sticky="ew")
        self._jac_rows: list[dict] = []
        ttk.Button(jac_lf, text="+ Add Jacobian",
                   command=lambda: self._add_jac_row()).grid(
            row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Label(jac_lf,
                  text="N levels → N−2 layer spectra; surface targets need nadir.",
                  foreground="gray", font=("TkDefaultFont", 8),
                  wraplength=200, justify=tk.LEFT).grid(row=3, column=0, sticky="w")

        ttk.Label(lf, text="XSC dir:").grid(row=2, column=0, sticky="w")
        self.xsc_dir_var = tk.StringVar(value="")
        ttk.Entry(lf, textvariable=self.xsc_dir_var, width=18).grid(
            row=2, column=1, sticky="ew")

        ttk.Label(lf, text="XSC files:").grid(row=3, column=0, sticky="nw")
        self.xsc_var = tk.StringVar(value="")
        ttk.Entry(lf, textvariable=self.xsc_var, width=18).grid(
            row=3, column=1, sticky="ew")
        ttk.Label(lf, text="e.g. CFC11=/path/cfc11.xsc",
                  foreground="gray", font=("TkDefaultFont", 8)).grid(
            row=4, column=1, sticky="w")

        self.binary_var = tk.BooleanVar(value=False)
        self.dbl_var = tk.BooleanVar(value=False)
        self.diag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="binary output", variable=self.binary_var).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(lf, text="double precision (dbl)", variable=self.dbl_var).grid(
            row=6, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(lf, text="diagnostics (PRF/PTH)", variable=self.diag_var).grid(
            row=7, column=0, columnspan=2, sticky="w")
        lf.columnconfigure(1, weight=1)

    def _build_sweep_frame(self, parent: tk.Widget) -> None:
        lf = self._collapsible(parent, "Batch Sweep").body

        self.sweep_on_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="Enable batch sweep", variable=self.sweep_on_var,
                        command=self._on_sweep_toggle).grid(
            row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(lf, text="Parameter:").grid(row=1, column=0, sticky="w")
        self.sweep_param_var = tk.StringVar(value="surf_temp")
        self.sweep_param_combo = ttk.Combobox(
            lf, textvariable=self.sweep_param_var, state="readonly", width=14)
        self.sweep_param_combo.grid(row=1, column=1, sticky="w")

        ttk.Label(lf, text="Mode:").grid(row=2, column=0, sticky="w")
        self.sweep_mode_var = tk.StringVar(value="range")
        mode_combo = ttk.Combobox(
            lf, textvariable=self.sweep_mode_var, state="readonly", width=14,
            values=["range", "list"])
        mode_combo.grid(row=2, column=1, sticky="w")
        mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_sweep_toggle())

        # range mode
        self.sweep_range_frame = ttk.Frame(lf)
        self.sweep_range_frame.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(self.sweep_range_frame, text="start").pack(side=tk.LEFT)
        self.sweep_start_var = tk.StringVar(value="270")
        ttk.Entry(self.sweep_range_frame, textvariable=self.sweep_start_var,
                  width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.sweep_range_frame, text="stop").pack(side=tk.LEFT)
        self.sweep_stop_var = tk.StringVar(value="310")
        ttk.Entry(self.sweep_range_frame, textvariable=self.sweep_stop_var,
                  width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.sweep_range_frame, text="N").pack(side=tk.LEFT)
        self.sweep_n_var = tk.StringVar(value="5")
        ttk.Entry(self.sweep_range_frame, textvariable=self.sweep_n_var,
                  width=4).pack(side=tk.LEFT, padx=2)

        # list mode
        self.sweep_list_frame = ttk.Frame(lf)
        self.sweep_list_frame.grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Label(self.sweep_list_frame, text="values:").pack(side=tk.LEFT)
        self.sweep_list_var = tk.StringVar(value="270, 290, 310")
        ttk.Entry(self.sweep_list_frame, textvariable=self.sweep_list_var,
                  width=20).pack(side=tk.LEFT, padx=2)

        ttk.Label(lf, text="Workers:").grid(row=5, column=0, sticky="w")
        self.workers_var = tk.StringVar(value=str(os.cpu_count() or 4))
        ttk.Entry(lf, textvariable=self.workers_var, width=6).grid(
            row=5, column=1, sticky="w")

        lf.columnconfigure(1, weight=1)
        self._refresh_sweep_params()
        self._on_sweep_toggle()

    def _refresh_sweep_params(self) -> None:
        """Populate sweep parameter choices: scalar kwargs + current gases."""
        params = list(_SWEEP_SCALARS)
        gases = self.gas_var.get().split() if hasattr(self, "gas_var") else []
        params += [f"VMR:{g}" for g in gases if g.lower() != "all"]
        self.sweep_param_combo.config(values=params)
        if self.sweep_param_var.get() not in params:
            self.sweep_param_var.set(params[0])

    def _on_sweep_toggle(self) -> None:
        """Show range/list inputs per mode and grey out when disabled."""
        on = self.sweep_on_var.get()
        state = "normal" if on else "disabled"
        self.sweep_param_combo.config(state="readonly" if on else "disabled")
        mode = self.sweep_mode_var.get()
        for child in self.sweep_range_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.config(state=state if mode == "range" else "disabled")
        for child in self.sweep_list_frame.winfo_children():
            if isinstance(child, ttk.Entry):
                child.config(state=state if mode == "list" else "disabled")
        self.run_btn.config(text="▶  RUN BATCH" if on else "▶  RUN")
        if on:
            self._refresh_sweep_params()

    # ------------------------------------------------------------------
    # Right panel — plot tabs
    # ------------------------------------------------------------------
    def _build_right_panel(self, parent: tk.Widget) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.spectra_tab = self._build_spectra_tab()
        self.notebook.add(self.spectra_tab, text="Spectra")
        self.atm_tab = self._build_atm_tab()
        self.notebook.add(self.atm_tab, text="Atmosphere")
        self.jac_tab = self._build_jac_tab()
        self.notebook.add(self.jac_tab, text="Jacobians")
        self.conv_tab = self._build_conv_tab()
        self.notebook.add(self.conv_tab, text="Filters / ILS")
        self.diag_tab = self._build_diag_tab()
        self.notebook.add(self.diag_tab, text="Diagnostics")

    def _embed_canvas(self, frame: tk.Widget, figsize=(9, 6)):
        """Create a Figure + canvas + navigation toolbar inside *frame*."""
        fig = plt.Figure(figsize=figsize, dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        return fig, canvas

    @staticmethod
    def _placeholder(fig, canvas, text: str) -> None:
        """Show a centered grey prompt on an empty figure."""
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center",
                color="gray", transform=ax.transAxes, wrap=True)
        ax.set_axis_off()
        canvas.draw()

    def _build_spectra_tab(self) -> ttk.Frame:
        frame = ttk.Frame(self.notebook)
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(ctrl, text="Output:").pack(side=tk.LEFT)
        self.spectra_which_var = tk.StringVar(value="grid")
        for val in ("grid", "rad", "bbt", "opt", "tra"):
            ttk.Radiobutton(ctrl, text=val.upper(), value=val,
                            variable=self.spectra_which_var,
                            command=self._plot_spectra).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(ctrl, text="Export…", command=self._on_export_spectra).pack(
            side=tk.RIGHT, padx=2)
        # Sweep colour map (used to colour a batch overlay by sweep value).
        self.sweep_cmap_var = tk.StringVar(value=_SWEEP_CMAPS[0])
        self.sweep_cmap_combo = ttk.Combobox(
            ctrl, textvariable=self.sweep_cmap_var, state="readonly",
            width=12, values=list(_SWEEP_CMAPS))
        self.sweep_cmap_combo.pack(side=tk.RIGHT, padx=(0, 6))
        self.sweep_cmap_combo.bind("<<ComboboxSelected>>",
                                   lambda _e: self._plot_spectra())
        ttk.Label(ctrl, text="Sweep cmap:").pack(side=tk.RIGHT, padx=(8, 2))
        self.spectra_fig, self.spectra_canvas = self._embed_canvas(frame)
        self._placeholder(self.spectra_fig, self.spectra_canvas,
                          "Configure a run and press RUN.")
        return frame

    def _build_atm_tab(self) -> ttk.Frame:
        frame = ttk.Frame(self.notebook)
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(ctrl, text="Field:").pack(side=tk.LEFT)
        self.atm_field_var = tk.StringVar(value="TEM")
        self.atm_field_combo = ttk.Combobox(
            ctrl, textvariable=self.atm_field_var, state="readonly", width=10,
            values=["TEM", "PRE"])
        self.atm_field_combo.pack(side=tk.LEFT, padx=(2, 8))
        self.atm_field_combo.bind("<<ComboboxSelected>>", lambda _e: self._plot_atm())
        ttk.Button(ctrl, text="Preview from config",
                   command=self._preview_atm).pack(side=tk.LEFT, padx=2)
        self.atm_fig, self.atm_canvas = self._embed_canvas(frame, figsize=(7, 6))
        self._placeholder(self.atm_fig, self.atm_canvas,
                          "Press 'Preview from config' or run a model.")
        return frame

    def _build_jac_tab(self) -> ttk.Frame:
        frame = ttk.Frame(self.notebook)
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(ctrl, text="Parameter:").pack(side=tk.LEFT)
        self.jac_param_var = tk.StringVar(value="")
        self.jac_param_combo = ttk.Combobox(
            ctrl, textvariable=self.jac_param_var, state="readonly", width=10)
        self.jac_param_combo.pack(side=tk.LEFT, padx=(2, 8))
        self.jac_param_combo.bind("<<ComboboxSelected>>", lambda _e: self._plot_jac())
        # Output quantity: a run with several output flags produces one
        # Jacobian per quantity (∂RAD/∂T, ∂BBT/∂T, …) — pick which to view.
        ttk.Label(ctrl, text="Output:").pack(side=tk.LEFT)
        self.jac_output_var = tk.StringVar(value="")
        self.jac_output_combo = ttk.Combobox(
            ctrl, textvariable=self.jac_output_var, state="readonly", width=6)
        self.jac_output_combo.pack(side=tk.LEFT, padx=(2, 8))
        self.jac_output_combo.bind("<<ComboboxSelected>>", lambda _e: self._plot_jac())
        # Sweep-member selector — only meaningful after a batch run.
        ttk.Label(ctrl, text="Member:").pack(side=tk.LEFT)
        self.jac_member_var = tk.StringVar(value="")
        self.jac_member_combo = ttk.Combobox(
            ctrl, textvariable=self.jac_member_var, state="disabled", width=14)
        self.jac_member_combo.pack(side=tk.LEFT, padx=(2, 8))
        self.jac_member_combo.bind("<<ComboboxSelected>>", lambda _e: self._plot_jac())
        self.jac_fig, self.jac_canvas = self._embed_canvas(frame)
        self._placeholder(self.jac_fig, self.jac_canvas,
                          "Set Jacobians in Advanced, then RUN.")
        return frame

    def _build_conv_tab(self) -> ttk.Frame:
        frame = ttk.Frame(self.notebook)
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))

        ttk.Label(ctrl, text="Output:").pack(side=tk.LEFT)
        self.conv_which_var = tk.StringVar(value="rad")
        ttk.Combobox(ctrl, textvariable=self.conv_which_var, state="readonly",
                     width=6, values=["rad", "bbt", "opt", "tra"]).pack(
            side=tk.LEFT, padx=(2, 10))

        ttk.Label(ctrl, text="Sensor:").pack(side=tk.LEFT)
        self.conv_sensor_var = tk.StringVar(value="")
        try:
            sensors = listSensors()
        except Exception:
            sensors = []
        ttk.Combobox(ctrl, textvariable=self.conv_sensor_var, state="readonly",
                     width=14, values=sensors).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Apply filter", command=self._on_apply_filter).pack(
            side=tk.LEFT, padx=2)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(ctrl, text="Instrument:").pack(side=tk.LEFT)
        self.conv_instr_var = tk.StringVar(value="")
        try:
            instruments = listInstruments()
        except Exception:
            instruments = []
        ttk.Combobox(ctrl, textvariable=self.conv_instr_var, state="readonly",
                     width=10, values=instruments).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Apply ILS", command=self._on_apply_ils).pack(
            side=tk.LEFT, padx=2)

        self.conv_fig, self.conv_canvas = self._embed_canvas(frame)
        self._placeholder(self.conv_fig, self.conv_canvas,
                          "Run a model, then apply a sensor filter or instrument ILS.")
        return frame

    def _build_diag_tab(self) -> ttk.Frame:
        frame = ttk.Frame(self.notebook)
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(ctrl, text="Mode:").pack(side=tk.LEFT)
        self.diag_mode_var = tk.StringVar(value="Limb map")
        diag_mode = ttk.Combobox(ctrl, textvariable=self.diag_mode_var,
                                 state="readonly", width=10,
                                 values=["Limb map", "Ray path"])
        diag_mode.pack(side=tk.LEFT, padx=(2, 8))
        diag_mode.bind("<<ComboboxSelected>>", lambda _e: self._plot_diag())
        ttk.Label(ctrl, text="Gas:").pack(side=tk.LEFT)
        self.diag_gas_var = tk.StringVar(value="")
        self.diag_gas_combo = ttk.Combobox(ctrl, textvariable=self.diag_gas_var,
                                           state="readonly", width=8)
        self.diag_gas_combo.pack(side=tk.LEFT, padx=2)
        self.diag_gas_combo.bind("<<ComboboxSelected>>", lambda _e: self._plot_diag())
        self.diag_fig, self.diag_canvas = self._embed_canvas(frame)
        self._placeholder(self.diag_fig, self.diag_canvas,
                          "Limb map needs limb geometry; Ray path needs "
                          "diagnostics=True (Advanced). Then RUN.")
        return frame

    def _refresh_diag(self) -> None:
        """Populate the ray-path gas list and draw the diagnostics view."""
        if self.rfm is None:
            return
        gases = [g for g in self.rfm.gas.split() if g] if isinstance(
            self.rfm.gas, str) else []
        self.diag_gas_combo.config(values=gases)
        if gases and self.diag_gas_var.get() not in gases:
            self.diag_gas_var.set(gases[0])
        # Default the mode to whatever the run supports.
        if self.rfm.type == "limb":
            self.diag_mode_var.set("Limb map")
        elif getattr(self.rfm, "diagnostics", False):
            self.diag_mode_var.set("Ray path")
        self._plot_diag()

    def _plot_diag(self) -> None:
        if self.rfm is None:
            return
        from rfmwrapper.plotting import plotLimbMap, plotRayPath
        from rfmwrapper.functions import parseRayPath
        mode = self.diag_mode_var.get()
        self.diag_fig.clear()
        ax = self.diag_fig.add_subplot(111)
        try:
            if mode == "Limb map":
                if self.rfm.type != "limb":
                    raise ValueError("Limb map requires limb geometry.")
                key = "rad" if "rad" in self.rfm._out else list(self.rfm._out)[0]
                plotLimbMap(self.rfm.getSpectra(key), ax=ax)
            else:
                pth = self.rfm.getRayPath()  # raises if diagnostics=False
                first = next(iter(pth.values()))
                parsed = parseRayPath(first)
                plotRayPath(parsed, self.diag_gas_var.get(), ax=ax)
        except Exception as e:
            self._placeholder(self.diag_fig, self.diag_canvas, str(e))
            return
        self.diag_fig.tight_layout()
        self.diag_canvas.draw()

    # ------------------------------------------------------------------
    # Log viewer (left panel)
    # ------------------------------------------------------------------
    def _build_log_viewer(self, parent: tk.Widget) -> None:
        outer = ttk.LabelFrame(parent, text="Log", padding=2)
        outer.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(2, 0))
        nb = ttk.Notebook(outer)
        nb.pack(fill=tk.BOTH, expand=True)

        # Console tab — wrapper logging.
        console_frame = ttk.Frame(nb)
        nb.add(console_frame, text="Console")
        c_sb = ttk.Scrollbar(console_frame, orient=tk.VERTICAL)
        c_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(console_frame, height=10, state=tk.DISABLED,
                                font=("TkFixedFont", 10), wrap=tk.WORD,
                                yscrollcommand=c_sb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        c_sb.config(command=self.log_text.yview)
        for tag, col in (("debug", "#999999"), ("info", "#555555"),
                         ("warning", "#bb7700"), ("error", "#cc2200")):
            self.log_text.tag_config(tag, foreground=col)

        self.gui_handler = _GuiLogHandler(self.log_text)
        self.gui_handler.setLevel(logging.DEBUG)
        self.gui_handler.setFormatter(self.gui_fmt)
        self.logger.addHandler(self.gui_handler)

        # rfm.log tab — raw RFM diagnostics.
        rfm_frame = ttk.Frame(nb)
        nb.add(rfm_frame, text="rfm.log")
        r_sb = ttk.Scrollbar(rfm_frame, orient=tk.VERTICAL)
        r_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.rfm_log_text = tk.Text(rfm_frame, height=10, state=tk.DISABLED,
                                    font=("TkFixedFont", 10), wrap=tk.NONE,
                                    yscrollcommand=r_sb.set)
        self.rfm_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        r_sb.config(command=self.rfm_log_text.yview)

    def _show_rfm_log(self, drv_path: str) -> None:
        """Load and display the RFM rfm.log written under *drv_path*."""
        log_path = os.path.join(drv_path, "rfm.log")
        try:
            with open(log_path, "r") as f:
                content = f.read()
        except OSError as e:
            content = f"(could not read {log_path}: {e})"
        self.rfm_log_text.config(state=tk.NORMAL)
        self.rfm_log_text.delete("1.0", tk.END)
        self.rfm_log_text.insert(tk.END, content)
        self.rfm_log_text.see(tk.END)
        self.rfm_log_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Progress helpers (thread-safe), mirroring mie_viewer
    # ------------------------------------------------------------------
    def _begin_op(self, message: str, total: int | None = None) -> None:
        self._status_var.set(message)
        self._op_message = message
        self.run_btn.config(state="disabled")
        try:
            self.progress_bar.stop()
        except tk.TclError:
            pass
        if total is None:
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start(12)
        else:
            self.progress_bar.config(mode="determinate", maximum=total, value=0)

    def _progress_step(self, current: int, total: int) -> None:
        try:
            self.progress_bar.config(value=current)
            self._status_var.set(f"{self._op_message} ({current}/{total})")
        except tk.TclError:
            pass

    def _post_progress(self, current: int, total: int) -> None:
        self._safe_after(lambda: self._progress_step(current, total))

    def _end_op(self, message: str = "Ready") -> None:
        try:
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", value=0)
        except tk.TclError:
            pass
        self._status_var.set(message)
        try:
            self.run_btn.config(state="normal")
        except tk.TclError:
            pass

    def _safe_after(self, callback) -> None:
        """Schedule a main-thread callback, ignoring a closed window."""
        try:
            if self.winfo_exists():
                self.after(0, callback)
        except (tk.TclError, RuntimeError):
            pass

    # ------------------------------------------------------------------
    # Config collection / application
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_scalar_or_list(text: str):
        """Parse '5' → 5.0, '0, 5, 10' → [0.0, 5.0, 10.0], '' → None."""
        text = text.strip()
        if not text:
            return None
        parts = [p for p in text.replace(",", " ").split() if p]
        vals = [float(p) for p in parts]
        return vals[0] if len(vals) == 1 else vals

    @staticmethod
    def _parse_list(text: str):
        """Parse a comma/space list of floats → list, or None if empty."""
        text = text.strip()
        if not text:
            return None
        return [float(p) for p in text.replace(",", " ").split() if p]

    @staticmethod
    def _parse_kv(text: str) -> dict:
        """Parse 'A=1, B=2' → {'A': 1.0, 'B': 2.0} (numeric values)."""
        out: dict = {}
        for chunk in text.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Expected key=value, got {chunk!r}")
            k, v = chunk.split("=", 1)
            out[k.strip()] = float(v.strip())
        return out

    @staticmethod
    def _parse_kv_str(text: str) -> dict:
        """Parse 'A=/path, B=/p2' → {'A': '/path', 'B': '/p2'} (string values)."""
        out: dict = {}
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Expected key=value, got {chunk!r}")
            k, v = chunk.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    @staticmethod
    def _parse_jacobians(text: str) -> dict:
        """Parse 'TEM, H2O' or 'TEM:0,5,10' → {param: None | [alts]}."""
        out: dict = {}
        # Split on ';' for multiple params with altitude lists, else commas.
        sep = ";" if ":" in text else ","
        for chunk in text.split(sep):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                name, alts = chunk.split(":", 1)
                out[name.strip()] = [float(a) for a in
                                     alts.replace(",", " ").split() if a]
            else:
                out[chunk] = None
        return out

    # ------------------------------------------------------------------
    # Structured Jacobian editor
    # ------------------------------------------------------------------
    def _add_jac_row(self, target: str = "", levels: str = "") -> None:
        """Append one Jacobian target row (editable target + levels entry)."""
        row_frame = ttk.Frame(self._jac_rows_frame)
        row_frame.pack(fill=tk.X, pady=1)
        target_var = tk.StringVar(value=target)
        levels_var = tk.StringVar(value=levels)
        ttk.Combobox(row_frame, textvariable=target_var, width=10,
                     values=list(_JAC_TARGETS)).pack(side=tk.LEFT)
        ttk.Entry(row_frame, textvariable=levels_var, width=12).pack(
            side=tk.LEFT, padx=(4, 0))
        row = {"target": target_var, "levels": levels_var, "frame": row_frame}
        ttk.Button(row_frame, text="✕", width=2,
                   command=lambda: self._remove_jac_row(row)).pack(
            side=tk.LEFT, padx=(2, 0))
        self._jac_rows.append(row)
        self._update_scroll_region()

    def _remove_jac_row(self, row: dict) -> None:
        """Destroy one Jacobian target row."""
        row["frame"].destroy()
        if row in self._jac_rows:
            self._jac_rows.remove(row)
        self._update_scroll_region()

    def _collect_jacobians(self) -> list[dict]:
        """Read the editor rows → JSON-friendly list of {target, levels} strings."""
        return [{"target": r["target"].get().strip(),
                 "levels": r["levels"].get().strip()}
                for r in self._jac_rows if r["target"].get().strip()]

    def _apply_jacobians(self, data) -> None:
        """Rebuild the editor rows from saved config (list, or legacy string)."""
        for r in list(self._jac_rows):
            self._remove_jac_row(r)
        if isinstance(data, str):
            # Legacy single-field format → convert via the old parser.
            for target, alts in (self._parse_jacobians(data).items()
                                 if data.strip() else []):
                lv = "" if alts is None else ", ".join(f"{a:g}" for a in alts)
                self._add_jac_row(target, lv)
        else:
            for row in (data or []):
                self._add_jac_row(row.get("target", ""), row.get("levels", ""))

    @staticmethod
    def _jacobians_dict(rows: list) -> dict:
        """Editor rows (or legacy string) → {target: None | [alts]} for RFM."""
        if isinstance(rows, str):
            return RFMViewer._parse_jacobians(rows) if rows.strip() else {}
        out: dict = {}
        for row in rows:
            target = row.get("target", "").strip()
            if not target:
                continue
            levels = row.get("levels", "").strip()
            if levels:
                out[target] = [float(a) for a in
                               levels.replace(",", " ").split() if a]
            else:
                out[target] = None
        return out

    def _validate_jacobians(self, jac: dict, cfg: dict) -> None:
        """Check Jacobian prerequisites; raise ValueError with a clear message."""
        gtype = cfg["type"]
        gases = cfg["gas"].upper().split()
        xsc_keys = [k.upper() for k in self._parse_kv_str(cfg["xsc"])] \
            if cfg.get("xsc", "").strip() else []
        nte = bool(cfg["physics"].get("nte", False))
        for target in jac:
            t = target.upper()
            if t in _JAC_SURFACE_TARGETS:
                if gtype != "nadir":
                    raise ValueError(
                        f"{target} Jacobian requires nadir geometry "
                        "(RFM only enables the surface/SFC flag for nadir).")
            elif t == "AEROSOL":
                if "AEROSOL" not in gases and "AEROSOL" not in xsc_keys:
                    raise ValueError(
                        "AEROSOL Jacobian requires an 'aerosol' species in the "
                        "gas list (or an AEROSOL entry in XSC files).")
            elif t.count("(") == 2:
                # Vibrational-temperature target, e.g. CO2(1)(2).
                if not nte:
                    raise ValueError(
                        f"Vibrational-temperature Jacobian {target} requires the "
                        "NTE physics flag (enable 'Non-LTE' in Physics).")

    def _collect_config(self) -> dict:
        """Read every widget into a JSON-friendly config dict."""
        cfg: dict = {}
        cfg["type"] = self.type_var.get()
        cfg["wn_start"] = self.wn_start_var.get()
        cfg["wn_stop"] = self.wn_stop_var.get()
        cfg["nperwn"] = self.nperwn_var.get()
        cfg["atm_type"] = self.atm_type_var.get()
        cfg["atm_path"] = self.atm_path_var.get()
        cfg["adjust"] = self.adjust_var.get()
        cfg["gas"] = self.gas_var.get()
        cfg["add_vmr"] = self.add_vmr_var.get()
        cfg["vmr"] = self.vmr_var.get()
        cfg["out"] = [f for f in _OUTPUT_FLAGS if self.out_vars[f].get()]
        cfg["physics"] = {k: v.get() for k, v in self.physics_vars.items()}
        cfg["jacobians"] = self._collect_jacobians()
        cfg["xsc_dir"] = self.xsc_dir_var.get()
        cfg["xsc"] = self.xsc_var.get()
        cfg["binary"] = self.binary_var.get()
        cfg["dbl"] = self.dbl_var.get()
        cfg["diagnostics"] = self.diag_var.get()
        cfg["geo"] = {k: row[2].get() for k, row in self._geo_rows.items()}
        cfg["sweep"] = {
            "on": self.sweep_on_var.get(),
            "param": self.sweep_param_var.get(),
            "mode": self.sweep_mode_var.get(),
            "start": self.sweep_start_var.get(),
            "stop": self.sweep_stop_var.get(),
            "n": self.sweep_n_var.get(),
            "list": self.sweep_list_var.get(),
            "workers": self.workers_var.get(),
        }
        return cfg

    def _apply_config(self, cfg: dict) -> None:
        """Repopulate every widget from a config dict (inverse of _collect)."""
        self.type_var.set(cfg.get("type", "nadir"))
        self.wn_start_var.set(cfg.get("wn_start", "2000"))
        self.wn_stop_var.set(cfg.get("wn_stop", "2600"))
        self.nperwn_var.set(cfg.get("nperwn", "10"))
        self.atm_type_var.set(cfg.get("atm_type", "FASSTD"))
        self.atm_path_var.set(cfg.get("atm_path", ""))
        self.adjust_var.set(cfg.get("adjust", ""))
        self.gas_var.set(cfg.get("gas", "all"))
        self.add_vmr_var.set(cfg.get("add_vmr", False))
        self.vmr_var.set(cfg.get("vmr", ""))
        out = cfg.get("out", list(_OUTPUT_FLAGS))
        for f in _OUTPUT_FLAGS:
            self.out_vars[f].set(f in out)
        for k, v in cfg.get("physics", {}).items():
            if k in self.physics_vars:
                self.physics_vars[k].set(v)
        self._apply_jacobians(cfg.get("jacobians", []))
        self.xsc_dir_var.set(cfg.get("xsc_dir", ""))
        self.xsc_var.set(cfg.get("xsc", ""))
        self.binary_var.set(cfg.get("binary", False))
        self.dbl_var.set(cfg.get("dbl", False))
        self.diag_var.set(cfg.get("diagnostics", False))
        # Clear the auto-fill memory so loaded (non-empty) values are preserved
        # rather than overwritten by geometry defaults below.
        self._geo_autofill = {}
        for k, val in cfg.get("geo", {}).items():
            if k in self._geo_rows:
                self._geo_rows[k][2].set(val)
        sw = cfg.get("sweep", {})
        self.sweep_on_var.set(sw.get("on", False))
        self.sweep_mode_var.set(sw.get("mode", "range"))
        self.sweep_start_var.set(sw.get("start", "270"))
        self.sweep_stop_var.set(sw.get("stop", "310"))
        self.sweep_n_var.set(sw.get("n", "5"))
        self.sweep_list_var.set(sw.get("list", "270, 290, 310"))
        self.workers_var.set(sw.get("workers", str(os.cpu_count() or 4)))
        self._on_geometry_changed()
        self._refresh_sweep_params()
        if sw.get("param") in self.sweep_param_combo["values"]:
            self.sweep_param_var.set(sw["param"])
        self._on_sweep_toggle()

    def _rfm_kwargs(self, cfg: dict) -> dict:
        """Translate a config dict into RFM(**kwargs). Raises ValueError on bad input."""
        kw: dict = {
            "type": cfg["type"],
            "wn_start": float(cfg["wn_start"]),
            "wn_stop": float(cfg["wn_stop"]),
            "nperwn": float(cfg["nperwn"]),
            "gas": cfg["gas"].strip() or "all",
            "add_vmr": bool(cfg["add_vmr"]),
            "out": " ".join(cfg["out"]) or "RAD",
            "physics": dict(cfg["physics"]),
            "binary": bool(cfg["binary"]),
            "dbl": bool(cfg["dbl"]),
            "diagnostics": bool(cfg["diagnostics"]),
            "silent": True,
            # Resolve paths fresh from config each run (no stale module globals).
            "exec_path": get_exec_path(),
            "hitran_path": get_hitran_path(),
        }

        # Atmosphere: an explicit custom file takes priority over the standard key.
        if cfg["atm_path"].strip():
            kw["atm_path"] = cfg["atm_path"].strip()
        else:
            kw["atm_type"] = cfg["atm_type"]

        if cfg["adjust"].strip():
            kw["adjust"] = self._parse_kv(cfg["adjust"])
        if cfg["vmr"].strip():
            kw["vmr"] = self._parse_kv(cfg["vmr"])
        jac_dict = self._jacobians_dict(cfg["jacobians"])
        if jac_dict:
            self._validate_jacobians(jac_dict, cfg)
            kw["jacobians"] = jac_dict
        if cfg["xsc_dir"].strip():
            kw["xsc_dir"] = cfg["xsc_dir"].strip()
        if cfg["xsc"].strip():
            kw["xsc"] = self._parse_kv_str(cfg["xsc"])

        gtype = cfg["type"]
        geo = cfg["geo"]
        if gtype in ("nadir", "zenith"):
            for key in ("elev", "h_obs"):
                v = self._parse_scalar_or_list(geo[key])
                if v is not None:
                    kw[key] = v
        if gtype == "limb":
            th = self._parse_list(geo["tan_heights"])
            if th is None:
                raise ValueError("limb geometry requires tangent heights")
            kw["tan_heights"] = th
            v = self._parse_scalar_or_list(geo["h_obs"])
            if v is not None:
                kw["h_obs"] = v
        if gtype in ("nadir", "zenith", "limb"):
            for key in ("h_surf", "surf_emiss", "surf_temp"):
                v = self._parse_scalar_or_list(geo[key])
                if v is not None:
                    kw[key] = v
            lv = self._parse_list(geo["levels"])
            if lv is not None:
                kw["levels"] = lv
        if gtype == "layer":
            for key in ("pathlength", "pres", "temp"):
                v = self._parse_scalar_or_list(geo[key])
                if v is not None:
                    kw[key] = v

        return kw

    def _build_rfm(self, cfg: dict) -> RFM:
        return RFM(**self._rfm_kwargs(cfg))

    def _sweep_values(self, sw: dict) -> list[float]:
        """Resolve the sweep specification into a list of numeric values."""
        if sw["mode"] == "range":
            start, stop, n = float(sw["start"]), float(sw["stop"]), int(sw["n"])
            if n < 1:
                raise ValueError("sweep N must be ≥ 1")
            return list(np.linspace(start, stop, n))
        vals = self._parse_list(sw["list"])
        if not vals:
            raise ValueError("sweep value list is empty")
        return vals

    def _build_rfm_list(self, cfg: dict) -> tuple[list[RFM], list[float], str]:
        """Build N RFM instances varying one parameter; return (jobs, values, label)."""
        sw = cfg["sweep"]
        param = sw["param"]
        values = self._sweep_values(sw)
        jobs: list[RFM] = []
        for v in values:
            kw = self._rfm_kwargs(cfg)
            if param.startswith("VMR:"):
                gas = param.split(":", 1)[1]
                adj = dict(kw.get("adjust", {}))
                adj[gas] = v
                kw["adjust"] = adj
            else:
                kw[param] = v
            jobs.append(RFM(**kw))
        return jobs, values, param

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        cfg = self._collect_config()
        if not cfg["out"]:
            messagebox.showwarning("No output", "Select at least one output flag.")
            return
        try:
            if cfg["sweep"]["on"]:
                jobs, values, label = self._build_rfm_list(cfg)
                self.logger.info(
                    f"Batch: sweeping {label} over {len(jobs)} values "
                    f"[{values[0]:g} … {values[-1]:g}]")
                workers = max(1, int(cfg["sweep"]["workers"]))
                self._begin_op("Running batch…", total=len(jobs))
                threading.Thread(
                    target=self._run_batch_worker,
                    args=(jobs, values, label, workers), daemon=False).start()
            else:
                rfm = self._build_rfm(cfg)
                self.logger.info(
                    f"Run: {cfg['type']} {cfg['wn_start']}–{cfg['wn_stop']} cm⁻¹ "
                    f"@ {cfg['nperwn']} pts/cm⁻¹, out={' '.join(cfg['out'])}")
                self._begin_op("Running RFM…")
                threading.Thread(
                    target=self._run_single_worker, args=(rfm,), daemon=False).start()
        except (ValueError, KeyError) as e:
            messagebox.showerror("Invalid Input", str(e))
            self.logger.error(f"Run setup failed: {e}")

    def _run_single_worker(self, rfm: RFM) -> None:
        try:
            rfm.run(silent=True)
            self.rfm = rfm
            self.batch = []
            self.batch_values = []
            self._safe_after(self._on_single_done)
        except Exception as e:
            msg = f"RFM run failed: {e}\n{traceback.format_exc()}"
            # Try to surface the RFM log even on failure.
            drv = getattr(rfm, "drv_path", None)
            self._safe_after(lambda: (self.logger.error(msg),
                                      self._show_rfm_log(drv) if drv else None,
                                      self._end_op("Error")))

    def _run_batch_worker(self, jobs: list[RFM], values: list[float],
                          label: str, workers: int) -> None:
        """GUI-side ProcessPoolExecutor with per-job progress.

        Mirrors objects.runBatch but advances the progress bar per completed
        future. The picklable callable is objects._run_one (module 'objects'),
        so spawn-based workers (macOS, Windows) never re-import this GUI module
        — provided launch() stays under the __main__ guard.
        """
        results: list[RFM | None] = [None] * len(jobs)
        done = 0
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                fut_index = {ex.submit(_run_one, j): i for i, j in enumerate(jobs)}
                for fut in as_completed(fut_index):
                    i = fut_index[fut]
                    results[i] = fut.result()
                    done += 1
                    self._post_progress(done, len(jobs))
            self.batch = [r for r in results if r is not None]
            self.batch_values = list(values)
            self.batch_label = label
            self.rfm = self.batch[0] if self.batch else None
            self._safe_after(self._on_batch_done)
        except Exception as e:
            msg = f"Batch failed: {e}\n{traceback.format_exc()}"
            self._safe_after(lambda: (self.logger.error(msg), self._end_op("Error")))

    def _on_single_done(self) -> None:
        self._end_op("Run complete")
        avail = list(self.rfm._out)
        self.logger.info(f"RFM done. Outputs: {avail}")
        self._show_rfm_log(self.rfm.drv_path)
        self._refresh_jac_params()
        self._refresh_diag()
        self._plot_spectra()
        self.notebook.select(self.spectra_tab)

    def _on_batch_done(self) -> None:
        self._end_op(f"Batch complete ({len(self.batch)} runs)")
        self.logger.info(f"Batch done: {len(self.batch)} runs.")
        if self.rfm is not None:
            self._show_rfm_log(self.rfm.drv_path)
        self._refresh_jac_params()   # sweep-member Jacobian selector
        self._plot_spectra()
        self.notebook.select(self.spectra_tab)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _available_outputs(self) -> list[str]:
        return list(self.rfm._out) if self.rfm is not None else []

    def _overlay_batch(self, ax, key: str, cmap, legend: bool = True) -> None:
        """Overlay one output across the batch, coloured by sweep value."""
        spectra_list = [rfm.getSpectra(key) for rfm in self.batch]
        plotSpectraSweep(
            spectra_list, self.batch_values, label=self.batch_label,
            ax=ax, cmap=cmap, legend_max=12 if legend else 0,
            title=f"{key.upper()} — batch over {self.batch_label}")

    def _plot_spectra(self) -> None:
        if self.rfm is None:
            return
        which = self.spectra_which_var.get()
        avail = self._available_outputs()
        self.spectra_fig.clear()

        if self.batch:
            if not avail:
                self._placeholder(self.spectra_fig, self.spectra_canvas,
                                  "No spectra available.")
                return
            cmap = _resolve_cmap(self.sweep_cmap_var.get())
            # 'grid' → one panel per output; otherwise a single chosen output.
            keys = avail if which == "grid" else (
                [which] if which in avail else [avail[0]])
            ncols = min(len(keys), 2)
            nrows = int(np.ceil(len(keys) / ncols))
            for i, key in enumerate(keys):
                ax = self.spectra_fig.add_subplot(nrows, ncols, i + 1)
                self._overlay_batch(ax, key, cmap, legend=(len(keys) == 1))
            self.spectra_fig.tight_layout()
            self.spectra_canvas.draw()
            return

        if which == "grid":
            keys = avail
            if not keys:
                self._placeholder(self.spectra_fig, self.spectra_canvas,
                                  "No spectra available.")
                return
            ncols = min(len(keys), 2)
            nrows = int(np.ceil(len(keys) / ncols))
            for i, key in enumerate(keys):
                ax = self.spectra_fig.add_subplot(nrows, ncols, i + 1)
                plotSpectra(self.rfm.getSpectra(key), ax=ax,
                            title=key.upper(), wl_axis=False)
        else:
            if which not in avail:
                self._placeholder(self.spectra_fig, self.spectra_canvas,
                                  f"Output {which.upper()} not available.")
                return
            ax = self.spectra_fig.add_subplot(111)
            plotSpectra(self.rfm.getSpectra(which), ax=ax, title=which.upper())
        self.spectra_fig.tight_layout()
        self.spectra_canvas.draw()

    def _preview_atm(self) -> None:
        """Read the configured atmosphere (without running) and plot it."""
        cfg = self._collect_config()
        try:
            if cfg["atm_path"].strip():
                atm = readAtm(cfg["atm_path"].strip())
            else:
                atm = readAtm(_std_atmospheres[cfg["atm_type"]])
        except Exception as e:
            messagebox.showerror("Atmosphere Error", str(e))
            self.logger.error(f"Atmosphere preview failed: {e}")
            return
        self._atm_dict = atm
        self._refresh_atm_fields(atm)
        self._plot_atm()
        self.notebook.select(self.atm_tab)

    def _refresh_atm_fields(self, atm: dict) -> None:
        fields = [k for k in atm if k not in ("HGT", "header")]
        self.atm_field_combo.config(values=fields)
        if self.atm_field_var.get() not in fields:
            self.atm_field_var.set("TEM" if "TEM" in fields else fields[0])

    def _plot_atm(self) -> None:
        atm = getattr(self, "_atm_dict", None)
        if atm is None and self.rfm is not None:
            atm = self.rfm.atm
        if atm is None:
            return
        self._refresh_atm_fields(atm)
        field = self.atm_field_var.get()
        self.atm_fig.clear()
        ax = self.atm_fig.add_subplot(111)
        try:
            plotAtmProfile(atm, field, ax=ax)
        except Exception as e:
            self._placeholder(self.atm_fig, self.atm_canvas, f"Cannot plot: {e}")
            return
        self.atm_fig.tight_layout()
        self.atm_canvas.draw()

    def _jac_rfm(self):
        """The RFM whose Jacobians the tab displays — a sweep member or the run."""
        if self.batch:
            idx = self.jac_member_combo.current()
            if 0 <= idx < len(self.batch):
                return self.batch[idx]
            return self.batch[0]
        return self.rfm

    def _refresh_jac_params(self) -> None:
        # Sweep-member selector: enabled and populated only for a batch run.
        if self.batch:
            members = [f"{self.batch_label}={v:g}" for v in self.batch_values]
            self.jac_member_combo.config(values=members, state="readonly")
            if self.jac_member_var.get() not in members:
                self.jac_member_var.set(members[0] if members else "")
        else:
            self.jac_member_combo.config(values=[], state="disabled")
            self.jac_member_var.set("")

        rfm = self._jac_rfm()
        if rfm is None:
            return
        try:
            params = list(rfm.getJacobians())
        except (RuntimeError, KeyError):
            params = []
        self.jac_param_combo.config(values=params)
        if params and self.jac_param_var.get() not in params:
            self.jac_param_var.set(params[0])
        # Output quantities that have a Jacobian (one per active output flag).
        jac_outs = sorted({r.get("output")
                           for r in getattr(rfm, "_jac_raw", [])})
        self.jac_output_combo.config(values=jac_outs)
        if jac_outs and self.jac_output_var.get() not in jac_outs:
            self.jac_output_var.set("rad" if "rad" in jac_outs else jac_outs[0])
        if params:
            self._plot_jac()

    def _plot_jac(self) -> None:
        rfm = self._jac_rfm()
        if rfm is None:
            return
        param = self.jac_param_var.get()
        if not param:
            return
        out = self.jac_output_var.get() or None
        try:
            jac = rfm.getJacobians(param, output=out)
        except (RuntimeError, KeyError) as e:
            self._placeholder(self.jac_fig, self.jac_canvas, str(e))
            return
        self.jac_fig.clear()
        ax = self.jac_fig.add_subplot(111)
        title = f"Jacobian — {param}" + (f" (∂{out.upper()})" if out else "")
        if self.batch and self.jac_member_var.get():
            title += f"  [{self.jac_member_var.get()}]"
        # A layer Jacobian (≥2 altitude levels) is an altitude×wavenumber map;
        # a column / single-level Jacobian is a spectrum per geometry → lines.
        if jac.label_type == "altitude" and jac.data.shape[0] > 1:
            plotJacobians(jac, ax=ax, title=title)
        else:
            plotSpectra(jac, ax=ax, title=title)
        self.jac_fig.tight_layout()
        self.jac_canvas.draw()

    def _on_apply_filter(self) -> None:
        if self.rfm is None and not self.batch:
            messagebox.showwarning("No run", "Run a model first.")
            return
        sensor = self.conv_sensor_var.get()
        if not sensor:
            messagebox.showwarning("No sensor", "Pick a sensor.")
            return
        which = self.conv_which_var.get()
        self.conv_fig.clear()
        ax = self.conv_fig.add_subplot(111)
        try:
            if self.batch:
                # Sweep: convolve every member; overlay coloured by sweep value.
                filter_list, spectra_list = [], []
                for rfm in self.batch:
                    rfm.applyFilters(sensor=sensor)
                    filter_list.append(rfm.getFilterOut(sensor, which))
                    spectra_list.append(rfm.getSpectra(which))
                plotFilterSweep(
                    filter_list, self.batch_values, label=self.batch_label,
                    spectra_list=spectra_list, ax=ax,
                    cmap=self.sweep_cmap_var.get(),
                    title=f"{sensor} — {which.upper()} over {self.batch_label}")
                n_total = len(self.batch)
            else:
                self.rfm.applyFilters(sensor=sensor)
                fout = self.rfm.getFilterOut(sensor, which)
                spectra = self.rfm.getSpectra(which)
                n_total = spectra.data.shape[0]
                plotSpectra(spectra, ax=ax, colors=light_colors(n_total),
                            title=f"{sensor} — {which.upper()}")
                plotFilterBands(fout, ax, sensor_label=sensor)
        except Exception as e:
            self._placeholder(self.conv_fig, self.conv_canvas, str(e))
            self.logger.error(f"applyFilters failed: {e}")
            return
        self.conv_fig.tight_layout()
        self.conv_canvas.draw()
        self.notebook.select(self.conv_tab)
        self.logger.info(
            f"Applied filter {sensor} to {which.upper()} "
            f"({'sweep ×' if self.batch else ''}{n_total} spectra)")

    def _on_apply_ils(self) -> None:
        if self.rfm is None and not self.batch:
            messagebox.showwarning("No run", "Run a model first.")
            return
        instr = self.conv_instr_var.get()
        if not instr:
            messagebox.showwarning("No instrument", "Pick an instrument.")
            return
        which = self.conv_which_var.get()
        self.conv_fig.clear()
        ax = self.conv_fig.add_subplot(111)
        try:
            if self.batch:
                # Sweep: convolve every member; overlay coloured by sweep value.
                ils_list, raw_list = [], []
                for rfm in self.batch:
                    rfm.applyILS(instrument=instr)
                    ils_list.append(rfm.getILSOut(which))
                    raw_list.append(rfm.getSpectra(which))
                plotILSSweep(
                    ils_list, self.batch_values, label=self.batch_label,
                    raw_list=raw_list, ax=ax, cmap=self.sweep_cmap_var.get(),
                    title=f"{instr} ILS — {which.upper()} over {self.batch_label}")
                n_total = len(self.batch)
            else:
                self.rfm.applyILS(instrument=instr)
                ils_out = self.rfm.getILSOut(which)
                raw = self.rfm.getSpectra(which)
                n_total = ils_out.data.shape[0]
                plotILS(ils_out, raw=raw, ax=ax, instrument=instr,
                        title=f"{instr} ILS — {which.upper()}")
        except Exception as e:
            self._placeholder(self.conv_fig, self.conv_canvas, str(e))
            self.logger.error(f"applyILS failed: {e}")
            return
        self.conv_fig.tight_layout()
        self.conv_canvas.draw()
        self.notebook.select(self.conv_tab)
        self.logger.info(
            f"Applied ILS {instr} to {which.upper()} "
            f"({'sweep ×' if self.batch else ''}{n_total} spectra)")

    # ------------------------------------------------------------------
    # Export spectra
    # ------------------------------------------------------------------
    def _on_export_spectra(self) -> None:
        if self.rfm is None:
            messagebox.showwarning("No run", "Run a model first.")
            return
        avail = self._available_outputs()
        which = self.spectra_which_var.get()
        key = which if which in avail else (avail[0] if avail else None)
        if key is None:
            messagebox.showwarning("Nothing to export", "No spectra available.")
            return
        path = filedialog.asksaveasfilename(
            title=f"Export {key.upper()} spectra",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("NumPy NPZ", "*.npz"),
                       ("RFM spectrum", "*.spc"), ("All files", "*.*")])
        if not path:
            return
        try:
            s = self.rfm.getSpectra(key)
            ext = os.path.splitext(path)[1].lower()
            if ext == ".npz":
                np.savez(path, wn=s.wn, data=s.data, labels=s.labels,
                         label_type=s.label_type, output=s.output, yunit=s.yunit)
            elif ext == ".spc":
                # RFM .spc holds one spectrum; write the first row.
                writeRFMspc(s.wn, s.data[0], path,
                            header=f"RFMViewer {key} export")
            else:  # CSV
                header = "wn," + ",".join(f"{key}_{lbl:g}" for lbl in s.labels)
                block = np.column_stack([s.wn, s.data.T])
                np.savetxt(path, block, delimiter=",", header=header, comments="")
            self.logger.info(f"Exported {key.upper()} spectra → {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            self.logger.error(f"Spectra export failed: {e}")

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------
    def _on_save_config(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save configuration", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._collect_config(), f, indent=2)
            self.logger.info(f"Saved configuration → {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
            self.logger.error(f"Config save failed: {e}")

    def _on_load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load configuration",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            self._apply_config(cfg)
            self.logger.info(f"Loaded configuration ← {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self.logger.error(f"Config load failed: {e}")

    # ------------------------------------------------------------------
    # Paths configuration
    # ------------------------------------------------------------------
    def _check_paths_on_startup(self) -> None:
        """Auto-open the Paths popup if the RFM executable can't be found."""
        if not os.path.exists(get_exec_path()):
            self.logger.warning("RFM executable not found — opening Paths dialog.")
            self._on_configure_paths()

    def _on_configure_paths(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Configure RFM paths")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        outer = ttk.Frame(dlg, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        rfm_dir_var = tk.StringVar(value=get_rfm_dir() or "")
        hitran_var = tk.StringVar(value=get_hitran_path() or "")

        ttk.Label(outer, text="RFM directory:").grid(row=0, column=0, sticky="w")
        ttk.Entry(outer, textvariable=rfm_dir_var, width=44).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(outer, text="Browse…",
                   command=lambda: self._browse_into(
                       rfm_dir_var, directory=True)).grid(row=0, column=2)

        ttk.Label(outer, text="HITRAN .bin:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(outer, textvariable=hitran_var, width=44).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Button(outer, text="Browse…",
                   command=lambda: self._browse_into(
                       hitran_var, directory=False)).grid(row=1, column=2, pady=(6, 0))

        status = ttk.Label(outer, text="", foreground="gray", justify=tk.LEFT)
        status.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        def _validate(*_a) -> None:
            warns = self._validate_rfm_dir(rfm_dir_var.get())
            if not warns:
                status.config(text="✓ RFM installation looks valid.",
                              foreground="green")
            else:
                status.config(text="\n".join(warns), foreground="orange")

        rfm_dir_var.trace_add("write", _validate)
        _validate()

        def _save() -> None:
            write_config(rfm_dir=rfm_dir_var.get().strip() or None,
                         hitran_path=hitran_var.get().strip() or None)
            self.logger.info(f"Wrote RFM paths to {_CONFIG_FILE}")
            dlg.destroy()

        btns = ttk.Frame(outer)
        btns.grid(row=3, column=0, columnspan=3, pady=(12, 0))
        ttk.Button(btns, text="Save", command=_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=4)
        outer.columnconfigure(1, weight=1)

    def _browse_into(self, var: tk.StringVar, directory: bool) -> None:
        if directory:
            path = filedialog.askdirectory(title="Select RFM directory")
        else:
            path = filedialog.askopenfilename(
                title="Select HITRAN binary",
                filetypes=[("HITRAN binary", "*.bin"), ("All files", "*.*")])
        if path:
            var.set(path)

    @staticmethod
    def _validate_rfm_dir(path: str) -> list[str]:
        """Warnings for missing expected sub-paths (mirror __main__._validate_rfm_dir)."""
        warns: list[str] = []
        if not path.strip():
            return ["  Enter an RFM directory."]
        p = Path(os.path.expanduser(path))
        if not p.exists():
            return [f"  Directory does not exist: {path}"]
        if not (p / "source" / "rfm").exists():
            warns.append(f"  RFM executable not found at: {p / 'source' / 'rfm'}")
        if not (p / "atm").exists():
            warns.append(f"  Atmosphere directory not found: {p / 'atm'}")
        hits = list((p / "Hitran").glob("hitran*.bin")) if (p / "Hitran").exists() else []
        if not hits:
            warns.append(f"  No hitran*.bin under: {p / 'Hitran'} (can be set separately)")
        return warns

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        for _h in (getattr(self, "gui_handler", None),
                   getattr(self, "file_handler", None)):
            if _h is not None:
                self.logger.removeHandler(_h)
                try:
                    _h.close()
                except Exception:
                    pass
        plt.close("all")
        self.destroy()
        if self.parent is not None and self.parent.winfo_exists():
            self.parent.destroy()


# ============================================================================
# Entry point
# ============================================================================

def launch() -> None:
    """Launch the RFMViewer GUI."""
    root = tk.Tk()
    root.withdraw()  # hide the root; RFMViewer is a Toplevel
    viewer = RFMViewer(root)
    viewer.protocol("WM_DELETE_WINDOW", viewer._on_close)
    viewer.mainloop()


if __name__ == "__main__":
    launch()
