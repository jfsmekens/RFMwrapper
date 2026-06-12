# CLAUDE.md

## Philosophy

This document defines the working relationship between the developer and Claude. Think of the developer as the conductor of an orchestra — Claude operates autonomously on small, focused, well-defined tasks (instrument sections), while the developer directs the overall composition and flow.

**Cardinal Rule:** Never take shortcuts. When complexity increases, break the problem down methodically. Do not simplify implementations to manage token usage or reduce scope. Shortcuts create technical debt that compounds into larger problems. If a task is too large, propose how to decompose it — do not silently reduce fidelity.

## Code Style and Practices

### Language and Dependencies

- **Primary language:** Python 3.x with NumPy and SciPy as foundational libraries
- **Acceptable dependencies:** scikit-learn, OpenCV, and other well-established scientific Python packages
- **Performance extensions:** When computationally justified, numba, multithreading, or C/C++/FORTRAN bindings are acceptable
- **Baseline assumption:** NumPy broadcasting and vectorization should be the default approach for array operations. Avoid Python loops over array elements unless necessary.

### Documentation

- **Docstrings:** NumPy style, comprehensive
- **Type hints:** Required for all function signatures
- **Inline comments:** Use sparingly to explain *intent*, not mechanics
- **Port projects:** When porting from another language, include inline comments with provenance: original module, function name, and line numbers
- **Strings**: Use f strings when appropriate instead of % place holders. Use double quotes for human readable text, single quotes for internal machine strings (dict keys, etc.)

Example:
```python
def calibrate_spectrum(
    raw_counts: np.ndarray,
    background: np.ndarray,
    responsivity: np.ndarray
) -> np.ndarray:
    """
    Apply radiometric calibration to raw detector counts.

    Parameters
    ----------
    raw_counts : np.ndarray
        Raw detector counts, shape (n_spectra, n_channels).
    background : np.ndarray
        Background signal to subtract, shape (n_channels,).
    responsivity : np.ndarray
        Instrument responsivity curve, shape (n_channels,).

    Returns
    -------
    np.ndarray
        Calibrated radiance values, shape (n_spectra, n_channels).
    """
    # Ref: original_calibration.f90::apply_cal, lines 142-156
    net_signal = raw_counts - background
    return net_signal / responsivity
```

### Naming Conventions

- **Functions and methods:** camelCase for compound verb+noun domain names (e.g. `readAtm`, `plotOut`, `insertPlume`, `runBatch`). Single-word verbs stay lowercase (`run`, `cleanup`). Private helpers use a leading underscore with the same camelCase rule (`_printSummary`, `_readOut`).
- **Classes:** PascalCase (`RFM`, `Spectra`).
- **Constants and module-level paths:** `ALL_CAPS` or `__DUNDER__` for module globals.
- **Unit-conversion shortcuts:** `unit2unit` pattern, lowercase throughout (`ppm2scd`, `c2k`, `RH2ppm`). Acronyms preserve their capitalisation (`RH`, `RFM`).

### Error Handling

- Use explicit exception handling with informative error messages
- Fail fast with clear diagnostics
- Error messages should help the developer locate and understand the problem

### Logging

- Use Python's `logging` module when appropriate
- Log messages must be focused and informative, not verbose
- No celebratory language or decorative emojis in logs
- Acceptable indicators: functional symbols (✓, ✗) for success/failure status

## Development Workflow

### Validation Before Tests

Ensure implementation matches intent before writing unit tests. The workflow is:
1. Implement the feature
2. Validate behavior against expectations (interactive testing, comparison with reference implementations)
3. Write unit tests to codify the validated behavior

### Git Operations

Claude may propose opportune times for commits and draft commit messages, but the developer controls all Git operations (add, commit, push, pull). Do not execute Git commands autonomously.

### Proactive Review

- Flag potential issues proactively: edge cases, performance concerns, maintainability
- If an approach appears problematic, ask for clarification rather than silently proceeding. The developer usually has rationale to provide.
- Prioritize readability and maintainability over cleverness
- Avoid excessive abstraction; use clear, descriptive naming

## Project Organization

### Directory Structure

- **Modules:** Organize to clearly separate concerns; structure evolves iteratively per project
- **Debug/test outputs:** Write to `test/` or `debug/` directories, never to project root
- **Intermediate/temporary files:** Use `/tmp` for artifacts that don't need to persist
- **Minimize bloat:** Keep the repository clean. Debugging artifacts should be structured for easy cleanup.

### Packaging

When the project reaches a testable state, structure it as an editable pip package:
```bash
pip install -e .
```

This requires a proper `pyproject.toml` or `setup.py` at the project root.

## What to Avoid

- **Shortcuts due to complexity or token limits** — decompose instead
- **Verbose or celebratory log messages**
- **Writing debug outputs to project root**
- **Autonomous Git operations**
- **Excessive abstraction or poor naming**
- **Python loops where NumPy broadcasting applies**

---

# Project-Specific Configuration

## Project Description

Python wrapper for the **RFM (Reference Forward Model)** Fortran radiative transfer code (Dudhia 2017, JQSRT 186, pp. 243–253). RFMwrapper handles driver-file generation, subprocess execution, and output parsing for RFM v5.21+. It supports nadir, zenith, layer, and limb geometry; Jacobians; binary I/O; cross-section (XSC) species including aerosols; spectral convolution with instrument line shape (ILS) functions and satellite sensor filter response functions; and batch parallelism via `ProcessPoolExecutor`.

## Key Concepts and Data Structures

- **`RFM` class** (`objects.py`) — central object: instantiate with geometry/atmosphere/output params → `run()` → `getSpectra()` → optionally `applyILS()` / `applyFilters()`.
- **`Spectra` dataclass** — `(wn, data, labels, label_type, output, xunit, yunit)`; `data` is always `(n_spectra, n_wn)`.
- **`FilterOut` / `ILSOut` dataclasses** — outputs of `applyFilters()` / `applyILS()`.
- **Geometry types:** `'nadir'`, `'zenith'`, `'layer'` (HOM), `'limb'` (requires `tan_heights`).
- **Spectral outputs:** `out='RAD BBT OPT TRA'`; Jacobians via `jacobians={'TEM': None, 'H2O': None}`.
- **Atmosphere:** standard profiles from `rfmwrapper/standards.txt`; `adjustAtm()` / `insertPlume()` / `sliceAtm()` for modification.
- Deep detail in `memory/project_overview.md`.

## External Dependencies

- **RFM binary** — Fortran executable at `$RFM_DIR/bin/rfm` (configured via env var or `rfmwrapper configure`).
- **HITRAN** — line-by-line database binary at `$RFM_DIR/hitran/`.
- **cmcrameri** — perceptually uniform scientific colormaps (`pip install cmcrameri`); used throughout `plotting.py`.
- **pyOptimalEstimation** — optimal estimation retrieval (`pip install pyOptimalEstimation`); used in `demo_retrieval.py`.
- **Standard scientific stack** — numpy, scipy, matplotlib, pandas.

## Reference Materials

- Dudhia, A. (2017). The Reference Forward Model (RFM). *JQSRT*, 186, 243–253.
- RFM v5.21 user guide: `$RFM_DIR/doc/` on the local machine.
- `docs/aerosol_xsc_interface.md` — derivation of aerosol XSC unit conventions and multi-species workaround.
- RTTOV SRF library: downloaded to `rfmwrapper/filters/`; fetch script at `tools/download_rttov_srfs.py`.

## Known Constraints and Quirks

- **XSC section format:** bare file paths only — `FILXSC = ...` keyword syntax is rejected by RFM (`F-SCNXSC` error).
- **XSC comment lines:** XSC files must contain no `!` comment lines — RFM aborts with `F-SCNXSC IOSTAT=5010`.
- **Aerosol species limit:** max 2 simultaneous user-defined XSC species per run; a 3rd causes a silent crash. `writeDrv()` raises `ValueError` as a guard.
- **Aerosol units:** species named `aerosol` (ID 100) are treated as km⁻¹ direct; all other XSC names go through the cm²/molec + ideal gas law path — see `docs/aerosol_xsc_interface.md`.
- **Jacobian filenames:** RFM writes `{out}{elev}_{param}{level_m}.out` — no `jac_` prefix. N altitude levels → N−2 output files (boundary levels are interpolation anchors only).
- **LEV output filenames:** RFM writes `{out}{elev}_{alt_m}{up|dn}.out` for `*LEV` section outputs. The `up`/`dn` suffix distinguishes LEV files from Jacobian files (which end in a parameter name like `tem05000`). `_readOut` uses `_decode_lev_suffix()` to tell them apart.
- **Binary output:** little-endian Fortran sequential unformatted; 4-byte record markers; float32 by default, float64 with `dbl=True`.
- **macOS multiprocessing:** `runBatch()` uses `ProcessPoolExecutor`; calling scripts **must** have `if __name__ == '__main__':` guard (macOS uses spawn, not fork).
- **`surf_emiss < 1` (nadir):** TRA and OPT correctly output the one-way atmospheric transmittance independent of surface emissivity (confirmed v5.21). Note: RFM v5.10 had an undocumented defect where TRA/OPT were contaminated by `(1−ε)·T_atm²` when `surf_emiss < 1`; this is fixed in v5.21.

## Project-Specific Conventions

- Naming follows the rules in the **Naming Conventions** section above; see also `memory/naming_conventions.md`.
- Filter CSVs: mandatory `wn` [cm⁻¹] and `T` columns; optional `wl` [µm]; sorted ascending by `wn`.
- All plotting functions live in `rfmwrapper/plotting.py` and return `(fig, ax[es])`; no `plt.show()` calls inside library code.
- Test outputs go to `test/demo_output/<demo_name>/`; intermediate/temp files go to `/tmp`.

## Session Memory

A persistent memory index lives at `memory/MEMORY.md` in this project root.
Read it at the start of each session to recover context from prior work.
The index points to individual files covering completed work, open TODOs,
naming conventions, conda environment, and project architecture.
