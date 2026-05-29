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

<!-- Brief description of this project's purpose and goals -->

## Key Concepts and Data Structures

<!-- Domain-specific terminology, core data structures, important algorithms -->

## External Dependencies

<!-- Non-standard dependencies and their purposes in this project -->

## Reference Materials

<!-- For ports: location of original codebase
     Relevant papers, documentation, or specifications -->

## Known Constraints and Quirks

<!-- Hardware limitations, data format peculiarities, legacy compatibility requirements -->

## Project-Specific Conventions

<!-- Any conventions that override or extend the general guidelines above -->
