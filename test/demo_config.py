"""
demo_config.py — Shared utilities for rfmwrapper demo scripts.

Provides output path management, a timing context manager, and the
canonical figure style.  Import this module at the top of every demo.

Usage
-----
    from demo_config import get_output_path, timer, apply_style
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt

# Central output directory for all demo figures
OUTPUT_BASE = Path(__file__).parent / 'demo_output'
OUTPUT_BASE.mkdir(exist_ok=True)


def get_output_path(demo_name: str, filename: str) -> Path:
    """
    Return the save path for a demo figure, creating subdirs as needed.

    Parameters
    ----------
    demo_name : str
        Demo identifier (e.g. ``'nadir'``, ``'limb'``).
    filename : str
        Filename including extension (e.g. ``'spectra.png'``).

    Returns
    -------
    Path
    """
    out_dir = OUTPUT_BASE / demo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


@contextmanager
def timer(label: str):
    """Print elapsed time for a block of code."""
    t0 = perf_counter()
    yield
    print(f"  {label}: {perf_counter() - t0:.1f} s")


def apply_style() -> None:
    """Apply the standard rfmwrapper demo figure style."""
    from rfmwrapper.plotting import STYLE
    plt.rcParams.update(STYLE)
