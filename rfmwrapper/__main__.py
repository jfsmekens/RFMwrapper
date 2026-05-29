"""
python -m rfmwrapper configure
python -m rfmwrapper show

Interactive setup for rfmwrapper path configuration.
Writes ~/.config/rfmwrapper/config.ini.
"""
import os
import sys
from pathlib import Path
from glob import glob

from .config import (
    _CONFIG_FILE,
    get_rfm_dir, get_hitran_path,
    get_exec_path, get_atm_dir, get_aria_dir,
    write_config,
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_rfm_dir(path: str) -> list[str]:
    """Return warning strings for missing expected sub-paths."""
    warnings = []
    p = Path(path)
    if not p.exists():
        warnings.append(f"  Directory does not exist: {path}")
        return warnings
    if not (p / 'source' / 'rfm').exists():
        warnings.append(f"  RFM executable not found at: {p / 'source' / 'rfm'}")
    if not (p / 'atm').exists():
        warnings.append(f"  Atmosphere directory not found: {p / 'atm'}")
    hits = list((p / 'Hitran').glob('hitran*.bin')) if (p / 'Hitran').exists() else []
    if not hits:
        warnings.append(f"  No hitran*.bin found under: {p / 'Hitran'} (can be set separately)")
    return warnings


def _validate_hitran(path: str) -> list[str]:
    warnings = []
    if not os.path.isfile(path):
        warnings.append(f"  File not found: {path}")
    elif not path.endswith('.bin'):
        warnings.append(f"  Unexpected extension — expected a .bin file: {path}")
    return warnings


def _prompt(label: str, current: str | None, optional: bool = False) -> str | None:
    """
    Prompt the user for a path value.

    Returns the entered value, the current default if the user hits enter,
    or None if the user enters 'skip' (for optional paths).
    """
    suffix = " [skip to leave unset]" if optional and current is None else f" [{current}]"
    response = input(f"{label}{suffix}: ").strip()
    if not response:
        return current
    if optional and response.lower() == 'skip':
        return None
    return os.path.expanduser(response)


def _warn_and_confirm(warnings: list[str], label: str) -> bool:
    """Print warnings and ask the user to confirm or abort. Returns True to proceed."""
    print(f"\n  Warnings for {label}:")
    for w in warnings:
        print(w)
    confirm = input("  Save anyway? [y/N]: ").strip().lower()
    return confirm == 'y'


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def configure() -> None:
    print("\nrfmwrapper — path configuration")
    print("=" * 45)
    print(f"Config file: {_CONFIG_FILE}")
    print()
    print("RFM installation instructions:")
    print("  https://eodg.atm.ox.ac.uk/RFM/sum/rfm_installation.html")
    print()
    print("Press Enter to keep the current value.  Enter 'skip' to leave an")
    print("optional path unset (it will be derived from rfm_dir instead).\n")

    # --- rfm_dir ---
    new_rfm_dir = _prompt("RFM installation directory (rfm_dir)", get_rfm_dir())
    if new_rfm_dir:
        warnings = _validate_rfm_dir(new_rfm_dir)
        if warnings and not _warn_and_confirm(warnings, 'rfm_dir'):
            print("Aborted. Configuration not saved.")
            return
    print()

    # --- hitran_path ---
    print("HITRAN binary path (hitran_path) — optional if hitran*.bin is under rfm_dir/Hitran/")
    current_hitran = get_hitran_path()
    new_hitran = _prompt("  HITRAN .bin file", current_hitran, optional=True)
    if new_hitran and new_hitran != current_hitran:
        warnings = _validate_hitran(new_hitran)
        if warnings and not _warn_and_confirm(warnings, 'hitran_path'):
            print("Aborted. Configuration not saved.")
            return
    print()

    write_config(
        rfm_dir=new_rfm_dir,
        hitran_path=new_hitran if new_hitran != get_hitran_path() else None,
    )

    print("Configuration saved.")
    print(f"  rfm_dir     = {new_rfm_dir}")
    if new_hitran:
        print(f"  hitran_path = {new_hitran}")
    print(f"\nConfig file: {_CONFIG_FILE}\n")


def show() -> None:
    def _source(env_key: str, cfg_key: str, derived: str) -> str:
        if os.environ.get(env_key):
            return f"env var ${env_key}"
        import configparser
        cfg = configparser.ConfigParser()
        if _CONFIG_FILE.exists():
            cfg.read(_CONFIG_FILE)
        if cfg.get('paths', cfg_key, fallback=None):
            return f"config file"
        return f"derived from rfm_dir"

    rfm_dir = get_rfm_dir()
    hitran  = get_hitran_path()

    print("\nrfmwrapper — current configuration")
    print("=" * 45)
    print(f"Config file : {_CONFIG_FILE} ({'exists' if _CONFIG_FILE.exists() else 'not found'})\n")
    print(f"{'Path':<16} {'Value':<45} {'Source'}")
    print("-" * 80)
    print(f"{'rfm_dir':<16} {rfm_dir:<45} {_source('RFM_DIR',    'rfm_dir',    '~/RFM')}")
    print(f"{'exec':<16} {get_exec_path():<45} derived from rfm_dir")
    print(f"{'atm_dir':<16} {get_atm_dir():<45} derived from rfm_dir")
    print(f"{'aria_dir':<16} {get_aria_dir():<45} derived from rfm_dir")
    print(f"{'hitran_path':<16} {str(hitran):<45} {_source('HITRAN_PATH', 'hitran_path', 'rfm_dir/Hitran/')}")
    print(f"{'filters':<16} {'bundled in package':<45} rfmwrapper/filters/")
    print()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'configure'
    if cmd == 'configure':
        configure()
    elif cmd == 'show':
        show()
    else:
        print(f"Unknown command: {cmd!r}")
        print("Usage: python -m rfmwrapper [configure|show]")
        sys.exit(1)
