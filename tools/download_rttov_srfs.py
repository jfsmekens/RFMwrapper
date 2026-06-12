#!/usr/bin/env python3
"""
download_rttov_srfs.py — Fetch RTTOV NWP-SAF spectral response functions
and convert to rfmwrapper dual-axis (wn, wl, T) CSV format.

Run from any directory:
    python tools/download_rttov_srfs.py [--force]

  --force   Re-download and overwrite existing CSVs (default: skip existing).

Each CSV file has:
  - A comment header line with the RTTOV band label
  - Columns: wn [cm-1], wl [um], T [0-1], sorted ascending by wn

Output structure under rfmwrapper/filters/:
  Multi-platform instruments:  <Instrument>/<Platform>/ch01.csv, ch02.csv ...
  Single-platform instruments: <Instrument>/ch01.csv, ch02.csv ...
"""

import io
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
FILTERS_DIR = _SCRIPT_DIR.parent / 'rfmwrapper' / 'filters'
BASE_URL = "https://nwp-saf.eumetsat.int/downloads/rtcoef_info/visir_srf"

# ---------------------------------------------------------------------------
# Manifest: (instrument_dir, platform_dir_or_None, rttov_slug, n_channels)
# platform_dir=None → files go directly into instrument_dir (single-platform)
# ---------------------------------------------------------------------------
MANIFEST = [
    # ---- GOES ABI (multi-platform) ----------------------------------------
    ('GOES-ABI',    'GOES-16',   'rtcoef_goes_16_abi',   16),
    ('GOES-ABI',    'GOES-17',   'rtcoef_goes_17_abi',   16),
    ('GOES-ABI',    'GOES-18',   'rtcoef_goes_18_abi',   16),
    ('GOES-ABI',    'GOES-19',   'rtcoef_goes_19_abi',   16),
    # ---- GOES Imager (multi-platform, 4–5 ch depending on satellite) -------
    ('GOES-Imager', 'GOES-8',    'rtcoef_goes_8_imager',  5),
    ('GOES-Imager', 'GOES-9',    'rtcoef_goes_9_imager',  5),
    ('GOES-Imager', 'GOES-10',   'rtcoef_goes_10_imager', 5),
    ('GOES-Imager', 'GOES-11',   'rtcoef_goes_11_imager', 5),
    ('GOES-Imager', 'GOES-12',   'rtcoef_goes_12_imager', 5),
    ('GOES-Imager', 'GOES-13',   'rtcoef_goes_13_imager', 5),
    ('GOES-Imager', 'GOES-14',   'rtcoef_goes_14_imager', 5),
    ('GOES-Imager', 'GOES-15',   'rtcoef_goes_15_imager', 5),
    # ---- GOES Sounder (multi-platform) -------------------------------------
    ('GOES-Sounder','GOES-8',    'rtcoef_goes_8_sounder', 18),
    ('GOES-Sounder','GOES-9',    'rtcoef_goes_9_sounder', 18),
    ('GOES-Sounder','GOES-10',   'rtcoef_goes_10_sounder',18),
    ('GOES-Sounder','GOES-11',   'rtcoef_goes_11_sounder',18),
    ('GOES-Sounder','GOES-12',   'rtcoef_goes_12_sounder',18),
    ('GOES-Sounder','GOES-13',   'rtcoef_goes_13_sounder',18),
    ('GOES-Sounder','GOES-14',   'rtcoef_goes_14_sounder',18),
    ('GOES-Sounder','GOES-15',   'rtcoef_goes_15_sounder',18),
    # ---- HIMAWARI AHI (multi-platform) -------------------------------------
    ('HIMAWARI-AHI','Himawari-8','rtcoef_himawari_8_ahi', 16),
    ('HIMAWARI-AHI','Himawari-9','rtcoef_himawari_9_ahi', 16),
    # ---- AVHRR (multi-platform) --------------------------------------------
    ('AVHRR',       'METOP-1',   'rtcoef_metop_1_avhrr',  6),
    ('AVHRR',       'METOP-2',   'rtcoef_metop_2_avhrr',  6),
    ('AVHRR',       'METOP-3',   'rtcoef_metop_3_avhrr',  6),
    ('AVHRR',       'NOAA-15',   'rtcoef_noaa_15_avhrr',  6),
    ('AVHRR',       'NOAA-16',   'rtcoef_noaa_16_avhrr',  6),
    ('AVHRR',       'NOAA-17',   'rtcoef_noaa_17_avhrr',  6),
    ('AVHRR',       'NOAA-18',   'rtcoef_noaa_18_avhrr',  6),
    ('AVHRR',       'NOAA-19',   'rtcoef_noaa_19_avhrr',  6),
    # ---- ECOSTRESS (single-platform) ---------------------------------------
    ('ECOSTRESS',   None,        'rtcoef_iss_1_ecostres',  5),
    # ---- MTG FCI (single-platform) -----------------------------------------
    ('MTG-FCI',     None,        'rtcoef_mtg_1_fci',      16),
    # ---- MODIS (multi-platform: Terra + Aqua) ------------------------------
    ('MODIS',       'Terra',     'rtcoef_eos_1_modis',    36),
    ('MODIS',       'Aqua',      'rtcoef_eos_2_modis',    36),
    # ---- MSG SEVIRI (multi-platform) ---------------------------------------
    ('MSG-SEVIRI',  'MSG-1',     'rtcoef_msg_1_seviri',   12),
    ('MSG-SEVIRI',  'MSG-2',     'rtcoef_msg_2_seviri',   12),
    ('MSG-SEVIRI',  'MSG-3',     'rtcoef_msg_3_seviri',   12),
    ('MSG-SEVIRI',  'MSG-4',     'rtcoef_msg_4_seviri',   12),
    # ---- Landsat TIRS (multi-platform) -------------------------------------
    ('Landsat-TIRS','Landsat-8', 'rtcoef_landsat_8_tirs',  2),
    ('Landsat-TIRS','Landsat-9', 'rtcoef_landsat_9_tirs',  2),
    # ---- Landsat TM / ETM+ (multi-platform) --------------------------------
    ('Landsat-TM',  'Landsat-4', 'rtcoef_landsat_4_tm',   6),
    ('Landsat-TM',  'Landsat-5', 'rtcoef_landsat_5_tm',   6),
    ('Landsat-TM',  'Landsat-7', 'rtcoef_landsat_7_tm',   6),
    # ---- Sentinel-3 SLSTR (multi-platform) ---------------------------------
    ('Sentinel3-SLSTR','Sentinel-3A','rtcoef_sentinel3_1_slstr', 9),
    ('Sentinel3-SLSTR','Sentinel-3B','rtcoef_sentinel3_2_slstr', 9),
    ('Sentinel3-SLSTR','Sentinel-3C','rtcoef_sentinel3_3_slstr', 9),
    ('Sentinel3-SLSTR','Sentinel-3D','rtcoef_sentinel3_4_slstr', 9),
    # ---- VIIRS (multi-platform) --------------------------------------------
    ('VIIRS',       'SNPP',      'rtcoef_jpss_0_viirs',   22),
    ('VIIRS',       'NOAA-20',   'rtcoef_noaa_20_viirs',  22),
    ('VIIRS',       'NOAA-21',   'rtcoef_noaa_21_viirs',  22),
]


# ---------------------------------------------------------------------------
# RTTOV SRF parser
# ---------------------------------------------------------------------------

def _parse_rttov_txt(raw_bytes: bytes) -> dict:
    """Parse raw bytes of a RTTOV SRF .txt file into wn/wl/T arrays."""
    lines = raw_bytes.decode('utf-8', errors='replace').splitlines()
    # Line 0: "<ch_num>, <label> ; <source>"
    # Line 1: "Number of data points:"
    # Line 2: "<npts>"
    # Line 3: "Wavenumber (cm-1)   Filter response"
    # Lines 4+: data
    line0 = lines[0] if lines else ''
    after_comma = line0.split(',', 1)[1] if ',' in line0 else line0
    label = after_comma.split(';')[0].strip()

    data_lines = lines[4:]
    data = np.loadtxt(io.StringIO('\n'.join(data_lines)))
    if data.ndim == 1:
        data = data[np.newaxis, :]

    wn = data[:, 0]
    T  = data[:, 1]
    if len(wn) > 1 and wn[0] > wn[-1]:
        wn = wn[::-1]
        T  = T[::-1]

    return {'label': label, 'wn': wn, 'wl': 1e4 / wn, 'T': T}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, retries: int = 3, delay: float = 1.0) -> bytes | None:
    """Fetch URL bytes; return None on HTTP 404, raise on other errors."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise
    return None


def _write_csv(path: Path, label: str, wn: np.ndarray,
               wl: np.ndarray, T: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"# label: {label}\n")
        f.write("wn,wl,T\n")
        for w, l, t in zip(wn, wl, T):
            f.write(f"{w:.6f},{l:.6f},{t:.6f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(force: bool = False) -> None:
    total_written = 0
    total_skipped = 0
    total_missing = 0

    for inst_dir, platform_dir, slug, max_ch in MANIFEST:
        if platform_dir:
            out_dir = FILTERS_DIR / inst_dir / platform_dir
            tag = f"{inst_dir}/{platform_dir}"
        else:
            out_dir = FILTERS_DIR / inst_dir
            tag = inst_dir

        ch_written = 0
        ch_skipped = 0

        for ch in range(1, max_ch + 1):
            ch_str = f"ch{ch:02d}"
            csv_path = out_dir / f"{ch_str}.csv"

            if csv_path.exists() and not force:
                ch_skipped += 1
                continue

            url = f"{BASE_URL}/{slug}_srf/{slug}_srf_{ch_str}.txt"
            raw = _fetch(url)

            if raw is None:
                # Channel does not exist for this platform — normal for instruments
                # where channel count varies across platforms (e.g. GOES Imager).
                if ch == 1:
                    print(f"  ✗ {tag}: ch01 not found — slug may be wrong, skipping")
                    total_missing += max_ch
                    break
                break   # Stop at first missing channel

            srf = _parse_rttov_txt(raw)
            _write_csv(csv_path, srf['label'], srf['wn'], srf['wl'], srf['T'])
            ch_written += 1

        if ch_written > 0:
            print(f"  ✓ {tag}: wrote {ch_written} channels")
        elif ch_skipped > 0:
            print(f"  — {tag}: {ch_skipped} channels already present")

        total_written += ch_written
        total_skipped += ch_skipped

    print(f"\nDone. Written: {total_written}  Skipped: {total_skipped}  "
          f"Missing slugs: {total_missing}")


if __name__ == '__main__':
    force = '--force' in sys.argv
    main(force=force)
