"""
rfmwrapper — Python wrapper for the Reference Forward Model (RFM).

Dudhia, A (2017): The Reference Forward Model (RFM), JQSRT 186, pp. 243-253.
doi:10.1016/j.jqsrt.2016.06.018
"""

__version__ = "0.2.0"

from .objects import (
    RFM,
    Spectra,
    FilterOut,
    applyFilters,
    ILSOut,
    applyILS,
    runBatch,
    __RFMDIR__,
    __EXECPATH__,
    __HITRANPATH__,
    __ATMPATH__,
    __ARIAPATH__,
    std_atmospheres,
)

from .functions import (
    # ILS kernel and instrument library
    makeILS,
    listInstruments,
    loadILSParams,
    # Atmosphere I/O and manipulation
    readAtm,
    writeAtm,
    adjustAtm,
    insertPlume,
    sliceAtm,
    readStandard,
    # Planck / radiometric
    rad,
    bbt,
    wn2wl,
    wl2wn,
    # Spectral file I/O
    writeRFMspc,
    # Filter convolution
    filterConvolve,
    listSensors,
    # Unit conversions
    ppm2scd,
    scd2ppm,
    ppmm2mcm,
    mcm2ppmm,
    mcm2gm2,
    gm2mcm2,
    ppm2mm3,
    mm32ppm,
    RH2ppm,
    ppm2RH,
    RH2mm3,
    mm32RH,
    RH2gm3,
    gm32RH,
    Psat,
    # Temperature conversions
    c2k, k2c,
    f2c, c2f,
    f2k,
    # Spectral resampling
    resampleSpectra,
    # Plotting
    plotGrid,
)

from . import constants
from .constants import plot_cmaps, plot_colors, pretty_names
from .config import (get_rfm_dir, get_hitran_path,
                     get_exec_path, get_atm_dir, get_aria_dir,
                     write_config, _CONFIG_FILE)

__all__ = [
    'RFM',
    'Spectra',
    'FilterOut',
    'applyFilters',
    'ILSOut',
    'applyILS',
    'makeILS',
    'listInstruments',
    'loadILSParams',
    'runBatch',
    # Path globals
    '__RFMDIR__', '__EXECPATH__', '__HITRANPATH__',
    '__ATMPATH__', '__ARIAPATH__',
    'std_atmospheres',
    # Functions
    'readAtm', 'writeAtm', 'adjustAtm', 'insertPlume', 'sliceAtm', 'readStandard',
    'rad', 'bbt', 'wn2wl', 'wl2wn',
    'writeRFMspc', 'filterConvolve', 'listSensors',
    'ppm2scd', 'scd2ppm', 'ppmm2mcm', 'mcm2ppmm',
    'mcm2gm2', 'gm2mcm2', 'ppm2mm3', 'mm32ppm',
    'RH2ppm', 'ppm2RH', 'RH2mm3', 'mm32RH', 'RH2gm3', 'gm32RH', 'Psat',
    'c2k', 'k2c', 'f2c', 'c2f', 'f2k',
    'resampleSpectra',
    'plotGrid',
    'constants',
    'plot_cmaps', 'plot_colors', 'pretty_names',
    'get_rfm_dir', 'get_hitran_path',
    'get_exec_path', 'get_atm_dir', 'get_aria_dir',
    'write_config', '_CONFIG_FILE',
]
