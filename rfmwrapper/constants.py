"""
Written by @jfsmekens - Last updated 04/14/2025

This module contains a series of constants associated with gas species, as well as
    lists of colors and descriptors used to format plots

"""
from matplotlib import colormaps as cm
import cmcrameri.cm as cmc


# Known extensions for MIDAC files
midac_extensions = ["spc", "sbm", "ifg", "sb", "abs"]

# Invalid Extensions: do not include those when searching for spectra in a directory
invalid_extensions = ["pkl", "csv", "txt", "ini", "dat", "xls", "xlsx", "doc", "docx", "jpg", "png"]

# Gas species in RFM + those extracted from NIST
rfm_gases = ["H2O", "CO2", "SO2", "CO", "NO", "O3", "N2O", "NO2", "NH3", "HNO3", "CH4", "O2", "HCl", "HF", "HBr",
             "HI", "H2S", "N2", "ClO", "SF6", "SO3", "OCS", "OH"]
special_gases = ["SiF4"]
possible_gases = rfm_gases + special_gases

# RFM atm fields
atm_fields = {"HGT": "Height", "PRE": "Pressure", "TEM": "Temperature"}
atm_units = {"HGT": "[km]", "PRE": "[mb]", "TEM": "[K]"}

# Known aerosol/particulate species for use with the XSC mechanism.
# These are treated differently from gas-phase molecules: the ATM profile
# value is a dimensionless scaling factor, not a VMR [ppmv].
#
# 'AEROSOL' is RFM molecule ID 100 — the only slot with native km⁻¹ treatment.
# All others require the synthetic multi-tabulation pre-scaling approach
# (see docs/aerosol_xsc_interface.md).  Add new species here as needed.
AEROSOL_SPECIES = {
    'AEROSOL',   # RFM molecule 100 — native km⁻¹ slot
    'ASH',       # volcanic ash
    'H2SO4',     # sulphuric acid aerosol
    'HNO3',      # nitric acid aerosol (distinct from gas-phase HNO3)
    'WATER',     # liquid water droplets
    'ICE',       # water ice
    'SOOT',      # black carbon / soot
}

# Keep old name as alias for backward compatibility
possible_aeros = sorted(AEROSOL_SPECIES)

# Interesting Ratios: If any combination of 2 targets makes one of those ratios, it will be plotted
interesting_ratios = ["CO2:SO2", "H2O:SO2", "H2O:CO2", "CO:SO2", "SO2:HCl", "SO2:HF", "SO2:H2S", "SO2:SO3",
                      "SO2:SiF4", "HCl:HF", "SO2:H2SO4", "SO4:SO2", "H2SO4:ASH", "H2O:WATER", "N/A:N/A"]

# Continuum Gases: gases in this list are treated differently when creating a reference
#       (include gases with continuum and/or line mixing issues)
continuum_gases = ["H2O"]

# Available Apodization functions for the ILS
apod_types = ["Boxcar", "Uniform", "Triangular", "Blackman-Harris", "Happ-Genzel", "Hamming", "Lorenz", "Gaussian",
                  "NB weak", "NB medium", "NB strong", "Cosine"]

# Molecular weights of all gases [g/mol]
# Atomic weights from IUPAC 2021 standard values.
weights = {"H2O":   18.01528,
           "CO2":   44.009,
           "O3":    47.997,
           "N2O":   44.013,
           "CH4":   16.043,
           "O2":    31.998,
           "SO2":   64.064,
           "SO3":   80.063,
           "HCl":   36.461,
           "HF":    20.006,
           "CO":    28.010,
           "OCS":   60.075,
           "NH3":   17.031,
           "NO":    30.006,
           "NO2":   46.005,
           "HNO3":  63.013,
           "HBr":   80.912,
           "HI":   127.912,
           "N2":    28.014,
           "ClO":   51.452,
           "SF6":  146.055,
           "OH":    17.007,
           "SiF4": 104.079,
           "H2S":   34.081,
           "H2SO4": 98.079,
           "ASH":   65.0,       # bulk proxy, not a molecular weight
           "WATER": 18.01528,   # H2O aerosol
           "ICE":   18.01528,   # H2O ice aerosol
           }

# Fallback colour and colourmap for species not in the lookup dicts.
_FALLBACK_COLOR: str = '#555555'
_FALLBACK_CMAP = cmc.grayC_r

# Plot Colors: A dedicated color for each target species
plot_colors = {"H2O_sc": "lightblue",
               "H2O": "#2166ac",       # deep ocean blue
               "OH": "tab:blue",
               "CO2": "#e07800",       # amber orange
               "O3": "#762a83",        # purple
               "O2": "tab:red",
               "N2O": "#b5810a",       # golden brown
               "N2": "darkgreen",
               "NO": "brown",
               "NO2": "brown",
               "CH4": "#4a148c",       # deep violet
               "SO2": "#d4001a",       # strong red
               "SO3": "tab:orange",
               "CO": "#7b1f1f",        # dark brown-red
               "OCS": "goldenrod",
               "HCl": "#006064",       # dark teal
               "HF": "#ad1457",        # deep rose
               "NH3": "#558b2f",       # forest green
               "HNO3": "goldenrod",
               "SiF4": "purple",
               "H2S": "#3e4906",       # dark olive
               "H2SO4": "tab:green",
               "ASH": "tab:grey",
               "WATER": "deepskyblue",
               "source_temp": "r",
               "gas_temp": "orange",
               "gasE_temp": "darkorange",
               "E_frac": "k",
               "fov": "k",
               "nu_shift": "k",
               "max_opd": "k",
               "R2": "k",
               "RMSE": "k",
               "N/A": "w",
               "meas": "k",
               "model": "r",
               "bkg": "r",
               "res": "k",
               "scat_all": "tab:grey",
               "scat_scroll": "tab:red",
               "scat_last": "b",
               "regress": "k",
               "intercept": "darkcyan",
               "confidence": "k",
               "pres": "k",
               "temp": "k"}
dark_colors = {"H2O_sc": "lightblue",
               "H2O": "tab:blue",
               "CO2": "tab:orange",
               "O3": "tab:pink",
               "N2O": "gold",
               "CH4": "darkviolet",
               "SO2": "tab:red",
               "CO": "maroon",
               "OCS": "goldenrod",
               "HCl": "turquoise",
               "HF": "magenta",
               "NH3": "yellow",
               "SiF4": "purple",
               "H2S": "olivedrab",
               "H2SO4": "tab:green",
               "ASH": "tab:grey",
               "WATER": "deepskyblue",
               "source_temp": "r",
               "gas_temp": "orange",
               "gasE_temp": "darkorange",
               "fov": "w",
               "nu_shift": "w",
               "max_opd": "w",
               "R2": "w",
               "RMSE": "w",
               "N/A": "k",
               "meas": "w",
               "model": "r",
               "bkg": "r",
               "res": "w",
               "scat_all": "w",
               "scat_scroll": "tab:pink",
               "scat_last": "dodgerblue",
               "regress": "w",
               "intercept": "tab:cyan",
               "confidence": "w"}

# Plot Color Maps: A dedicated sequential colourmap for each species
plot_cmaps = {
    # Primary gas species — all cmcrameri _r variants
    'H2O':      cmc.nuuk_r,
    'CO2':      cmc.bilbao_r,
    'O3':       cmc.acton_r,
    'SO2':      cmc.lajolla_r,
    'CH4':      cmc.devon_r,
    'CO':       cmc.tokyo_r,
    'HCl':      cmc.davos_r,
    'HF':       cmc.lipari_r,
    'N2O':      cmc.hawaii_r,
    'NH3':      cmc.bamako_r,
    'H2S':      cmc.turku_r,
    # Aerosol / non-gas species
    'H2SO4':    cm['Greens'],
    'ASH':      cm['Greys'],
    'WATER':    cmc.nuuk_r,
    'SiF4':     cm['Purples'],
    # Diagnostic variables
    'R2':       cm['Greys'],
    'RMSE':     cm['Greys'],
    'dt_prox':  cm['Oranges'],
    'dt_plume': cm['Reds'],
    'temp':     cmc.hawaii_r,
    'pres':     cmc.bamako_r,
}

# Pretty name: Mathtext for each species to use in labels
pretty_names = {"H2O": "$H_2O$",
                "OH": "$OH$",
                "SO2": "$SO_2$",
                "SO3": "$SO_3$",
                "CO2": "$CO_2$",
                "NH3": "$NH_3$",
                "HNO3": "$HNO_3$", 
                "O2": "$O_2$",
                "O3": "$O_3$",
                "N2": "$N_2$", 
                "N2O": "$N_2O$",
                "NO": "$NO$", 
                "NO2": "$NO_2$",
                "CH4": "$CH_4$",
                "CO": "$CO$",
                "OCS": "$OCS$",
                "HCl": "$HCl$",
                "CLO": "$ClO$",
                "HF": "$HF$",
                "SiF4": "$SiF_4$",
                "H2S": "$H_2S$",
                "SO4": "${SO_4}^{2-}$",
                "H2SO4": "${SO_4}^{2-}$ $aerosol$",
                "ASH": "$ash$",
                "WATER": "$H_2O$ $aerosol$",
                "ICE": "$ICE$",
                "time": "Local Time",
                "R2": "$R^2$",
                "RMSE": "RMS error",
                "source_temp": "$T_{source}$",
                "gas_temp": "$T_{gas}$",
                "gasE_temp": "$T_{gas}(E)$",
                "E_frac": r"$X{\epsilon}$",
                "fov": "$FOV$",
                "max_opd": "$OPD_{max}$",
                "nu_shift": r"$\nu_{shift}$",
                "N/A": "$n/a$"}

# Pretty Ratios: Use the math text in pretty_names to create pretty ratios for labels
pretty_ratios = {}
for ratio in interesting_ratios:
    yname, xname = ratio.split(":")
    pretty_ratios[ratio] = ":".join([pretty_names[yname], pretty_names[xname]])

# Pretty Units: Mathtext with the units of SCD in molar or mass
pretty_labels = {"molar": r" SCD [$molec \cdot cm^{-2}$]",
                 "mass": r" SCD [$g \cdot m^{-2}$]"}

# Logo: The plumeIR_dev logo, version and fonts to display them
logotext = "plumeIR"
logofont = {"family": "Palatino",
            "color":  "tab:red",
            "weight": "bold",
            "style": "italic",
            "size": 18}
versiontext = "v0.8"
versionfont = {"family": "Palatino",
               "color":  "tab:red",
               "weight": "normal",
               "style": "italic",
               "size": 10}

# Primary windows
primary_windows = {"SO2": [2400, 2550],
                   "HCl": [2600, 2900],
                   "HF": [4000, 4100],
                   "CO2": [2020, 2150],
                   "SiF4": [1010, 1040]}

# Secondary windows
secondary_windows = {"SO2": [1080, 1120],
                     "HCl": [5730.0, 5780.0],
                     "HF": [4000, 4100],
                     "CO2": [2020, 2150]}
# Microwindows
micro_windows = {"SO2": [2400, 2550],
                 "HCl": [[2727.0, 2728.5], [2775.0, 2776.50], [2818.75, 2820.35], [2820.75, 2822.35], [2843.0, 2844.4],
                         [2903.35, 2904.85], [2923.0, 2924.50], [2925.0, 2926.75], [2942.0, 2943.5], [2960.3, 2961.825],
                         [2962.3, 2964.0], [2995.0, 2996.5]],
                 "HF": [4000, 4100],
                 "CO2": [2020, 2150]}

# Unit notations
units = {"rad": r"$mW$ / ($m^{2} \cdot sr \cdot cm^{-1}$)",
         "bbt": r"$K$",
         "opt": r"$a.u.$",
         "wn": r"$cm^{-1}$",
         "um": r"$\mu$$m$",
         "nm": r"$nm$",
         "molec.cm^-2": r"$molec \cdot cm^{-2}$",
         "ppmm": r"$ppm \cdot m$",
         "g.m^-2": r"$g \cdot m^{-2}$",
         "ppmv": r"$ppm$",
         "relh": r"$%$",
         "N_density": r"$cm^{-3}",
         "N_density_SI": r"$m^{-3}",
         "g.cm^-3": r"$g \cdot cm^{-3}$",
         "g.m^-3": r"$g \cdot m^{-3}$",
         "fov": r"$mrad$",
         "nu_shift": r"$cm^{-1}$",
         "gas_temp": r"$K$",
         "gasE_temp": r"$K$",
         "source_temp": r"$K$",
         "max_opd": r"$cm$"}



