# rfmwrapper

Python wrapper for the [Reference Forward Model (RFM)](http://eodg.atm.ox.ac.uk/RFM/) radiative transfer code.

> Dudhia, A (2017): The Reference Forward Model (RFM), *Journal of Quantitative Spectroscopy and Radiative Transfer* 186, pp. 243–253. doi:[10.1016/j.jqsrt.2016.06.018](https://doi.org/10.1016/j.jqsrt.2016.06.018)

---

## Overview

`rfmwrapper` provides a clean Python interface for setting up, running, and reading output from the RFM Fortran radiative transfer code. It is designed for atmospheric spectroscopy workflows that require:

- Multiple RFM simulations run in parallel (batch processing across CPU cores)
- Atmospheric profiles manipulated programmatically (plume insertion, gas adjustment, altitude slicing)
- Spectral output as NumPy arrays for post-processing (ILS convolution, filter integration, resampling)
- Unit-aware spectral objects with wavenumber ↔ wavelength conversion

It also ships **RFMViewer**, an interactive desktop GUI for configuring runs (single or
batch), launching them, and exploring the results — see [Graphical interface](#graphical-interface-rfmviewer).

---

## Requirements

- Python ≥ 3.9
- A compiled RFM executable, a HITRAN binary database, and standard atmosphere files.
  See the [RFM installation guide](https://eodg.atm.ox.ac.uk/RFM/sum/rfm_installation.html) for instructions.

Python dependencies (`numpy`, `scipy`, `matplotlib`, `pandas`, `openpyxl`, `cmcrameri`) are declared in `pyproject.toml` and installed automatically.

---

## Installation

```bash
conda create -n rfm python=3.11
conda activate rfm
pip install -e .
```

### Configure paths

After installation, tell `rfmwrapper` where your RFM is installed:

```bash
python -m rfmwrapper configure
```

This writes to `~/.config/rfmwrapper/config.ini` and persists across sessions. To review:

```bash
python -m rfmwrapper show
```

### Path resolution

Three paths are configured independently:

| Path | What it points to |
|---|---|
| `rfm_dir` | RFM installation root |
| `hitran_path` | HITRAN `.bin` file |
| `atm_dir` | Standard atmosphere files |

Each path is resolved in priority order: runtime override → environment variable → config file → default/derived. Runtime override example:

```python
import rfmwrapper
rfmwrapper.__HITRANPATH__ = '/data/shared/HITRAN/hitran2020.bin'
```

---

## Quick Start

### Single simulation

```python
import rfmwrapper as rw

model = rw.RFM(
    type     = 'zenith',
    wn_start = 2000,
    wn_stop  = 2600,
    nperwn   = 10,
    atm_type = 'FASMLS',
    gas      = 'H2O CO2',
    h_obs    = 0.0,
    adjust   = {'CO2': 420},
)
model.run()

tra = model.getSpectra('tra')   # Spectra object
print(tra.wn.shape, tra.data.shape)
```

### Nadir geometry with surface

```python
nadir = rw.RFM(
    type       = 'nadir',
    wn_start   = 650,
    wn_stop    = 1350,
    nperwn     = 25,
    atm_type   = 'FASMLS',
    gas        = 'all',
    surf_temp  = 300,     # K — surface skin temperature
    surf_emiss = 1.0,     # NOTE: values < 1 change the meaning of TRA/OPT output;
                          # see "Surface emissivity" section below
)
nadir.run()
```

### Parallel batch runs

```python
import numpy as np

jobs = [
    rw.RFM('layer', 1050, 1250, 25, gas='SO2',
            vmr={'SO2': vmr}, silent=True)
    for vmr in np.geomspace(10, 50_000, 16)
]
results = rw.runBatch(jobs, max_workers=8)
spectra = [r.getSpectra('tra') for r in results]
```

Each instance gets its own isolated temporary work directory automatically.

---

## Graphical interface (RFMViewer)

`RFMViewer` is an interactive Tkinter desktop GUI for building, running, and visualizing
RFM simulations without writing a script. After `pip install -e .` it is available as a
console command:

```bash
rfm-viewer
# equivalently:
python -m rfmwrapper.rfm_viewer
```

**Layout**

- **Left panel** — collapsible configuration sections: *Geometry* (with geometry-aware
  default values and spinbox steppers), *Spectral Range*, *Gases* (with a species/VMR
  picker popup), *Output*, *Physics*, *Advanced* (structured Jacobian editor, XSC,
  binary/diagnostics), and *Batch Sweep*. A live log viewer mirrors `rfm.log`.
- **Toolbar** — **Load**/**Save** a configuration to JSON, a prominent **RUN** button, and
  a progress bar for batch runs.
- **Right panel** — result tabs: **Spectra**, **Atmosphere**, **Jacobians**,
  **Filters / ILS**, and **Diagnostics**. Each embeds a Matplotlib canvas with the standard
  navigation toolbar.

**Highlights**

- **Batch sweeps** — vary one parameter (e.g. `surf_temp`, a gas VMR, elevation) over a
  range or list; results overlay in the Spectra tab coloured by sweep value (selectable
  cmcrameri/Matplotlib colour map).
- **Sweep-aware post-processing** — *Filters / ILS* and *Jacobians* operate across every
  sweep member, not just the first.
- **Structured Jacobian editor** — add Jacobian targets as rows (`TEM`, `PRE`, `SFCTEM`,
  `SFCEMS`, `AEROSOL`, gases, isotopes…), each column- or layer-resolved, with prerequisite
  validation (e.g. surface Jacobians require nadir geometry).
- **Gas/VMR picker** — tick species from the full RFM gas list and enter per-gas layer VMRs.

Runs execute on a background thread (single) or a process pool (batch), so the UI stays
responsive. macOS/Windows batch runs require the usual `if __name__ == '__main__':` guard,
which the launcher provides.

---

## Geometry types

| `type`    | Description                                                                  |
|-----------|------------------------------------------------------------------------------|
| `'layer'` | Homogeneous slab; specify `temp`, `pres`, `vmr`, `pathlength`                |
| `'zenith'`| Upward-looking from `h_obs`; elevation in [0.1, 90] deg                     |
| `'nadir'` | Downward-looking toward surface; elevation in [-90, −0.1] deg                |
| `'limb'`  | Limb-viewing; specify `tan_heights=[...]` [km] and optionally `h_obs` [km]  |

---

## Output access

`model.run()` produces up to four output types: `'rad'`, `'bbt'`, `'tra'`, `'opt'`. Access via:

```python
s = model.getSpectra('rad')   # Spectra object
s.wn          # np.ndarray (n_wn,)          wavenumber axis [cm⁻¹]
s.data        # np.ndarray (n_spectra, n_wn)
s.labels      # np.ndarray (n_spectra,)     elevation [deg] / altitude [km] / pathlength [km]
s.label_type  # str: 'elevation' | 'altitude' | 'layer'
s.output      # str: 'tra' | 'rad' | 'bbt' | 'opt'
s.xunit       # str: 'wn' | 'wl'
s.yunit       # str: e.g. 'mW/(m²·sr·cm⁻¹)' for radiance
```

---

## Spectra: unit conversion and resampling

`Spectra` objects carry axis-unit state and support in-place conversion:

```python
rad_wl = model.getSpectra('rad').toWavelength()
# rad_wl.wn holds wavelength [µm]; data converted with Jacobian dν/dλ
# rad_wl.yunit = 'mW/(m²·sr·µm)'

rad_wn = rad_wl.toWavenumber()   # inverse conversion; raises ValueError if already in wn
```

Resample onto a coarser grid (e.g. after ILS convolution):

```python
s_coarse = model.getSpectra('tra').resample(dx=0.25)        # uniform spacing [cm⁻¹]
s_custom = model.getSpectra('tra').resample(x_out=my_grid)  # explicit axis
```

The standalone `resampleSpectra(spectra, dx, x_out)` works identically and also accepts `ILSOut` objects.

---

## ILS convolution

Compute and apply an Instrument Line Shape kernel:

```python
# Using a named instrument from the built-in library
model.applyILS('IASI')
ils_out = model.getILSOut('rad')   # ILSOut object, same structure as Spectra

# Or with explicit parameters
model.applyILS(fov=0.030, max_opd=1.7, apod_type='NB medium')

# Standalone function
ils = rw.makeILS(fov=0.030, max_opd=1.7, nper_wn=25, wn_center=1000,
                 apod_type='Gaussian', fwhm_wn=0.5)
ils_out = rw.applyILS(spectra, ils)
```

### Instrument library

```python
rw.listInstruments()           # ['IASI', 'Midac']
params = rw.loadILSParams('IASI')
# {'max_opd': 2.0, 'fov': 0.01465, 'apod_type': 'Gaussian', 'fwhm_wn': 0.5, ...}
```

Bundled instruments:

| Name    | Description                                         |
|---------|-----------------------------------------------------|
| `Midac` | Midac M2401 open-path FTIR (NB medium, OPD=1.7 cm)  |
| `IASI`  | IASI on MetOp L1C (Gaussian, 0.5 cm⁻¹ FWHM)        |

### Supported apodization types

`Boxcar`, `Triangular`, `Hann`, `Hamming`, `Happ-Genzel`, `Blackman-Harris`, `Nuttall`, `Kaiser`, `Tukey`, `Cosine`, `Lorentz`, `Gaussian`, `NB weak`, `NB medium`, `NB strong`

For `Gaussian`, pass `fwhm_wn` to pin the spectral FWHM; otherwise a default width proportional to `1/max_opd` is used.

---

## Filter convolution

Integrate spectra over imager filter response functions:

```python
model.applyFilters(sensor='ASTER')
filt = model.getFilterOut('ASTER', 'rad')   # FilterOut object
filt.wl          # band centre wavelengths [µm]
filt.data        # band-integrated values (n_spectra, n_bands)
filt.band_names  # ['Band_10', 'Band_11', ...]

rw.listSensors()   # list all bundled sensors
```

Convolution is performed in wavenumber space (∫L(ν)·T(ν)dν / ∫T(ν)dν) for all output types. Use `listPlatforms(sensor)` for multi-platform instruments (e.g. `'GOES-ABI'`) and `readRTTOVsrf(path)` to import custom RTTOV-format SRF files.

---

## Limb geometry

```python
limb = rw.RFM(
    type        = 'limb',
    wn_start    = 550,
    wn_stop     = 1350,
    nperwn      = 10,
    atm_type    = 'FASMLS',
    gas         = 'H2O CO2 O3',
    tan_heights = [5, 10, 15, 20, 30],   # [km]
    h_obs       = 800,                    # satellite altitude [km]
)
limb.run()
s = limb.getSpectra('rad')   # labels = tangent heights [km]
```

---

## Cross-sections (XSC species)

Aerosol and heavy molecules (SF₆, CFCs, etc.) can be included via HITRAN XSC files:

```python
model = rw.RFM(
    type     = 'nadir',
    wn_start = 900, wn_stop = 1000, nperwn = 25,
    gas      = 'H2O AEROSOL',
    xsc      = {'AEROSOL': '/path/to/aerosol.xsc'},
)
```

Prepare a HITRAN XSC dataset for RFM with `concatXSC`:

```python
from glob import glob
rw.concatXSC(glob('/data/sf6_xsc/*.xsc'), T_range=(200, 320))
# → writes $RFM_DIR/xsc/sf6.xsc
```

---

## Ray-path diagnostics

```python
model = rw.RFM(..., diagnostics=True)
model.run()
pth = model.getRayPath()               # raw text lines from RFM PTH file
parsed = rw.parseRayPath(pth)          # {header, gases, totals}

fig, ax = rw.plotRayGeometry(parsed)   # slant path length per layer
fig, ax = rw.plotRayPath(parsed, 'H2O')  # slant column per layer for one gas
```

---

## Jacobians

Request Jacobians (∂spectrum/∂parameter) via the `jacobians` dict at construction, then
read them back per target — and per output quantity:

```python
model = rw.RFM('nadir', 800, 1000, 25, gas='H2O CO2', out='RAD BBT',
               jacobians={'TEM': None,             # column (whole-profile) perturbation
                          'H2O': [0, 5, 10, 20]})   # layer perturbation at these altitudes
model.run()

jac = model.getJacobians('TEM')                # ∂RAD/∂T (default output)
jac = model.getJacobians('TEM', output='bbt')  # ∂BBT/∂T
all_jac = model.getJacobians()                 # {param: Spectra} for the default output
```

When several output flags are active, RFM writes one Jacobian per quantity (∂RAD, ∂BBT, …);
`getJacobians` keys results by `(param, output)` and never mixes them. Layer Jacobians follow
RFM's rule that *N* altitudes produce *N−2* triangular-layer spectra. Surface targets
(`SFCTEM`, `SFCEMS`) require nadir geometry.

---

## Plotting

All standalone figure functions live in `rfmwrapper.plotting` and return `(fig, ax)` or `(fig, axes)`:

```python
import rfmwrapper as rw

rw.applyStyle()                               # set rcParams to the canonical style
fig, ax  = rw.plotSpectra(spectra, species='SO2')
fig, axes = rw.plotSpectraGrid({'rad': s_rad, 'tra': s_tra})
fig, ax  = rw.plotAtmProfile(atm, 'H2O')
fig, axes = rw.plotAtmGrid(atm, ['H2O', 'CO2', 'TEM', 'PRE'])
fig, ax  = rw.plotLimbMap(spectra)
fig, ax  = rw.plotJacobians(jac_spectra)       # 2-D altitude × wavenumber map
fig, ax  = rw.plotFilterBands(filt, ax)        # band-integrated points over a spectrum
fig, ax  = rw.plotILS(ils_out, raw=spectra)    # ILS-convolved curves + raw background
```

Multiple spectra (output levels, elevations, tangent heights) are coloured with a
qualitative "Dark2" cycle automatically.

### Sweep overlays

The same overlays the GUI uses are available standalone — pass a list of containers (one per
batch member) plus the sweep values to colour each member by its value:

```python
results = rw.runBatch(jobs)
values  = [...]                                # the swept parameter values

spectra = [r.getSpectra('rad') for r in results]
fig, ax = rw.plotSpectraSweep(spectra, values, label='SO2 VMR', cmap='cmc.batlow')

# filters / ILS / column Jacobians across the sweep
fig, ax = rw.plotFilterSweep(filter_list, values, spectra_list=spectra, label='elev')
fig, ax = rw.plotILSSweep(ils_list, values, raw_list=raw_list, label='elev')
fig, ax = rw.plotJacobianSweep(jac_list, values, label='surf_temp')

colors  = rw.sweep_colors(values, cmap='cmc.batlow')   # the underlying value→colour map
```

Species-specific colours and colormaps (cmcrameri-based) are looked up automatically:

```python
color = rw.get_species_color('SO2')   # '#d4001a'
cmap  = rw.get_species_cmap('H2O')   # cmc.nuuk_r
```

---

## Atmosphere manipulation

```python
atm = rw.readAtm('/path/to/std.atm')
atm = rw.adjustAtm(atm, 'CO2', 420)              # scale surface VMR to 420 ppmv
atm = rw.insertPlume(atm, h=2.0, th=0.5,
                     plume={'SO2': 500})           # triangular plume at 2 km
atm = rw.sliceAtm(atm, h1=0.0, h2=20.0)          # trim to 0–20 km
rw.writeAtm(atm, 'modified.atm')
```

---

## Surface emissivity

In nadir geometry, `surf_emiss` controls surface thermal emission. With RFM v5.21+, `TRA` and `OPT` correctly output the one-way atmospheric transmittance regardless of `surf_emiss`; the surface emissivity only affects the `RAD` and `BBT` outputs via the surface emission term. Use `surf_emiss=1.0` (blackbody) as the default unless you are explicitly modelling surface reflectance in `RAD`.

---

## Unit conversion utilities

```python
# Radiometric
rw.rad(wn, temp)               # Planck radiance [mW/(m²·sr·cm⁻¹)]
rw.bbt(wn, radiance)           # brightness temperature [K]
rw.wn2wl(wn, L_wn)            # wavenumber → wavelength space (with Jacobian)
rw.wl2wn(wl, L_wl)            # wavelength → wavenumber space

# Column / concentration
rw.ppm2scd(ppm, pathlength)    # VMR [ppmv] → slant column density [molec/cm²]
rw.scd2ppm(scd, pathlength)
rw.mcm2gm2(mcm, species)       # molec/cm² → g/m²
rw.ppmm2mcm(ppmm, species)     # ppm·m → molec/cm²

# Humidity
rw.RH2ppm(RH, pres, temp)      # relative humidity [%] → H₂O VMR [ppmv]
rw.RH2gm3(RH, pres, temp)      # relative humidity [%] → absolute humidity [g/m³]
rw.gm32RH(gm3, pres, temp)     # absolute humidity [g/m³] → relative humidity [%]

# Temperature
rw.c2k(deg); rw.k2c(kelv)      # Celsius ↔ Kelvin
rw.f2c(deg); rw.c2f(deg)       # Fahrenheit ↔ Celsius
```

---

## Standard atmospheres

| Key      | Description            |
|----------|------------------------|
| `FASSTD` | US Standard            |
| `FASMLS` | Mid-Latitude Summer    |
| `FASMLW` | Mid-Latitude Winter    |
| `FASSAS` | Sub-Arctic Summer      |
| `FASSAW` | Sub-Arctic Winter      |
| `FASTRO` | Tropical               |
| `MIPDAY` | MIPAS Day              |
| `MIPNGT` | MIPAS Night            |

---

## Module structure

```
rfmwrapper/
├── __init__.py       Public API surface
├── __main__.py       python -m rfmwrapper configure / show
├── objects.py        RFM class, Spectra / ILSOut / FilterOut dataclasses, runBatch
├── functions.py      Atmosphere I/O, Planck, ILS kernel, filter convolution,
│                     XSC utilities, unit conversions, spectral resampling,
│                     ray-path parsing
├── plotting.py       All standalone figure functions (no plt.show()),
│                     including the sweep-overlay helpers used by the GUI
├── rfm_viewer.py     RFMViewer desktop GUI (console script: rfm-viewer)
├── config.py         Path resolution: config file, env var, fallback chain
├── constants.py      Gas species lists, AEROSOL_SPECIES, molecular weights,
│                     plot colours/cmaps
├── standards.txt     Default VMR / aerosol properties for all species
├── instruments.json  Bundled ILS parameters (Midac, IASI)
└── filters/          Imager filter response CSVs (ASTER, MMT-gasCam,
                      GOES-ABI, HIMAWARI-AHI, MODIS, MSG-SEVIRI, ...)
```

---

## Context manager

```python
with rw.RFM('layer', 2000, 2600, 10, gas='SO2') as model:
    model.run()
    s = model.getSpectra('tra')
# temporary work directory cleaned up automatically
```

---

## License

This wrapper is provided as-is for research use. The RFM itself is subject to its own [license terms](http://eodg.atm.ox.ac.uk/RFM/).
