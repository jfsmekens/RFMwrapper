# Aerosol XSC Interface: Mie package → rfmwrapper → RFM

## Overview

RFM provides exactly **one** slot for aerosol extinction treated with km⁻¹ units:
**molecule 100, named `aerosol`**. All other user-defined XSC species (including
`X109`, `X110`, and arbitrary names) are treated as standard molecular cross-sections
in cm²/molec with an ideal gas law conversion applied — wrong units for aerosol
extinction.

For a single aerosol species per run, use `aerosol` directly. For multiple simultaneous
species, use the synthetic multi-tabulation approach described below.

---

## Single aerosol species (direct km⁻¹ path)

### What RFM does for `aerosol` (molecule 100)

```
τ = k_ext(ν) [km⁻¹]  ×  ATM_profile(z) [dimensionless]  ×  path(z) [km]
```

The ATM profile value is a **dimensionless scaling factor**. No ideal gas law
conversion is applied.

### XSC file requirements

- **No comment lines** (`!`) — RFM's `SCNXSC` reader does not support them
- **Molecule name must be `aerosol`** (case-insensitive) in the 20-character
  fixed-width name field of the HITRAN header
- Standard HITRAN XSC format: one header line, data values in cm²/molec field
  but interpreted as km⁻¹ by RFM for molecule 100
- p = 0.0 Torr (vacuum) is acceptable; T can be any representative value

### HITRAN XSC header format (fixed-width, 1-based columns)

```
cols  1–20   molecule name (right-justified, e.g. '             aerosol')
cols 21–30   WN_start [cm⁻¹]  (F10.4)
cols 31–40   WN_end   [cm⁻¹]  (F10.4)
cols 41–47   NPTS              (I7)
cols 48–54   T [K]             (F7.2)
cols 55–60   p [Torr]          (F6.1)
cols 61+     max_k, resolution, broadener, reference (not parsed by rfmwrapper)
```

Example header for a single-block aerosol file:
```
             aerosol  700.0000 5000.0000   4301 298.00   0.0 1.000E+00 1.00
```

### ATM profile

Add a field named `AEROSOL` (or `aerosol`, case-insensitive) to the `.atm` file.
Values are dimensionless scaling factors:

- `1.0` everywhere → full reference extinction as tabulated in the XSC file
- `0.0` → no aerosol
- Vertical profile → spatially varying aerosol loading relative to the XSC reference

```python
from rfmwrapper import adjustAtm, readAtm
atm = adjustAtm(readAtm('std.atm'), 'AEROSOL', 1.0)   # uniform scaling = 1
```

### rfmwrapper usage

```python
from rfmwrapper import RFM, __XSCPATH__

m = RFM('nadir', 800, 1200, 10,
        gas='H2O CO2 AEROSOL',
        atm_path='my_atm.atm',
        xsc={'AEROSOL': f'{__XSCPATH__}/aerosol.xsc'},
        out='RAD OPT')
m.run()
```

Or with `xsc_dir` (file must be named `aerosol.xsc`):

```python
from rfmwrapper import RFM, __XSCPATH__

m = RFM('nadir', 800, 1200, 10,
        gas='H2O CO2 AEROSOL',
        atm_path='my_atm.atm',
        xsc_dir=__XSCPATH__,
        out='RAD OPT')
m.run()
```

---

## Multiple aerosol species: synthetic multi-tabulation approach

For N > 1 simultaneous aerosol species, RFM's single `aerosol` slot is a
limitation. The workaround: use a user-defined molecule name and supply a
**synthetic multi-tabulation XSC file** where the extinction has been
pre-converted to effective cm²/molec at each (T, p) node, such that RFM's
ideal gas law conversion cancels exactly.

### Derivation

RFM applies for a user-defined XSC molecule:

```
τ = k_eff(ν) [cm²/molec]  ×  n_air(T,p) [molec/cm³]  ×  VMR × 10⁻⁶  ×  path [cm]
```

We want:

```
τ = k_ext(ν) [km⁻¹]  ×  scaling  ×  path [km]
```

Setting `VMR = 1 ppmv` (ATM profile value = `scaling`), equating and solving:

```
k_eff(T, p) = k_ext(ν) [km⁻¹]  ×  k_B × T  /  (p [Pa] × 10⁻⁷)
```

where k_B = 1.380649 × 10⁻²³ J/K.

Equivalently:

```
k_eff(T, p) = k_ext(ν)  ×  R × T  /  (N_A × p [Pa] × 10⁻⁷)
```

where R = 8.31446 J/(mol·K), N_A = 6.02214 × 10²³ mol⁻¹.

**The n_air factors cancel exactly at every (T, p) node.** RFM's triangulation
interpolates the smooth k_eff(T, p) surface between nodes, correctly tracking
the atmospheric variation throughout the column.

### Numerical values at representative conditions

| T [K] | p [mbar] | k_B·T/(p·10⁻⁷) [cm²·km / (molec·km⁻¹)] |
|-------|----------|------------------------------------------|
|  200  |    1     | 2.762 × 10⁻¹⁸                           |
|  220  |   10     | 3.038 × 10⁻¹⁹                           |
|  260  |  100     | 3.590 × 10⁻²⁰                           |
|  280  |  500     | 7.714 × 10⁻²¹                           |
|  300  | 1013     | 4.093 × 10⁻²¹                           |

### Recommended (T, p) grid for atmospheric use

Covering troposphere + lower stratosphere:

```python
import numpy as np

T_grid = np.array([180, 200, 220, 240, 260, 280, 300])       # K
p_grid = np.array([1, 5, 10, 30, 100, 300, 500, 800, 1013])  # mbar
```

This gives 63 (T, p) blocks. RFM triangulates freely within the convex hull;
boundary values are extrapolated at edges (high altitude / low p is the
critical boundary — ensure T_min and p_min span the top of your atmosphere).

### Expected Mie package output format

The Mie package should output:

1. `k_ext(ν)` — extinction cross-section in **km⁻¹** at a **single reference
   condition** (e.g. T=298 K, p=1013 mbar, particle concentration = reference).
   This is the spectral shape. All (T, p) blocks share the same shape; only
   the amplitude changes via the 1/n_air(T, p) scaling.

2. `wn` — wavenumber axis [cm⁻¹] on a regular grid.

3. (Optional) `T_ref`, `p_ref` — the reference conditions used for k_ext,
   so rfmwrapper can verify the pre-scaling is consistent.

### `makeUserAerosolXSC()` — planned rfmwrapper utility

To be implemented in `rfmwrapper/functions.py`:

```python
def makeUserAerosolXSC(
    k_ext: np.ndarray,       # extinction [km⁻¹], shape (n_wn,)
    wn: np.ndarray,          # wavenumber axis [cm⁻¹], shape (n_wn,)
    mol_name: str,           # molecule name for *GAS and ATM field
    T_grid: np.ndarray,      # temperature nodes [K]
    p_grid: np.ndarray,      # pressure nodes [mbar]
    output: str = None,      # defaults to __XSCPATH__/{mol_name}.xsc
) -> str:
    """
    Generate a synthetic multi-tabulation XSC file for a user-defined aerosol
    species, pre-scaling k_ext to effective cm²/molec at each (T, p) node so
    that RFM's ideal gas law conversion cancels and the correct km⁻¹ extinction
    is recovered at every atmospheric layer.

    The ATM profile for this species should be a dimensionless scaling factor
    (1.0 = reference aerosol loading as given in k_ext).
    """
```

The corresponding ATM field convention is identical to the single-species case:
values are dimensionless scaling factors, and `adjustAtm(atm, mol_name, 1.0)`
adds a uniform reference loading.

---

## Summary table

| Approach | Species name | XSC units | ATM units | Max species/run |
|---|---|---|---|---|
| Direct (ID 100) | `aerosol` | km⁻¹ | dimensionless | 1 |
| Synthetic multi-tab | any name | cm²/molec (pre-scaled per T,p) | dimensionless (= scaling) | many* |
| Standard molecular | any HITRAN name | cm²/molec | ppmv | many |

\* **confirmed limit: 2 simultaneous user-defined species** (X109, X110).
A third causes a silent crash in `*GAS` parsing with no fatal error message.
Combined with the single `aerosol` slot, the maximum is **3 aerosol species per run**.

---

## Slot capacity (tested against RFM v5.10)

- **`aerosol` (ID 100)**: 1 slot, km⁻¹ direct path
- **User-defined names**: exactly **2 slots** (X109, X110 internally). A third
  user-defined species causes a silent crash during `*GAS` parsing — no fatal
  error message, log just truncates. Total aerosol capacity = **3 species per run**.

---

## Confirmed behaviour (tested against RFM v5.10)

- `aerosol` (ID 100): km⁻¹ direct path confirmed, ΔOptDepth ≈ 0.23 for ASH
  extinction ~2×10⁻³ km⁻¹ × 120 km column × scaling=1.0 ✓
- User-defined name (`ash`, `X109`, `X110`): cm²/molec path confirmed,
  ΔOptDepth ≈ 4×10¹⁶ (wrong by factor ~n_air ≈ 1.7×10¹⁷) ✓
- Comment lines (`!`) in XSC file: rejected by RFM (`F-SCNXSC IOSTAT=5010`) ✓
- `makeUserAerosolXSC()` utility: planned, not yet implemented ✓
