"""
demo_retrieval.py — Temperature profile retrieval from the CO₂ 15 µm band.

Demonstrates:
  1. Computing temperature Jacobians (∂R/∂T) using RFM's JAC flag.
  2. Constructing a synthetic noisy observation with a known temperature
     perturbation.
  3. Optimal Estimation retrieval using pyOptimalEstimation with a
     *linearized* forward model (F(x) = F₀ + K·δx) — exact because K is
     computed from the analytical RFM Jacobian, not finite differences.
  4. Four-panel diagnostic plot: Jacobians | spectral fit | profile.

Requires pyOptimalEstimation ≥ 1.3:
    pip install pyOptimalEstimation

Run with:
    python test/demo_retrieval.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import rfmwrapper as rw
from rfmwrapper.plotting import (
    plotJacobians, plotRetrievalProfile, _add_wl_axis,
)
from demo_config import get_output_path, timer, apply_style

apply_style()

try:
    from pyOptimalEstimation import optimalEstimation
except ImportError:
    raise ImportError(
        "pyOptimalEstimation is required for this demo.\n"
        "Install with: pip install pyOptimalEstimation"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START   = 667.0     # cm⁻¹ — CO₂ 15 µm band
WN_STOP    = 750.0
NPERWN     = 2
ATM_TYPE   = 'FASMLS'

# Jacobian altitude levels — RFM produces N-2 output files; boundaries
# act as interpolation anchors only and are NOT included in the state vector.
JAC_LEVELS = [0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 18.0, 22.0,
              28.0, 35.0, 45.0, 55.0, 65.0, 75.0, 80.0]   # km

# True temperature perturbation at the retrievable levels (JAC_LEVELS[1:-1])
# +5 K warm anomaly in lower troposphere, -4 K cool anomaly in upper trop.
DELTA_T_PROFILE = {
    2.0:  5.0,
    5.0:  5.0,
    8.0:  3.0,
    11.0: 0.0,
    14.0: -4.0,
    18.0: -4.0,
    22.0: -2.0,
    28.0:  0.0,
    35.0:  0.0,
    45.0:  0.0,
    55.0:  0.0,
    65.0:  0.0,
}

NESR        = 0.05   # mW m⁻² sr⁻¹ (cm⁻¹)⁻¹ — noise equivalent spectral radiance
T_PRIOR_STD = 5.0    # K — prior temperature uncertainty at each level

# ---------------------------------------------------------------------------
# Phase 1: Prior run + Jacobians
# ---------------------------------------------------------------------------

print("Phase 1 — RFM at prior state (MLS), computing Jacobians...")
model = rw.RFM(
    type='nadir',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas='CO2 H2O N2O',
    atm_type=ATM_TYPE,
    jacobians={'TEM': JAC_LEVELS},
    out='RAD',
    silent=True,
)

with timer("prior RFM run + Jacobians"):
    model.run(silent=True)

rad_prior = model.getSpectra('rad')    # shape (1, n_wn)
jac       = model.getJacobians('TEM') # shape (N-2, n_wn), label_type='altitude'

F0     = rad_prior.data[0]            # prior spectrum [mW m⁻² sr⁻¹ (cm⁻¹)⁻¹]
wn     = rad_prior.wn
K_mat  = jac.data                     # (n_levels, n_wn)  — dR/dT at each level

ret_levels = jac.labels               # retrievable altitudes (excluding boundaries)
n_levels   = len(ret_levels)
n_wn       = len(wn)

print(f"  Spectral points  : {n_wn}")
print(f"  Retrievable levels: {n_levels} ({ret_levels[0]:.0f}–{ret_levels[-1]:.0f} km)")
print(f"  |K|_max          : {np.abs(K_mat).max():.4f}")

# ---------------------------------------------------------------------------
# Phase 2: True-state run + synthetic observation
# ---------------------------------------------------------------------------

print("\nPhase 2 — RFM at true state, building synthetic observation...")

# Build temperature perturbation at the retrievable levels
delta_T    = np.array([DELTA_T_PROFILE.get(float(z), 0.0) for z in ret_levels])

# Prior temperature at retrievable levels (interpolated from ATM grid)
x_ap_vals  = np.interp(ret_levels, model.atm['HGT'], model.atm['TEM'])
x_true_vals = x_ap_vals + delta_T

# Perturb the full ATM temperature profile (interpolated from ret_levels to ATM grid)
delta_T_atm = np.interp(model.atm['HGT'], ret_levels, delta_T)

atm_true = {k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in model.atm.items()}
atm_true['TEM'] = model.atm['TEM'] + delta_T_atm

# Write perturbed atmosphere and run RFM
atm_fname = os.path.join(model._work_dir, 'true_state.atm')
rw.writeAtm(atm_true, atm_fname)

model_true = rw.RFM(
    type='nadir',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas='CO2 H2O N2O',
    atm_path=atm_fname,
    out='RAD',
    silent=True,
)

with timer("true-state RFM run"):
    model_true.run(silent=True)

F_true = model_true.getSpectra('rad').data[0]

# Add Gaussian noise
rng       = np.random.default_rng(seed=42)
y_obs_arr = F_true + rng.normal(0.0, NESR, size=n_wn)

print(f"  δT range: {delta_T.min():.1f} to {delta_T.max():.1f} K")
print(f"  NESR     : {NESR} mW m⁻² sr⁻¹ (cm⁻¹)⁻¹")

# ---------------------------------------------------------------------------
# Phase 3: OEM retrieval (linearized forward model)
# ---------------------------------------------------------------------------

print("\nPhase 3 — Optimal Estimation retrieval...")

x_vars = [f"T_{z:.0f}km" for z in ret_levels]
y_vars = [f"nu_{v:.4f}" for v in wn]

x_a    = pd.Series(x_ap_vals, index=x_vars)
x_true = pd.Series(x_true_vals, index=x_vars)
y_obs  = pd.Series(y_obs_arr, index=y_vars)

# Prior covariance: diagonal ±T_PRIOR_STD K at each level
S_a = np.diag(np.full(n_levels, T_PRIOR_STD ** 2))

# Observation covariance: diagonal NESR² at each spectral point
S_y = np.diag(np.full(n_wn, NESR ** 2))

# Linearized forward model: F(x) = F0 + K.T @ (x - x_ap)
# K_mat: (n_levels, n_wn)  →  K_mat.T: (n_wn, n_levels)
def forward(x_series):
    dx = x_series.values - x_ap_vals
    return pd.Series(F0 + K_mat.T @ dx, index=y_vars)

oe = optimalEstimation(
    x_vars=x_vars,
    x_a=x_a,
    S_a=S_a,
    y_vars=y_vars,
    y_obs=y_obs,
    S_y=S_y,
    forward=forward,
    x_truth=x_true,
)

with timer("OEM retrieval"):
    converged = oe.doRetrieval(maxIter=10)

x_ret_vals = oe.x_op.values if oe.x_op is not None else x_ap_vals
print(f"  Converged: {converged}  ({oe.convI} iterations)")
if hasattr(oe, 'dgf'):
    print(f"  Degrees of freedom: {oe.dgf:.2f}")

# ---------------------------------------------------------------------------
# Phase 4: Diagnostic figure (3 panels)
# ---------------------------------------------------------------------------

print("\nBuilding diagnostic figure...")

fig = plt.figure(figsize=(14, 6), constrained_layout=True)
gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.6, 1.8, 1.0])
fig.suptitle(
    f"Temperature retrieval — CO₂ 15 µm band  |  {ATM_TYPE}\n"
    f"NESR = {NESR} mW m⁻² sr⁻¹ (cm⁻¹)⁻¹    "
    f"σ_prior = {T_PRIOR_STD} K    "
    f"converged: {converged}"
)

# Panel 1 — Jacobian 2D map
ax1 = fig.add_subplot(gs[0])
plotJacobians(jac, ax=ax1, cmap='cmc.vik',
              title=r'$\partial R / \partial T$ Jacobians')

# Panel 2 — Spectral fit
ax2 = fig.add_subplot(gs[1])
ax2.plot(wn, y_obs_arr,   color='k',       lw=0.7, alpha=0.8, label='y_obs (noisy)')
ax2.plot(wn, F_true,      color='tab:gray', lw=0.9, ls='--',  label='F(x_true)')
ax2.plot(wn, F0,          color='tab:blue', lw=1.0, ls='-.',  label='F(x_prior)')
ax2.plot(wn, forward(oe.x_op).values,
              color='tab:red',  lw=1.2, label='F(x_ret)')
ax2.set_xlabel(r'Wavenumber [cm$^{-1}$]')
ax2.set_ylabel(r'Radiance [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$]')
ax2.set_title('Spectral fit')
ax2.legend(fontsize=8)
_add_wl_axis(ax2)

# Panel 3 — Retrieved temperature profile
ax3 = fig.add_subplot(gs[2])
plotRetrievalProfile(
    altitudes=ret_levels,
    x_prior=x_ap_vals,
    x_ret=x_ret_vals,
    x_true=x_true_vals,
    ax=ax3,
    xlabel='Temperature [K]',
    title='Temperature profile',
)

out = get_output_path('retrieval', 'retrieval_diagnostic.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"  Saved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# Residual figure
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                            constrained_layout=True)
fig2.suptitle("Spectral residuals")

residual_prior = y_obs_arr - F0
residual_ret   = y_obs_arr - forward(oe.x_op).values

axes2[0].plot(wn, residual_prior, color='tab:blue', lw=0.8, label='y_obs − F(x_prior)')
axes2[0].axhline(0, color='k', lw=0.5, ls='--')
axes2[0].set_ylabel(r'$\Delta R$ [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$]')
axes2[0].set_title('Prior residual')
axes2[0].legend(fontsize=8)
_add_wl_axis(axes2[0])

axes2[1].plot(wn, residual_ret, color='tab:red', lw=0.8, label='y_obs − F(x_ret)')
axes2[1].axhline(0, color='k', lw=0.5, ls='--')
axes2[1].set_ylabel(r'$\Delta R$ [mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$]')
axes2[1].set_title('Retrieved residual')
axes2[1].set_xlabel(r'Wavenumber [cm$^{-1}$]')
axes2[1].legend(fontsize=8)

out2 = get_output_path('retrieval', 'residuals.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")
plt.show()

model.cleanup()
model_true.cleanup()
print("Done.")
