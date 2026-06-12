"""
test_gra_limb.py — Integration test for GRA flag (2D horizontally
inhomogeneous atmosphere) in limb-viewing geometry.

Uses three standard RFM atmosphere profiles at different horizontal PSI
angles to simulate a horizontally varying atmosphere along the ray path.

For TH=10 km / h_obs=800 km the actual ray-path PSI range is approximately
-38° to -17°.  Profiles are placed at PSI = -40°, -20°, 0° so that both
ray legs traverse genuinely different horizontal columns:

  PSI = -40 deg  →  Sub-arctic winter (FASSAW)   [far side of ray path]
  PSI = -20 deg  →  Mid-latitude summer (FASMLS)  [near tangent point]
  PSI =   0 deg  →  Tropical (FASTRO)             [observer side reference]

Tests:
  1. RFM runs without error with the GRA flag.
  2. Spectra have the expected shape (n_tan × n_wn).
  3. PTH files are produced (diagnostics=True).
  4. parseRayPath correctly separates the two ray-path segments (downward
     and upward) and returns 'gases_up' / 'totals_up' keys.
  5. Upward and downward totals differ — confirming that RFM actually used
     different horizontal columns on each leg.

Run with:
    python test/test_gra_limb.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

import rfmwrapper as rw
from rfmwrapper.objects import _std_atmospheres
from demo_config import get_output_path, timer, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WN_START    = 700.0
WN_STOP     = 800.0
NPERWN      = 1
TAN_HEIGHTS = [5.0, 10.0, 15.0, 20.0]
H_SAT       = 800.0

# Three horizontally distinct profiles spanning the actual ray-path PSI
# range (~-38° to -17° for TH=10 km, h_obs=800 km).  Profiles at ±5°
# are insufficient — both ray legs would extrapolate to the same values.
GRA_PROFILES = {
    -40.0: _std_atmospheres['FASSAW'],   # far side of ray path (sub-arctic winter)
    -20.0: _std_atmospheres['FASMLS'],   # near tangent point  (mid-lat summer)
      0.0: _std_atmospheres['FASTRO'],   # observer side reference (tropical)
}

print("=" * 65)
print("GRA flag integration test — 2D limb atmosphere")
print("=" * 65)
print(f"  Tangent heights : {TAN_HEIGHTS} km")
print(f"  Satellite       : {H_SAT:.0f} km")
print(f"  PSI profiles    : {list(GRA_PROFILES.keys())} deg")
print(f"   -40°  →  {os.path.basename(_std_atmospheres['FASSAW'])}")
print(f"   -20°  →  {os.path.basename(_std_atmospheres['FASMLS'])}")
print(f"     0°  →  {os.path.basename(_std_atmospheres['FASTRO'])}")
print()

# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

rfm = rw.RFM(
    type='limb',
    wn_start=WN_START, wn_stop=WN_STOP, nperwn=NPERWN,
    gas='H2O CO2 O3',
    atm_type='FASMLS',      # sets self.atm_path; overridden at PSI=0 by gra_profiles
    tan_heights=TAN_HEIGHTS,
    h_obs=H_SAT,
    diagnostics=True,
    out='RAD',
    gra_profiles=GRA_PROFILES,
    silent=True,
)

with timer("GRA limb RFM run"):
    rfm.run(silent=True)

# ---------------------------------------------------------------------------
# Test 1 — spectra shape
# ---------------------------------------------------------------------------

spectra = rfm.getSpectra('rad')
assert spectra.data.shape[0] == len(TAN_HEIGHTS), \
    f"Expected {len(TAN_HEIGHTS)} spectra, got {spectra.data.shape[0]}"
assert spectra.data.shape[1] > 0, "Empty wavenumber axis"
print(f"Test 1 PASS — spectra shape {spectra.data.shape}  "
      f"(mean RAD = {np.mean(spectra.data):.3e})")

# ---------------------------------------------------------------------------
# Test 2 — PTH files produced and contain GRA two-segment data
# ---------------------------------------------------------------------------

pth_raw = rfm.getRayPath()
assert len(pth_raw) == len(TAN_HEIGHTS), \
    f"Expected {len(TAN_HEIGHTS)} PTH files, got {len(pth_raw)}"
print(f"Test 2 PASS — {len(pth_raw)} PTH files retrieved")

# ---------------------------------------------------------------------------
# Test 3 — parseRayPath: GRA header flag and two-segment keys present
# ---------------------------------------------------------------------------

pth = {k: rw.parseRayPath(v) for k, v in sorted(pth_raw.items())}

for key, p in pth.items():
    assert p['header']['gra'], \
        f"PTH {key}: expected header['gra']=True, got False"
    assert 'gases_up' in p, \
        f"PTH {key}: 'gases_up' key missing — second segment not parsed"
    assert 'totals_up' in p, \
        f"PTH {key}: 'totals_up' key missing"

print(f"Test 3 PASS — all {len(pth)} PTH files have gra=True and "
      "both segment keys")

# ---------------------------------------------------------------------------
# Test 4 — downward and upward totals differ (different horizontal columns)
# ---------------------------------------------------------------------------

print()
print("─" * 65)
print("Per-tangent-height segment comparison (H2O total column):")
print(f"  {'TH [km]':>8}  {'dn amt':>14}  {'up amt':>14}  {'ratio up/dn':>12}")
all_differ = True
for key, p in sorted(pth.items(), key=lambda kv: kv[1]['header']['tan_hgt']):
    th = p['header']['tan_hgt']
    if 'H2O' not in p['totals'] or 'H2O' not in p['totals_up']:
        print(f"  {th:>8.1f}  H2O not available")
        continue
    amt_dn = p['totals']['H2O']['amt']
    amt_up = p['totals_up']['H2O']['amt']
    ratio  = amt_up / amt_dn if amt_dn > 0 else float('nan')
    differs = not np.isclose(amt_dn, amt_up, rtol=1e-4, atol=0)
    all_differ = all_differ and differs
    flag = "✓ differ" if differs else "✗ SAME"
    print(f"  {th:>8.1f}  {amt_dn:>14.4e}  {amt_up:>14.4e}  {ratio:>12.4f}  {flag}")

if all_differ:
    print()
    print("Test 4 PASS — downward and upward columns differ for all TH")
else:
    print()
    print("Test 4 FAIL — some segments have identical columns (GRA may not "
          "be using different horizontal columns)")

# ---------------------------------------------------------------------------
# Figure — downward vs upward H2O column loading for TH=10 km
# ---------------------------------------------------------------------------

key_10 = next(k for k, p in pth.items() if p['header']['tan_hgt'] == 10.0)
p10 = pth[key_10]

if 'H2O' in p10['gases'] and 'H2O' in p10['gases_up']:
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    df_dn = p10['gases']['H2O']
    df_up = p10['gases_up']['H2O']

    ax.plot(df_dn['amt'], df_dn['z_low'], color='#2166ac', lw=1.5,
            label='Downward leg  (PSI ≈ −38° → −27°,  FASSAW–FASMLS)')
    ax.fill_betweenx(df_dn['z_low'], 0, df_dn['amt'],
                     color='#2166ac', alpha=0.12)
    ax.plot(df_up['amt'], df_up['z_low'], color='#d4001a', lw=1.5,
            ls='--', label='Upward leg  (PSI ≈ −27° → −17°,  FASMLS–FASTRO)')
    ax.fill_betweenx(df_up['z_low'], 0, df_up['amt'],
                     color='#d4001a', alpha=0.12)

    ax.set_xscale('log')
    ax.set_xlabel(r'H$_2$O column amount  [kmol cm$^{-2}$]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title(f"GRA limb — H₂O column loading per layer\n"
                 f"TH = 10 km  |  PSI: FASSAW(−40°) / FASMLS(−20°) / FASTRO(0°)")
    ax.legend(fontsize=9)

    out = get_output_path('gra', 'gra_h2o_column_dn_vs_up.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {out}")
    plt.show()

rfm.cleanup()
print()
print("Done.")
