import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_jup, R_sun

import seda

# ----------------------------
# Helpers
# ----------------------------
def _bundled_grid_inputs(metallicity=0.0, idx=500):
	"""Return (Lbol, R, Teff, logg, age, mass) for one row of a bundled Bobcat table."""
	path = seda.models.get_evolutionary_table_path(model='Sonora_Bobcat', metallicity=metallicity)
	grid = seda.models.read_evolutionary_model(filename=path, model='Sonora_Bobcat')
	if idx < 0:
		idx = len(grid['mass']) + idx

	Lbol = 10.0 ** grid['logL'][idx]
	R_rjup = (grid['radius'][idx] * R_sun).to(R_jup).value
	mass_mjup = (grid['mass'][idx] * u.M_sun).to(u.M_jup).value
	return Lbol, R_rjup, grid['Teff'][idx], grid['logg'][idx], grid['age'][idx], mass_mjup

# ----------------------------
# Tests
# ----------------------------
def test_evol_params_round_trip():
	"""Feeding a grid row's (Lbol, R) should recover that row's mass/age/logg/Teff."""
	np.random.seed(0)
	Lbol, R_rjup, Teff_exp, logg_exp, age_exp, mass_mjup_exp = _bundled_grid_inputs()

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		metallicity=0.0, n_mc=2000, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_mjup_exp, rel=0.05), (
		f"Expected mass ~{mass_mjup_exp} M_jup, got {out['mass']}"
	)
	assert out['age'] == pytest.approx(age_exp, rel=0.05), (
		f"Expected age ~{age_exp} Gyr, got {out['age']}"
	)
	assert out['logg'] == pytest.approx(logg_exp, rel=0.05), (
		f"Expected logg ~{logg_exp}, got {out['logg']}"
	)
	assert out['Teff'] == pytest.approx(Teff_exp, rel=0.05), (
		f"Expected Teff ~{Teff_exp} K, got {out['Teff']}"
	)
	assert 'n_outside_grid' in out and 'frac_outside_grid' in out, (
		"Output should report out-of-grid sample bookkeeping"
	)

def test_evol_params_outside_grid_raises():
	"""A luminosity far outside the grid leaves no valid samples and must raise."""
	np.random.seed(0)

	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=1e10, eLbol=1e8, R=1.0, eR=1e-4,
			metallicity=0.0, n_mc=500, verbose=False,
		)

def test_evol_params_invalid_metallicity_raises():
	"""An unsupported metallicity should raise a clear error."""
	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=1e-3, eLbol=1e-4, R=1.0, eR=0.1,
			metallicity=1.0, n_mc=100, verbose=False,
		)

def test_evol_params_std_error_mode():
	"""With error='std' the uncertainties should be returned as scalars."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs()

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		metallicity=0.0, error="std", n_mc=1000, verbose=False,
	)

	assert np.isscalar(out["emass"]), "emass should be a scalar when error='std'"
	assert np.isscalar(out["eage"]), "eage should be a scalar when error='std'"
	assert np.isscalar(out["elogg"]), "elogg should be a scalar when error='std'"
	assert np.isscalar(out["eTeff"]), "eTeff should be a scalar when error='std'"

def test_evol_params_nonpositive_lbol_raises():
	"""A non-positive bolometric luminosity should raise an error."""
	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=0.0, eLbol=1e-5, R=1.0, eR=0.1,
			metallicity=0.0, n_mc=500, verbose=False,
		)

def test_evol_params_reproducible_with_seed():
	"""Fixing the random seed should make the Monte Carlo results reproducible."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs()

	np.random.seed(0)
	out1 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		metallicity=0.0, n_mc=500, verbose=False,
	)

	np.random.seed(0)
	out2 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		metallicity=0.0, n_mc=500, verbose=False,
	)

	assert out1["mass"] == pytest.approx(out2["mass"]), (
		"Mass should be reproducible with a fixed random seed"
	)
	assert out1["age"] == pytest.approx(out2["age"]), (
		"Age should be reproducible with a fixed random seed"
	)

@pytest.mark.parametrize('metallicity', [-0.5, 0.0, 0.5])
def test_evol_params_bundled_metallicities(metallicity):
	"""Each bundled Sonora Bobcat metallicity table should recover a grid row."""
	np.random.seed(0)
	Lbol, R_rjup, Teff_exp, logg_exp, age_exp, mass_mjup_exp = _bundled_grid_inputs(
		metallicity=metallicity, idx=500,
	)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		metallicity=metallicity, n_mc=500, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_mjup_exp, rel=0.05)
	assert out['age'] == pytest.approx(age_exp, rel=0.05)
	assert np.isfinite(out['logg'])
	assert np.isfinite(out['Teff'])

def test_evol_params_bobcat_file():
	"""Run evol_params() on the bundled solar Bobcat table and print the derived parameters."""
	np.random.seed(0)

	Lbol, eLbol = 6.324e-5, 6.978e-6  # Lsun
	R, eR = 1.018, 0.059              # Rjup

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=eLbol, R=R, eR=eR,
		metallicity=0.0, verbose=True,
	)

	print("\nDerived fundamental parameters from bundled Bobcat table:")
	print(f"   mass = {out['mass']:.4g} M_jup  (err {out['emass']})")
	print(f"   age  = {out['age']:.4g} Gyr     (err {out['eage']})")
	print(f"   logg = {out['logg']:.4g} dex    (err {out['elogg']})")
	print(f"   Teff = {out['Teff']:.4g} K      (err {out['eTeff']})")
	print(f"   samples outside grid: {out['frac_outside_grid'] * 100:.1f}%")

	assert np.isfinite(out['mass'])
	assert np.isfinite(out['age'])
	assert np.isfinite(out['logg'])
	assert np.isfinite(out['Teff'])
	assert out['frac_outside_grid'] < 0.75

def test_evol_params_regular_user_output():
	"""Show what a regular user sees when calling evol_params(verbose=True)."""
	np.random.seed(0)

	seda.phy_params.evol_params(
		Lbol=6.324e-5, eLbol=6.978e-6, R=1.018, eR=0.059,
		metallicity=0.0, error="percentile", verbose=True,
	)
