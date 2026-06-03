import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_jup, R_sun
from importlib import resources
from pathlib import Path

import seda

# ----------------------------
# Test data
# ----------------------------
# A toy grid whose (logL, R) -> (mass, age) mapping is bijective (linear with a
# non-singular Jacobian), so interpolation recovers the node values exactly.
MASS_MSUN = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # M_sun
AGE_GYR = np.array([0.1, 0.5, 1.0, 5.0, 10.0])        # Gyr

BOBCAT_MASS_FILE = str(
	Path(resources.files("seda.models_aux")) / "Sonora_Bobcat_evolution" / "nc+0.0_co1.0_mass"
)

# ----------------------------
# Helpers
# ----------------------------
def _toy_grid():
	"""Build a synthetic evolutionary grid as a dictionary of arrays."""
	mass_mesh, age_mesh = np.meshgrid(MASS_MSUN, AGE_GYR, indexing='ij')
	mass = mass_mesh.ravel()
	age = age_mesh.ravel()

	logL = -4.0 + 60.0 * mass - 0.1 * age      # log10(L/Lsun)
	radius = 1.0 - 5.0 * mass - 0.02 * age     # R_sun (kept positive over the grid)
	Teff = 1000.0 + 10000.0 * mass - 50.0 * age  # K
	logg = 4.0 + 10.0 * mass - 0.05 * age      # cgs dex

	return {'mass': mass, 'age': age, 'Teff': Teff, 'logL': logL,
	        'logg': logg, 'radius': radius}

def _node_values(mass_msun, age_gyr):
	"""Return (Lbol[Lsun], R[Rjup], Teff[K], logg, mass[Mjup]) for a grid node."""
	logL = -4.0 + 60.0 * mass_msun - 0.1 * age_gyr
	radius_rsun = 1.0 - 5.0 * mass_msun - 0.02 * age_gyr
	Teff = 1000.0 + 10000.0 * mass_msun - 50.0 * age_gyr
	logg = 4.0 + 10.0 * mass_msun - 0.05 * age_gyr

	Lbol = 10.0 ** logL
	R_rjup = (radius_rsun * R_sun).to(R_jup).value
	mass_mjup = (mass_msun * u.M_sun).to(u.M_jup).value
	return Lbol, R_rjup, Teff, logg, mass_mjup

def _write_toy_mass_table(tmp_path, grid):
	"""Write a tiny Bobcat-style *_mass table and return its path."""
	path = tmp_path / "toy_bobcat_mass"
	lines = [" M/Msun   age(Gyr)  log L/Lsun  Teff(K)  log g  R/Rsun   log I\n"]
	for mass, age, logL, Teff, logg, radius in zip(
		grid['mass'], grid['age'], grid['logL'], grid['Teff'], grid['logg'], grid['radius']
	):
		lines.append(
			f"{mass:.6f} {age:.6f} {logL:.8f} {Teff:.6f} {logg:.8f} {radius:.8f} 0.0\n"
		)
	path.write_text("".join(lines))
	return str(path)

# ----------------------------
# Tests
# ----------------------------
def test_evol_params_round_trip(tmp_path):
	"""Feeding a node's (Lbol, R) should recover that node's mass/age/logg/Teff."""
	np.random.seed(0)
	grid = _toy_grid()
	path = _write_toy_mass_table(tmp_path, grid)

	mass_msun, age_gyr = 0.03, 1.0
	Lbol, R_rjup, Teff_exp, logg_exp, mass_mjup_exp = _node_values(mass_msun, age_gyr)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		evolutionary_model=path, n_mc=2000, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_mjup_exp, rel=0.05), (
		f"Expected mass ~{mass_mjup_exp} M_jup, got {out['mass']}"
	)
	assert out['age'] == pytest.approx(age_gyr, rel=0.05), (
		f"Expected age ~{age_gyr} Gyr, got {out['age']}"
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

def test_evol_params_outside_grid_raises(tmp_path):
	"""A luminosity far outside the grid leaves no valid samples and must raise."""
	np.random.seed(0)
	path = _write_toy_mass_table(tmp_path, _toy_grid())

	# logL = 10 (Lbol = 1e10 Lsun) is far above the grid coverage
	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=1e10, eLbol=1e8, R=1.0, eR=1e-4,
			evolutionary_model=path, n_mc=500, verbose=False,
		)

def test_evol_params_bad_mass_table_raises(tmp_path):
	"""A file with no Bobcat seven-column rows should raise a clear error."""
	path = tmp_path / "bad_bobcat_mass"
	path.write_text("not a valid Bobcat mass table\n24\n")

	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=1e-3, eLbol=1e-4, R=1.0, eR=0.1,
			evolutionary_model=str(path), n_mc=100, verbose=False,
		)

def test_evol_params_std_error_mode(tmp_path):
	"""With error='std' the uncertainties should be returned as scalars."""
	path = _write_toy_mass_table(tmp_path, _toy_grid())
	Lbol, R_rjup, _, _, _ = _node_values(0.03, 1.0)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		evolutionary_model=path, error="std", n_mc=1000, verbose=False,
	)

	assert np.isscalar(out["emass"]), "emass should be a scalar when error='std'"
	assert np.isscalar(out["eage"]), "eage should be a scalar when error='std'"
	assert np.isscalar(out["elogg"]), "elogg should be a scalar when error='std'"
	assert np.isscalar(out["eTeff"]), "eTeff should be a scalar when error='std'"

def test_evol_params_nonpositive_lbol_raises(tmp_path):
	"""A non-positive bolometric luminosity should raise an error."""
	path = _write_toy_mass_table(tmp_path, _toy_grid())

	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=0.0, eLbol=1e-5, R=1.0, eR=0.1,
			evolutionary_model=path, n_mc=500, verbose=False,
		)

def test_evol_params_reproducible_with_seed(tmp_path):
	"""Fixing the random seed should make the Monte Carlo results reproducible."""
	path = _write_toy_mass_table(tmp_path, _toy_grid())
	Lbol, R_rjup, _, _, _ = _node_values(0.03, 1.0)

	np.random.seed(0)
	out1 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		evolutionary_model=path, n_mc=500, verbose=False,
	)

	np.random.seed(0)
	out2 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		evolutionary_model=path, n_mc=500, verbose=False,
	)

	assert out1["mass"] == pytest.approx(out2["mass"]), (
		"Mass should be reproducible with a fixed random seed"
	)
	assert out1["age"] == pytest.approx(out2["age"]), (
		"Age should be reproducible with a fixed random seed"
	)

def test_evol_params_bobcat_file():
	"""Run evol_params() on the bundled Bobcat table and print the derived parameters."""
	np.random.seed(0)

	# docstring example values
	Lbol, eLbol = 6.324e-5, 6.978e-6  # Lsun
	R, eR = 1.018, 0.059              # Rjup

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=eLbol, R=R, eR=eR,
		evolutionary_model=BOBCAT_MASS_FILE, verbose=True,
	)

	# print the derived parameters the function is intended to infer
	print("\nDerived fundamental parameters from bundled Bobcat table:")
	print(f"   mass = {out['mass']:.4g} M_jup  (err {out['emass']})")
	print(f"   age  = {out['age']:.4g} Gyr     (err {out['eage']})")
	print(f"   logg = {out['logg']:.4g} dex    (err {out['elogg']})")
	print(f"   Teff = {out['Teff']:.4g} K      (err {out['eTeff']})")
	print(f"   samples outside grid: {out['frac_outside_grid'] * 100:.1f}%")

	# the query point is inside the grid, so values must be finite
	assert np.isfinite(out['mass']), "Inferred mass should be finite"
	assert np.isfinite(out['age']), "Inferred age should be finite"
	assert np.isfinite(out['logg']), "Inferred logg should be finite"
	assert np.isfinite(out['Teff']), "Inferred Teff should be finite"
	assert out['frac_outside_grid'] < 0.75, (
		"Almost all Monte Carlo samples should fall inside the grid"
	)

def test_evol_params_regular_user_output():
	"""Show what a regular user sees when calling evol_params(verbose=True)."""
	np.random.seed(0)

	seda.phy_params.evol_params(
		Lbol=6.324e-5, eLbol=6.978e-6, R=1.018, eR=0.059,
		evolutionary_model=BOBCAT_MASS_FILE, error="percentile", verbose=True,
	)
