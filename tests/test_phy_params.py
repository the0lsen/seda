import numpy as np
import pytest
from astropy.constants import M_jup, M_sun, R_jup, R_sun

import seda
from tests.conftest import load_evolutionary_model_catalog, load_evolutionary_table_catalog

BOBCAT_FILENAME = 'nc+0.0_co1.0_mass'
DIAMONDBACK_FILENAME = 'nc_m0.0_mass'

# ----------------------------
# Helpers
# ----------------------------
def _grid_radius_in_rjup(model, radius):
	"""Convert a native grid radius value to R_jup using ``config.json`` units."""
	radius_unit = seda.models.EvolutionaryModels(model).units['radius']
	if radius_unit == 'R_sun':
		return (radius * R_sun).to(R_jup).value
	if radius_unit == 'R_jup':
		return float(radius)
	raise ValueError(f'Unsupported evolutionary grid radius unit: {radius_unit!r}')

def _grid_radius_native(model, filename, idx=500):
	"""Return native-grid radius for one row of a bundled evolutionary table."""
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	if idx < 0:
		idx = len(grid['mass']) + idx
	return float(grid['radius'][idx])

def _bundled_grid_inputs(model, filename, idx=500):
	"""Return (Lbol, R, Teff, logg, age, mass) for one row of a bundled evolutionary table."""
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	if idx < 0:
		idx = len(grid['mass']) + idx

	Lbol = 10.0 ** grid['logL'][idx]
	R_rjup = _grid_radius_in_rjup(model, grid['radius'][idx])
	mass_msun = grid['mass'][idx]
	return Lbol, R_rjup, grid['Teff'][idx], grid['logg'][idx], grid['age'][idx], mass_msun

def _grid_sample_indices(n_rows, n_samples=5):
	"""Return evenly spaced row indices across an evolutionary table."""
	if n_rows == 1:
		return [0]
	step = max((n_rows - 1) // (n_samples - 1), 1)
	indices = sorted({min(i * step, n_rows - 1) for i in range(n_samples)})
	return indices

def _evol_sb_teff_cases():
	"""(model, filename, grid_index) cases spanning every bundled evolutionary table."""
	cases = []
	for model, filename in load_evolutionary_table_catalog():
		grid = seda.models.read_evolutionary_model(filename=filename, model=model)
		for idx in _grid_sample_indices(len(grid['mass'])):
			cases.append(
				pytest.param(
					model, filename, idx,
					id=f'{model}-{filename}-row{idx}',
				)
			)
	return cases

# ----------------------------
# Tests
# ----------------------------
@pytest.mark.parametrize('model, filename, idx', _evol_sb_teff_cases())
def test_evol_teff_matches_stefan_boltzmann(model, filename, idx):
	"""Teff from evol_params should match phy_params.teff (SB law) for the same (Lbol, R)."""
	np.random.seed(0)
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename, idx=idx)
	eLbol = 1e-12 * Lbol
	eR = 1e-12 * R_rjup

	teff_evol = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=eLbol, R=R_rjup, eR=eR,
		model=model, filename=filename, n_mc=1000, verbose=False,
	)['Teff']

	teff_sb, _ = seda.phy_params.teff(
		Lbol=Lbol, eLbol=eLbol, R=R_rjup, eR=eR, n_mc=1000,
	)

	assert teff_evol == pytest.approx(teff_sb, rel=0.02), (
		f'{model}/{filename} row {idx}: evol_params Teff={teff_evol:.2f} K '
		f'differs from Stefan-Boltzmann Teff={teff_sb:.2f} K'
	)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_round_trip(model, filename):
	"""Feeding a grid row's (Lbol, R) should recover that row's mass/age/logg/Teff."""
	np.random.seed(0)
	Lbol, R_rjup, Teff_exp, logg_exp, age_exp, mass_msun_exp = _bundled_grid_inputs(
		model, filename,
	)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, n_mc=2000, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_msun_exp, rel=0.05), (
		f"Expected mass ~{mass_msun_exp} M_sun, got {out['mass']}"
	)
	assert out['age'] == pytest.approx(age_exp, rel=0.05), (
		f"Expected age ~{age_exp}, got {out['age']}"
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

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_outside_grid_raises(model, filename):
	"""A luminosity far outside the grid leaves no valid samples and must raise."""
	np.random.seed(0)

	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=1e10, eLbol=1e8, R=1.0, eR=1e-4,
			model=model, filename=filename, n_mc=500, verbose=False,
		)

def test_evol_params_invalid_filename_lists_available(capsys):
	"""An unrecognized filename should list available tables and raise."""
	with pytest.raises(ValueError, match='not recognized'):
		seda.phy_params.evol_params(
			Lbol=1e-3, eLbol=1e-4, R=1.0, eR=0.1,
			model='Sonora_Bobcat',
			filename='not_a_real_table', n_mc=100, verbose=False,
		)

	captured = capsys.readouterr()
	assert 'nc+0.0_co1.0_mass' in captured.out
	assert 'nc-0.5_co1.0_mass' in captured.out

@pytest.mark.parametrize(
	'model',
	[
		model for model in seda.models.EvolutionaryModels().available_models
		if len(seda.models.EvolutionaryModels(model).available_tables) > 1
	],
)
def test_evol_params_multiple_tables_without_filename_raises(model, capsys):
	"""Models with multiple tables must list them when filename is omitted."""
	with pytest.raises(ValueError, match='Multiple evolutionary tables'):
		seda.phy_params.evol_params(
			Lbol=1e-3, eLbol=1e-4, R=1.0, eR=0.1,
			model=model, n_mc=100, verbose=False,
		)

	captured = capsys.readouterr()
	for filename in seda.models.EvolutionaryModels(model).available_tables:
		assert filename in captured.out

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_dynamic_output_keys(model, filename):
	"""Returned keys should match grid columns with e-prefix uncertainties."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, n_mc=500, verbose=False,
	)

	for param in ('mass', 'age', 'logg', 'Teff'):
		assert param in out
		assert f'e{param}' in out

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_std_error_mode(model, filename):
	"""With error='std' the uncertainties should be returned as scalars."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, error="std", n_mc=1000, verbose=False,
	)

	assert np.isscalar(out["emass"]), "emass should be a scalar when error='std'"
	assert np.isscalar(out["eage"]), "eage should be a scalar when error='std'"
	assert np.isscalar(out["elogg"]), "elogg should be a scalar when error='std'"
	assert np.isscalar(out["eTeff"]), "eTeff should be a scalar when error='std'"

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_nonpositive_lbol_raises(model, filename):
	"""A non-positive bolometric luminosity should raise an error."""
	with pytest.raises(ValueError):
		seda.phy_params.evol_params(
			Lbol=0.0, eLbol=1e-5, R=1.0, eR=0.1,
			model=model, filename=filename, n_mc=500, verbose=False,
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_evol_params_reproducible_with_seed(model, filename):
	"""Fixing the random seed should make the Monte Carlo results reproducible."""
	Lbol, R_rjup, _, _, _, _ = _bundled_grid_inputs(model, filename)
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	interp_params = [p for p in grid if p not in ('logL', 'radius')]

	np.random.seed(0)
	out1 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		model=model, filename=filename, n_mc=5000, verbose=False,
	)

	np.random.seed(0)
	out2 = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-4 * Lbol, R=R_rjup, eR=1e-4 * R_rjup,
		model=model, filename=filename, n_mc=5000, verbose=False,
	)

	for param in interp_params:
		assert out1[param] == pytest.approx(out2[param]), (
			f"{param} should be reproducible with a fixed random seed"
		)
		assert out1[f'e{param}'] == pytest.approx(out2[f'e{param}']), (
			f"e{param} should be reproducible with a fixed random seed"
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_table_catalog())
def test_evol_params_bundled_filenames(model, filename):
	"""Each bundled evolutionary table should recover a grid row."""
	np.random.seed(0)
	Lbol, R_rjup, Teff_exp, logg_exp, age_exp, mass_msun_exp = _bundled_grid_inputs(
		model, filename, idx=500,
	)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, n_mc=500, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_msun_exp, rel=0.05)
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
		model='Sonora_Bobcat', filename=BOBCAT_FILENAME, verbose=True,
	)

	print("\nDerived fundamental parameters from bundled Bobcat table:")
	print(f"   mass = {out['mass']:.4g} M_sun  (err {out['emass']})")
	print(f"   age  = {out['age']:.4g} Gyr     (err {out['eage']})")
	print(f"   logg = {out['logg']:.4g} dex    (err {out['elogg']})")
	print(f"   Teff = {out['Teff']:.4g} K      (err {out['eTeff']})")
	print(f"   samples outside grid: {out['frac_outside_grid'] * 100:.1f}%")

	assert np.isfinite(out['mass'])
	assert np.isfinite(out['age'])
	assert np.isfinite(out['logg'])
	assert np.isfinite(out['Teff'])
	assert out['frac_outside_grid'] < 0.75

def test_evol_params_diamondback():
	"""Sonora Diamondback evolutionary tables should round-trip through evol_params."""
	np.random.seed(0)
	model = 'Sonora_Diamondback'
	Lbol, R_rjup, Teff_exp, logg_exp, age_exp, mass_exp = _bundled_grid_inputs(
		model, DIAMONDBACK_FILENAME, idx=500,
	)

	out = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=DIAMONDBACK_FILENAME, n_mc=1000, verbose=False,
	)

	assert out['mass'] == pytest.approx(mass_exp, rel=0.01)
	assert out['age'] == pytest.approx(age_exp, rel=0.01)
	assert out['Teff'] == pytest.approx(Teff_exp, rel=0.01)
	assert out['logg'] == pytest.approx(logg_exp, rel=0.01)
	assert out['frac_outside_grid'] < 0.1

def test_evol_params_regular_user_output():
	"""Show what a regular user sees when calling evol_params(verbose=True)."""
	np.random.seed(0)

	seda.phy_params.evol_params(
		Lbol=6.324e-5, eLbol=6.978e-6, R=1.018, eR=0.059,
		model='Sonora_Bobcat', filename=BOBCAT_FILENAME, error="percentile", verbose=True,
	)

def test_list_evolutionary_tables():
	"""EvolutionaryModels should expose bundled table basenames for each model."""
	bobcat_tables = seda.models.EvolutionaryModels('Sonora_Bobcat').available_tables
	assert 'nc+0.0_co1.0_mass' in bobcat_tables
	assert len(bobcat_tables) == 3

	diamondback_tables = seda.models.EvolutionaryModels('Sonora_Diamondback').available_tables
	assert 'nc_m0.0_mass' in diamondback_tables
	assert len(diamondback_tables) == 9

	atmo_tables = seda.models.EvolutionaryModels('ATMO2020').available_tables
	assert 'ATMO_CEQ_mass.txt' in atmo_tables
	assert len(atmo_tables) == 3

	bhac_tables = seda.models.EvolutionaryModels('BHAC2015').available_tables
	assert 'BHAC15_tracks+structure.txt' in bhac_tables
	assert len(bhac_tables) == 1

def test_evolutionary_models_params_requires_model():
	"""params should require a model name, like available_tables."""
	with pytest.raises(Exception, match='Pass a model name'):
		_ = seda.models.EvolutionaryModels().params

@pytest.mark.parametrize('model', sorted(seda.models.EvolutionaryModels().available_models))
def test_evolutionary_models_params_structure(model):
	"""params should list min/max for every grid column in each bundled table."""
	model_obj = seda.models.EvolutionaryModels(model)
	params = model_obj.params

	assert set(params) == set(model_obj.available_tables)
	for filename in model_obj.available_tables:
		grid = seda.models.read_evolutionary_model(filename=filename, model=model)
		assert set(params[filename]) == set(grid)
		for col, (vmin, vmax) in params[filename].items():
			assert vmin == pytest.approx(float(grid[col].min()))
			assert vmax == pytest.approx(float(grid[col].max()))

def test_evolutionary_models_params_bobcat_spot_check():
	"""Spot-check known coverage for the solar-metallicity Bobcat table."""
	params = seda.models.EvolutionaryModels('Sonora_Bobcat').params['nc+0.0_co1.0_mass']

	assert params['mass'] == [0.0005, 0.08]
	assert params['age'] == [0.001, 15.0]
	assert params['logL'] == pytest.approx([-9.213, -2.662])
	assert params['Teff'] == [91.0, 2537.0]
	assert params['logg'] == pytest.approx([2.654, 5.484])
	assert params['radius'] == pytest.approx([0.0769, 0.2657])
	assert 'logI' not in params

def test_evolutionary_models_params_atmo_spot_check():
	"""Spot-check known coverage for the ATMO 2020 CEQ table."""
	params = seda.models.EvolutionaryModels('ATMO2020').params['ATMO_CEQ_mass.txt']

	assert params['mass'] == [0.001, 0.075]
	assert params['age'] == [0.001, 10.0]
	assert params['logL'] == pytest.approx([-7.74437436, -1.27027279])
	assert params['Teff'] == pytest.approx([206.71029843, 3156.67625353])
	assert params['logg'] == pytest.approx([3.01108287, 5.51013179])
	assert params['radius'] == pytest.approx([0.07585432, 0.79547701])

def test_evolutionary_models_params_bhac_spot_check():
	"""Spot-check known coverage for the BHAC15 tracks+structure table."""
	params = seda.models.EvolutionaryModels('BHAC2015').params['BHAC15_tracks+structure.txt']

	assert params['mass'] == [0.01, 1.4]
	assert params['age'] == pytest.approx([5.68945, 10.000343])
	assert params['logL'] == pytest.approx([-4.716, 0.74])
	assert params['Teff'] == [1206.0, 6768.0]
	assert params['logg'] == pytest.approx([3.224, 5.391])
	assert params['radius'] == pytest.approx([0.086, 3.621])
	assert params['logLi'] == pytest.approx([-11.1759, 0.0])
	assert params['logTc'] == pytest.approx([5.417, 7.398])
	assert params['logRho_c'] == pytest.approx([-0.6068, 2.8806])
	assert params['Mrad'] == pytest.approx([0.0, 1.4])
	assert params['Rrad'] == pytest.approx([0.0, 1.745])
	assert params['k2conv'] == pytest.approx([0.00124, 0.4944])
	assert params['k2rad'] == pytest.approx([0.0, 0.3072])

# ----------------------------
# Isochrone parameters
# ----------------------------
@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_isochrone_params_round_trip(model, filename):
	np.random.seed(0)
	Lbol, R_rjup, Teff_exp, logg_exp, age_exp, mass_msun_exp = _bundled_grid_inputs(
		model, filename,
	)
	R_native = _grid_radius_native(model, filename)

	out = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, age=age_exp, eage=0.0,
		model=model, filename=filename, n_mc=1000, verbose=False,
	)

	assert out['radius'] == pytest.approx(R_native, rel=0.05)
	assert out['mass'] == pytest.approx(mass_msun_exp, rel=0.05)
	assert out['logg'] == pytest.approx(logg_exp, rel=0.05)
	if 'Teff' in out:
		assert out['Teff'] == pytest.approx(Teff_exp, rel=0.05)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_isochrone_params_matches_evol_params(model, filename):
	np.random.seed(0)
	Lbol, R_rjup, _, _, age_exp, mass_msun_exp = _bundled_grid_inputs(model, filename)
	R_native = _grid_radius_native(model, filename)

	iso = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, age=age_exp, eage=0.0,
		model=model, filename=filename, n_mc=1000, verbose=False,
	)
	evol = seda.phy_params.evol_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, R=R_rjup, eR=1e-10 * R_rjup,
		model=model, filename=filename, n_mc=1000, verbose=False,
	)

	assert iso['radius'] == pytest.approx(R_native, rel=0.05)
	assert evol['age'] == pytest.approx(age_exp, rel=0.05)
	assert iso['mass'] == pytest.approx(mass_msun_exp, rel=0.05)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_isochrone_output_keys_match_evol_style(model, filename):
	Lbol, R_rjup, _, _, age_exp, _ = _bundled_grid_inputs(model, filename)
	grid = seda.models.read_evolutionary_model(filename=filename, model=model)
	interp_params = [p for p in grid if p not in ('logL', 'age')]

	out = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, age=age_exp, eage=0.0,
		model=model, filename=filename, n_mc=500, verbose=False,
	)

	for param in interp_params:
		assert param in out
		assert f'e{param}' in out
		assert not isinstance(out[param], tuple)

def test_isochrone_eage_propagation():
	np.random.seed(0)
	model, filename = 'Sonora_Bobcat', BOBCAT_FILENAME
	Lbol, _, _, _, age_exp, _ = _bundled_grid_inputs(model, filename)

	out_fixed = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, age=age_exp, eage=0.0,
		model=model, filename=filename, n_mc=2000, verbose=False,
	)
	np.random.seed(0)
	out_unc = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol, age=age_exp, eage=0.05 * age_exp,
		model=model, filename=filename, n_mc=2000, verbose=False,
	)

	assert out_unc['eradius'][0] + out_unc['eradius'][1] > out_fixed['eradius'][0] + out_fixed['eradius'][1]

def test_isochrone_eage_outside_grid_excluded():
	"""Age MC samples outside grid coverage are excluded, not clipped to the edges."""
	np.random.seed(0)
	model, filename = 'Sonora_Bobcat', BOBCAT_FILENAME
	Lbol, _, _, _, age_exp, _ = _bundled_grid_inputs(model, filename)

	out = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=1e-10 * Lbol,
		age=age_exp * 100, eage=age_exp * 50,
		model=model, filename=filename, n_mc=2000, verbose=False,
	)

	assert out['n_outside_grid'] > 0
	assert out['frac_outside_grid'] > 0.5

def test_isochrone_params_zhang2021_gj570d():
	"""Spot-check against Zhang et al. (2021) Table 6 evolutionary values for GJ 570D."""
	# Table 6 (evolutionary column): log(L/L_sun) = -5.54 +/- 0.03;
	# primary-star age range 1.4-5.2 Gyr (Sec. 5). Representative age 2.7 Gyr.
	logL = -5.54
	e_logL = 0.03
	Lbol = 10.0 ** logL
	eLbol = Lbol * np.log(10) * e_logL
	age_gyr = 2.7

	np.random.seed(0)
	out = seda.phy_params.isochrone_params(
		Lbol=Lbol, eLbol=eLbol, age=age_gyr, eage=0.0,
		model='Sonora_Bobcat', filename=BOBCAT_FILENAME,
		n_mc=5000, verbose=False,
	)

	mass_mjup = out['mass'] * (M_sun / M_jup).value
	radius_rjup = _grid_radius_in_rjup('Sonora_Bobcat', out['radius'])

	# Table 6 evolutionary: R = 0.89 +/- 0.05 R_Jup, Teff = 786 +/- 20 K,
	# logg = 5.04 +/- 0.13, M = 34.6 (+7.5/-6.5) M_Jup (Sonora Bobcat).
	assert radius_rjup == pytest.approx(0.89, abs=0.06)
	assert out['Teff'] == pytest.approx(786, abs=25)
	assert out['logg'] == pytest.approx(5.04, abs=0.15)
	assert mass_mjup == pytest.approx(34.6, abs=5.0)

def test_isochrone_params_missing_age_raises():
	with pytest.raises(TypeError):
		seda.phy_params.isochrone_params(
			Lbol=1e-3, eLbol=1e-4, model='Sonora_Bobcat',
			filename=BOBCAT_FILENAME, n_mc=100, verbose=False,
		)

@pytest.mark.parametrize('model, filename', load_evolutionary_model_catalog())
def test_isochrone_params_outside_grid(model, filename):
	with pytest.raises(ValueError):
		seda.phy_params.isochrone_params(
			Lbol=1e10, eLbol=1e8, age=1.0, eage=0.0,
			model=model, filename=filename, n_mc=500, verbose=False,
		)
