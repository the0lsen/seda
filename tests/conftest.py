import os
from pathlib import Path
from importlib import resources
import fnmatch
import json
import seda

def load_model_spectra_catalog():
    """Return list of (model, example_spectrum_path)."""

    # base path to models_aux
    base = Path(resources.files('seda.models_aux'))

    # get available models from the API
    available_models = seda.models.Models().available_models

    catalog = []

    # loop over available models
    for model in available_models:

        # path to model folder
        model_dir = base / model

        # read config.json
        config_path = model_dir / 'config.json'
        with open(config_path) as f:
            config = json.load(f)

        pattern = config['filename_pattern']

        # find spectra matching pattern
        spectra = [
            f for f in os.listdir(model_dir)
            if fnmatch.fnmatch(f, pattern)
        ]

        if not spectra:
            continue

        # pick one example spectrum (sorted for reproducibility)
        spectra = sorted(spectra)
        example_file = model_dir / spectra[0]

        # append (model, spectrum path)
        catalog.append((model, str(example_file)))

    return catalog

def load_evolutionary_model_catalog():
	"""Return list of (model, filename) with one example table per evolutionary model."""

	catalog = []
	for model in sorted(seda.models.EvolutionaryModels().available_models):
		tables = seda.models.EvolutionaryModels(model).available_tables
		if not tables:
			continue
		catalog.append((model, tables[0]))

	return catalog

def load_evolutionary_table_catalog():
	"""Return list of (model, filename) for every bundled evolutionary table."""

	catalog = []
	for model in sorted(seda.models.EvolutionaryModels().available_models):
		for filename in seda.models.EvolutionaryModels(model).available_tables:
			catalog.append((model, filename))

	return catalog
