from pathlib import Path

import numpy as np

_MODEL_DIR = Path(__file__).parent

def _get_table_path(metallicity=0.0):
    """
    Return the path to a bundled Sonora Bobcat ``*_mass`` evolutionary table.
    """

    metallicity_files = {
        -0.5: 'nc-0.5_co1.0_mass',
        0.0: 'nc+0.0_co1.0_mass',
        0.5: 'nc+0.5_co1.0_mass',
    }
    if metallicity not in metallicity_files:
        raise ValueError(
            f'metallicity={metallicity!r} is not recognized for Sonora Bobcat. '
            f'Valid options: {list(metallicity_files)}.'
        )

    table_path = _MODEL_DIR / metallicity_files[metallicity]
    if not table_path.exists():
        raise FileNotFoundError(f'Bundled evolutionary table not found: "{table_path}"')

    return str(table_path)

def _read_evolutionary_model(filename):
    """
    Read a Sonora Bobcat ``*_mass`` evolutionary table and return its grid arrays.
     The seventh column (log I) is parsed out and not returned.
    """

    with open(filename) as evo_file:
        rows = [line.split() for line in evo_file]
    data = np.array([row for row in rows if len(row) == 7], dtype=float)
    if data.size == 0:
        raise ValueError(f'No seven-column data rows were found in "{filename}". '
                         f'Pass a Sonora Bobcat *_mass evolutionary table.')

    out = {'mass': data[:, 0], 'age': data[:, 1], 'logL': data[:, 2],
           'Teff': data[:, 3], 'logg': data[:, 4], 'radius': data[:, 5]}

    return out
