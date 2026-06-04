import numpy as np

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
