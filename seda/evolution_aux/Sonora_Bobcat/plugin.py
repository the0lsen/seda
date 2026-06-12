from pathlib import Path

import numpy as np
from astropy.constants import R_jup, R_sun

_MODEL_DIR = Path(__file__).parent

def _read_evolutionary_model(filename):
    """
    Read a Sonora Bobcat ``*_mass`` evolutionary table and return a dictionary of grid arrays.
    The seventh column (log I) is parsed out and not returned.
    """

    table_path = _MODEL_DIR / filename
    with open(table_path) as evo_file:
        rows = [line.split() for line in evo_file]
    data = np.array([row for row in rows if len(row) == 7], dtype=float)
    if data.size == 0:
        raise ValueError(f'No seven-column data rows were found in "{table_path}". '
                         f'Pass a Sonora Bobcat *_mass evolutionary table.')

    out = {'mass': data[:, 0], 'age': data[:, 1], 'logL': data[:, 2],
           'Teff': data[:, 3], 'logg': data[:, 4], 'radius': data[:, 5]}

    return out

def _convert_inputs(Lbol, eLbol, R, eR):
    """
    Convert user inputs to grid interpolation-axis units.

    Users pass Lbol in L_sun and R in R_jup. Returns 
    the ``logL`` and ``radius`` axes used by the evolutionary grid.
    """

    R_rsun = (R * R_jup).to(R_sun).value
    eR_rsun = (eR * R_jup).to(R_sun).value
    logL = np.log10(Lbol)
    e_logL = eLbol / (Lbol * np.log(10))
    return {
        'logL': logL, 'e_logL': e_logL,
        'radius': R_rsun, 'e_radius': eR_rsun,
    }
