# utils.py

import os
import xarray as xr
import numpy as np
import pandas as pd
import cftime
import json
from typing import Tuple, Optional, Dict

def to_np_datetime64_safe(t):
    """Konversi waktu (termasuk cftime) ke numpy.datetime64 dengan aman."""
    try:
        if isinstance(t, (cftime.DatetimeNoLeap, cftime.DatetimeGregorian)):
            return np.datetime64(t.strftime('%Y-%m-%d'), 'D')
        return np.datetime64(str(t), 'D')
    except Exception:
        return np.datetime64(pd.to_datetime(t).date(), 'D')

class NumpyEncoder(json.JSONEncoder):
    """ Kelas helper untuk encoding NumPy array ke JSON saat menyimpan model. """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
