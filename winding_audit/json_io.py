"""Portable report JSON: missing numeric observations are null, never NaN."""
import json
import math
import numpy as np


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    return value


def dumps(value, **kwargs):
    return json.dumps(_clean(value), allow_nan=False, **kwargs)
