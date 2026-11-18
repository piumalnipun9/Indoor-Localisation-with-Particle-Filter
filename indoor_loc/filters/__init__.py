"""Three BLE particle filters that share motion and resampling but differ in measurement model."""
import numpy as np

from .common import FilterConfig, run
from .fingerprint import FingerprintFilter
from .multi_wall import MultiWallFilter
from .plain import PlainFilter

METHODS = {
    "plain": (PlainFilter, {}),
    "multi_wall": (MultiWallFilter, {}),
    "fingerprint_3m": (FingerprintFilter, {"map_spacing_m": 3.0}),
    "fingerprint_6m": (FingerprintFilter, {"map_spacing_m": 6.0}),
    "fingerprint_12m": (FingerprintFilter, {"map_spacing_m": 12.0}),
}

__all__ = ["FilterConfig", "FingerprintFilter", "METHODS", "MultiWallFilter", "PlainFilter", "make", "run"]


def make(method, survey, seed=0, config=FilterConfig()):
    """Build a named filter with its own random generator."""
    cls, options = METHODS[method]
    return cls(survey, config, np.random.default_rng(seed), **options)
