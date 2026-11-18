"""Filter 3: fingerprint map from the site survey.

    h_i(p) = M_i(p)

M_i is the surveyed RSSI of tag i on a regular grid, read between grid points by
bilinear interpolation. The survey is taken every 3 m; map_spacing_m keeps every
second point (6 m) or every fourth (12 m), which is how much survey work is needed.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from .common import FilterConfig, ParticleFilter


class FingerprintFilter(ParticleFilter):
    def __init__(self, survey, config=FilterConfig(), rng=None, map_spacing_m=6.0):
        super().__init__(survey, config, rng)
        survey_step = float(survey["xs"][1] - survey["xs"][0])
        every = int(round(map_spacing_m / survey_step))
        if every < 1 or not np.isclose(every * survey_step, map_spacing_m):
            raise ValueError(f"map_spacing_m must be a multiple of the {survey_step:g} m survey step")
        self.map_spacing_m = map_spacing_m
        self.origin = float(survey["xs"][0])
        self.ble_map = survey["ble_map"][:, ::every, ::every]

    def expected_rssi(self, tags):
        coords = (self.xy[:, ::-1].T - self.origin) / self.map_spacing_m  # Rows are y, columns are x.
        return np.stack([map_coordinates(self.ble_map[t], coords, order=1, mode="nearest") for t in tags], axis=1)
