"""Filter 1: log-distance path loss, with no knowledge of walls.

    h_i(p) = A - 10 n log10(d_i(p))

A and n are fitted to the site survey (p0_plain and n_plain in data/survey.npz).
"""
import numpy as np

from .common import FilterConfig, ParticleFilter, tag_distance


class PlainFilter(ParticleFilter):
    def __init__(self, survey, config=FilterConfig(), rng=None):
        super().__init__(survey, config, rng)
        self.a_dbm = float(survey["p0_plain"])
        self.n = float(survey["n_plain"])

    def expected_rssi(self, tags):
        return self.a_dbm - 10 * self.n * np.log10(tag_distance(self.xy, self.tags[tags]))
