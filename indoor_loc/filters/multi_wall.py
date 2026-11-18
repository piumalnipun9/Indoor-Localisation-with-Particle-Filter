"""Filter 2: log-distance path loss plus a loss for every wall crossed (Motley-Keenan).

    h_i(p) = A - 10 n log10(d_i(p)) - sum_t N_i,t(p) L_t

N_i,t(p) is the number of walls of type t on the straight line from the phone to tag i.
It comes from the floor plan, worked out in advance on a 1 m grid (wall_counts in
data/survey.npz). A, n and the losses L_t are fitted to the site survey.
"""
import numpy as np

from ..wall_model import GRID_M
from .common import FilterConfig, ParticleFilter, tag_distance


class MultiWallFilter(ParticleFilter):
    def __init__(self, survey, config=FilterConfig(), rng=None):
        super().__init__(survey, config, rng)
        self.a_dbm = float(survey["multi_a"])
        self.n = float(survey["multi_n"])
        self.losses_db = np.asarray(survey["multi_losses"], dtype=float)
        # Total wall loss from every 1 m cell to every tag: (tags, ny, nx).
        self.wall_db = (survey["wall_counts"] @ self.losses_db).astype(np.float32)

    def wall_loss(self, tags):
        gy = np.clip((self.xy[:, 1] / GRID_M).astype(int), 0, self.wall_db.shape[1] - 1)
        gx = np.clip((self.xy[:, 0] / GRID_M).astype(int), 0, self.wall_db.shape[2] - 1)
        return self.wall_db[tags[:, None], gy, gx].T

    def expected_rssi(self, tags):
        free_space = self.a_dbm - 10 * self.n * np.log10(tag_distance(self.xy, self.tags[tags]))
        return free_space - self.wall_loss(tags)
