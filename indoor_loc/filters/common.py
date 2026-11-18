"""Parts shared by the three BLE particle filters: motion, weighting, resampling and the run loop.

The filters differ only in their measurement model, `expected_rssi`, which each one
defines in its own file. They read the phone log and data/survey.npz, never the true
path or the Sionna maps.

State per particle: x, y, heading, gyro bias and a stride scale.
"""
from dataclasses import dataclass

import numpy as np

from .. import floorplan as fp

CELL = 0.5
PHONE_HEIGHT_M = 1.2
MIN_DISTANCE_M = 0.5


@dataclass(frozen=True)
class FilterConfig:
    particles: int = 3000
    step_scale_sd: float = 0.1  # Spread of each particle's own stride correction.
    step_noise_m: float = 0.08  # Random error added to each step length.
    heading_noise_rad: float = np.deg2rad(3.0)  # Random turn added to each step.
    bias_sd_rad_s: float = 0.003  # Spread of each particle's own gyro drift.
    ble_top_k: int = 10  # Strongest readings used per scan.
    ble_sigma_db: float = 6.0  # Spread of reading minus expected value.
    offset_shrink: float = 0.5  # Share of the common reading offset removed per particle.
    offset_limit_db: float = 8.0  # Cap on that common offset before the share is removed.
    roughen_m: float = 0.15  # Jitter given to copied particles after resampling.
    inject_fraction: float = 0.01  # Share of particles moved near the strongest tags after resampling.
    inject_sd_m: float = 6.0  # How far from those tags they are placed.


def gaussian_logpdf(residual, sigma):
    z = residual / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def systematic_resample(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions).clip(0, len(weights) - 1)


def tag_distance(xy, tags):
    """3D distance from a phone at PHONE_HEIGHT_M to each tag: (particles, tags)."""
    dx = xy[:, None, 0] - tags[:, 0]
    dy = xy[:, None, 1] - tags[:, 1]
    return np.maximum(np.sqrt(dx * dx + dy * dy + (tags[:, 2] - PHONE_HEIGHT_M) ** 2), MIN_DISTANCE_M)


class ParticleFilter:
    """Step-driven motion, wall constraint and BLE weighting. Subclasses give expected_rssi."""

    def __init__(self, survey, config=FilterConfig(), rng=None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.free = fp.occupancy_grid(CELL)
        self.tags = survey["tags"]
        self.xy = None

    def expected_rssi(self, tags):
        """Expected RSSI of each chosen tag at each particle: (particles, tags)."""
        raise NotImplementedError

    def is_free(self, xy):
        ix = np.floor(xy[:, 0] / CELL).astype(int)
        iy = np.floor(xy[:, 1] / CELL).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.free.shape[1]) & (iy < self.free.shape[0])
        result = np.zeros(len(xy), dtype=bool)
        result[inside] = self.free[iy[inside], ix[inside]]
        return result

    @staticmethod
    def strongest(rssi, k):
        heard = np.flatnonzero(~np.isnan(rssi))
        return heard[np.argsort(-rssi[heard])[:k]]

    def _sample_near_tags(self, tags, count):
        points = np.empty((count, 2))
        missing = np.arange(count)
        while len(missing):
            chosen = self.rng.choice(tags, len(missing))
            points[missing] = self.tags[chosen, :2] + self.rng.normal(0, self.config.inject_sd_m, (len(missing), 2))
            missing = missing[~self.is_free(points[missing])]
        return points

    def initialise(self, rssi):
        c = self.config
        self.xy = self._sample_near_tags(self.strongest(rssi, 3), c.particles)
        self.heading = self.rng.uniform(-np.pi, np.pi, c.particles)
        self.bias = self.rng.normal(0, c.bias_sd_rad_s, c.particles)
        self.scale = np.maximum(0.5, self.rng.normal(1.0, c.step_scale_sd, c.particles))
        self.weights = np.full(c.particles, 1.0 / c.particles)

    def predict_step(self, length, turn, elapsed):
        """Move every particle by one detected step; particles that hit a wall are dropped."""
        c = self.config
        n = len(self.xy)
        self.heading = self.heading + turn - self.bias * elapsed + self.rng.normal(0, c.heading_noise_rad, n)
        stride = self.scale * length + self.rng.normal(0, c.step_noise_m, n)
        move = np.column_stack([stride * np.cos(self.heading), stride * np.sin(self.heading)])
        target = self.xy + move
        blocked = ~(self.is_free(target) & self.is_free(self.xy + move / 2))
        target[blocked] = self.xy[blocked]
        self.weights[blocked] = 0.0
        self._normalise()
        self.xy = target

    def _normalise(self):
        total = self.weights.sum()
        if total <= 0 or not np.isfinite(total):
            self.weights[:] = 1.0 / len(self.weights)
        else:
            self.weights /= total

    def log_likelihood(self, residual):
        """Gaussian log-likelihood per particle after removing part of the common offset."""
        c = self.config
        offset = np.clip(residual.mean(axis=1, keepdims=True), -c.offset_limit_db, c.offset_limit_db)
        return gaussian_logpdf(residual - c.offset_shrink * offset, c.ble_sigma_db).sum(axis=1)

    def update(self, rssi):
        tags = self.strongest(rssi, self.config.ble_top_k)
        if len(tags) == 0:
            return
        log_like = self.log_likelihood(rssi[tags] - self.expected_rssi(tags))
        alive = self.weights > 0
        peak = log_like[alive].max() if alive.any() else log_like.max()
        self.weights = self.weights * np.exp(np.clip(log_like - peak, -300, 0))
        self._normalise()

    def resample_if_needed(self, rssi):
        c = self.config
        if 1.0 / np.sum(self.weights**2) >= len(self.weights) / 2:
            return
        index = systematic_resample(self.weights, self.rng)
        self.xy, self.heading = self.xy[index].copy(), self.heading[index].copy()
        self.bias, self.scale = self.bias[index].copy(), self.scale[index].copy()
        self.weights = np.full(len(index), 1.0 / len(index))
        jitter = self.xy + self.rng.normal(0, c.roughen_m, self.xy.shape)
        ok = self.is_free(jitter)
        self.xy[ok] = jitter[ok]
        count = int(c.inject_fraction * len(index))
        top = self.strongest(rssi, 3)
        if count and len(top):
            pick = self.rng.choice(len(index), count, replace=False)
            self.xy[pick] = self._sample_near_tags(top, count)
            self.heading[pick] = self.rng.uniform(-np.pi, np.pi, count)

    def estimate(self):
        position = np.average(self.xy, axis=0, weights=self.weights)
        heading = np.arctan2(np.sum(self.weights * np.sin(self.heading)), np.sum(self.weights * np.cos(self.heading)))
        spread = np.sqrt(np.average(np.sum((self.xy - position) ** 2, axis=1), weights=self.weights))
        return position, heading, spread


def run(log, pf, cloud_size=0):
    """Replay one phone log through a filter. Returns time, position, heading, spread and a cloud."""
    ble_time, ble = log["ble_time"], log["ble_rssi"]
    step_time, step_length, step_turn = log["step_time"], log["step_length"], log["step_turn"]

    pf.initialise(ble[0])
    step_index, last_step_time = 0, 0.0
    positions, headings, spreads, clouds = [], [], [], []
    for n, now in enumerate(ble_time):
        while step_index < len(step_time) and step_time[step_index] <= now:
            pf.predict_step(step_length[step_index], step_turn[step_index], step_time[step_index] - last_step_time)
            last_step_time = step_time[step_index]
            step_index += 1
        pf.update(ble[n])
        pf.resample_if_needed(ble[n])
        position, heading, spread = pf.estimate()
        positions.append(position)
        headings.append(heading)
        spreads.append(spread)
        if cloud_size:
            chosen = pf.rng.choice(len(pf.xy), cloud_size, p=pf.weights)
            clouds.append(pf.xy[chosen].astype(np.float32))
    return {
        "time": ble_time,
        "position": np.array(positions),
        "heading": np.array(headings),
        "spread": np.array(spreads),
        "cloud": np.array(clouds) if cloud_size else np.empty((0, 0, 2)),
    }
