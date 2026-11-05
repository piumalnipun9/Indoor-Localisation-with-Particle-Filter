"""Particle filter for position and heading from steps, BLE, Wi-Fi and the floor plan.

It only reads the phone log and data/survey.npz, never the true path or Sionna maps.
State per particle: x, y, heading, gyro bias and a stride scale.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import gammaln

from . import floorplan as fp

CELL = 0.5
PHONE_HEIGHT_M = 1.2


@dataclass(frozen=True)
class FilterConfig:
    particles: int = 3000
    nominal_step_m: float = 0.7
    step_scale_sd: float = 0.1
    step_noise_m: float = 0.08
    heading_noise_rad: float = np.deg2rad(3.0)
    bias_sd_rad_s: float = 0.003
    ble_top_k: int = 10
    ble_sigma_db: float = 6.0
    wifi_top_k: int = 8
    wifi_sigma_db: float = 7.0
    student_dof: float = 3.0
    likelihood: str = "student"  # "student" or "gaussian"
    tempering: float = 0.5  # Readings share errors, so each counts for less than one.
    offset_shrink: float = 0.5
    offset_limit_db: float = 8.0
    roughen_m: float = 0.15
    inject_fraction: float = 0.01
    inject_sd_m: float = 6.0
    use_walls: bool = True
    use_wifi: bool = False  # BLE only until its error is reasonable.
    use_steps: bool = True
    idle_noise_m: float = 0.2  # Spread per scan when no steps are available.
    ble_model: str = "multi_wall"  # Or "zone", "zone_off" and "plain" for comparison.


def student_logpdf(residual, sigma, dof):
    z = residual / sigma
    return (
        gammaln((dof + 1) / 2) - gammaln(dof / 2) - 0.5 * np.log(dof * np.pi) - np.log(sigma)
        - (dof + 1) / 2 * np.log1p(z * z / dof)
    )


def gaussian_logpdf(residual, sigma):
    z = residual / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def systematic_resample(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions).clip(0, len(weights) - 1)


class ParticleFilter:
    def __init__(self, survey, config, rng):
        self.config = config
        self.rng = rng
        self.free = fp.occupancy_grid(CELL)
        self.zones = fp.zone_grid(CELL)
        self.tags = survey["tags"]
        self.tag_zones = np.array([fp.zone_of(x, y) for x, y in self.tags[:, :2]])
        self.p0, self.n, self.wall_loss = float(survey["p0"]), float(survey["n"]), float(survey["wall_loss"])
        if config.ble_model != "zone":
            self.wall_loss = 0.0
        if config.ble_model == "plain":
            self.p0, self.n = float(survey["p0_plain"]), float(survey["n_plain"])
        self.wall_db = None
        if config.ble_model == "multi_wall":
            self.p0, self.n = float(survey["multi_a"]), float(survey["multi_n"])
            self.wall_db = (survey["wall_counts"] @ survey["multi_losses"]).astype(np.float32)
        self.wifi_map = survey["wifi_map"]
        self.wifi_step = float(survey["xs"][1] - survey["xs"][0])
        self.wifi_origin = float(survey["xs"][0])
        self.xy = None

    def is_free(self, xy):
        ix = np.floor(xy[:, 0] / CELL).astype(int)
        iy = np.floor(xy[:, 1] / CELL).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < self.free.shape[1]) & (iy < self.free.shape[0])
        result = np.zeros(len(xy), dtype=bool)
        result[inside] = self.free[iy[inside], ix[inside]]
        return result

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
        top = self._strongest(rssi, 3)
        self.xy = self._sample_near_tags(top, c.particles)
        self.heading = self.rng.uniform(-np.pi, np.pi, c.particles)
        self.bias = self.rng.normal(0, c.bias_sd_rad_s, c.particles)
        self.scale = np.maximum(0.5, self.rng.normal(1.0, c.step_scale_sd, c.particles))
        self.weights = np.full(c.particles, 1.0 / c.particles)

    @staticmethod
    def _strongest(rssi, k):
        heard = np.flatnonzero(~np.isnan(rssi))
        return heard[np.argsort(-rssi[heard])[:k]]

    def predict_step(self, length, turn, elapsed):
        c = self.config
        n = len(self.xy)
        self.heading = self.heading + turn - self.bias * elapsed + self.rng.normal(0, c.heading_noise_rad, n)
        stride = c.nominal_step_m * self.scale * (length / c.nominal_step_m) + self.rng.normal(0, c.step_noise_m, n)
        move = np.column_stack([stride * np.cos(self.heading), stride * np.sin(self.heading)])
        target = self.xy + move
        if c.use_walls:
            blocked = ~(self.is_free(target) & self.is_free(self.xy + move / 2))
            target[blocked] = self.xy[blocked]
            self.weights[blocked] = 0.0
            self._normalise()
        self.xy = target

    def spread_without_steps(self):
        c = self.config
        moved = self.xy + self.rng.normal(0, c.idle_noise_m, self.xy.shape)
        if c.use_walls:
            keep = self.is_free(moved)
            moved[~keep] = self.xy[~keep]
        self.xy = moved

    def _normalise(self):
        total = self.weights.sum()
        if total <= 0 or not np.isfinite(total):
            self.weights[:] = 1.0 / len(self.weights)
        else:
            self.weights /= total

    def _score(self, residual, sigma):
        c = self.config
        offset = np.clip(residual.mean(axis=1, keepdims=True), -c.offset_limit_db, c.offset_limit_db)
        residual = residual - c.offset_shrink * offset
        if c.likelihood == "student":
            log_like = student_logpdf(residual, sigma, c.student_dof)
        else:
            log_like = gaussian_logpdf(residual, sigma)
        return c.tempering * log_like.sum(axis=1)

    def _apply(self, log_like):
        peak = log_like[self.weights > 0].max() if np.any(self.weights > 0) else log_like.max()
        self.weights = self.weights * np.exp(np.clip(log_like - peak, -300, 0))
        self._normalise()

    def expected_ble(self, tags):
        tag = self.tags[tags]
        dx = self.xy[:, None, 0] - tag[:, 0]
        dy = self.xy[:, None, 1] - tag[:, 1]
        d = np.sqrt(dx * dx + dy * dy + (tag[:, 2] - PHONE_HEIGHT_M) ** 2)
        ix = np.clip((self.xy[:, 0] / CELL).astype(int), 0, self.zones.shape[1] - 1)
        iy = np.clip((self.xy[:, 1] / CELL).astype(int), 0, self.zones.shape[0] - 1)
        different = self.zones[iy, ix][:, None] != self.tag_zones[tags]
        expected = self.p0 - 10 * self.n * np.log10(np.maximum(d, 0.5))
        if self.wall_db is None:
            return expected - self.wall_loss * different
        gy = np.clip(self.xy[:, 1].astype(int), 0, self.wall_db.shape[1] - 1)
        gx = np.clip(self.xy[:, 0].astype(int), 0, self.wall_db.shape[2] - 1)
        return expected - self.wall_db[tags[:, None], gy, gx].T

    def update_ble(self, rssi):
        tags = self._strongest(rssi, self.config.ble_top_k)
        if len(tags) == 0:
            return
        self._apply(self._score(rssi[tags] - self.expected_ble(tags), self.config.ble_sigma_db))

    def update_wifi(self, rssi):
        heard = np.flatnonzero(~np.isnan(rssi))
        if len(heard) == 0:
            return
        aps = heard[np.argsort(-rssi[heard])[: self.config.wifi_top_k]]
        coords = (self.xy[:, ::-1].T - self.wifi_origin) / self.wifi_step
        coords[0] = (self.xy[:, 1] - self.wifi_origin) / self.wifi_step
        coords[1] = (self.xy[:, 0] - self.wifi_origin) / self.wifi_step
        expected = np.stack([map_coordinates(self.wifi_map[a], coords, order=1, mode="nearest") for a in aps], axis=1)
        self._apply(self._score(rssi[aps] - expected, self.config.wifi_sigma_db))

    def resample_if_needed(self, rssi):
        c = self.config
        effective = 1.0 / np.sum(self.weights**2)
        if effective >= len(self.weights) / 2:
            return
        index = systematic_resample(self.weights, self.rng)
        self.xy, self.heading = self.xy[index].copy(), self.heading[index].copy()
        self.bias, self.scale = self.bias[index].copy(), self.scale[index].copy()
        self.weights = np.full(len(index), 1.0 / len(index))
        jitter = self.xy + self.rng.normal(0, c.roughen_m, self.xy.shape)
        ok = self.is_free(jitter) if c.use_walls else np.ones(len(jitter), dtype=bool)
        self.xy[ok] = jitter[ok]
        count = int(c.inject_fraction * len(index))
        top = self._strongest(rssi, 3)
        if count and len(top):
            pick = self.rng.choice(len(index), count, replace=False)
            self.xy[pick] = self._sample_near_tags(top, count)
            self.heading[pick] = self.rng.uniform(-np.pi, np.pi, count)

    def estimate(self):
        position = np.average(self.xy, axis=0, weights=self.weights)
        heading = np.arctan2(np.sum(self.weights * np.sin(self.heading)), np.sum(self.weights * np.cos(self.heading)))
        spread = np.sqrt(np.average(np.sum((self.xy - position) ** 2, axis=1), weights=self.weights))
        return position, heading, spread


def run(log, survey, config=FilterConfig(), seed=0, cloud_size=0):
    """Replay one phone log. Returns time, position, heading, spread and an optional cloud."""
    rng = np.random.default_rng(seed)
    pf = ParticleFilter(survey, config, rng)
    ble_time, ble = log["ble_time"], log["ble_rssi"]
    wifi_time = log["wifi_time"]
    wifi = np.concatenate([log["wifi_2g4_rssi"], log["wifi_5g_rssi"]], axis=1)
    steps = (log["step_time"], log["step_length"], log["step_turn"])

    pf.initialise(ble[0])
    step_index, wifi_index = 0, 0
    last_step_time = 0.0
    positions, headings, spreads, clouds = [], [], [], []
    for n, now in enumerate(ble_time):
        stepped = False
        if config.use_steps:
            while step_index < len(steps[0]) and steps[0][step_index] <= now:
                elapsed = steps[0][step_index] - last_step_time
                pf.predict_step(steps[1][step_index], steps[2][step_index], elapsed)
                last_step_time = steps[0][step_index]
                step_index += 1
                stepped = True
        if not stepped and not config.use_steps:
            pf.spread_without_steps()
        pf.update_ble(ble[n])
        if config.use_wifi and wifi_index < len(wifi_time) and wifi_time[wifi_index] <= now:
            pf.update_wifi(wifi[wifi_index])
            wifi_index += 1
        pf.resample_if_needed(ble[n])
        position, heading, spread = pf.estimate()
        positions.append(position)
        headings.append(heading)
        spreads.append(spread)
        if cloud_size:
            chosen = rng.choice(len(pf.xy), cloud_size, p=pf.weights)
            clouds.append(pf.xy[chosen].astype(np.float32))
    return {
        "time": ble_time,
        "position": np.array(positions),
        "heading": np.array(headings),
        "spread": np.array(spreads),
        "cloud": np.array(clouds) if cloud_size else np.empty((0, 0, 2)),
    }


ALL_TAGS = 1000

VARIANTS = {
    "full": FilterConfig(),
    "gaussian": replace(FilterConfig(), likelihood="gaussian"),
    "no_walls": replace(FilterConfig(), use_walls=False),
    "with_wifi": replace(FilterConfig(), use_wifi=True),
    "no_steps": replace(FilterConfig(), use_steps=False),
    "zone_off": replace(FilterConfig(), ble_model="zone_off"),
    "plain": replace(FilterConfig(), ble_model="plain"),
    "old_zone": replace(FilterConfig(), ble_model="zone"),
    "all_tags": replace(FilterConfig(), ble_top_k=ALL_TAGS),
    "all_tags_soft": replace(FilterConfig(), ble_top_k=ALL_TAGS, tempering=0.5 * 10 / 66),
}
