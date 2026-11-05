"""Turn a true path into phone readings.

The Sionna maps are only used here. The particle filter must never read them, so it
always works with a simpler model than the world that produced its measurements.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

MAP_CELL = 0.5
FIELD_CELL = 2.0
BUILDING_SEED = 2026


@dataclass(frozen=True)
class RadioConfig:
    tx_power_dbm: float
    sensitivity_dbm: float
    period_s: float
    receive_probability: float
    rician_k_db: float = 6.0
    shadow_sd_db: float = 3.0
    shadow_corr_m: float = 3.0
    measurement_sd_db: float = 1.5
    device_offset_sd_db: float = 2.0
    body_loss_max_db: float = 10.0
    person_loss_db: float = 5.0
    crowd_loss_cap_db: float = 15.0
    deep_fade_probability: float = 0.03
    deep_fade_min_db: float = 10.0
    deep_fade_max_db: float = 20.0


BLE = RadioConfig(tx_power_dbm=-12.0, sensitivity_dbm=-95.0, period_s=1.0, receive_probability=0.85)
WIFI = RadioConfig(tx_power_dbm=15.0, sensitivity_dbm=-90.0, period_s=30.0, receive_probability=0.95)

PERSON_RADIUS_M = 0.3
PERSON_HEIGHT_M = 1.75
PHONE_HEIGHT_M = 1.2
BODY_BLOCK_ANGLE = np.deg2rad(120.0)


@dataclass(frozen=True)
class ImuConfig:
    stride_m: float = 0.7
    stride_person_sd: float = 0.05
    step_length_sd: float = 0.1
    heading_step_sd_rad: float = np.deg2rad(1.0)
    gyro_bias_sd_rad_s: float = 0.002
    missed_step_probability: float = 0.02


def load_path_gain_db(path):
    """Path gain per transmitter in dB. Cells no ray reached, such as inside pillars,
    take their nearest neighbour's value so smoothing cannot create false slopes."""
    gain = np.load(path)["path_gain"].copy()
    for i in range(len(gain)):
        holes = gain[i] <= 0
        if holes.any():
            nearest = distance_transform_edt(holes, return_distances=False, return_indices=True)
            gain[i] = gain[i][tuple(nearest)]
    return 10 * np.log10(gain)


class RadioField:
    """Mean RSSI and fixed shadowing for every transmitter in one map file."""

    def __init__(self, path, config, building_seed=BUILDING_SEED):
        """The shadowing belongs to the building, so every run shares building_seed."""
        data = np.load(path)
        rng = np.random.default_rng(building_seed)
        self.config = config
        self.ids = data["ids"]
        self.positions = data["positions"]
        self.mean_db = (config.tx_power_dbm + load_path_gain_db(path)).astype(np.float32)
        ny, nx = self.mean_db.shape[1:]
        shape = (len(self.ids), int(ny * MAP_CELL / FIELD_CELL) + 2, int(nx * MAP_CELL / FIELD_CELL) + 2)
        field = rng.normal(size=shape)
        sigma = config.shadow_corr_m / FIELD_CELL
        field = gaussian_filter(field, sigma=(0, sigma, sigma))
        field /= field.std(axis=(1, 2), keepdims=True)
        self.shadow_db = (field * config.shadow_sd_db).astype(np.float32)

    def mean_at(self, xy):
        """Deterministic RSSI for each transmitter at the points xy: shape (tx, points)."""
        coords = np.array([xy[:, 1] / MAP_CELL - 0.5, xy[:, 0] / MAP_CELL - 0.5])
        return np.stack([map_coordinates(m, coords, order=1, mode="nearest") for m in self.mean_db])

    def shadow_at(self, xy):
        coords = np.array([xy[:, 1] / FIELD_CELL, xy[:, 0] / FIELD_CELL])
        return np.stack([map_coordinates(s, coords, order=1, mode="nearest") for s in self.shadow_db])


def rician_gain_db(rng, k_db, shape):
    k = 10 ** (k_db / 10)
    line_of_sight = np.sqrt(k / (k + 1))
    scatter = np.sqrt(1 / (2 * (k + 1)))
    h = line_of_sight + scatter * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    return 20 * np.log10(np.maximum(np.abs(h), 1e-6))


def body_loss_db(xy, heading, tx_xy, max_loss):
    """Own-body loss: largest when the transmitter is directly behind the person."""
    to_tx = np.arctan2(tx_xy[:, 1] - xy[1], tx_xy[:, 0] - xy[0])
    angle = np.abs(np.angle(np.exp(1j * (to_tx - heading))))
    scale = np.clip((angle - BODY_BLOCK_ANGLE) / (np.pi - BODY_BLOCK_ANGLE), 0, 1)
    return max_loss * scale


def crowd_blockers(xy, tx_positions, people):
    """People standing on the phone-to-transmitter line, below head height there."""
    if len(people) == 0:
        return np.zeros(len(tx_positions))
    to_tx = tx_positions[:, :2] - xy
    length = np.maximum(np.hypot(to_tx[:, 0], to_tx[:, 1]), 1e-6)
    unit = to_tx / length[:, None]
    relative = people - xy
    along = relative @ unit.T
    across = np.abs(relative[:, 0, None] * unit[:, 1] - relative[:, 1, None] * unit[:, 0])
    height = PHONE_HEIGHT_M + (tx_positions[:, 2] - PHONE_HEIGHT_M) * along / length
    blocks = (across < PERSON_RADIUS_M) & (along > PERSON_RADIUS_M) & (along < length) & (height < PERSON_HEIGHT_M)
    return blocks.sum(axis=0)


def simulate_radio(field, truth, heading, crowd, dt, rng):
    """One scan per period. Returns (times, rssi); rssi is NaN when not received."""
    config = field.config
    stride = int(round(config.period_s / dt))
    indices = np.arange(stride, len(truth), stride)
    xy = truth[indices]
    mean = field.mean_at(xy) + field.shadow_at(xy)
    tx_xy = field.positions[:, :2]
    device_offset = rng.normal(0, config.device_offset_sd_db)

    rssi = np.full((len(indices), len(tx_xy)), np.nan, dtype=np.float32)
    for n, i in enumerate(indices):
        loss = body_loss_db(truth[i], heading[i], tx_xy, config.body_loss_max_db)
        people = crowd[i] if crowd is not None else np.empty((0, 2))
        blockers = crowd_blockers(truth[i], field.positions, people)
        loss = loss + np.minimum(blockers * config.person_loss_db, config.crowd_loss_cap_db)
        deep = rng.random(len(tx_xy)) < config.deep_fade_probability
        loss = loss + deep * rng.uniform(config.deep_fade_min_db, config.deep_fade_max_db, len(tx_xy))
        value = (
            mean[:, n]
            - loss
            + rician_gain_db(rng, config.rician_k_db, len(tx_xy))
            + rng.normal(0, config.measurement_sd_db, len(tx_xy))
            + device_offset
        )
        received = (value >= config.sensitivity_dbm) & (rng.random(len(tx_xy)) < config.receive_probability)
        rssi[n, received] = np.round(value[received])
    return indices * dt, rssi


def simulate_steps(truth, heading, dt, rng, config=ImuConfig()):
    """Pedestrian dead-reckoning input: one (time, length, heading change) per step."""
    moved = np.concatenate([[0.0], np.hypot(*np.diff(truth, axis=0).T)])
    distance = np.cumsum(moved)
    stride = max(0.4, rng.normal(config.stride_m, config.stride_person_sd))
    marks = np.arange(stride, distance[-1], stride)
    indices = np.searchsorted(distance, marks)
    bias = rng.normal(0, config.gyro_bias_sd_rad_s)

    last_time, last_heading = 0.0, heading[0]
    times, lengths, turns = [], [], []
    for i in indices:
        change = np.angle(np.exp(1j * (heading[i] - last_heading)))
        elapsed = i * dt - last_time
        last_time, last_heading = i * dt, heading[i]
        turn = change + bias * elapsed + rng.normal(0, config.heading_step_sd_rad)
        if rng.random() < config.missed_step_probability:
            continue
        times.append(i * dt)
        lengths.append(stride + rng.normal(0, config.step_length_sd))
        turns.append(turn)
    return np.array(times), np.array(lengths), np.array(turns), stride, bias
