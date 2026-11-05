"""Checks the filter's building blocks and that it can track a simple walk."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import particle_filter as pf  # noqa: E402


def small_survey():
    xs, ys = np.arange(5.0, 250.0, 10.0), np.arange(5.0, 60.0, 10.0)
    tags = np.array([[x, y, 3.0] for x in xs for y in ys], dtype=np.float32)
    return {
        "p0": -55.0, "n": 2.0, "wall_loss": 0.0, "xs": xs, "ys": ys,
        "wifi_map": np.zeros((1, len(ys), len(xs)), dtype=np.float32), "tags": tags,
    }


def test_student_keeps_outliers_alive_and_gaussian_does_not():
    gap_student = pf.student_logpdf(0.0, 5.0, 3.0) - pf.student_logpdf(20.0, 5.0, 3.0)
    gap_gaussian = pf.gaussian_logpdf(0.0, 5.0) - pf.gaussian_logpdf(20.0, 5.0)
    assert gap_student < gap_gaussian / 2


def test_resampling_follows_the_weights():
    weights = np.array([0.0, 0.0, 1.0, 0.0])
    index = pf.systematic_resample(weights, np.random.default_rng(0))
    assert (index == 2).all()


def test_particles_cannot_cross_a_wall():
    filter_ = pf.ParticleFilter(small_survey(), pf.FilterConfig(particles=100), np.random.default_rng(0))
    filter_.xy = np.full((100, 2), [20.5, 10.0]) - [1.0, 0.0]  # Just west of a room wall at x = 20.
    filter_.heading = np.zeros(100)
    filter_.bias = np.zeros(100)
    filter_.scale = np.ones(100)
    filter_.weights = np.full(100, 0.01)
    before = filter_.xy.copy()
    filter_.predict_step(0.9, 0.0, 0.5)
    crossed = (filter_.xy[:, 0] > 20.0) & (filter_.weights > 0)
    assert not crossed.any()
    assert np.all(filter_.weights >= 0) and before.shape == filter_.xy.shape


def test_filter_tracks_a_straight_walk():
    survey = small_survey()
    rng = np.random.default_rng(1)
    times = np.arange(1.0, 121.0)
    x = 40.0 + 1.2 * times
    truth = np.column_stack([x, np.full_like(x, 30.0)])
    d = np.sqrt(np.sum((truth[:, None] - survey["tags"][:, :2]) ** 2, axis=2) + 1.8**2)
    rssi = -55.0 - 20 * np.log10(d) + rng.normal(0, 4.0, d.shape)
    rssi[rssi < -85] = np.nan
    step_time = np.arange(0.6, 120.0, 0.6)
    log = {
        "ble_time": times, "ble_rssi": rssi,
        "wifi_time": np.array([]), "wifi_2g4_rssi": np.empty((0, 1)), "wifi_5g_rssi": np.empty((0, 1)),
        "step_time": step_time, "step_length": np.full(len(step_time), 0.72), "step_turn": np.zeros(len(step_time)),
    }
    result = pf.run(log, survey, pf.FilterConfig(particles=1500), seed=2)
    error = np.hypot(*(result["position"] - truth).T)
    assert np.median(error[30:]) < 3.0
