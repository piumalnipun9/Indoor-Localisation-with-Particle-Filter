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
    filter_ = pf.ParticleFilter(small_survey(), pf.FilterConfig(particles=100, ble_model="zone"), np.random.default_rng(0))
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
    result = pf.run(log, survey, pf.FilterConfig(particles=1500, ble_model="zone"), seed=2)
    error = np.hypot(*(result["position"] - truth).T)
    assert np.median(error[30:]) < 3.0


def wall_filter(pick, top_k=1):
    survey = small_survey()
    survey["tags"] = np.array([[10.0, 30.0, 3.0], [12.0, 30.0, 3.0]], dtype=np.float32)
    survey["multi_a"], survey["multi_n"] = -60.0, 1.5
    survey["multi_losses"] = np.array([10.0])
    survey["wall_counts"] = np.zeros((2, 60, 250, 1), dtype=np.uint8)
    survey["wall_counts"][1, :, 100:] = 1  # Tag 1 is behind a wall for particles past x = 100.
    filter_ = pf.ParticleFilter(survey, pf.FilterConfig(particles=2, ble_model="multi_wall", ble_top_k=top_k, ble_pick=pick), np.random.default_rng(0))
    filter_.xy = np.array([[50.0, 30.0], [150.0, 30.0]])
    filter_.weights = np.array([0.5, 0.5])
    return filter_


def test_wall_loss_changes_which_tag_is_strongest():
    rssi = np.array([-80.0, -82.0])  # Tag 0 reads stronger, tag 1 is stronger once its wall is added back.
    filter_ = wall_filter("wall_each")
    residual = rssi[None, :] - filter_.expected_ble(np.array([0, 1]))
    chosen = filter_._each_particle_residuals(rssi, np.array([0, 1]))
    assert chosen[0, 0] == residual[0, 0]
    assert chosen[1, 0] == residual[1, 1]


def test_every_particle_shares_the_tags_with_the_averaged_correction():
    filter_ = wall_filter("wall_shared")
    seen = []
    filter_.expected_ble = lambda tags: seen.append(tags.copy()) or np.zeros((2, len(tags)))
    filter_.update_ble(np.array([-80.0, -82.0]))  # Average wall loss on tag 1 is 5 dB, so it wins.
    assert seen[-1].tolist() == [1]


def test_fingerprint_reads_between_survey_points():
    survey = small_survey()
    survey["xs"] = np.array([1.5, 4.5, 7.5, 10.5, 13.5])
    grid = np.tile(survey["xs"], (5, 1))  # RSSI equals x, so a lookup returns the position.
    survey["ble_map"] = np.stack([grid, grid + 10.0]).astype(np.float32)
    config = pf.FilterConfig(particles=2, ble_model="fingerprint")
    filter_ = pf.ParticleFilter(survey, config, np.random.default_rng(0))
    filter_.xy = np.array([[3.0, 3.0], [12.0, 9.0]])
    assert np.allclose(filter_.expected_ble(np.array([0, 1])), [[3.0, 13.0], [12.0, 22.0]])
    filter_ = pf.ParticleFilter(survey, pf.FilterConfig(particles=2, ble_model="fingerprint", map_every=2), np.random.default_rng(0))
    filter_.xy = np.array([[3.0, 3.0], [12.0, 9.0]])
    assert np.allclose(filter_.expected_ble(np.array([0])), [[3.0], [12.0]])
