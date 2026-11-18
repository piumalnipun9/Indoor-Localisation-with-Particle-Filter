"""Checks the filters' shared parts, each measurement model, and that they track a simple walk."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor_loc import filters  # noqa: E402
from indoor_loc.filters import common  # noqa: E402


def small_survey():
    xs = np.array([1.5, 4.5, 7.5, 10.5, 13.5])
    tags = np.array([[x, y, 3.0] for x in np.arange(5.0, 250.0, 10.0) for y in np.arange(5.0, 60.0, 10.0)], dtype=np.float32)
    return {"p0_plain": -55.0, "n_plain": 2.0, "xs": xs, "tags": tags}


def test_gaussian_log_density():
    assert np.isclose(common.gaussian_logpdf(0.0, 2.0), -np.log(2.0) - 0.5 * np.log(2 * np.pi))
    assert np.isclose(common.gaussian_logpdf(4.0, 2.0) - common.gaussian_logpdf(0.0, 2.0), -2.0)


def test_resampling_follows_the_weights():
    index = common.systematic_resample(np.array([0.0, 0.0, 1.0, 0.0]), np.random.default_rng(0))
    assert (index == 2).all()


def test_offset_removal_is_partial_and_capped():
    pf = filters.PlainFilter(small_survey(), filters.FilterConfig(ble_sigma_db=1.0))
    shared = pf.log_likelihood(np.full((1, 4), -4.0))  # Phone 4 dB low: 2 dB removed.
    capped = pf.log_likelihood(np.full((1, 4), -20.0))  # Capped at 8 dB, so only 4 dB removed.
    assert np.isclose(shared[0], 4 * common.gaussian_logpdf(-2.0, 1.0))
    assert np.isclose(capped[0], 4 * common.gaussian_logpdf(-16.0, 1.0))


def test_particles_cannot_cross_a_wall():
    pf = filters.PlainFilter(small_survey(), filters.FilterConfig(particles=100), np.random.default_rng(0))
    pf.xy = np.full((100, 2), [19.5, 10.0])  # Just west of a room wall at x = 20.
    pf.heading, pf.bias, pf.scale = np.zeros(100), np.zeros(100), np.ones(100)
    pf.weights = np.full(100, 0.01)
    pf.predict_step(0.9, 0.0, 0.5)
    assert not ((pf.xy[:, 0] > 20.0) & (pf.weights > 0)).any()


def test_plain_model_falls_with_distance():
    pf = filters.PlainFilter(small_survey())
    pf.xy = np.array([[5.0, 5.0], [15.0, 5.0]])
    expected = pf.expected_rssi(np.array([0]))[:, 0]
    d = np.array([1.8, np.hypot(10.0, 1.8)])
    assert np.allclose(expected, -55.0 - 20 * np.log10(d))


def test_multi_wall_subtracts_the_wall_loss():
    survey = small_survey()
    survey["multi_a"], survey["multi_n"] = -55.0, 2.0
    survey["multi_losses"] = np.array([10.0])
    survey["wall_counts"] = np.zeros((len(survey["tags"]), 60, 250, 1), dtype=np.uint8)
    survey["wall_counts"][0, :, 100:] = 1  # Tag 0 is behind one wall for particles past x = 100.
    pf = filters.MultiWallFilter(survey)
    pf.xy = np.array([[50.0, 5.0], [150.0, 5.0]])
    plain = filters.PlainFilter(survey)
    plain.xy = pf.xy
    assert np.allclose(pf.expected_rssi(np.array([0]))[:, 0] - plain.expected_rssi(np.array([0]))[:, 0], [0.0, -10.0])


def test_fingerprint_reads_between_survey_points():
    survey = small_survey()
    grid = np.tile(survey["xs"], (5, 1))  # RSSI equals x, so a lookup returns the position.
    survey["ble_map"] = np.stack([grid, grid + 10.0]).astype(np.float32)
    pf = filters.FingerprintFilter(survey, map_spacing_m=3.0)
    pf.xy = np.array([[3.0, 3.0], [12.0, 9.0]])
    assert np.allclose(pf.expected_rssi(np.array([0, 1])), [[3.0, 13.0], [12.0, 22.0]])
    pf = filters.FingerprintFilter(survey, map_spacing_m=6.0)
    pf.xy = np.array([[3.0, 3.0], [12.0, 9.0]])
    assert np.allclose(pf.expected_rssi(np.array([0])), [[3.0], [12.0]])


def test_fingerprint_rejects_a_spacing_off_the_survey_grid():
    survey = small_survey()
    survey["ble_map"] = np.zeros((1, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        filters.FingerprintFilter(survey, map_spacing_m=5.0)


def test_filter_tracks_a_straight_walk():
    survey = small_survey()
    rng = np.random.default_rng(1)
    times = np.arange(1.0, 121.0)
    truth = np.column_stack([40.0 + 1.2 * times, np.full_like(times, 30.0)])
    d = np.sqrt(np.sum((truth[:, None] - survey["tags"][:, :2]) ** 2, axis=2) + 1.8**2)
    rssi = -55.0 - 20 * np.log10(d) + rng.normal(0, 4.0, d.shape)
    rssi[rssi < -85] = np.nan
    step_time = np.arange(0.6, 120.0, 0.6)
    log = {
        "ble_time": times, "ble_rssi": rssi,
        "step_time": step_time, "step_length": np.full(len(step_time), 0.72), "step_turn": np.zeros(len(step_time)),
    }
    pf = filters.PlainFilter(survey, filters.FilterConfig(particles=1500), np.random.default_rng(2))
    result = filters.run(log, pf)
    error = np.hypot(*(result["position"] - truth).T)
    assert np.median(error[30:]) < 3.0
