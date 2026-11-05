"""Simulate the site survey an installer would do in the empty building.

The result is all the filter may know about radio: a fitted path-loss model for BLE
and a coarse Wi-Fi fingerprint map. It never sees the Sionna maps directly.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import sensors as se  # noqa: E402
from indoor_loc import wall_model as wm  # noqa: E402

SPACING_M = 3.0
READINGS_PER_POINT = 8
READING_SD_DB = 5.0
LAYOUT = "E_B_plus_extras"
FAR_DBM = -95.0


def survey_points():
    xs = np.arange(SPACING_M / 2, fp.WIDTH, SPACING_M)
    ys = np.arange(SPACING_M / 2, fp.DEPTH, SPACING_M)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    return xs, ys, grid


def measured(field, points, rng):
    """Average of a few noisy readings at each point."""
    values = field.mean_at(points) + field.shadow_at(points)
    return values + rng.normal(0, READING_SD_DB / np.sqrt(READINGS_PER_POINT), values.shape)


def fit_path_loss(field, points, readings, use_zones=True):
    """Fit RSSI = P0 - 10 n log10(d) - L * (different zone); L is 0 without zones."""
    point_zones = np.array([fp.zone_of(x, y) for x, y in points])
    tag_zones = np.array([fp.zone_of(x, y) for x, y in field.positions[:, :2]])
    rows, targets = [], []
    for t, tag in enumerate(field.positions):
        d = np.sqrt(np.sum((points - tag[:2]) ** 2, axis=1) + (tag[2] - se.PHONE_HEIGHT_M) ** 2)
        heard = readings[t] >= field.config.sensitivity_dbm
        crossing = -(point_zones[heard] != tag_zones[t]).astype(float)
        columns = [np.ones(heard.sum()), -10 * np.log10(d[heard])]
        rows.append(np.column_stack(columns + [crossing] if use_zones else columns))
        targets.append(readings[t, heard])
    design, target = np.concatenate(rows), np.concatenate(targets)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return coefficients, residual.std()


def fit_multi_wall(field, points, readings):
    """Fit RSSI = A - 10 n log10(d) - sum(N_i L_i), keeping n and every L_i at or above 0."""
    pieces = wm.wall_pieces()
    rows, targets = [], []
    for t, tag in enumerate(field.positions):
        d = np.sqrt(np.sum((points - tag[:2]) ** 2, axis=1) + (tag[2] - se.PHONE_HEIGHT_M) ** 2)
        heard = readings[t] >= field.config.sensitivity_dbm
        crossings = wm.crossing_counts(points[heard], tag, pieces).astype(float)
        rows.append(np.column_stack([np.ones(heard.sum()), -10 * np.log10(d[heard]), -crossings]))
        targets.append(readings[t, heard])
    design, target = np.concatenate(rows), np.concatenate(targets)
    lower = [-np.inf] + [0.0] * (design.shape[1] - 1)
    solution = lsq_linear(design, target, bounds=(lower, np.inf)).x
    residual = target - design @ solution
    return solution, residual.std()


def main():
    rng = np.random.default_rng(99)
    maps = ROOT / "data" / "radio_maps"
    xs, ys, points = survey_points()

    ble = se.RadioField(maps / f"{LAYOUT}__ble.npz", se.BLE)
    ble_readings = measured(ble, points, rng)
    (p0, n, wall), spread = fit_path_loss(ble, points, ble_readings)
    (p0_plain, n_plain), spread_plain = fit_path_loss(ble, points, ble_readings, use_zones=False)

    multi, spread_multi = fit_multi_wall(ble, points, ble_readings)
    grid = wm.count_grid(ble.positions)

    wifi = []
    for band in ("wifi_2g4", "wifi_5g"):
        field = se.RadioField(maps / f"wifi__{band}.npz", se.WIFI)
        wifi.append(np.maximum(measured(field, points, rng), FAR_DBM))
    wifi = np.concatenate(wifi).reshape(-1, len(ys), len(xs)).astype(np.float32)

    path = ROOT / "data" / "survey.npz"
    np.savez_compressed(
        path, p0=p0, n=n, wall_loss=wall, p0_plain=p0_plain, n_plain=n_plain,
        multi_a=multi[0], multi_n=multi[1], multi_losses=multi[2:], wall_counts=grid,
        xs=xs, ys=ys, wifi_map=wifi, tags=ble.positions,
    )
    print(f"BLE model: P0 {p0:.1f} dBm, n {n:.2f}, zone change {wall:.1f} dB, fit error {spread:.1f} dB")
    print(f"Without zones: P0 {p0_plain:.1f} dBm, n {n_plain:.2f}, fit error {spread_plain:.1f} dB")
    losses = ", ".join(f"{name} {value:.1f} dB" for name, value in zip(wm.TYPES, multi[2:]))
    print(f"Multi-wall: A {multi[0]:.1f} dBm, n {multi[1]:.2f}, {losses}, fit error {spread_multi:.1f} dB")
    print(f"Wi-Fi map: {wifi.shape[0]} AP bands on {len(xs)} x {len(ys)} survey points, {SPACING_M:.0f} m apart")


if __name__ == "__main__":
    main()
