"""Compare beacon layouts from the radio maps, before any filter exists.

Accuracy here is the Cramer-Rao bound: the best position error any estimator could
reach from one snapshot of BLE RSSI. A particle filter adds motion, so it can beat it.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import layouts as ly  # noqa: E402
from indoor_loc import sensors  # noqa: E402

TX_POWERS_DBM = (0.0, -12.0)
SENSITIVITY_DBM = -90.0
NOISE_DB = 5.0
SMOOTH_METRES = 2.0
CELL = 0.5
NOISE_TRIALS = 10
HALL = fp.HALL


def rssi_maps(name, tx_power):
    rssi = tx_power + sensors.load_path_gain_db(ROOT / "data" / "radio_maps" / f"{name}__ble.npz")
    return gaussian_filter(rssi, sigma=(0, SMOOTH_METRES / CELL, SMOOTH_METRES / CELL)).astype(np.float32)


def beacon_zones(beacons):
    return np.array([fp.zone_of(b.x, b.y) for b in beacons])


def position_bound(rssi):
    heard = rssi >= SENSITIVITY_DBM
    dy, dx = np.gradient(rssi, CELL, axis=(1, 2))
    weight = heard / NOISE_DB**2
    a = np.sum(weight * dx * dx, axis=0)
    b = np.sum(weight * dx * dy, axis=0)
    c = np.sum(weight * dy * dy, axis=0)
    det = a * c - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.sqrt((a + c) / det)
    bound[~np.isfinite(bound) | (det <= 0)] = 100.0
    return bound, heard.sum(axis=0)


def zone_rates(rssi, beacons, zones, free, rng):
    """How often the strongest noisy beacon is in the person's zone: (rooms %, hall %)."""
    tx_zone = beacon_zones(beacons)
    cells = rssi[:, free]
    person = zones[free]
    in_room = person >= 0
    room_hits, hall_hits = 0, 0
    for _ in range(NOISE_TRIALS):
        noisy = cells + rng.normal(0, NOISE_DB, cells.shape).astype(np.float32)
        correct = tx_zone[np.argmax(noisy, axis=0)] == person
        room_hits += int(correct[in_room].sum())
        hall_hits += int(correct[~in_room].sum())
    return (
        100 * room_hits / (NOISE_TRIALS * in_room.sum()),
        100 * hall_hits / (NOISE_TRIALS * (~in_room).sum()),
    )


def main():
    free = fp.occupancy_grid(CELL)
    zones = fp.zone_grid(CELL)
    hall = free & (zones == HALL)
    rooms = free & (zones >= 0)
    rng = np.random.default_rng(7)
    layouts = ly.all_layouts()

    fig, axes = plt.subplots(len(layouts), 1, figsize=(13, 2.7 * len(layouts)), constrained_layout=True)
    print(f"Assumes {SENSITIVITY_DBM:.0f} dBm sensitivity, {NOISE_DB:.0f} dB independent noise, no crowd")
    print(f"{'layout':<22}{'TX':>5}{'beacons':>8}{'heard':>7}{'<3 heard %':>11}"
          f"{'hall m':>8}{'room m':>8}{'p90 m':>7}{'room ok %':>11}{'hall ok %':>11}")
    for name, beacons in layouts.items():
        for tx_power in TX_POWERS_DBM:
            rssi = rssi_maps(name, tx_power)
            bound, heard = position_bound(rssi)
            room_ok, hall_ok = zone_rates(rssi, beacons, zones, free, rng)
            print(
                f"{name:<22}{tx_power:>5.0f}{len(beacons):>8}{np.median(heard[free]):>7.0f}"
                f"{100 * np.mean(heard[free] < 3):>11.1f}"
                f"{np.median(bound[hall]):>8.2f}{np.median(bound[rooms]):>8.2f}"
                f"{np.percentile(bound[free], 90):>7.2f}{room_ok:>11.1f}{hall_ok:>11.1f}",
                flush=True,
            )
            if tx_power == TX_POWERS_DBM[-1]:
                ax = axes[list(layouts).index(name)]
                shown = np.where(free, bound, np.nan)
                image = ax.imshow(shown, origin="lower", extent=(0, fp.WIDTH, 0, fp.DEPTH), vmin=0, vmax=6)
                ax.scatter([b.x for b in beacons], [b.y for b in beacons], c="white", edgecolors="black", s=12, lw=0.4)
                ax.set_title(f"{name}: best possible error from one RSSI snapshot at {tx_power:.0f} dBm (m)", loc="left", fontsize=11)
                ax.set_ylabel("y (m)")
                fig.colorbar(image, ax=ax, shrink=0.8)
    axes[-1].set_xlabel("x (m)")
    out = ROOT / "outputs" / "layout_accuracy.png"
    fig.savefig(out, dpi=120)

    rssi = rssi_maps("D_uneven", TX_POWERS_DBM[-1])
    beacons = layouts["D_uneven"]
    choices = [("Room beacon", 33 + 16 + 12), ("Concourse pillar", 8)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    for ax, (label, index) in zip(axes, choices):
        image = ax.imshow(rssi[index], origin="lower", extent=(0, fp.WIDTH, 0, fp.DEPTH), vmin=-100, vmax=-40, cmap="magma")
        ax.scatter([beacons[index].x], [beacons[index].y], c="cyan", edgecolors="black", s=45)
        for wall in fp.build_walls():
            ax.plot([wall.x0, wall.x1], [wall.y0, wall.y1], color="white", lw=0.5, alpha=0.7)
        ax.set_title(f"{label} {beacons[index].id}: RSSI in dBm at phone height", loc="left", fontsize=11)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(ROOT / "outputs" / "rssi_examples.png", dpi=120)
    print("Wrote outputs/layout_accuracy.png and outputs/rssi_examples.png")


if __name__ == "__main__":
    main()
