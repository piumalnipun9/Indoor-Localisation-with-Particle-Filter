"""Build layout E: start from layout B and add the most useful tags one at a time."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_maps import CELL, HALL, NOISE_DB, SENSITIVITY_DBM, beacon_zones, rssi_maps, zone_grid  # noqa: E402
from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import layouts as ly  # noqa: E402

MAX_TAGS = 100
TX_POWER = -12.0
STRIDE = 4  # Score every 2 m; the final check uses every cell.
TRIALS = 4
METRES_PENALTY = 5.0
BOUND_CLIP = 10.0
BASE = "B_even_ceiling_15m"


def information(rssi, mask):
    heard = rssi >= SENSITIVITY_DBM
    dy, dx = np.gradient(rssi, CELL, axis=(1, 2))
    weight = heard / NOISE_DB**2
    return (
        (weight * dx * dx)[:, mask].astype(np.float64),
        (weight * dx * dy)[:, mask].astype(np.float64),
        (weight * dy * dy)[:, mask].astype(np.float64),
    )


def mean_bound(a, b, c):
    det = a * c - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.sqrt((a + c) / det)
    bound[~np.isfinite(bound) | (det <= 0)] = BOUND_CLIP
    return float(np.mean(np.minimum(bound, BOUND_CLIP)))


def main():
    free = fp.occupancy_grid(CELL)
    zones = zone_grid(free.shape)
    mask = np.zeros_like(free)
    mask[::STRIDE, ::STRIDE] = True
    mask &= free
    person = zones[mask]
    in_room = person >= 0

    base_tx = ly.even_ceiling(ly.BASE_SPACING)
    pool = ly.candidate_pool()
    base_rssi = rssi_maps(BASE, TX_POWER)
    pool_rssi = rssi_maps("candidates", TX_POWER)
    base_zone, pool_zone = beacon_zones(base_tx), beacon_zones(pool)

    ia, ib, ic = information(pool_rssi, mask)
    a, b, c = (x.sum(axis=0) for x in information(base_rssi, mask))

    rng = np.random.default_rng(11)
    pool_noisy = pool_rssi[:, mask][None] + rng.normal(0, NOISE_DB, (TRIALS, len(pool), mask.sum())).astype(np.float32)
    base_noisy = base_rssi[:, mask][None] + rng.normal(0, NOISE_DB, (TRIALS, len(base_tx), mask.sum())).astype(np.float32)
    best_value = base_noisy.max(axis=1)
    best_zone = base_zone[base_noisy.argmax(axis=1)]

    def score(values, labels, aa, bb, cc):
        correct = labels == person
        room_ok = 100 * correct[:, in_room].mean()
        hall_ok = 100 * correct[:, ~in_room].mean()
        return (room_ok + hall_ok) / 2 - METRES_PENALTY * mean_bound(aa, bb, cc), room_ok, hall_ok

    remaining = list(range(len(pool)))
    chosen = []
    print(f"start: score {score(best_value, best_zone, a, b, c)[0]:.1f}")
    for step in range(MAX_TAGS - len(base_tx)):
        best = None
        for i in remaining:
            wins = pool_noisy[:, i] > best_value
            labels = np.where(wins, pool_zone[i], best_zone)
            result = score(None, labels, a + ia[i], b + ib[i], c + ic[i])
            if best is None or result[0] > best[0]:
                best = (result[0], i, result[1], result[2])
        i = best[1]
        wins = pool_noisy[:, i] > best_value
        best_value = np.maximum(best_value, pool_noisy[:, i])
        best_zone = np.where(wins, pool_zone[i], best_zone)
        a, b, c = a + ia[i], b + ib[i], c + ic[i]
        remaining.remove(i)
        chosen.append(i)
        if (step + 1) % 8 == 0:
            print(f"+{step + 1} tags: score {best[0]:.1f}, room {best[2]:.1f}%, hall {best[3]:.1f}%", flush=True)

    extras = [{"x": pool[i].x, "y": pool[i].y, "z": pool[i].z} for i in chosen]
    ly.EXTRAS_FILE.write_text(json.dumps(extras, indent=2))

    base_data = np.load(ROOT / "data" / "radio_maps" / f"{BASE}__ble.npz")
    pool_data = np.load(ROOT / "data" / "radio_maps" / "candidates__ble.npz")
    layout = ly.layout_e()
    np.savez_compressed(
        ROOT / "data" / "radio_maps" / "E_B_plus_extras__ble.npz",
        path_gain=np.concatenate([base_data["path_gain"], pool_data["path_gain"][chosen]]),
        ids=np.array([t.id for t in layout]),
        positions=np.array([[t.x, t.y, t.z] for t in layout], dtype=np.float32),
        frequency=base_data["frequency"],
        cell_size=base_data["cell_size"],
        receiver_height=base_data["receiver_height"],
        samples_per_tx=base_data["samples_per_tx"],
        max_depth=base_data["max_depth"],
    )

    in_rooms = sum(pool_zone[i] != HALL for i in chosen)
    counts = {}
    for i in chosen:
        name = "hall" if pool_zone[i] == HALL else fp.build_rooms()[pool_zone[i]].name
        counts[name] = counts.get(name, 0) + 1
    print(f"Chose {len(chosen)} tags: {in_rooms} in rooms, {len(chosen) - in_rooms} in hall")
    print(counts)


if __name__ == "__main__":
    main()
