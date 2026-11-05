"""Find the ray count and bounce depth the radio maps need, using one room beacon."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_radio_maps import BANDS, CELL_SIZE, load, trace  # noqa: E402
from indoor_loc import layouts as ly  # noqa: E402
from sionna.rt import RadioMapSolver  # noqa: E402

TX = ly.Transmitter("probe", "ble", 110.0, 30.0, 3.0, "pillar")


def db(values):
    return 10 * np.log10(np.maximum(values, 1e-30))


def main():
    scene = load(BANDS["ble"])
    solver = RadioMapSolver()
    reference = None
    print(f"{'rays':>9}{'depth':>7}{'sec':>7}{'hit %':>8}{'dB noise':>10}{'dB vs ref':>11}")
    for samples, depth in [(1e6, 3), (5e6, 3), (2e7, 3), (2e7, 6), (2e7, 10), (1e8, 6)]:
        start = time.time()
        a = trace(scene, solver, [TX], int(samples), depth, 1)[0]
        seconds = time.time() - start
        b = trace(scene, solver, [TX], int(samples), depth, 2)[0]
        mask = (a > 0) & (b > 0)
        noise = np.std(db(a[mask]) - db(b[mask])) / np.sqrt(2)
        if samples == 1e8:
            reference = a
        print(f"{samples:>9.0e}{depth:>7}{seconds:>7.1f}{100 * mask.mean():>8.1f}{noise:>10.2f}", flush=True)
    ny, nx = reference.shape
    iy, ix = int(30 / CELL_SIZE), int(110 / CELL_SIZE)
    print("shape", reference.shape)
    for dist in (1, 5, 10, 20, 40, 80):
        cell = reference[iy, min(ix + int(dist / CELL_SIZE), nx - 1)]
        free = -20 * np.log10(4 * np.pi * np.hypot(dist, 1.8) * BANDS["ble"] / 3e8)
        print(f"along hall {dist:>3} m: sionna {db(cell):7.1f} dB, free space {free:7.1f} dB")
    room = reference[int(10 / CELL_SIZE), ix]
    print(f"inside a room 20 m away: {db(room):.1f} dB")


if __name__ == "__main__":
    main()
