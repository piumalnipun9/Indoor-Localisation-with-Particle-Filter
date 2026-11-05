"""Ray-trace a path-gain map per transmitter and save it to data/radio_maps/.

Run on elgin with:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python scripts/generate_radio_maps.py --set wifi --band wifi_2g4
"""
import argparse
import os
import sys
import time
from pathlib import Path

import mitsuba as mi

mi.set_variant(os.environ.get("SIONNA_VARIANT", "cuda_ad_mono_polarized"))
import numpy as np
from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import layouts as ly  # noqa: E402

BANDS = {"ble": 2.426e9, "wifi_2g4": 2.437e9, "wifi_5g": 5.5e9}
CELL_SIZE = 0.5
RECEIVER_HEIGHT = 1.2


def transmitters_for(name):
    if name == "wifi":
        return ly.wifi_access_points()
    if name == "candidates":
        return ly.candidate_pool()
    return ly.all_layouts()[name]


def trace(scene, solver, transmitters, samples, depth, seed):
    for name in list(scene.transmitters):
        scene.remove(name)
    for tx in transmitters:
        scene.add(Transmitter(tx.id, position=[tx.x, tx.y, tx.z]))
    radio_map = solver(
        scene,
        center=[fp.WIDTH / 2, fp.DEPTH / 2, RECEIVER_HEIGHT],
        orientation=[0, 0, 0],
        size=[fp.WIDTH, fp.DEPTH],
        cell_size=[CELL_SIZE, CELL_SIZE],
        samples_per_tx=samples,
        max_depth=depth,
        refraction=True,
        seed=seed,
    )
    return np.array(radio_map.path_gain, dtype=np.float32)


def load(frequency):
    scene = load_scene(str(ROOT / "data" / "scene" / "airport.xml"), merge_shapes=False)
    scene.frequency = frequency
    array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.tx_array = array
    scene.rx_array = array
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, help="wifi or a BLE layout name")
    parser.add_argument("--band", required=True, choices=sorted(BANDS))
    parser.add_argument("--samples", type=float, default=2e7)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    transmitters = transmitters_for(args.set)
    scene = load(BANDS[args.band])
    solver = RadioMapSolver()

    start = time.time()
    maps = []
    for i in range(0, len(transmitters), args.batch):
        maps.append(trace(scene, solver, transmitters[i : i + args.batch], int(args.samples), args.depth, args.seed))
        print(f"{i + len(maps[-1])}/{len(transmitters)} done, {time.time() - start:.0f} s", flush=True)

    out = ROOT / "data" / "radio_maps"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.set}__{args.band}.npz"
    np.savez_compressed(
        path,
        path_gain=np.concatenate(maps).astype(np.float32),
        ids=np.array([t.id for t in transmitters]),
        positions=np.array([[t.x, t.y, t.z] for t in transmitters], dtype=np.float32),
        frequency=BANDS[args.band],
        cell_size=CELL_SIZE,
        receiver_height=RECEIVER_HEIGHT,
        samples_per_tx=int(args.samples),
        max_depth=args.depth,
    )
    print(f"Wrote {path} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
