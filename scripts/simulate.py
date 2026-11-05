"""Create one replayable run: a tracked person, a crowd and all phone readings."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import paths as pa  # noqa: E402
from indoor_loc import sensors as se  # noqa: E402

CROWDS = {"none": 0, "low": 100, "medium": 300, "high": 800}
LAYOUT = "E_B_plus_extras"
WIFI_BANDS = ("wifi_2g4", "wifi_5g")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crowd", choices=sorted(CROWDS), default="medium")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", type=float, default=600.0)
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    maps = ROOT / "data" / "radio_maps"
    router = pa.Router()
    config = pa.WalkConfig()
    dt = config.dt

    truth = pa.walk(router, rng, args.duration, config)
    heading = pa.headings(truth, dt)
    crowd = None
    if CROWDS[args.crowd]:
        people = [pa.walk(router, rng, args.duration, config) for _ in range(CROWDS[args.crowd])]
        crowd = np.stack(people, axis=1).astype(np.float32)

    ble_field = se.RadioField(maps / f"{LAYOUT}__ble.npz", se.BLE, rng)
    ble_time, ble = se.simulate_radio(ble_field, truth, heading, crowd, dt, rng)

    wifi = []
    for band in WIFI_BANDS:
        field = se.RadioField(maps / f"wifi__{band}.npz", se.WIFI, rng)
        wifi_time, rssi = se.simulate_radio(field, truth, heading, crowd, dt, rng)
        wifi.append(rssi)

    step_time, step_length, step_turn, stride, bias = se.simulate_steps(truth, heading, dt, rng)

    out = ROOT / "data" / "sim"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{LAYOUT}__{args.crowd}__seed{args.seed}.npz"
    np.savez_compressed(
        path,
        dt=dt,
        truth=truth.astype(np.float32),
        heading=heading.astype(np.float32),
        crowd=crowd if crowd is not None else np.empty((0, 0, 2), dtype=np.float32),
        ble_time=ble_time,
        ble_rssi=ble,
        ble_positions=ble_field.positions,
        wifi_time=wifi_time,
        wifi_2g4_rssi=wifi[0],
        wifi_5g_rssi=wifi[1],
        step_time=step_time,
        step_length=step_length,
        step_turn=step_turn,
        true_stride=stride,
        true_gyro_bias=bias,
    )
    print(f"Wrote {path} in {time.time() - started:.0f} s")
    heard = np.sum(~np.isnan(ble), axis=1)
    walked = np.hypot(*np.diff(truth, axis=0).T).sum()
    print(f"walked {walked:.0f} m, {len(step_time)} steps, {CROWDS[args.crowd]} people in crowd")
    print(f"BLE scans {len(ble_time)}, tags heard per scan: median {np.median(heard):.0f}, min {heard.min()}")
    print(f"Wi-Fi scans {len(wifi_time)}, APs heard per scan: 2.4 GHz {np.median(np.sum(~np.isnan(wifi[0]), 1)):.0f}, 5 GHz {np.median(np.sum(~np.isnan(wifi[1]), 1)):.0f}")


if __name__ == "__main__":
    main()
