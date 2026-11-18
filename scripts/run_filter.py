"""Run the filter on one log, print accuracy, and save the estimate for the replay."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import filters  # noqa: E402

SETTLE_S = 30.0
CLOUD = 300


def truth_at(log, times):
    index = np.clip((times / float(log["dt"])).round().astype(int), 0, len(log["truth"]) - 1)
    return log["truth"][index], log["heading"][index]


def zone_ok(est, true):
    """Correct-zone rate for people in rooms and in the hall, in percent."""
    ours = np.array([fp.zone_of(x, y) for x, y in est])
    real = np.array([fp.zone_of(x, y) for x, y in true])
    in_room = real != fp.HALL
    room = 100 * np.mean(ours[in_room] == real[in_room]) if in_room.any() else float("nan")
    hall = 100 * np.mean(ours[~in_room] == real[~in_room]) if (~in_room).any() else float("nan")
    return float(room), float(hall), float(100 * in_room.mean())


def score(log, result):
    keep = result["time"] >= SETTLE_S
    true_xy, true_heading = truth_at(log, result["time"][keep])
    est = result["position"][keep]
    error = np.hypot(*(est - true_xy).T)
    turn = np.abs(np.angle(np.exp(1j * (result["heading"][keep] - true_heading))))
    room_pct, hall_pct, time_in_rooms = zone_ok(est, true_xy)
    return {
        "median_m": float(np.median(error)),
        "p90_m": float(np.percentile(error, 90)),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "heading_deg": float(np.degrees(np.median(turn))),
        "room_pct": room_pct,
        "hall_pct": hall_pct,
        "time_in_rooms_pct": time_in_rooms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--filter", default="fingerprint_6m", choices=sorted(filters.METHODS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log = dict(np.load(args.log))
    survey = dict(np.load(ROOT / "data" / "survey.npz"))
    result = filters.run(log, filters.make(args.filter, survey, args.seed), cloud_size=CLOUD)
    print(args.log.stem, args.filter, {k: round(v, 2) for k, v in score(log, result).items()})
    out = args.log.with_name(f"{args.log.stem}__{args.filter}_estimate.npz")
    np.savez_compressed(out, **result)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
