"""Score filter variants and baselines on chosen runs, with paired per-run comparisons."""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from indoor_loc import filters  # noqa: E402
from run_filter import SETTLE_S, score, truth_at  # noqa: E402

BOOTSTRAP_DRAWS = 10000
BLE_ONLY_TAGS = 4
STEP_M = 0.7


def log_path(crowd, seed):
    return ROOT / "data" / "sim" / f"E_B_plus_extras__{crowd}__seed{seed}.npz"


def ble_only(log, survey):
    """Weighted centroid of the strongest tags; no motion, walls or filter."""
    rssi, tags = log["ble_rssi"], survey["tags"][:, :2]
    filled = np.nan_to_num(rssi, nan=-200.0)
    top = np.argsort(-filled, axis=1)[:, :BLE_ONLY_TAGS]
    weights = 10 ** (np.take_along_axis(filled, top, axis=1) / 20)
    position = np.sum(tags[top] * weights[..., None], axis=1) / weights.sum(axis=1, keepdims=True)
    return {"time": log["ble_time"], "position": position, "heading": np.zeros(len(position))}


def pdr_only(log):
    """Steps and heading only, given the true start: no radio at all."""
    start_xy, start_heading = log["truth"][0], log["heading"][0]
    heading = start_heading + np.cumsum(log["step_turn"])
    moves = (log["step_length"] * np.array([np.cos(heading), np.sin(heading)])).T
    track = start_xy + np.cumsum(moves, axis=0)
    index = np.searchsorted(log["step_time"], log["ble_time"], side="right") - 1
    position = np.where(index[:, None] >= 0, track[np.maximum(index, 0)], start_xy)
    head = np.where(index >= 0, heading[np.maximum(index, 0)], start_heading)
    return {"time": log["ble_time"], "position": position, "heading": head}


def evaluate(job):
    crowd, seed, method = job
    file = np.load(log_path(crowd, seed))
    log = {key: file[key] for key in file.files if key != "crowd"}
    survey = dict(np.load(ROOT / "data" / "survey.npz"))
    if method == "ble_only":
        result = ble_only(log, survey)
    elif method == "pdr_only":
        result = pdr_only(log)
    else:
        result = filters.run(log, filters.make(method, survey, seed))
    metrics = score(log, result)
    keep = result["time"] >= SETTLE_S
    errors = np.hypot(*(result["position"][keep] - truth_at(log, result["time"][keep])[0]).T)
    return crowd, seed, method, metrics, errors


def paired_difference(rows, baseline, method, crowd, seeds):
    """Mean per-run change in median error against baseline, with a 95% interval."""
    base = {seed: np.median(e) for seed, _, e in rows[(baseline, crowd)]}
    other = {seed: np.median(e) for seed, _, e in rows[(method, crowd)]}
    change = np.array([other[s] - base[s] for s in seeds])
    rng = np.random.default_rng(0)
    draws = rng.choice(change, (BOOTSTRAP_DRAWS, len(change))).mean(axis=1)
    return change.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5), int(np.sum(change < 0))


def crowd_effect(rows, method, crowd, seeds):
    """Mean per-run change in median error against the same path with no crowd."""
    empty = {seed: np.median(e) for seed, _, e in rows[(method, "none")]}
    busy = {seed: np.median(e) for seed, _, e in rows[(method, crowd)]}
    change = np.array([busy[s] - empty[s] for s in seeds])
    draws = np.random.default_rng(0).choice(change, (BOOTSTRAP_DRAWS, len(change))).mean(axis=1)
    return change.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5), int(np.sum(change > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--crowds", nargs="+", default=["none"])
    parser.add_argument("--methods", nargs="+", default=["fingerprint_6m"])
    parser.add_argument("--baseline", default="fingerprint_6m")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="results")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    jobs = [(c, s, m) for c in args.crowds for s in args.seeds for m in args.methods]
    with Pool(args.workers) as pool:
        results = pool.map(evaluate, jobs, chunksize=1)

    rows = {}
    for crowd, seed, method, metrics, errors in results:
        rows.setdefault((method, crowd), []).append((seed, metrics, errors))

    keys = ("median_m", "p90_m", "heading_deg", "room_pct", "hall_pct")
    print(f"{'method':<14}{'crowd':<8}{'median m':>9}{'p90 m':>7}{'heading':>9}{'room %':>8}{'hall %':>8}{'run range m':>14}")
    table = {}
    for method in args.methods:
        for crowd in args.crowds:
            items = rows[(method, crowd)]
            pooled = np.concatenate([e for _, _, e in items])
            per_run = [np.median(e) for _, _, e in items]
            mean = {k: float(np.nanmean([m[k] for _, m, _ in items])) for k in keys}
            mean["median_m"], mean["p90_m"] = float(np.median(pooled)), float(np.percentile(pooled, 90))
            table[f"{method}/{crowd}"] = mean
            print(f"{method:<14}{crowd:<8}{mean['median_m']:>9.2f}{mean['p90_m']:>7.2f}{mean['heading_deg']:>9.1f}"
                  f"{mean['room_pct']:>8.1f}{mean['hall_pct']:>8.1f}{min(per_run):>8.1f}-{max(per_run):<5.1f}")
    print(f"Time spent in rooms: {np.mean([m['time_in_rooms_pct'] for _, _, _, m, _ in results]):.0f}%")
    print(f"\nPaired change in median error against {args.baseline} (negative is better):")
    for method in args.methods:
        for crowd in args.crowds:
            if method != args.baseline:
                mean, low, high, wins = paired_difference(rows, args.baseline, method, crowd, args.seeds)
                print(f"  {method:<14}{crowd:<8}{mean:+.2f} m  (95% interval {low:+.2f} to {high:+.2f}), better in {wins}/{len(args.seeds)} runs")
    if "none" in args.crowds and len(args.crowds) > 1:
        print("\nSame paths compared with no crowd (positive is worse):")
        for method in args.methods:
            for crowd in args.crowds:
                if crowd != "none":
                    mean, low, high, worse = crowd_effect(rows, method, crowd, args.seeds)
                    print(f"  {method:<14}{crowd:<8}{mean:+.2f} m  (95% interval {low:+.2f} to {high:+.2f}), worse in {worse}/{len(args.seeds)} runs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(table, indent=2))
    lines = ["method,crowd,seed," + ",".join(keys)]
    for crowd, seed, method, metrics, _ in sorted(results, key=lambda r: (r[2], r[0], r[1])):
        lines.append(f"{method},{crowd},{seed}," + ",".join(f"{metrics[k]:.3f}" for k in keys))
    (args.out_dir / f"{args.name}_runs.csv").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in args.methods:
        pooled = np.sort(np.concatenate([e for c in args.crowds for _, _, e in rows[(method, c)]]))
        ax.plot(pooled, np.linspace(0, 1, len(pooled)), label=method)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Fraction of time error is below this")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(args.out_dir / f"{args.name}_cdf.png", dpi=130)


if __name__ == "__main__":
    main()
