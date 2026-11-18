"""Run every experiment behind the report and save the numbers to outputs/report/.

1. Main comparison: the three filters (fingerprint at 3, 6 and 12 m) and two simple
   baselines, on all crowd levels.
2. Parameter sweeps on the 6 m fingerprint filter, with no crowd and a high crowd.
3. Example runs with particle clouds, for the trajectory and error-over-time figures.

Uses seeds 90-99, which no earlier experiment touched.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from indoor_loc import filters  # noqa: E402
from evaluate import ble_only, log_path, pdr_only  # noqa: E402
from run_filter import SETTLE_S, score, truth_at  # noqa: E402

SEEDS = list(range(90, 100))
CROWDS = ["none", "low", "medium", "high"]
SWEEP_CROWDS = ["none", "high"]
FILTERS = list(filters.METHODS)
BASELINES = ["ble_only", "pdr_only"]
SWEEP_FILTER = "fingerprint_6m"
SWEEPS = {
    "particles": [300, 1000, 3000, 10000],
    "ble_sigma_db": [3.0, 6.0, 10.0],
    "ble_top_k": [3, 5, 10, 20],
    "offset_shrink": [0.0, 0.5, 1.0],
}
EXAMPLE = ("medium", 90)
EXAMPLE_FILTERS = ["plain", "multi_wall", "fingerprint_6m"]
CLOUD = 300
OUT = ROOT / "outputs" / "report"


def load(crowd, seed):
    file = np.load(log_path(crowd, seed))
    return {key: file[key] for key in file.files if key != "crowd"}


def errors_of(log, result):
    keep = result["time"] >= SETTLE_S
    return np.hypot(*(result["position"][keep] - truth_at(log, result["time"][keep])[0]).T)


def main_job(job):
    crowd, seed, method = job
    log = load(crowd, seed)
    survey = dict(np.load(ROOT / "data" / "survey.npz"))
    started = time.perf_counter()
    if method == "ble_only":
        result = ble_only(log, survey)
    elif method == "pdr_only":
        result = pdr_only(log)
    else:
        result = filters.run(log, filters.make(method, survey, seed))
    seconds = time.perf_counter() - started
    return crowd, seed, method, score(log, result), errors_of(log, result), seconds / len(result["time"])


def sweep_job(job):
    crowd, seed, setting, value = job
    log = load(crowd, seed)
    survey = dict(np.load(ROOT / "data" / "survey.npz"))
    config = replace(filters.FilterConfig(), **{setting: value})
    started = time.perf_counter()
    result = filters.run(log, filters.make(SWEEP_FILTER, survey, seed, config))
    seconds = time.perf_counter() - started
    return crowd, seed, setting, value, score(log, result), seconds / len(result["time"])


def example_job(method):
    crowd, seed = EXAMPLE
    log = load(crowd, seed)
    survey = dict(np.load(ROOT / "data" / "survey.npz"))
    result = filters.run(log, filters.make(method, survey, seed), cloud_size=CLOUD)
    np.savez_compressed(OUT / f"example_{method}.npz", **result)
    return method, score(log, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        main_jobs = [(c, s, m) for c in CROWDS for s in SEEDS for m in FILTERS + BASELINES]
        main_results = pool.map(main_job, main_jobs, chunksize=1)
        sweep_jobs = [(c, s, k, v) for k, values in SWEEPS.items() for v in values for c in SWEEP_CROWDS for s in SEEDS]
        sweep_results = pool.map(sweep_job, sweep_jobs, chunksize=1)
        examples = pool.map(example_job, EXAMPLE_FILTERS)

    keys = ["median_m", "p90_m", "rmse_m", "heading_deg", "room_pct", "hall_pct", "time_in_rooms_pct"]
    lines = ["method,crowd,seed," + ",".join(keys) + ",seconds_per_scan"]
    errors = {}
    for crowd, seed, method, metrics, err, per_scan in main_results:
        lines.append(f"{method},{crowd},{seed}," + ",".join(f"{metrics[k]:.4f}" for k in keys) + f",{per_scan:.6f}")
        errors[f"{method}__{crowd}__{seed}"] = err.astype(np.float32)
    (OUT / "main_runs.csv").write_text("\n".join(lines) + "\n")
    np.savez_compressed(OUT / "main_errors.npz", **errors)

    lines = ["setting,value,crowd,seed," + ",".join(keys) + ",seconds_per_scan"]
    for crowd, seed, setting, value, metrics, per_scan in sweep_results:
        lines.append(f"{setting},{value},{crowd},{seed}," + ",".join(f"{metrics[k]:.4f}" for k in keys) + f",{per_scan:.6f}")
    (OUT / "sweep_runs.csv").write_text("\n".join(lines) + "\n")

    (OUT / "examples.json").write_text(json.dumps({m: s for m, s in examples}, indent=2))
    summarise(main_results, sweep_results)


def summarise(main_results, sweep_results):
    """Print and save the tables used in the report."""
    table = {}
    print(f"{'method':<17}{'crowd':<8}{'median':>8}{'p90':>8}{'rmse':>8}{'head':>7}{'room%':>7}{'hall%':>7}{'ms/scan':>9}")
    for method in FILTERS + BASELINES:
        for crowd in CROWDS:
            rows = [r for r in main_results if r[2] == method and r[0] == crowd]
            pooled = np.concatenate([r[4] for r in rows])
            entry = {
                "median_m": float(np.median(pooled)),
                "p90_m": float(np.percentile(pooled, 90)),
                "rmse_m": float(np.sqrt(np.mean(pooled**2))),
                "heading_deg": float(np.nanmean([r[3]["heading_deg"] for r in rows])),
                "room_pct": float(np.nanmean([r[3]["room_pct"] for r in rows])),
                "hall_pct": float(np.nanmean([r[3]["hall_pct"] for r in rows])),
                "ms_per_scan": float(1000 * np.mean([r[5] for r in rows])),
                "run_medians": [float(np.median(r[4])) for r in sorted(rows, key=lambda r: r[1])],
            }
            table[f"{method}/{crowd}"] = entry
            print(f"{method:<17}{crowd:<8}{entry['median_m']:>8.2f}{entry['p90_m']:>8.2f}{entry['rmse_m']:>8.2f}"
                  f"{entry['heading_deg']:>7.1f}{entry['room_pct']:>7.1f}{entry['hall_pct']:>7.1f}{entry['ms_per_scan']:>9.2f}")
    (OUT / "main_summary.json").write_text(json.dumps(table, indent=2))

    sweeps = {}
    print("\nSweeps on", SWEEP_FILTER)
    for setting, values in SWEEPS.items():
        for value in values:
            for crowd in SWEEP_CROWDS:
                rows = [r for r in sweep_results if r[2] == setting and r[3] == value and r[0] == crowd]
                medians = [r[4]["median_m"] for r in rows]
                entry = {
                    "median_of_runs_m": float(np.median(medians)),
                    "mean_of_run_medians_m": float(np.mean(medians)),
                    "mean_p90_m": float(np.mean([r[4]["p90_m"] for r in rows])),
                    "worst_run_median_m": float(np.max(medians)),
                    "room_pct": float(np.nanmean([r[4]["room_pct"] for r in rows])),
                    "ms_per_scan": float(1000 * np.mean([r[5] for r in rows])),
                }
                sweeps[f"{setting}={value}/{crowd}"] = entry
                print(f"  {setting:<14}{value!s:<8}{crowd:<6} mean run median {entry['mean_of_run_medians_m']:.2f} m, "
                      f"mean p90 {entry['mean_p90_m']:.2f} m, worst run {entry['worst_run_median_m']:.2f} m, "
                      f"room {entry['room_pct']:.1f}%, {entry['ms_per_scan']:.1f} ms/scan")
    (OUT / "sweep_summary.json").write_text(json.dumps(sweeps, indent=2))


if __name__ == "__main__":
    main()
