"""How the 6 m fingerprint filter copes when the building has changed since the survey.

The map is perturbed by a smooth random error per tag (a few metres across) plus a
constant error per tag, as furniture, shops or a replaced beacon would cause. The
phone readings are unchanged, so the map no longer matches the building.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from indoor_loc import filters  # noqa: E402
from report_experiments import OUT, SEEDS, SWEEP_CROWDS, errors_of, load  # noqa: E402

LEVELS = {"0 dB": (0.0, 0.0), "3 dB": (3.0, 1.0), "5 dB": (5.0, 2.0), "8 dB": (8.0, 3.0)}
METHOD = "fingerprint_6m"


def stale(ble_map, field_sd, tag_sd, seed=7):
    rng = np.random.default_rng(seed)
    out = ble_map.copy()
    for t in range(len(ble_map)):
        smooth = gaussian_filter(rng.normal(size=ble_map[t].shape), 1.0)
        out[t] += field_sd * smooth / smooth.std() + rng.normal(0, tag_sd)
    return np.maximum(out, -95.0).astype(np.float32)


def job(args):
    crowd, seed, level = args
    log = load(crowd, seed)
    survey = dict(np.load(ROOT / "data" / "survey.npz"))
    survey["ble_map"] = stale(survey["ble_map"], *LEVELS[level])
    return crowd, seed, level, float(np.median(errors_of(log, filters.run(log, filters.make(METHOD, survey, seed)))))


def main():
    jobs = [(c, s, level) for level in LEVELS for c in SWEEP_CROWDS for s in SEEDS]
    with Pool(7) as pool:
        results = pool.map(job, jobs, chunksize=1)
    summary = {}
    for level in LEVELS:
        for crowd in SWEEP_CROWDS:
            medians = [m for c, _, lv, m in results if c == crowd and lv == level]
            summary[f"{level}/{crowd}"] = {"mean_run_median_m": float(np.mean(medians)), "worst_run_median_m": float(np.max(medians))}
            print(f"{level:<6}{crowd:<6} mean run median {np.mean(medians):.2f} m, worst {np.max(medians):.2f} m")
    (OUT / "stale_map.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
