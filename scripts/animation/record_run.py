"""Replay one walk through the current filter and record the cloud after every step and scan."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "animation"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from indoor_loc import filters
from run_filter import truth_at

crowd, seed, method = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SHOW = 500
f = np.load(ROOT / f"data/sim/E_B_plus_extras__{crowd}__seed{seed}.npz")
log = {k: f[k] for k in f.files if k != "crowd"}
survey = dict(np.load(ROOT / "data/survey.npz"))
pf = filters.make(method, survey, seed)

ble_time, ble = log["ble_time"], log["ble_rssi"]
st, sl, sturn = log["step_time"], log["step_length"], log["step_turn"]
events = []  # (time, kind, est_xy, cloud_xy, identity_kept)
def pick():
    return pf.rng.choice(len(pf.xy), SHOW, p=pf.weights)
pf.initialise(ble[0]); idx = pick()
events.append((ble_time[0], 0, pf.estimate()[0], pf.xy[idx].copy(), False))
si, last = 0, 0.0
for n, now in enumerate(ble_time):
    while si < len(st) and st[si] <= now:
        pf.predict_step(sl[si], sturn[si], st[si] - last); last = st[si]; si += 1
        events.append((last, 1, pf.estimate()[0], pf.xy[idx].copy(), True))
    pf.update(ble[n]); before = pf.xy
    pf.resample_if_needed(ble[n])
    kept = pf.xy is before
    if not kept: idx = pick()
    events.append((now, 2, pf.estimate()[0], pf.xy[idx].copy(), kept))
t = np.array([e[0] for e in events]); est = np.array([e[2] for e in events])
err = np.hypot(*(est - truth_at(log, t)[0]).T)
scan = np.array([e[1] == 2 for e in events])
print(f"{crowd} seed{seed} {method}: median err after 30 s {np.median(err[scan & (t>=30)]):.2f} m, events {len(t)}")
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"run_{crowd}_{seed}_{method}.npz", t=t, kind=np.array([e[1] for e in events]), est=est,
    cloud=np.array([e[3] for e in events]).astype(np.float32), kept=np.array([e[4] for e in events]),
    err=err, truth=log["truth"], dt=log["dt"], truth_heading=log["heading"], tags=log["ble_positions"])
