"""Turn one recorded filter run into per-frame animation data for the Isaac Sim render.

Writes scene.json (static geometry) and frames.npz (everything that moves, per video frame).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "animation"
sys.path.insert(0, str(ROOT))
from indoor_loc import floorplan as fp  # noqa: E402

HERE = OUT
RUN = HERE / "run_none_60_fingerprint_6m.npz"
LOG = ROOT / "data/sim/E_B_plus_extras__none__seed60.npz"
FPS, SECONDS, INTRO_S, SPEED = 30, 30, 2.5, 3.0
WALL_H = 3.0  # Cut-away: every wall drawn at room height so the camera sees inside.
WALL_T = 0.18
TOP_K = 10


def scene_json():
    boxes = []  # (material, cx, cy, cz, sx, sy, sz)
    for w in fp.build_walls():
        cx, cy = (w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2
        sx = abs(w.x1 - w.x0) + WALL_T if w.y0 == w.y1 else WALL_T
        sy = abs(w.y1 - w.y0) + WALL_T if w.x0 == w.x1 else WALL_T
        h = min(w.height, WALL_H)
        boxes.append((w.material, cx, cy, h / 2, sx, sy, h))
    for x, y in fp.build_pillars():
        boxes.append(("pillar", x, y, WALL_H / 2, fp.PILLAR_SIZE, fp.PILLAR_SIZE, WALL_H))
    log = np.load(LOG)
    tags = log["ble_positions"]
    zones = [dict(x0=r.x0, x1=r.x1, y0=r.y0, y1=r.y1) for r in fp.build_rooms()]
    boxes = [(m, *map(float, v)) for m, *v in boxes]
    return dict(width=fp.WIDTH, depth=fp.DEPTH, boxes=boxes, rooms=zones, tags=tags.tolist())


def make_crowd(t, seed=7):
    """Other people: walk back and forth along open lanes (concourse, gates, two nearby rooms)."""
    rng = np.random.default_rng(seed)
    # (y, x_min, x_max, start x) lanes clear of pillars (rows at y 26 and 34) and gate dividers (from y 46).
    lanes = [(22.5, 2, 248, 112), (24.0, 2, 248, 150), (28.5, 2, 248, 92), (30.0, 2, 248, 175),
             (31.5, 2, 248, 128), (36.5, 2, 248, 105), (38.0, 2, 248, 160), (42.5, 2, 248, 85),
             (44.0, 2, 248, 140), (23.2, 2, 248, 196), (37.2, 2, 248, 70),
             (13.0, 87, 113, 92), (9.0, 42, 68, 55)]
    xy, hd, dist, moving = [], [], [], []
    for y, x0, x1, start in lanes:
        speed = rng.uniform(1.0, 1.45) * rng.choice([1, 1, -1])
        pause = rng.random() < 0.2
        span = x1 - x0
        u = (start - x0 + (0 if pause else speed) * t) % (2 * span)
        x = x0 + np.where(u < span, u, 2 * span - u)
        forward = np.where(u < span, 1, -1) * np.sign(speed)
        xy.append(np.c_[x, y + 0.3 * np.sin(t / 7 + start)])
        hd.append(np.where(forward > 0, 0.0, np.pi) if not pause else np.full_like(t, rng.uniform(-3, 3)))
        dist.append(np.abs(speed) * t * (0 if pause else 1))
        moving.append(np.full(len(t), not pause))
    xy = np.stack(xy, 1).astype(np.float32)
    hd = np.stack(hd, 1)
    hd = gaussian_filter1d(np.unwrap(hd, axis=0), 3, axis=0).astype(np.float32)  # Smooth the turn-arounds.
    return xy, hd, np.stack(dist, 1).astype(np.float32), np.stack(moving, 1)


def interp_events(run, T):
    """Estimate and displayed particle cloud at walk time T, moving smoothly between filter events."""
    t, est, cloud, kept = run["t"], run["est"], run["cloud"], run["kept"]
    i = int(np.clip(np.searchsorted(t, T, side="right") - 1, 0, len(t) - 1))
    if i + 1 < len(t) and t[i + 1] > t[i]:
        a = (T - t[i]) / (t[i + 1] - t[i])
        e = (1 - a) * est[i] + a * est[i + 1]
        c = (1 - a) * cloud[i] + a * cloud[i + 1] if kept[i + 1] else cloud[i]
        return e, c
    return est[i], cloud[i]


def main():
    run = np.load(RUN)
    log = np.load(LOG)
    truth, dt, heading = log["truth"], float(log["dt"]), log["heading"]
    ble_time, ble = log["ble_time"], log["ble_rssi"]
    n_frames = FPS * SECONDS
    walk_t = np.maximum(0.0, (np.arange(n_frames) / FPS - INTRO_S) * SPEED)

    def truth_at(T):
        k = T / dt
        i = np.clip(np.floor(k).astype(int), 0, len(truth) - 2)
        a = (k - i)[..., None]
        return (1 - a) * truth[i] + a * truth[i + 1]

    walker = truth_at(walk_t)
    # Heading from the truth log, unwrapped and smoothed so the figure turns rather than snaps.
    h = np.unwrap(heading.astype(float))
    h = gaussian_filter1d(h, 4)
    walker_heading = np.interp(walk_t, np.arange(len(h)) * dt, h)
    dist = np.r_[0, np.cumsum(np.hypot(*np.diff(walker, axis=0).T))]
    moving = np.r_[0, np.hypot(*np.diff(walker, axis=0).T)] > 1e-3

    est, cloud = zip(*(interp_events(run, T) for T in walk_t))
    est, cloud = np.array(est), np.array(cloud, dtype=np.float32)

    # Beacons: which tags the filter used at the latest scan, and how long ago that scan was (video seconds).
    n_tags = ble.shape[1]
    active = np.zeros((n_frames, n_tags), bool)
    scan_age = np.full(n_frames, 99.0)
    for f, T in enumerate(walk_t):
        if f / FPS < INTRO_S:
            continue
        s = np.searchsorted(ble_time, T, side="right") - 1
        if s < 0:
            continue
        r = np.nan_to_num(ble[s], nan=-200.0)
        heard = np.where(r > -150)[0]
        active[f, heard[np.argsort(r[heard])[::-1][:TOP_K]]] = True
        scan_age[f] = (T - ble_time[s]) / SPEED

    # Camera: follow a heavily smoothed walker; start wide, settle close, pull back at the end.
    target_xy = gaussian_filter1d(walker, 45, axis=0, mode="nearest")
    tsec = np.arange(n_frames) / FPS
    def ease(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)
    close = ease((tsec - 0.8) / 3.5) * (1 - ease((tsec - 26.0) / 3.5))
    radius = 50 - 18 * close
    height = 36 - 12 * close
    azimuth = np.deg2rad(105 - 20 * tsec / SECONDS)  # Slow orbit from behind-left to behind-right.
    eye = np.c_[target_xy[:, 0] + radius * np.cos(azimuth), target_xy[:, 1] + radius * np.sin(azimuth), height]
    target = np.c_[target_xy + np.c_[np.zeros(n_frames), -4 * close], np.full(n_frames, 0.5)]

    crowd = make_crowd(np.arange(n_frames) / FPS * SPEED)

    # Trails: estimate and truth history, resampled every 0.5 s of walk time.
    np.savez_compressed(
        HERE / "frames.npz",
        fps=FPS, intro_s=INTRO_S, walk_t=walk_t, walker=walker.astype(np.float32),
        walker_heading=walker_heading.astype(np.float32), walker_dist=dist.astype(np.float32),
        walker_moving=moving, est=est.astype(np.float32), cloud=cloud, active=active, scan_age=scan_age,
        eye=eye.astype(np.float32), target=target.astype(np.float32),
        crowd_xy=crowd[0], crowd_heading=crowd[1], crowd_dist=crowd[2], crowd_moving=crowd[3],
    )
    (HERE / "scene.json").write_text(json.dumps(scene_json()))
    err = np.hypot(*(est - walker).T)[walk_t > 0]
    print(f"{n_frames} frames, walk 0-{walk_t[-1]:.0f} s, shown error median {np.median(err):.2f} m, "
          f"walker ends at {walker[-1].round(1)}")


if __name__ == "__main__":
    main()
