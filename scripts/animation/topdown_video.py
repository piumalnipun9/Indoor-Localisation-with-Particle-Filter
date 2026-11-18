"""Top-down matplotlib version of the Isaac Sim animation: same walk, same frames, same timing.

Light style of scripts/replay.py. Reads frames.npz written by prep.py and writes
ble_tracking_30s_topdown.mp4 (1920x600, 30 fps).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_DIR = ROOT / "outputs" / "animation"
sys.path.insert(0, str(ROOT))
from indoor_loc import floorplan as fp  # noqa: E402

FRAMES = OUT_DIR / "frames.npz"
LOG = ROOT / "data/sim/E_B_plus_extras__none__seed60.npz"
OUT = OUT_DIR / "ble_tracking_30s_topdown.mp4"
WALL_COLORS = {"concrete": "#333333", "drywall": "#8a8a8a", "glass": "#3aa0c8"}
TRUE_COLOR, ESTIMATE_COLOR, PARTICLE_COLOR = "#1f77b4", "#ff7f00", "#ff00c8"
TAG_COLOR, LINK_COLOR, PEOPLE_COLOR = "#d1495b", "#2e9e5b", "#7a7a7a"
LINK_COUNT = 5


def draw_map(ax, tags):
    ax.add_patch(Rectangle((0, 0), fp.WIDTH, 20, color="#f4efe6", zorder=0))
    ax.add_patch(Rectangle((0, 20), fp.WIDTH, 20, color="#eef3f7", zorder=0))
    ax.add_patch(Rectangle((0, 40), fp.WIDTH, 20, color="#f6f6f6", zorder=0))
    for wall in fp.build_walls():
        ax.plot([wall.x0, wall.x1], [wall.y0, wall.y1], color=WALL_COLORS[wall.material], lw=1.3, zorder=1)
    for room in fp.build_rooms():
        ax.text(room.door_x, 1.5, room.name, ha="center", fontsize=8, color="#666666")
    ax.scatter(tags[:, 0], tags[:, 1], s=12, c=TAG_COLOR, zorder=2)
    ax.set_xlim(-2, fp.WIDTH + 2)
    ax.set_ylim(-2, fp.DEPTH + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    d = np.load(FRAMES)
    log = np.load(LOG)
    tags, ble_time, ble = log["ble_positions"], log["ble_time"], log["ble_rssi"]
    fps, intro = int(d["fps"]), float(d["intro_s"])
    walker, est, cloud, crowd, walk_t = d["walker"], d["est"], d["cloud"], d["crowd_xy"], d["walk_t"]
    start = int(intro * fps)

    fig, ax = plt.subplots(figsize=(19.2, 6.0), constrained_layout=True)
    draw_map(ax, tags)
    people = ax.scatter([], [], s=14, c=PEOPLE_COLOR, alpha=0.7, zorder=3)
    links = [ax.plot([], [], color=LINK_COLOR, lw=0.9, alpha=0.75, zorder=4)[0] for _ in range(LINK_COUNT)]
    trail, = ax.plot([], [], color=TRUE_COLOR, lw=1.6, zorder=4)
    predicted, = ax.plot([], [], color=ESTIMATE_COLOR, lw=1.6, zorder=4)
    cloud_art = ax.scatter([], [], s=10, c=PARTICLE_COLOR, alpha=0.85, zorder=5, linewidths=0)
    body, = ax.plot([], [], color=TRUE_COLOR, lw=3, solid_capstyle="round", zorder=6)
    head = Circle((0, 0), 0.9, color=TRUE_COLOR, zorder=6)
    ax.add_patch(head)
    guess, = ax.plot([], [], marker="o", color=ESTIMATE_COLOR, markersize=9, markeredgecolor="black", ls="", zorder=7)
    title = ax.set_title("", loc="left", fontsize=12)
    ax.legend(
        handles=[
            Line2D([], [], color=TRUE_COLOR, lw=2, label="True path"),
            Line2D([], [], color=TAG_COLOR, marker="o", ls="", label="BLE tag"),
            Line2D([], [], color=LINK_COLOR, label=f"{LINK_COUNT} strongest tags"),
            Line2D([], [], color=ESTIMATE_COLOR, lw=2, marker="o", label="Predicted path"),
            Line2D([], [], color=PARTICLE_COLOR, marker="o", ls="", markersize=4, label="Particles"),
            Line2D([], [], color=PEOPLE_COLOR, marker="o", ls="", markersize=4, label="Other people"),
        ],
        loc="upper right", ncol=6, fontsize=10, frameon=True,
    )

    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"])
    with writer.saving(fig, str(OUT), dpi=100):
        for f in range(len(walk_t)):
            tracking = f >= start
            x, y = walker[f]
            body.set_data([x, x], [y - 2.2, y - 0.6])
            head.center = (x, y)
            people.set_offsets(crowd[f])
            t = walk_t[f]
            text = f"t = {t:4.0f} s   (3x speed)"
            if tracking:
                trail.set_data(*walker[start:f + 1].T)
                predicted.set_data(*est[start:f + 1].T)
                cloud_art.set_offsets(cloud[f])
                guess.set_data([est[f, 0]], [est[f, 1]])
                scan = np.searchsorted(ble_time, t, side="right") - 1
                rssi = ble[max(scan, 0)]
                order = np.argsort(np.nan_to_num(-rssi, nan=np.inf))
                for line, tag in zip(links, order):
                    line.set_data([x, tags[tag, 0]], [y, tags[tag, 1]])
                heard = int(np.sum(~np.isnan(rssi)))
                text += f"   tags heard: {heard}   position error: {np.hypot(*(est[f] - walker[f])):.1f} m"
            title.set_text(text)
            writer.grab_frame()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
