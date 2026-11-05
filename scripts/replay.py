"""Draw a simulation log as a top-down video. Later this also shows the particle cloud."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import floorplan as fp  # noqa: E402

WALL_COLORS = {"concrete": "#333333", "drywall": "#8a8a8a", "glass": "#3aa0c8"}
FPS = 10
TRAIL_S = 30.0
LINK_COUNT = 5


def draw_person(ax, color):
    """A small top-down-map person: head dot above a body line."""
    body, = ax.plot([], [], color=color, lw=3, solid_capstyle="round", zorder=6)
    head = Circle((0, 0), 0.9, color=color, zorder=6)
    ax.add_patch(head)
    return body, head


def draw_map(ax, positions):
    ax.add_patch(Rectangle((0, 0), fp.WIDTH, 20, color="#f4efe6", zorder=0))
    ax.add_patch(Rectangle((0, 20), fp.WIDTH, 20, color="#eef3f7", zorder=0))
    ax.add_patch(Rectangle((0, 40), fp.WIDTH, 20, color="#f6f6f6", zorder=0))
    for wall in fp.build_walls():
        ax.plot([wall.x0, wall.x1], [wall.y0, wall.y1], color=WALL_COLORS[wall.material], lw=1.3, zorder=1)
    for room in fp.build_rooms():
        ax.text(room.door_x, 1.5, room.name, ha="center", fontsize=7, color="#666666")
    ax.scatter(positions[:, 0], positions[:, 1], s=9, c="#d1495b", zorder=2)
    ax.set_xlim(-2, fp.WIDTH + 2)
    ax.set_ylim(-2, fp.DEPTH + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--speed", type=float, default=4.0, help="Playback speed")
    parser.add_argument("--still", type=float, help="Save one frame at this time instead of a video")
    args = parser.parse_args()

    log = np.load(args.log)
    dt = float(log["dt"])
    truth, crowd = log["truth"], log["crowd"]
    ble_time, ble, positions = log["ble_time"], log["ble_rssi"], log["ble_positions"]

    fig, ax = plt.subplots(figsize=(14, 4.4), constrained_layout=True)
    draw_map(ax, positions)
    people = ax.scatter([], [], s=9, c="#7a7a7a", alpha=0.6, zorder=3)
    trail, = ax.plot([], [], color="#1f77b4", lw=1.2, alpha=0.7, zorder=4)
    links = [ax.plot([], [], color="#2e9e5b", lw=0.8, alpha=0.7, zorder=4)[0] for _ in range(LINK_COUNT)]
    body, head = draw_person(ax, "#1f77b4")
    title = ax.set_title("", loc="left", fontsize=10)
    ax.legend(
        handles=[
            Line2D([], [], color="#1f77b4", lw=3, label="Tracked person (true path)"),
            Line2D([], [], color="#7a7a7a", marker="o", ls="", label="Crowd"),
            Line2D([], [], color="#d1495b", marker="o", ls="", label="BLE tag"),
            Line2D([], [], color="#2e9e5b", label=f"{LINK_COUNT} strongest tags"),
        ],
        loc="upper right", ncol=4, fontsize=8, frameon=True,
    )

    def update(t):
        i = min(int(t / dt), len(truth) - 1)
        x, y = truth[i]
        body.set_data([x, x], [y - 2.2, y - 0.6])
        head.center = (x, y)
        begin = max(0, i - int(TRAIL_S / dt))
        trail.set_data(truth[begin:i, 0], truth[begin:i, 1])
        if len(crowd):
            people.set_offsets(crowd[i])
        scan = np.searchsorted(ble_time, t, side="right") - 1
        heard = 0
        if scan >= 0:
            rssi = ble[scan]
            order = np.argsort(np.nan_to_num(-rssi, nan=np.inf))
            heard = int(np.sum(~np.isnan(rssi)))
            for line, tag in zip(links, order):
                line.set_data([x, positions[tag, 0]], [y, positions[tag, 1]])
        title.set_text(f"t = {t:5.0f} s   tags heard: {heard}   crowd: {crowd.shape[1] if len(crowd) else 0}")

    if args.still is not None:
        update(args.still)
        out = ROOT / "outputs" / f"{args.log.stem}__t{args.still:.0f}.png"
        fig.savefig(out, dpi=130)
    else:
        out = ROOT / "outputs" / f"{args.log.stem}.mp4"
        writer = FFMpegWriter(fps=FPS, bitrate=2500)
        with writer.saving(fig, str(out), dpi=110):
            for t in np.arange(args.start, args.start + args.seconds, args.speed / FPS):
                update(t)
                writer.grab_frame()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
