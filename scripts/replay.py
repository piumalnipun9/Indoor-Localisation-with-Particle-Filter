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
PARTICLE_COLOR = "#ff00c8"
ESTIMATE_COLOR = "#ff7f00"
LINK_COUNT = 5
ARROW_M = 7.0


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
    parser.add_argument("--estimate", type=Path, help="Particle filter output to overlay")
    parser.add_argument("--show-crowd", action="store_true", help="Also draw the other people")
    parser.add_argument("--full-path", action="store_true", help="Save one picture of both whole paths")
    args = parser.parse_args()

    log = np.load(args.log)
    dt = float(log["dt"])
    truth, crowd, heading = log["truth"], log["crowd"], log["heading"]
    ble_time, ble, positions = log["ble_time"], log["ble_rssi"], log["ble_positions"]

    estimate = dict(np.load(args.estimate)) if args.estimate else None
    fig, ax = plt.subplots(figsize=(14, 4.4), constrained_layout=True)
    draw_map(ax, positions)
    cloud = ax.scatter([], [], s=9, c=PARTICLE_COLOR, alpha=0.9, zorder=5, linewidths=0)
    guess, = ax.plot([], [], marker="o", color=ESTIMATE_COLOR, markersize=8, markeredgecolor="black", ls="", zorder=7)
    predicted, = ax.plot([], [], color=ESTIMATE_COLOR, lw=1.3, zorder=4)
    arrow = ax.quiver([0], [0], [0], [0], color=ESTIMATE_COLOR, scale=1, scale_units="xy", angles="xy", width=0.0028, zorder=7)
    arrow.set_visible(False)
    true_arrow = ax.quiver([0], [0], [0], [0], color="#1f77b4", scale=1, scale_units="xy", angles="xy", width=0.0028, zorder=7)
    people = ax.scatter([], [], s=9, c="#7a7a7a", alpha=0.6, zorder=3)
    show_crowd = args.show_crowd and len(crowd) > 0
    trail, = ax.plot([], [], color="#1f77b4", lw=1.3, zorder=4)
    links = [ax.plot([], [], color="#2e9e5b", lw=0.8, alpha=0.7, zorder=4)[0] for _ in range(LINK_COUNT)]
    body, head = draw_person(ax, "#1f77b4")
    title = ax.set_title("", loc="left", fontsize=10)
    ax.legend(
        handles=[
            Line2D([], [], color="#1f77b4", lw=2, label="True path"),
            Line2D([], [], color="#d1495b", marker="o", ls="", label="BLE tag"),
            Line2D([], [], color="#2e9e5b", label=f"{LINK_COUNT} strongest tags"),
        ] + ([
            Line2D([], [], color=ESTIMATE_COLOR, lw=2, marker="o", label="Predicted path"),
            Line2D([], [], color=PARTICLE_COLOR, marker="o", ls="", markersize=4, label="Particles"),
        ] if estimate else []),
        loc="upper right", ncol=6, fontsize=8, frameon=True,
    )

    def update(t):
        i = min(int(t / dt), len(truth) - 1)
        x, y = truth[i]
        body.set_data([x, x], [y - 2.2, y - 0.6])
        head.center = (x, y)
        trail.set_data(truth[:i, 0], truth[:i, 1])
        if show_crowd:
            people.set_offsets(crowd[i])
        scan = np.searchsorted(ble_time, t, side="right") - 1
        heard = 0
        if scan >= 0:
            rssi = ble[scan]
            order = np.argsort(np.nan_to_num(-rssi, nan=np.inf))
            heard = int(np.sum(~np.isnan(rssi)))
            for line, tag in zip(links, order):
                line.set_data([x, positions[tag, 0]], [y, positions[tag, 1]])
        text = f"t = {t:5.0f} s   tags heard: {heard}"
        true_arrow.set_offsets([[x, y]])
        true_arrow.set_UVC([ARROW_M * np.cos(heading[i])], [ARROW_M * np.sin(heading[i])])
        if estimate is not None and scan >= 0:
            ex, ey = estimate["position"][scan]
            eh = estimate["heading"][scan]
            cloud.set_offsets(estimate["cloud"][scan])
            guess.set_data([ex], [ey])
            predicted.set_data(estimate["position"][: scan + 1, 0], estimate["position"][: scan + 1, 1])
            arrow.set_visible(True)
            arrow.set_offsets([[ex, ey]])
            arrow.set_UVC([ARROW_M * np.cos(eh)], [ARROW_M * np.sin(eh)])
            position_error = np.hypot(ex - x, ey - y)
            heading_error = np.degrees(abs(np.angle(np.exp(1j * (eh - heading[i])))))
            text += f"   position error: {position_error:.1f} m   heading error: {heading_error:.0f} deg"
        title.set_text(text)

    suffix = "__" + args.estimate.stem.split("__")[-1] if estimate else ""
    if args.full_path:
        for artist in (cloud, guess, arrow, true_arrow, body, head, *links):
            artist.set_visible(False)
        trail.set_data(truth[:, 0], truth[:, 1])
        predicted.set_data(estimate["position"][:, 0], estimate["position"][:, 1])
        ax.plot(*truth[0], marker="s", color="#2ca02c", markersize=9, markeredgecolor="black", ls="", zorder=8, label="Start")
        ax.plot(*truth[-1], marker="X", color="black", markersize=10, ls="", zorder=8, label="Finish")
        ax.set_title("True path and predicted path, start to finish", loc="left", fontsize=10)
        out = ROOT / "outputs" / f"{args.log.stem}{suffix}__paths.png"
        fig.savefig(out, dpi=140)
    elif args.still is not None:
        update(args.still)
        out = ROOT / "outputs" / f"{args.log.stem}{suffix}__t{args.still:.0f}.png"
        fig.savefig(out, dpi=130)
    else:
        out = ROOT / "outputs" / f"{args.log.stem}{suffix}.mp4"
        writer = FFMpegWriter(fps=FPS, bitrate=2500)
        with writer.saving(fig, str(out), dpi=110):
            for t in np.arange(args.start, args.start + args.seconds, args.speed / FPS):
                update(t)
                writer.grab_frame()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
