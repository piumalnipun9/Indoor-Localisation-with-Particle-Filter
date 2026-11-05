"""Write data/layouts.json and draw the floor plan with all four beacon layouts."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import layouts as ly  # noqa: E402

WALL_COLORS = {"concrete": "#333333", "drywall": "#8a8a8a", "glass": "#3aa0c8"}
MARKERS = {
    "ceiling": ("o", "#d1495b"),
    "pillar": ("^", "#2e7d32"),
    "wall": ("s", "#e08e0b"),
    "pole": ("D", "#1f77b4"),
}


def draw_floorplan(ax):
    ax.add_patch(Rectangle((0, fp.ROOMS_Y[0]), fp.WIDTH, 20, color="#f4efe6", zorder=0))
    ax.add_patch(Rectangle((0, fp.CONCOURSE_Y[0]), fp.WIDTH, 20, color="#eef3f7", zorder=0))
    ax.add_patch(Rectangle((0, fp.GATES_Y[0]), fp.WIDTH, 20, color="#f6f6f6", zorder=0))
    for wall in fp.build_walls():
        ax.plot([wall.x0, wall.x1], [wall.y0, wall.y1], color=WALL_COLORS[wall.material], lw=1.4, zorder=1)
    half = fp.PILLAR_SIZE / 2
    for x, y in fp.build_pillars():
        ax.add_patch(Rectangle((x - half, y - half), fp.PILLAR_SIZE, fp.PILLAR_SIZE, color="#555555", zorder=1))
    for room in fp.build_rooms():
        ax.text(room.door_x, 2, room.name, ha="center", fontsize=6.5, color="#555555")
    for gate in fp.build_gates():
        ax.text((gate.x0 + gate.x1) / 2, 57, gate.name, ha="center", fontsize=6.5, color="#555555")
    ax.text(fp.WIDTH / 2, 29.2, "Main concourse", ha="center", fontsize=7, color="#555555")
    ax.set_xlim(-3, fp.WIDTH + 3)
    ax.set_ylim(-3, fp.DEPTH + 3)
    ax.set_aspect("equal")
    ax.set_ylabel("y (m)")


def draw_transmitters(ax, beacons, access_points):
    for mount, (marker, color) in MARKERS.items():
        points = [t for t in beacons if t.mount == mount]
        ax.scatter([t.x for t in points], [t.y for t in points], marker=marker, c=color, s=20, zorder=3)
    ax.scatter([t.x for t in access_points], [t.y for t in access_points], marker="*", c="#6a3fb5", s=55, zorder=3)


def main():
    layouts = ly.all_layouts()
    access_points = ly.wifi_access_points()

    data = {
        "map": {"width_m": fp.WIDTH, "depth_m": fp.DEPTH},
        "wifi": ly.to_records(access_points),
        "ble_layouts": {name: ly.to_records(items) for name, items in layouts.items()},
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "layouts.json").write_text(json.dumps(data, indent=2))

    fig, axes = plt.subplots(len(layouts), 1, figsize=(13, 16), constrained_layout=True)
    for ax, (name, beacons) in zip(axes, layouts.items()):
        draw_floorplan(ax)
        draw_transmitters(ax, beacons, access_points)
        ax.set_title(f"{name}: {len(beacons)} BLE beacons, {len(access_points)} Wi-Fi APs", loc="left", fontsize=11)
    axes[-1].set_xlabel("x (m)")

    handles = [
        plt.Line2D([], [], marker=m, color=c, ls="", label=f"BLE ({mount})") for mount, (m, c) in MARKERS.items()
    ]
    handles.append(plt.Line2D([], [], marker="*", color="#6a3fb5", ls="", markersize=10, label="Wi-Fi AP"))
    handles += [plt.Line2D([], [], color=c, label=f"{name} wall") for name, c in WALL_COLORS.items()]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=8, frameon=False)

    (ROOT / "outputs").mkdir(exist_ok=True)
    path = ROOT / "outputs" / "layouts_overview.png"
    fig.savefig(path, dpi=130)
    print(f"Wrote {path}")
    for name, beacons in layouts.items():
        print(f"{name}: {len(beacons)} beacons")
    print(f"Wi-Fi APs: {len(access_points)}")


if __name__ == "__main__":
    main()
