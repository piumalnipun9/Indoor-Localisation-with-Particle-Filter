"""Export the airport to data/scene/ and draw a 3D preview."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import scene_export as se  # noqa: E402

COLORS = {
    "wall_concrete": "#6b6b6b",
    "wall_drywall": "#d2a35c",
    "wall_glass": "#3aa0c8",
    "pillars": "#2b2b2b",
    "floor": "#efefef",
}
ALPHAS = {"wall_glass": 0.45}
DRAW_ORDER = {"floor": 1, "wall_concrete": 2, "wall_drywall": 3, "wall_glass": 4, "pillars": 5}
HIDDEN_IN_PREVIEW = {"room_ceiling", "hall_roof"}


def blocks_view(quad):
    """True for the two outer walls between the camera and the building."""
    ys = {corner[1] for corner in quad}
    xs = {corner[0] for corner in quad}
    return ys == {0} or xs == {fp.WIDTH}


def main():
    path = se.write_scene(ROOT / "data" / "scene")
    groups = se.build_surfaces()
    for name, (material, quads) in groups.items():
        print(f"{name}: {len(quads)} quads, {material.itu_type}, {material.thickness} m")
    print(f"Wrote {path}")

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    for name, (_, quads) in groups.items():
        if name in HIDDEN_IN_PREVIEW:
            continue
        visible = [q for q in quads if not blocks_view(q)]
        ax.add_collection3d(
            Poly3DCollection(
                visible,
                facecolor=COLORS[name],
                edgecolor="#222222",
                linewidth=0.35,
                alpha=ALPHAS.get(name, 1.0),
                zorder=DRAW_ORDER[name],
            )
        )
    ax.set_xlim(0, fp.WIDTH)
    ax.set_ylim(0, fp.DEPTH)
    ax.set_zlim(0, fp.HALL_CEILING)
    ax.set_box_aspect((fp.WIDTH, fp.DEPTH, fp.HALL_CEILING * 2), zoom=1.0)
    ax.view_init(elev=42, azim=-75)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_title("Airport scene with ceilings and the near outer walls hidden (height drawn at 2x)")
    (ROOT / "outputs").mkdir(exist_ok=True)
    preview = ROOT / "outputs" / "scene_preview.png"
    fig.savefig(preview, dpi=130)
    print(f"Wrote {preview}")


if __name__ == "__main__":
    main()
