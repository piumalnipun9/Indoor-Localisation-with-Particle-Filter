"""Turn the floor plan into a 3D Mitsuba scene that Sionna RT can load.

Walls are flat sheets, not boxes. Sionna treats every surface a ray crosses as a
full slab of the material's `thickness`, so a box wall with two faces would count
its loss twice. Pillars are solid, so they stay closed boxes and each face carries
half the pillar's real depth.
"""
import struct
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from . import floorplan as fp

FLOOR_THICKNESS = 0.30
ROOF_THICKNESS = 0.20
ROOM_CEILING_THICKNESS = 0.02


@dataclass(frozen=True)
class RadioMaterial:
    itu_type: str
    thickness: float


# Maps floorplan.py wall materials to Sionna ITU materials.
WALL_MATERIALS = {
    "concrete": RadioMaterial("concrete", 0.20),
    "drywall": RadioMaterial("plasterboard", 0.10),
    "glass": RadioMaterial("glass", 0.02),
}

SURFACE_MATERIALS = {
    "floor": RadioMaterial("concrete", FLOOR_THICKNESS),
    "room_ceiling": RadioMaterial("ceiling_board", ROOM_CEILING_THICKNESS),
    "hall_roof": RadioMaterial("concrete", ROOF_THICKNESS),
    "pillars": RadioMaterial("concrete", fp.PILLAR_SIZE / 2),
}


def _wall_quad(x0, y0, x1, y1, z0, z1):
    return [(x0, y0, z0), (x1, y0 if y0 == y1 else y1, z0), (x1, y1, z1), (x0, y0, z1)]


def _horizontal_quad(x0, x1, y0, y1, z):
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def _pillar_quads(x, y, height):
    h = fp.PILLAR_SIZE / 2
    x0, x1, y0, y1 = x - h, x + h, y - h, y + h
    return [
        _wall_quad(x0, y0, x1, y0, 0, height),
        _wall_quad(x1, y0, x1, y1, 0, height),
        _wall_quad(x1, y1, x0, y1, 0, height),
        _wall_quad(x0, y1, x0, y0, 0, height),
    ]


def build_surfaces():
    """Return {group name: (RadioMaterial, list of 4-corner quads)}."""
    groups = {f"wall_{name}": (material, []) for name, material in WALL_MATERIALS.items()}

    for wall in fp.build_walls():
        quads = groups[f"wall_{wall.material}"][1]
        quads.append(_wall_quad(wall.x0, wall.y0, wall.x1, wall.y1, 0, wall.height))

    half_door = fp.DOOR_WIDTH / 2
    for room, (_, _, front) in zip(fp.build_rooms(), fp.ROOM_ROW):
        lintel = _wall_quad(
            room.door_x - half_door, room.y1, room.door_x + half_door, room.y1, fp.DOOR_HEIGHT, fp.HALL_CEILING
        )
        groups[f"wall_{front}"][1].append(lintel)

    groups["floor"] = (SURFACE_MATERIALS["floor"], [_horizontal_quad(0, fp.WIDTH, 0, fp.DEPTH, 0)])
    groups["room_ceiling"] = (
        SURFACE_MATERIALS["room_ceiling"],
        [_horizontal_quad(0, fp.WIDTH, *fp.ROOMS_Y, fp.ROOM_CEILING)],
    )
    groups["hall_roof"] = (
        SURFACE_MATERIALS["hall_roof"],
        [_horizontal_quad(0, fp.WIDTH, fp.CONCOURSE_Y[0], fp.DEPTH, fp.HALL_CEILING)],
    )
    groups["pillars"] = (
        SURFACE_MATERIALS["pillars"],
        [quad for x, y in fp.build_pillars() for quad in _pillar_quads(x, y, fp.HALL_CEILING)],
    )
    return groups


def _write_ply(path, quads):
    vertices = np.asarray(quads, dtype="<f4").reshape(-1, 3)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {2 * len(quads)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())
        for i in range(len(quads)):
            a, b, c, d = (4 * i + k for k in range(4))
            f.write(struct.pack("<B3iB3i", 3, a, b, c, 3, a, c, d))


def write_scene(out_dir):
    """Write airport.xml and meshes/*.ply; return the path of the XML file."""
    out_dir = Path(out_dir)
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    groups = build_surfaces()

    lines = ['<scene version="2.1.0">', ""]
    for name, (material, _) in groups.items():
        lines += [
            f'    <bsdf type="itu-radio-material" id={quoteattr(name + "-mat")}>',
            f'        <string name="type" value={quoteattr(material.itu_type)}/>',
            f'        <float name="thickness" value="{material.thickness}"/>',
            "    </bsdf>",
        ]
    lines.append("")
    for name, (_, quads) in groups.items():
        _write_ply(out_dir / "meshes" / f"{name}.ply", quads)
        lines += [
            f'    <shape type="ply" id={quoteattr("mesh-" + name)}>',
            f'        <string name="filename" value={quoteattr("meshes/" + name + ".ply")}/>',
            '        <boolean name="face_normals" value="true"/>',
            f'        <ref id={quoteattr(name + "-mat")} name="bsdf"/>',
            "    </shape>",
        ]
    lines += ["", "</scene>", ""]

    path = out_dir / "airport.xml"
    path.write_text("\n".join(lines))
    return path
