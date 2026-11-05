"""Beacon and access-point layouts. Each layout is a list of Transmitter records."""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from . import floorplan as fp

EXTRAS_FILE = Path(__file__).resolve().parents[1] / "data" / "layout_E_extras.json"
BASE_SPACING = 15.0
MIN_CANDIDATE_GAP = 3.0

LOW_MOUNT_Z = 3.0
CEILING_GAP = 0.05
MIN_WALL_CLEARANCE = 0.25
DOOR_FLANK = 3.0
WALL_STANDOFF = 0.3


@dataclass(frozen=True)
class Transmitter:
    id: str
    kind: str  # "ble" or "wifi"
    x: float
    y: float
    z: float
    mount: str  # "ceiling", "pillar", "wall", "pole"


def _clear_of_walls(x, y):
    """Shift a point sideways so it does not land on a wall."""
    while fp.distance_to_nearest_wall(x, y) < MIN_WALL_CLEARANCE:
        x += 0.5
    return x, y


def _ceiling_ble(i, x, y):
    x, y = _clear_of_walls(x, y)
    return Transmitter(f"ble_{i:03d}", "ble", x, y, fp.ceiling_height(x, y) - CEILING_GAP, "ceiling")


def _grid(spacing):
    columns = math.ceil(fp.WIDTH / spacing)
    rows = math.ceil(fp.DEPTH / spacing)
    x_margin = (fp.WIDTH - (columns - 1) * spacing) / 2
    y_margin = (fp.DEPTH - (rows - 1) * spacing) / 2
    return [
        (x_margin + c * spacing, y_margin + r * spacing)
        for r in range(rows)
        for c in range(columns)
    ]


def even_ceiling(spacing):
    return [_ceiling_ble(i, x, y) for i, (x, y) in enumerate(_grid(spacing))]


def even_low(spacing):
    points = [_clear_of_walls(x, y) for x, y in _grid(spacing)]
    return [
        Transmitter(f"ble_{i:03d}", "ble", x, y, min(LOW_MOUNT_Z, fp.ceiling_height(x, y) - CEILING_GAP), "pole")
        for i, (x, y) in enumerate(points)
    ]


def _on_pillar_face(y):
    """Mount on the side of the pillar facing the middle of the concourse."""
    middle = sum(fp.CONCOURSE_Y) / 2
    offset = fp.PILLAR_SIZE / 2 + WALL_STANDOFF / 3
    return y + offset if y < middle else y - offset


def uneven():
    """Dense at rooms and doors, sparse in the open hall."""
    points = [(x, _on_pillar_face(y), LOW_MOUNT_Z, "pillar") for x, y in fp.build_pillars()]

    for gate in fp.build_gates():
        width = gate.x1 - gate.x0
        for fraction in (0.25, 0.75):
            points.append((gate.x0 + fraction * width, 50.0, LOW_MOUNT_Z, "pole"))

    for room in fp.build_rooms():
        x, y = room.center
        points.append((x, y, fp.ROOM_CEILING - CEILING_GAP, "ceiling"))
        for offset in (-DOOR_FLANK, DOOR_FLANK):
            points.append((room.door_x + offset, room.y1 + WALL_STANDOFF, LOW_MOUNT_Z, "wall"))

    return [Transmitter(f"ble_{i:03d}", "ble", x, y, z, mount) for i, (x, y, z, mount) in enumerate(points)]


def wifi_access_points():
    """Three staggered rows with about 25 m between neighbouring access points."""
    spacing = fp.WIDTH / 8
    points = [(spacing * (k + 0.5), y) for y in (10.0, 50.0) for k in range(8)]
    points += [(spacing * (k + 1), 30.0) for k in range(7)]
    result = []
    for i, (x, y) in enumerate(points):
        x, y = _clear_of_walls(x, y)
        result.append(Transmitter(f"wifi_{i:02d}", "wifi", x, y, fp.ceiling_height(x, y) - CEILING_GAP, "ceiling"))
    return result


def candidate_pool():
    """Ceiling positions the extra tags of layout E may choose from."""
    points = []
    for room in fp.build_rooms():
        x = room.x0 + 3.75
        while x < room.x1:
            points += [(x, y) for y in (5.0, 12.0, 18.0)]
            x += 7.5
    for x in (12.5 + 15 * k for k in range(16)):
        points += [(x, 30.0), (x, 45.0)]
    for x in (2.5, fp.WIDTH - 2.5):
        points += [(x, y) for y in (25.0, 35.0, 45.0, 55.0)]

    base = even_ceiling(BASE_SPACING)
    kept = []
    for x, y in points:
        if all(math.hypot(x - b.x, y - b.y) >= MIN_CANDIDATE_GAP for b in base):
            kept.append((x, y))
    return [_ceiling_ble(i, x, y) for i, (x, y) in enumerate(kept)]


def layout_e():
    """Layout B plus the extra tags chosen by scripts/select_layout.py."""
    base = even_ceiling(BASE_SPACING)
    extras = json.loads(EXTRAS_FILE.read_text())
    return base + [
        Transmitter(f"ble_{len(base) + i:03d}", "ble", e["x"], e["y"], e["z"], "ceiling")
        for i, e in enumerate(extras)
    ]


def all_layouts():
    layouts = {
        "A_even_ceiling_10m": even_ceiling(10.0),
        "B_even_ceiling_15m": even_ceiling(BASE_SPACING),
        "C_even_low_15m": even_low(BASE_SPACING),
        "D_uneven": uneven(),
    }
    if EXTRAS_FILE.exists():
        layouts["E_B_plus_extras"] = layout_e()
    return layouts


def to_records(transmitters):
    return [asdict(t) for t in transmitters]
