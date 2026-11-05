"""One-floor airport concourse: the single source of truth for geometry.

Coordinates are metres. x runs along the concourse, y across it, z is up.
Sionna RT scene export and the particle filter's wall map both build on this.
"""
from dataclasses import dataclass

import numpy as np

WIDTH = 250.0
DEPTH = 60.0

ROOMS_Y = (0.0, 20.0)
CONCOURSE_Y = (20.0, 40.0)
GATES_Y = (40.0, 60.0)

HALL_CEILING = 10.0
ROOM_CEILING = 3.0

DOOR_WIDTH = 2.4
DOOR_HEIGHT = 2.2
PILLAR_SIZE = 0.6
PILLAR_SPACING = 15.0
PILLAR_ROWS_Y = (26.0, 34.0)
GATE_COUNT = 8
GATE_DIVIDER_START_Y = 46.0  # Dividers stop short of the concourse.

# (name, width in metres, material of the wall facing the concourse)
ROOM_ROW = [
    ("Shop 1", 20.0, "glass"),
    ("Shop 2", 20.0, "glass"),
    ("Food 1", 30.0, "glass"),
    ("WC 1", 15.0, "drywall"),
    ("Lounge", 30.0, "glass"),
    ("Staff", 20.0, "drywall"),
    ("Prayer", 15.0, "drywall"),
    ("Shop 3", 20.0, "glass"),
    ("Food 2", 30.0, "glass"),
    ("WC 2", 15.0, "drywall"),
    ("Store", 35.0, "drywall"),
]


@dataclass(frozen=True)
class Wall:
    x0: float
    y0: float
    x1: float
    y1: float
    material: str
    height: float


@dataclass(frozen=True)
class Room:
    name: str
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def door_x(self):
        return (self.x0 + self.x1) / 2

    @property
    def center(self):
        return (self.door_x, (self.y0 + self.y1) / 2)


@dataclass(frozen=True)
class Gate:
    name: str
    x0: float
    x1: float


def build_rooms():
    rooms, x = [], 0.0
    for name, width, _ in ROOM_ROW:
        rooms.append(Room(name, x, x + width, *ROOMS_Y))
        x += width
    return rooms


def build_gates():
    width = WIDTH / GATE_COUNT
    return [Gate(f"Gate {i + 1}", i * width, (i + 1) * width) for i in range(GATE_COUNT)]


def build_walls():
    walls = [
        Wall(0, 0, WIDTH, 0, "concrete", ROOM_CEILING),
        Wall(0, 0, 0, ROOMS_Y[1], "concrete", ROOM_CEILING),
        Wall(WIDTH, 0, WIDTH, ROOMS_Y[1], "concrete", ROOM_CEILING),
        Wall(0, ROOMS_Y[1], 0, DEPTH, "concrete", HALL_CEILING),
        Wall(WIDTH, ROOMS_Y[1], WIDTH, DEPTH, "concrete", HALL_CEILING),
        Wall(0, DEPTH, WIDTH, DEPTH, "glass", HALL_CEILING),
    ]
    rooms = build_rooms()
    for room, (_, _, front) in zip(rooms, ROOM_ROW):
        y = room.y1
        half = DOOR_WIDTH / 2
        walls.append(Wall(room.x0, y, room.door_x - half, y, front, HALL_CEILING))
        walls.append(Wall(room.door_x + half, y, room.x1, y, front, HALL_CEILING))
    for room in rooms[:-1]:
        walls.append(Wall(room.x1, room.y0, room.x1, room.y1, "drywall", ROOM_CEILING))
    for gate in build_gates()[:-1]:
        walls.append(Wall(gate.x1, GATE_DIVIDER_START_Y, gate.x1, DEPTH, "glass", ROOM_CEILING))
    return walls


def build_pillars():
    """Pillar centres. The two rows are offset so beacons can sit staggered."""
    pillars = []
    for row, y in enumerate(PILLAR_ROWS_Y):
        x = PILLAR_SPACING / 2 if row == 0 else PILLAR_SPACING
        while x < WIDTH:
            pillars.append((x, y))
            x += PILLAR_SPACING
    return pillars


def ceiling_height(x, y):
    return ROOM_CEILING if y < ROOMS_Y[1] else HALL_CEILING


def distance_to_nearest_wall(x, y):
    best = float("inf")
    for w in build_walls():
        dx, dy = w.x1 - w.x0, w.y1 - w.y0
        t = np.clip(((x - w.x0) * dx + (y - w.y0) * dy) / (dx * dx + dy * dy), 0, 1)
        best = min(best, float(np.hypot(x - w.x0 - t * dx, y - w.y0 - t * dy)))
    return best


def occupancy_grid(resolution=0.5):
    """Boolean grid indexed [iy, ix]; True is walkable. Used for the wall check."""
    nx, ny = int(WIDTH / resolution), int(DEPTH / resolution)
    free = np.ones((ny, nx), dtype=bool)
    xs = (np.arange(nx) + 0.5) * resolution
    ys = (np.arange(ny) + 0.5) * resolution
    half_thickness = resolution / 2
    for w in build_walls():
        x0, x1 = sorted((w.x0, w.x1))
        y0, y1 = sorted((w.y0, w.y1))
        ix = (xs >= x0 - half_thickness) & (xs <= x1 + half_thickness)
        iy = (ys >= y0 - half_thickness) & (ys <= y1 + half_thickness)
        free[np.ix_(iy, ix)] = False
    half = PILLAR_SIZE / 2
    for px, py in build_pillars():
        ix = np.abs(xs - px) <= half
        iy = np.abs(ys - py) <= half
        free[np.ix_(iy, ix)] = False
    return free
