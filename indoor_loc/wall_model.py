"""Count the walls of each type on the straight line from a phone to a tag.

This is the geometry for the Motley-Keenan multi-wall path-loss model. It only uses
the floor plan, which a real deployment would also have.
"""
import numpy as np

from . import floorplan as fp
from . import scene_export as se

TYPES = ("concrete", "drywall", "glass", "ceiling")
GRID_M = 1.0
PHONE_HEIGHT_M = 1.2


def wall_pieces():
    """Rows of (x0, y0, x1, y1, z0, z1, type index), including the wall above each door."""
    pieces = []
    surfaces = se.build_surfaces()
    for index, name in enumerate(TYPES[:3]):
        for quad in surfaces[f"wall_{name}"][1]:
            corners = np.array(quad)
            pieces.append(
                (corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1], corners[:, 2].min(), corners[:, 2].max(), index)
            )
    return np.array(pieces)


def crossing_counts(points, tag, pieces):
    """Walls crossed per type from each point (at phone height) to one tag: (points, types)."""
    counts = np.zeros((len(points), len(TYPES)), dtype=np.uint8)
    ray = np.asarray(tag[:2], dtype=float) - points
    rise = float(tag[2]) - PHONE_HEIGHT_M
    for x0, y0, x1, y1, z0, z1, kind in pieces:
        wall = np.array([x1 - x0, y1 - y0])
        offset = np.array([x0, y0]) - points
        denominator = ray[:, 0] * wall[1] - ray[:, 1] * wall[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            along_ray = (offset[:, 0] * wall[1] - offset[:, 1] * wall[0]) / denominator
            along_wall = (offset[:, 0] * ray[:, 1] - offset[:, 1] * ray[:, 0]) / denominator
        height = PHONE_HEIGHT_M + along_ray * rise
        hit = (
            (denominator != 0) & (along_ray > 0) & (along_ray < 1)
            & (along_wall >= 0) & (along_wall < 1) & (height >= z0) & (height <= z1)
        )
        counts[hit, int(kind)] += 1

    ceiling = fp.ROOM_CEILING
    if float(tag[2]) > ceiling:
        fraction = (ceiling - PHONE_HEIGHT_M) / rise
        y_at_ceiling = points[:, 1] + fraction * ray[:, 1]
        counts[y_at_ceiling < fp.ROOMS_Y[1], TYPES.index("ceiling")] += 1
    return counts


def grid_points():
    xs = np.arange(GRID_M / 2, fp.WIDTH, GRID_M)
    ys = np.arange(GRID_M / 2, fp.DEPTH, GRID_M)
    return xs, ys, np.array(np.meshgrid(xs, ys)).reshape(2, -1).T


def count_grid(tags):
    """Crossings for every tag on a 1 m grid: (tags, ny, nx, types)."""
    xs, ys, points = grid_points()
    pieces = wall_pieces()
    grid = np.stack([crossing_counts(points, tag, pieces) for tag in tags])
    return grid.reshape(len(tags), len(ys), len(xs), len(TYPES))
