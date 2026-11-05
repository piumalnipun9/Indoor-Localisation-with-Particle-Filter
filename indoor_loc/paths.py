"""Move people through the airport without crossing walls."""
import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from . import floorplan as fp

CELL = 0.5
COMFORT_M = 2.5  # People prefer to stay this far from walls.
WALL_PENALTY = 3.0
SMOOTHING_CELLS = 5
NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass(frozen=True)
class WalkConfig:
    speed_mean: float = 1.3
    speed_sd: float = 0.2
    pause_probability: float = 0.5
    pause_min_s: float = 5.0
    pause_max_s: float = 40.0
    dt: float = 0.1


def destinations():
    """Places people walk to: every room, every gate and points along the concourse."""
    points = [room.center for room in fp.build_rooms()]
    points += [((g.x0 + g.x1) / 2, 52.0) for g in fp.build_gates()]
    points += [(x, 30.0) for x in np.arange(10.0, fp.WIDTH, 40.0)]
    return points


class Router:
    """Routes that avoid walls where they can and still pass through doors."""

    def __init__(self):
        self.free = fp.occupancy_grid(CELL)
        self.walkable = binary_erosion(self.free, border_value=0)
        clearance = distance_transform_edt(self.walkable) * CELL
        self.cost = 1.0 + WALL_PENALTY * np.maximum(0.0, COMFORT_M - clearance)
        self._fields = {}

    def _cell(self, point):
        ix = int(np.clip(point[0] / CELL, 0, self.free.shape[1] - 1))
        iy = int(np.clip(point[1] / CELL, 0, self.free.shape[0] - 1))
        if self.walkable[iy, ix]:
            return iy, ix
        ys, xs = np.nonzero(self.walkable)
        nearest = np.argmin((ys - iy) ** 2 + (xs - ix) ** 2)
        return int(ys[nearest]), int(xs[nearest])

    def _steps(self, y, x):
        ny, nx = self.free.shape
        for dy, dx in NEIGHBOURS:
            v, u = y + dy, x + dx
            if not (0 <= v < ny and 0 <= u < nx) or not self.walkable[v, u]:
                continue
            if dy and dx and not (self.walkable[y, u] and self.walkable[v, x]):
                continue
            yield v, u, (1.414 if dy and dx else 1.0) * self.cost[v, u]

    def _field(self, goal):
        """Cost to reach goal from every cell (Dijkstra)."""
        if goal in self._fields:
            return self._fields[goal]
        field = np.full(self.free.shape, np.inf)
        field[goal] = 0.0
        queue = [(0.0, goal)]
        while queue:
            value, (y, x) = heapq.heappop(queue)
            if value > field[y, x]:
                continue
            for v, u, step in self._steps(y, x):
                if value + step < field[v, u]:
                    field[v, u] = value + step
                    heapq.heappush(queue, (value + step, (v, u)))
        self._fields[goal] = field
        return field

    def route(self, start, goal):
        """Polyline in metres from start to goal."""
        goal_cell = self._cell(goal)
        field = self._field(goal_cell)
        current = self._cell(start)
        cells = [current]
        while current != goal_cell:
            current = min(self._steps(*current), key=lambda s: field[s[0], s[1]] + s[2])[:2]
            cells.append(current)

        points = (np.array(cells, dtype=float)[:, ::-1] + 0.5) * CELL
        if len(points) <= SMOOTHING_CELLS:
            return points
        half = SMOOTHING_CELLS // 2
        padded = np.pad(points, ((half, half), (0, 0)), mode="edge")
        kernel = np.ones(SMOOTHING_CELLS) / SMOOTHING_CELLS
        return np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in (0, 1)])


def walk(router, rng, duration, config=WalkConfig(), start=None):
    """Return (x, y) per config.dt for a person visiting random destinations."""
    places = destinations()
    count = int(duration / config.dt)
    position = np.array(start if start is not None else places[rng.integers(len(places))], dtype=float)
    speed = max(0.6, rng.normal(config.speed_mean, config.speed_sd))
    track = []
    while len(track) < count:
        polyline = np.vstack([position, router.route(position, places[rng.integers(len(places))])])
        segments = np.diff(polyline, axis=0)
        lengths = np.hypot(segments[:, 0], segments[:, 1])
        total = lengths.sum()
        if total > 0:
            distance = np.arange(0, total, speed * config.dt)
            cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
            leg = np.column_stack(
                [np.interp(distance, cumulative, polyline[:, 0]), np.interp(distance, cumulative, polyline[:, 1])]
            )
            track.extend(leg)
        position = polyline[-1]
        if rng.random() < config.pause_probability:
            pause = rng.uniform(config.pause_min_s, config.pause_max_s)
            track.extend([position.copy()] * int(pause / config.dt))
    return np.array(track[:count])


def headings(track, dt, window_s=0.5):
    """Walking direction per sample; holds the last direction while standing."""
    window = max(1, int(window_s / dt))
    delta = np.zeros_like(track)
    delta[window:] = track[window:] - track[:-window]
    moving = np.hypot(delta[:, 0], delta[:, 1]) > 1e-6
    angle = np.zeros(len(track))
    angle[moving] = np.arctan2(delta[moving, 1], delta[moving, 0])
    last = 0.0
    for i in range(len(track)):
        if moving[i]:
            last = angle[i]
        angle[i] = last
    return angle
