"""Checks that the floor plan and layouts are physically sensible."""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import layouts as ly  # noqa: E402


def test_room_row_spans_full_width():
    assert sum(width for _, width, _ in fp.ROOM_ROW) == fp.WIDTH


def test_transmitters_are_inside_and_clear_of_walls():
    groups = list(ly.all_layouts().values()) + [ly.wifi_access_points()]
    for group in groups:
        assert len({t.id for t in group}) == len(group)
        for t in group:
            assert 0 < t.x < fp.WIDTH and 0 < t.y < fp.DEPTH, t
            assert 0 < t.z < fp.ceiling_height(t.x, t.y), t
            assert fp.distance_to_nearest_wall(t.x, t.y) >= 0.25, t


def test_transmitters_are_outside_pillars():
    half = fp.PILLAR_SIZE / 2
    for group in list(ly.all_layouts().values()) + [ly.wifi_access_points()]:
        for t in group:
            for x, y in fp.build_pillars():
                assert abs(t.x - x) > half or abs(t.y - y) > half, t


def test_every_walkable_cell_is_reachable():
    free = fp.occupancy_grid()
    ny, nx = free.shape
    start = (int(30 / 0.5), int(125.25 / 0.5))
    assert free[start]
    seen = {start}
    queue = deque([start])
    while queue:
        y, x = queue.popleft()
        for ny_, nx_ in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny_ < ny and 0 <= nx_ < nx and free[ny_, nx_] and (ny_, nx_) not in seen:
                seen.add((ny_, nx_))
                queue.append((ny_, nx_))
    assert len(seen) == int(free.sum()), f"{int(free.sum()) - len(seen)} cells unreachable"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"PASS {name}")
