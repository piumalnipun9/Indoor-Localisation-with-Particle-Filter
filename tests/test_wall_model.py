"""Checks the wall counting behind the multi-wall model."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import wall_model as wm  # noqa: E402

PIECES = wm.wall_pieces()


def counts(point, tag):
    result = wm.crossing_counts(np.array([point], dtype=float), np.array(tag, dtype=float), PIECES)[0]
    return dict(zip(wm.TYPES, result.tolist()))


def test_nothing_between_two_points_in_the_hall():
    assert sum(counts((50, 30), (60, 30, 3.0)).values()) == 0


def test_room_tag_seen_from_hall_crosses_its_front_wall():
    room = fp.build_rooms()[0]  # Shop 1 has a glass front.
    result = counts((room.door_x + 5, 25), (room.door_x + 5, 10, 2.95))
    assert result["glass"] == 1 and result["drywall"] == 0


def test_neighbouring_rooms_cross_one_drywall():
    a, b = fp.build_rooms()[:2]
    result = counts((a.x1 - 3, 10), (b.x0 + 3, 10, 2.95))
    assert result["drywall"] == 1


def test_open_door_is_not_a_wall_but_the_space_above_it_is():
    room = fp.build_rooms()[0]
    assert sum(counts((room.door_x, 15), (room.door_x, 25, 1.5)).values()) == 0
    assert sum(counts((room.door_x, 15), (room.door_x, 25, 5.0)).values()) >= 1


def test_high_hall_tag_seen_from_a_room_also_crosses_the_ceiling():
    result = counts((50, 15), (50, 22.5, 9.95))
    assert result["ceiling"] == 1


def test_grid_shape():
    grid = wm.count_grid(np.array([[10.0, 10.0, 2.95]]))
    assert grid.shape == (1, int(fp.DEPTH), int(fp.WIDTH), len(wm.TYPES))
