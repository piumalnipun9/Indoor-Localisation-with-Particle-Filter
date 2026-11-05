"""Checks that simulated people and sensors behave as designed."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import paths as pa  # noqa: E402
from indoor_loc import sensors as se  # noqa: E402


def test_walks_never_cross_a_wall():
    router = pa.Router()
    rng = np.random.default_rng(3)
    track = pa.walk(router, rng, 300.0)
    ix = (track[:, 0] / pa.CELL).astype(int)
    iy = (track[:, 1] / pa.CELL).astype(int)
    assert router.free[iy, ix].all()
    speed = np.hypot(*np.diff(track, axis=0).T) / pa.WalkConfig().dt
    assert speed.max() < 3.0


def test_walks_reach_rooms_through_doors():
    router = pa.Router()
    room = fp.build_rooms()[0]
    polyline = router.route((125.0, 30.0), room.center)
    assert np.hypot(*(polyline[-1] - np.array(room.center))) < 1.5
    wall_y = fp.ROOMS_Y[1]
    for (x0, y0), (x1, y1) in zip(polyline[:-1], polyline[1:]):
        if y0 > wall_y >= y1:
            crossing_x = x0 + (x1 - x0) * (y0 - wall_y) / (y0 - y1)
            assert abs(crossing_x - room.door_x) < fp.DOOR_WIDTH / 2
            return
    raise AssertionError("route never entered the room")


def test_empty_map_cells_are_filled(tmp_path):
    gain = np.full((1, 6, 6), 1e-7, dtype=np.float32)
    gain[0, 2:4, 2:4] = 0
    path = tmp_path / "map.npz"
    np.savez(path, path_gain=gain)
    assert np.allclose(se.load_path_gain_db(path), -70.0, atol=1e-3)


def test_body_loss_only_from_behind():
    tx = np.array([[10.0, 0.0], [-10.0, 0.0]])
    loss = se.body_loss_db(np.zeros(2), 0.0, tx, 10.0)
    assert loss[0] == 0.0
    assert loss[1] == 10.0


def test_person_on_the_line_blocks_low_transmitter_only():
    transmitters = np.array([[10.0, 0.0, 1.6], [10.0, 0.0, 10.0]])
    person = np.array([[8.0, 0.0]])
    blocked = se.crowd_blockers(np.zeros(2), transmitters, person)
    assert blocked.tolist() == [1, 0]


def test_rician_fading_has_expected_spread():
    gain = se.rician_gain_db(np.random.default_rng(0), 6.0, 200000)
    assert 2.0 < gain.std() < 6.0


def test_steps_match_distance_walked():
    truth = np.column_stack([np.linspace(0, 70, 701), np.zeros(701)])
    rng = np.random.default_rng(1)
    times, lengths, _, stride, _ = se.simulate_steps(truth, np.zeros(701), 0.1, rng)
    assert abs(lengths.sum() - 70) < 5
    assert abs(lengths.mean() - stride) < 0.05
