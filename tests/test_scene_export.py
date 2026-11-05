"""Checks that the exported scene is complete and that Sionna RT can load it."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indoor_loc import floorplan as fp  # noqa: E402
from indoor_loc import scene_export as se  # noqa: E402


def test_every_group_has_quads_and_material():
    for name, (material, quads) in se.build_surfaces().items():
        assert quads, name
        assert material.thickness > 0, name


def test_door_gaps_stay_open_below_the_lintel():
    quads = [q for _, group in se.build_surfaces().items() for q in group[1]]
    for room in fp.build_rooms():
        for quad in quads:
            xs = [c[0] for c in quad]
            ys = [c[1] for c in quad]
            zs = [c[2] for c in quad]
            on_front_wall = min(ys) == max(ys) == room.y1
            covers_door = min(xs) < room.door_x < max(xs)
            if on_front_wall and covers_door:
                assert min(zs) >= fp.DOOR_HEIGHT, room.name


def test_sionna_loads_the_scene(tmp_path):
    pytest.importorskip("sionna.rt")
    import mitsuba as mi

    mi.set_variant(os.environ.get("SIONNA_VARIANT", "llvm_ad_mono_polarized"))
    from sionna.rt import load_scene

    scene = load_scene(str(se.write_scene(tmp_path)), merge_shapes=False)
    groups = se.build_surfaces()
    assert set(scene.objects) == set(groups)
    for name, (material, _) in groups.items():
        loaded = scene.objects[name].radio_material
        assert float(loaded.thickness[0] if hasattr(loaded.thickness, "__len__") else loaded.thickness) == pytest.approx(material.thickness)
