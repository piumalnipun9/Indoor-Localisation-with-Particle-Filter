"""Measure what one wall costs in Sionna, as a flat sheet and as a two-faced box."""
import os
import tempfile
import sys
from pathlib import Path

import mitsuba as mi

mi.set_variant(os.environ.get("SIONNA_VARIANT", "cuda_ad_mono_polarized"))
import numpy as np
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indoor_loc import scene_export as se  # noqa: E402

FREQUENCY = 2.44e9
HALF = 20.0


def write_scene(directory, material, x_faces):
    meshes = Path(directory) / "meshes"
    meshes.mkdir()
    quads = [se._wall_quad(x, -HALF, x, HALF, -HALF, HALF) for x in x_faces]
    se._write_ply(meshes / "wall.ply", quads)
    xml = Path(directory) / "s.xml"
    xml.write_text(
        '<scene version="2.1.0">'
        f'<bsdf type="itu-radio-material" id="m"><string name="type" value="{material.itu_type}"/>'
        f'<float name="thickness" value="{material.thickness}"/></bsdf>'
        '<shape type="ply" id="mesh-wall"><string name="filename" value="meshes/wall.ply"/>'
        '<boolean name="face_normals" value="true"/><ref id="m" name="bsdf"/></shape></scene>'
    )
    return xml


def received_power(xml):
    scene = load_scene(str(xml), merge_shapes=False)
    scene.frequency = FREQUENCY
    array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.tx_array = array
    scene.rx_array = array
    scene.add(Transmitter("tx", position=[-3.0, 0.0, 0.0]))
    scene.add(Receiver("rx", position=[3.0, 0.0, 0.0]))
    paths = PathSolver()(scene, max_depth=2, los=True, specular_reflection=False, refraction=True)
    a, _ = paths.cir(out_type="numpy")
    return float(np.sum(np.abs(a) ** 2))


def main():
    print(f"{'material':<12}{'thick m':>9}{'sheet dB':>11}{'box dB':>9}")
    free = None
    for name, material in se.WALL_MATERIALS.items():
        with tempfile.TemporaryDirectory() as far, tempfile.TemporaryDirectory() as sheet, tempfile.TemporaryDirectory() as box:
            if free is None:
                free = received_power(write_scene(far, material, [15.0]))
            one = received_power(write_scene(sheet, material, [0.0]))
            two = received_power(write_scene(box, material, [-material.thickness / 2, material.thickness / 2]))
        print(
            f"{name:<12}{material.thickness:>9}"
            f"{10 * np.log10(free / one):>11.2f}{10 * np.log10(free / two):>9.2f}"
        )


if __name__ == "__main__":
    main()
