"""Render the BLE tracking animation in Isaac Sim (headless, RTX real-time).

Copy scene.json and frames.npz (from prep.py) into the container, then run inside it:
    /isaac-sim/python.sh render_isaac.py --data /root/anim --out /root/anim/frames [--start 0 --end 900 --step 1]
"""
import argparse
import json
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/root/anim")
parser.add_argument("--out", default="/root/anim/frames")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=900)
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--subframes", type=int, default=8)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": args.width, "height": args.height})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
settings.set("/rtx/post/motionblur/enabled", False)
settings.set("/rtx/post/dof/enabled", False)
settings.set("/rtx/directLighting/sampledLighting/enabled", True)
settings.set("/rtx/post/lensFlares/enabled", True)  # Gives the emissive parts a soft bloom.
settings.set("/rtx/post/lensFlares/flareScale", 0.08)
settings.set("/rtx/post/lensFlares/cutoffPoint", [2.0, 2.0, 2.0])
settings.set("/rtx/post/tonemap/op", 6)  # ACES
settings.set("/rtx/post/histogram/enabled", False)  # Fixed exposure keeps the dark look.

scene = json.load(open(os.path.join(args.data, "scene.json")))
data = np.load(os.path.join(args.data, "frames.npz"))
n_frames = len(data["walk_t"])
tags = np.array(scene["tags"], dtype=float)
tags[tags[:, 2] > 5, 2] = 4.0  # Hall beacons drawn at 4 m (really 10 m) so they stay in shot.

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")


def hexrgb(h):
    h = h.lstrip("#")
    return Gf.Vec3f(*(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)))


def material(name, color, rough=0.6, metal=0.0, emit=None, emit_gain=1.0, opacity=1.0):
    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(hexrgb(color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if emit:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(hexrgb(emit) * emit_gain)
    if opacity < 1.0:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def no_shadow(prim):
    prim.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim() if hasattr(prim, "GetPrim") else prim).Bind(mat)


# Palette: dark floor, light architecture, glowing data.
M = {
    "floor": material("Floor", "#161a21", rough=0.35),
    "room_floor": material("RoomFloor", "#1f2530", rough=0.5),
    "concrete": material("Concrete", "#8e949c", rough=0.8),
    "drywall": material("Drywall", "#d5d8de", rough=0.7),
    "glass": material("Glass", "#9cc8e8", rough=0.05, opacity=0.28),
    "pillar": material("Pillar", "#5b616b", rough=0.6),
    "beam": material("Beam", "#3a404a", rough=0.6, metal=0.3),
    "beacon_dim": material("BeaconDim", "#1b4f7a", emit="#2f8fd8", emit_gain=1.5),
    "beacon_on": material("BeaconOn", "#3aa0ff", emit="#3aa0ff", emit_gain=5.0),
    "ring_a": material("RingA", "#3aa0ff", emit="#3aa0ff", emit_gain=3.5),
    "ring_b": material("RingB", "#3aa0ff", emit="#3aa0ff", emit_gain=2.2),
    "ring_c": material("RingC", "#3aa0ff", emit="#3aa0ff", emit_gain=1.4),
    "particle": material("Particle", "#ff4d00", emit="#ff3c00", emit_gain=0.9),
    "estimate": material("Estimate", "#18e0b0", emit="#18e0b0", emit_gain=3.0),
    "est_trail": material("EstTrail", "#18e0b0", emit="#18e0b0", emit_gain=2.0),
    "true_trail": material("TrueTrail", "#ffffff", emit="#ffffff", emit_gain=0.6),
    "walker": material("Walker", "#eef0f4", rough=0.45),
}


def cube(path, center, size, mat):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(c, mat)
    return c


# Static architecture.
cube("/World/Floor", (scene["width"] / 2, scene["depth"] / 2, -0.05), (scene["width"] + 40, scene["depth"] + 40, 0.1), M["floor"])
for i, r in enumerate(scene["rooms"]):
    cube(f"/World/RoomFloor_{i}", ((r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2, 0.005),
         (r["x1"] - r["x0"] - 0.2, r["y1"] - r["y0"] - 0.2, 0.01), M["room_floor"])
for i, (mat, cx, cy, cz, sx, sy, sz) in enumerate(scene["boxes"]):
    cube(f"/World/Arch/{mat}_{i}", (cx, cy, cz), (sx, sy, sz), M[mat])


def annulus_mesh(path, inner, outer, mat, segments=72):
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pts = [(outer * math.cos(t), outer * math.sin(t), 0.0) for t in a] + [(inner * math.cos(t), inner * math.sin(t), 0.0) for t in a]
    counts, idx = [], []
    for k in range(segments):
        j = (k + 1) % segments
        counts.append(4)
        idx += [k, j, segments + j, segments + k]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDoubleSidedAttr(True)
    m.CreateNormalsAttr([(0, 0, 1)] * len(pts))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    bind(m, mat)
    no_shadow(m)
    return m


def instancer(path, protos):
    inst = UsdGeom.PointInstancer.Define(stage, path)
    rel = inst.CreatePrototypesRel()
    for p in protos:
        rel.AddTarget(p.GetPath())
    return inst


# Beacons: a dim and a lit puck, switched per frame; expanding rings when a beacon is used.
UsdGeom.Scope.Define(stage, "/World/Protos")
protos = []
for name in ("beacon_dim", "beacon_on"):
    cyl = UsdGeom.Cylinder.Define(stage, f"/World/Protos/{name}")
    cyl.CreateRadiusAttr(0.32)
    cyl.CreateHeightAttr(0.14)
    cyl.CreateAxisAttr("Z")
    bind(cyl, M[name])
    no_shadow(cyl)
    protos.append(cyl)
beacons = instancer("/World/Beacons", protos)
beacons.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(*t) for t in tags]))
beacons.CreateProtoIndicesAttr(Vt.IntArray([0] * len(tags)))

ring_protos = [annulus_mesh(f"/World/Protos/ring_{k}", 0.93, 1.0, M[f"ring_{k}"]) for k in "abc"]
rings = instancer("/World/Rings", ring_protos)
rings.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(*t) for t in tags]))
rings.CreateProtoIndicesAttr(Vt.IntArray([0] * len(tags)))
rings.CreateScalesAttr(Vt.Vec3fArray([Gf.Vec3f(1e-4)] * len(tags)))

# Particles.
sphere = UsdGeom.Sphere.Define(stage, "/World/Protos/particle")
sphere.CreateRadiusAttr(0.065)
bind(sphere, M["particle"])
no_shadow(sphere)
particles = instancer("/World/Particles", [sphere])
n_particles = 320
particles.CreateProtoIndicesAttr(Vt.IntArray([0] * n_particles))
particles.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, -50)] * n_particles))

# Estimate marker: glowing ring plus a centre dot on the floor.
est = UsdGeom.Xform.Define(stage, "/World/Estimate")
est_op = est.AddTranslateOp()
est_scale = est.AddScaleOp()
annulus_mesh("/World/Estimate/ring", 0.85, 1.05, M["estimate"])
dot = UsdGeom.Cylinder.Define(stage, "/World/Estimate/dot")
dot.CreateRadiusAttr(0.18)
dot.CreateHeightAttr(0.04)
dot.CreateAxisAttr("Z")
bind(dot, M["estimate"])
no_shadow(dot)


def curve(path, mat, width):
    c = UsdGeom.BasisCurves.Define(stage, path)
    c.CreateTypeAttr(UsdGeom.Tokens.linear)
    c.CreateCurveVertexCountsAttr([2])
    c.CreatePointsAttr([(0, 0, -50), (0, 0.01, -50)])
    c.CreateWidthsAttr([width])
    c.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    bind(c, mat)
    no_shadow(c)
    return c


est_trail = curve("/World/EstTrail", M["est_trail"], 0.16)
true_trail = curve("/World/TrueTrail", M["true_trail"], 0.06)

# People: simple figures whose arms and legs swing with distance walked.
def capsule(path, radius, height, z, mat):
    c = UsdGeom.Capsule.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr("Z")
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
    bind(c, mat)
    return c


def person(path, mat):
    root = UsdGeom.Xform.Define(stage, path)
    t, r = root.AddTranslateOp(), root.AddRotateZOp()
    capsule(f"{path}/torso", 0.17, 0.42, 1.22, mat)
    head = UsdGeom.Sphere.Define(stage, f"{path}/head")
    head.CreateRadiusAttr(0.12)
    UsdGeom.Xformable(head).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.67))
    bind(head, mat)
    limbs = []
    for name, y, z, radius, length, sign in (("leg_l", 0.1, 0.92, 0.075, 0.7, 1), ("leg_r", -0.1, 0.92, 0.075, 0.7, -1),
                                             ("arm_l", 0.25, 1.46, 0.055, 0.5, -1), ("arm_r", -0.25, 1.46, 0.055, 0.5, 1)):
        pivot = UsdGeom.Xform.Define(stage, f"{path}/{name}")
        pivot.AddTranslateOp().Set(Gf.Vec3d(0, y, z))
        rot = pivot.AddRotateYOp()
        capsule(f"{path}/{name}/limb", radius, length, -(length / 2 + radius), mat)
        limbs.append((rot, sign, 1.0 if name.startswith("leg") else 0.8))
    return t, r, limbs


def pose(person_ops, xy, heading_rad, dist, gain):
    t, r, limbs = person_ops
    t.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), 0.0))
    r.Set(float(np.degrees(heading_rad)))
    phase = float(dist) / 0.7 * math.pi
    for rot, sign, amp in limbs:
        rot.Set(float(sign * amp * 26.0 * gain * math.sin(phase)))


# Our man starts white like everyone else and turns yellow when tracking starts.
hero_mat = material("Hero", "#eef0f4", rough=0.45, emit="#000000")
hero_shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Hero/Shader"))
hero = person("/World/Walker", hero_mat)
others = [person(f"/World/People/p{i}", M["walker"]) for i in range(data["crowd_xy"].shape[1])]

# Lights.
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(90)
dome.CreateColorAttr(Gf.Vec3f(0.75, 0.82, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(700)
sun.CreateAngleAttr(1.5)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(38, 0, -30))

# Camera.
cam = UsdGeom.Camera.Define(stage, "/World/Cam")
cam.CreateFocalLengthAttr(22.0)
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955 * args.height / args.width)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 2000.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()

render_product = rep.create.render_product("/World/Cam", (args.width, args.height))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])

walk_t, fps, intro = data["walk_t"], float(data["fps"]), float(data["intro_s"])
moving = data["walker_moving"].astype(float)
swing_gain = np.convolve(moving, np.ones(9) / 9, mode="same")  # Ease limbs in and out of pauses.
cloud, est_xy, walker_xy = data["cloud"], data["est"], data["walker"]
os.makedirs(args.out, exist_ok=True)


def set_frame(f):
    tsec = f / fps
    shown = tsec >= intro  # The filter starts after the intro.

    # Beacons: lit while used by the latest scan; rings expand for 0.8 s after each scan.
    active = data["active"][f]
    beacons.GetProtoIndicesAttr().Set(Vt.IntArray(active.astype(int).tolist()))
    age = float(data["scan_age"][f])
    if tsec < intro:  # Intro: a soft wave across every beacon.
        phase = (tsec * 1.6 - tags[:, 0] / 120.0) % 1.0
        scale = np.where(phase < 0.6, 0.3 + 6.0 * phase, 1e-4)
        level = np.minimum((phase / 0.6 * 3).astype(int), 2)
    else:
        grow = ((tsec - intro) * 1.2 + tags[:, 0] / 37.0) % 1.0  # Steady ripple from the beacons in use.
        scale = np.where(active, 0.4 + 4.5 * grow, 1e-4)
        level = np.minimum((grow * 3).astype(int), 2)
    rings.GetScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(s), float(s), 1.0) for s in scale]))
    rings.GetProtoIndicesAttr().Set(Vt.IntArray(level.tolist()))

    # Particles fade in by shrinking from below the floor for the first half second.
    if shown:
        z = 0.08 - 0.4 * max(0.0, 1.0 - (tsec - intro) / 0.5)
        particles.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), z) for x, y in cloud[f][:n_particles]]))
        e = est_xy[f]
        est_op.Set(Gf.Vec3d(float(e[0]), float(e[1]), 0.2))
        est_scale.Set(Gf.Vec3f(1.0))
        k = np.r_[np.arange(int(intro * fps), f, 3), f]
        if len(k) >= 2:
            est_trail.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(a), float(b), 0.035) for a, b in est_xy[k]]))
            est_trail.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(k)]))
            true_trail.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(a), float(b), 0.025) for a, b in walker_xy[k]]))
            true_trail.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(k)]))
    else:
        est_scale.Set(Gf.Vec3f(1e-4))

    # Walker pose.
    pose(hero, walker_xy[f], data["walker_heading"][f], data["walker_dist"][f], swing_gain[f])
    for i, ops in enumerate(others):
        pose(ops, data["crowd_xy"][f, i], data["crowd_heading"][f, i], data["crowd_dist"][f, i],
             float(data["crowd_moving"][f, i]))
    k = min(max((tsec - intro + 0.3) / 0.6, 0.0), 1.0)  # Fade to yellow over 0.6 s.
    white, yellow = np.array([0.93, 0.94, 0.96]), np.array([1.0, 0.82, 0.0])
    hero_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*map(float, (1 - k) * white + k * yellow)))
    hero_shader.GetInput("emissiveColor").Set(Gf.Vec3f(*map(float, k * 1.2 * yellow)))

    # Camera look-at.
    eye, target = Gf.Vec3d(*map(float, data["eye"][f])), Gf.Vec3d(*map(float, data["target"][f]))
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
    cam_op.Set(view.GetInverse())


# Warm up shaders and the denoiser before the first saved frame.
set_frame(args.start)
for _ in range(30):
    app.update()

for f in range(args.start, min(args.end, n_frames), args.step):
    set_frame(f)
    rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=True)
    img = rgb.get_data()
    Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(args.out, f"{f:04d}.png"))
    if f % 30 == 0:
        print(f"frame {f}", flush=True)

rep.orchestrator.wait_until_complete()
app.close()
