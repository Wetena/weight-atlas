# ruff: noqa — runs inside Blender's bundled Python (bpy/bmesh), not the project venv
"""bpy script: render the UMAP embedding-terrain landscape (Cycles).

Runs INSIDE Blender (which bundles its own Python, so ``bpy``/``bmesh``
are only importable there — all handles typed Any on the weight-atlas
side). All knobs arrive after ``--``. Deterministic: seed 0, CPU,
metadata stripped by the caller.
"""
import json
import math
import os as _os
import sys
from typing import Any

import bpy  # type: ignore[import-not-found]
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name: str, default: str) -> str:
    return argv[argv.index("--" + name) + 1] if "--" + name in argv else default


def argf(name: str, default: float) -> float:
    return float(arg(name, str(default)))


DENSITY = arg("density", "")
PEAKS = arg("peaks", "")
LABELS = arg("labels", "1") == "1"
PITCH = argf("pitch", 55.0)
YAW = argf("yaw", 18.0)
DIST = argf("dist-factor", 2.1)
LENS = argf("lens", 42.0)
Z_SCALE = argf("z-scale", 7.0)
GAMMA = argf("gamma", 1.35)
LOG_HEIGHT = arg("log-height", "0") == "1"
SUN_ALT = argf("sun-alt", 14.0)
SUN_AZI = argf("sun-azi", 305.0)
RES = int(argf("resolution", 1800))
SAMPLES = int(argf("samples", 96))
OUT = arg("out", "")

SIZE = 8.0

FAM_COLORS = {
    "expert": (0.95, 0.42, 0.18), "shared_expert": (0.90, 0.55, 0.30),
    "router": (0.98, 0.85, 0.25), "attn": (0.98, 0.85, 0.25),
    "mlp": (0.35, 0.85, 0.55), "ssm": (0.25, 0.70, 0.95),
    "hc": (0.80, 0.45, 0.95), "ngram": (0.95, 0.30, 0.55),
    "embed": (0.20, 0.95, 0.85), "norm": (0.85, 0.85, 0.85),
    "v": (0.60, 0.60, 0.65), "other": (0.55, 0.50, 0.40),
}
FAM_LABEL = {
    "expert": "expert massif", "shared_expert": "shared experts",
    "router": "routers", "attn": "attention", "mlp": "mlp",
    "ssm": "linear attn", "hc": "hyper-conn", "ngram": "n-gram",
    "embed": "embed", "norm": "norms", "v": "vision", "other": "other",
}

d = np.load(DENSITY)
Hn = d["Hn"]; dom = d["dom"]; present = d["present"]
FAMS = [str(x) for x in d["fams"]]
G = Hn.shape[0]
peaks = json.load(open(PEAKS)) if PEAKS else []

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

import bmesh  # type: ignore[import-not-found]

mesh = bpy.data.meshes.new("umap_terrain")
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=G, y_segments=G, size=SIZE / 2)
bm.to_mesh(mesh); bm.free()
obj: Any = bpy.data.objects.new("UMAPTerrain", mesh)
scene.collection.objects.link(obj)

positions = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
mesh.attributes["position"].data.foreach_get("vector", positions)
xs = positions[0::3]; ys = positions[1::3]
ix = np.clip(((xs + SIZE / 2) / SIZE * (G - 1)).astype(int), 0, G - 1)
iy = np.clip(((ys + SIZE / 2) / SIZE * (G - 1)).astype(int), 0, G - 1)
h = Hn[iy, ix].astype(np.float64)
present_g = present[iy, ix]
h_vis = np.log1p(h * 9) / np.log1p(9) if LOG_HEIGHT else h  # log1p normalised to [0,1]
positions[2::3] = np.where(present_g, (h_vis ** GAMMA) * Z_SCALE + 0.05, 0.0)
mesh.attributes["position"].data.foreach_set("vector", positions)

colors = np.empty(len(mesh.vertices) * 4, dtype=np.float32)
h_c = np.clip(h, 0, 1)
for i in range(len(mesh.vertices)):
    f = FAMS[dom[iy[i], ix[i]]] if present_g[i] else "other"
    c = FAM_COLORS.get(f, FAM_COLORS["other"])
    shade = 0.60 + 0.50 * h_c[i]
    colors[i * 4 + 0] = c[0] * shade
    colors[i * 4 + 1] = c[1] * shade
    colors[i * 4 + 2] = c[2] * shade
    colors[i * 4 + 3] = 1.0
col_attr = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
col_attr.data.foreach_set("color", colors.ravel())

mat = bpy.data.materials.new("UMAPMat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.55
bsdf.inputs["Specular IOR Level"].default_value = 0.2
vcol = mat.node_tree.nodes.new("ShaderNodeVertexColor"); vcol.layer_name = "Col"
mat.node_tree.links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])
obj.data.materials.append(mat)
for p in mesh.polygons:
    p.use_smooth = True

# sea (empty UMAP space)
wmat = bpy.data.materials.new("Sea")
wmat.use_nodes = True
wb = wmat.node_tree.nodes["Principled BSDF"]
wb.inputs["Roughness"].default_value = 0.12
wb.inputs["Base Color"].default_value = (0.01, 0.04, 0.09, 1.0)
wb.inputs["Transmission Weight"].default_value = 0.6
wmesh = bpy.data.meshes.new("Sea")
wbm = bmesh.new()
bmesh.ops.create_grid(wbm, x_segments=1, y_segments=1, size=SIZE * 1.5)
wbm.to_mesh(wmesh); wbm.free()
wobj: Any = bpy.data.objects.new("Sea", wmesh)
wobj.location = (0.0, 0.0, 0.045)
wobj.visible_shadow = False
scene.collection.objects.link(wobj)
wobj.data.materials.append(wmat)

# family peak labels (emission text)
if LABELS:
    lbl_mat = bpy.data.materials.new("Label")
    lbl_mat.use_nodes = True
    lnodes = lbl_mat.node_tree.nodes
    for n in list(lnodes):
        if n.type != "OUTPUT_MATERIAL":
            lnodes.remove(n)
    em = lnodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs["Strength"].default_value = 2.2
    lbl_mat.node_tree.links.new(em.outputs["Emission"],
                                lnodes["Material Output"].inputs["Surface"])
    _meta_path = DENSITY.replace("_density.npz", "_meta.json")
    _fracs = {}
    _total = 0
    if _os.path.exists(_meta_path):
        _m = json.load(open(_meta_path))
        _fracs = _m.get("family_param_fractions", {})
        _total = _m.get("total_params", 0)
    for pk in peaks:
        _f = pk["family"]
        _lbl = FAM_LABEL.get(_f, _f)
        if _f in _fracs and _total > 0:
            _pct = _fracs[_f] * 100
            _bp = _total * _fracs[_f]
            _lbl = f"{_lbl}\n{_pct:.1f}% · {_bp/1e9:.1f}B"
        wx, wy = (pk["col"] / (G - 1) * SIZE - SIZE / 2,
                  pk["row"] / (G - 1) * SIZE - SIZE / 2)
        _hv = np.log1p(Hn[pk["row"], pk["col"]] * 9) / np.log1p(9) if LOG_HEIGHT else Hn[pk["row"], pk["col"]]
        wz = float((_hv ** GAMMA) * Z_SCALE + 0.05)
        crv = bpy.data.curves.new(f"lbl_{pk['family']}", type="FONT")
        crv.body = _lbl
        crv.size = 0.22
        crv.align_x = "CENTER"; crv.align_y = "CENTER"
        txt: Any = bpy.data.objects.new(f"lbl_{pk['family']}", crv)
        txt.location = (wx, wy, wz + 0.22)
        scene.collection.objects.link(txt)
        txt.data.materials.append(lbl_mat)

# ── scale bar + family percentages (on a plinth in front) ────────────────
meta_path = DENSITY.replace("_density.npz", "_meta.json")
import os as _os
if _os.path.exists(meta_path):
    _meta = json.load(open(meta_path))
    _total = _meta.get("total_params", 0)
    _fracs = _meta.get("family_param_fractions", {})
    if _total > 0:
        _major = sorted(_fracs.items(), key=lambda kv: -kv[1])[:5]
        _lines = []
        for _f, _pct in _major:
            _p = _pct * 100
            _label = FAM_LABEL.get(_f, _f)
            _lines.append(f"{_label}: {_p:.1f}%")
        _lines.append(f"Total: {_total/1e9:.1f}B params")
        scale_body = " | ".join(_lines)
        scrv = bpy.data.curves.new("scale_bar", type="FONT")
        scrv.body = scale_body
        scrv.size = 0.18
        scrv.align_x = "CENTER"
        smat = bpy.data.materials.new("ScaleBar")
        smat.use_nodes = True
        sem = smat.node_tree.nodes.new("ShaderNodeEmission")
        sem.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1.0)
        sem.inputs["Strength"].default_value = 1.5
        smat.node_tree.links.new(sem.outputs["Emission"],
                                  smat.node_tree.nodes["Material Output"].inputs["Surface"])
        sobj = bpy.data.objects.new("ScaleBar", scrv)
        sobj.location = (0.0, -SIZE / 2 - 0.35, 0.05)
        scene.collection.objects.link(sobj)
        sobj.data.materials.append(smat)

# golden hour
sun = bpy.data.lights.new("Sun", type="SUN")
sun.energy = 5.0; sun.angle = math.radians(1.8)
sun.color = (1.0, 0.70, 0.42)
so: Any = bpy.data.objects.new("Sun", sun)
so.rotation_euler = (math.radians(SUN_ALT), 0.0, math.radians(SUN_AZI))
scene.collection.objects.link(so)
fill = bpy.data.lights.new("Fill", type="SUN")
fill.energy = 0.7; fill.color = (0.40, 0.55, 0.95)
fo: Any = bpy.data.objects.new("Fill", fill)
fo.rotation_euler = (math.radians(58.0), 0.0, math.radians(125.0))
scene.collection.objects.link(fo)

# hero camera
cam = bpy.data.cameras.new("Cam"); cam.type = "PERSP"; cam.lens = LENS
co: Any = bpy.data.objects.new("Cam", cam)
scene.collection.objects.link(co); scene.camera = co
pitch, yaw = math.radians(PITCH), math.radians(YAW)
dist = SIZE * DIST
co.location = (dist * math.cos(pitch) * math.sin(yaw),
               -dist * math.cos(pitch) * math.cos(yaw),
               dist * math.sin(pitch) + SIZE * 0.06)
co.rotation_euler = (pitch, 0.0, yaw)

world = bpy.data.worlds.new("World"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.03, 0.05, 0.10, 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.seed = 0
scene.cycles.use_denoising = False
scene.cycles.samples = SAMPLES
scene.render.resolution_x = RES
scene.render.resolution_y = int(RES * 0.66)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = OUT
scene.view_settings.view_transform = "Standard"

bpy.ops.render.render(write_still=True)
print("EMBEDDING TERRAIN DONE")
