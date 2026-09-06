"""Embedding-terrain renderer: UMAP density landscape of per-tensor records.

Registry ID ``"embedding_terrain"``. Consumes the UMAP projection of the
scan's per-tensor metric vectors (produced by ``compute_embedding_terrain``
in ``scan.py``: ``embedding_umap_records.npz`` + ``..._meta.json``), builds
a family-colored density landscape (peaks = clusters of statistically
similar tensors), and renders it via Blender Cycles with camera/light
parameters exposed as render knobs.

Determinism: fixed seed, CPU Cycles, metadata-stripped PNG — same
contract as the fractal/blender renderers.

Blender script args (after ``--``):
  --density, --out, --pitch, --yaw, --dist-factor, --lens, --z-scale,
  --gamma, --sun-alt, --sun-azi, --resolution, --samples, --labels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec, Field2D
from weight_atlas.render.blender.blender_wrapper import (
    build_blender_env,
    resolve_blender_path,
    run_blender_command,
)
from weight_atlas.render.blender.render_terrain import _strip_png_metadata

_SCRIPT = Path(__file__).resolve().parent / "render_umap_terrain.py"

_FAMILIES = ("expert", "shared_expert", "router", "attn", "mlp", "ssm",
             "hc", "ngram", "embed", "norm", "v", "other")

FAM_COLORS: dict[str, tuple[float, float, float]] = {
    "expert": (0.95, 0.42, 0.18), "shared_expert": (0.90, 0.55, 0.30),
    "router": (0.98, 0.85, 0.25), "attn": (0.98, 0.85, 0.25),
    "mlp": (0.35, 0.85, 0.55), "ssm": (0.25, 0.70, 0.95),
    "hc": (0.80, 0.45, 0.95), "ngram": (0.95, 0.30, 0.55),
    "embed": (0.20, 0.95, 0.85), "norm": (0.85, 0.85, 0.85),
    "v": (0.60, 0.60, 0.65), "other": (0.55, 0.50, 0.40),
}


def fam(slot: str) -> str:
    """Coarse family for a slot name (mirrors the density builder)."""
    if slot in _FAMILIES:
        return slot
    for f in _FAMILIES:
        if slot.startswith(f + "_") or slot == f:
            return f
    if slot.startswith("v_"):
        return "v"
    return "other"


# ── analysis step: records → UMAP → density grid ─────────────────────────


def record_matrix(tensors: dict[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
    """(n, d) metric matrix (log-compressed, z-scored) + names + slots.

    Metrics available in every fingerprint: spectral_norm, stable_rank,
    kurtosis, sparsity, effective_rank, frobenius (+ log10 numel).
    """
    metrics = ("spectral_norm", "stable_rank", "kurtosis", "sparsity",
               "effective_rank", "frobenius")
    rows, names, slots = [], [], []
    for name, v in tensors.items():
        vals = [v.get(m) for m in metrics]
        if any(x is None or not np.isfinite(x) for x in vals):
            continue
        s = v.get("shape") or []
        rows.append([float(x) for x in vals] +
                    [float(np.log10(max(int(np.prod(s)) if s else 1, 1)))])
        names.append(name)
        slots.append(str(v.get("slot", "other")))
    matrix = np.array(rows, dtype=np.float64)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if (col > 0).all():
            matrix[:, j] = np.log10(col)
    std = matrix.std(0)
    std[std < 1e-12] = 1.0
    return (matrix - matrix.mean(0)) / std, names, slots


def build_density_field(
    tensors: dict[str, Any], out_dir: Path, spec: AtlasSpec,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """UMAP the records and rasterise a (G, G) family-dominance density.

    Returns (Hn grid in [0,1], family peak list, meta dict) and persists
    ``embedding_umap_records.npz`` + ``..._meta.json`` next to the scan.
    """
    from scipy import ndimage

    x_std, names, slots = record_matrix(tensors)
    if x_std.shape[0] < 50:
        raise ValueError(
            f"too few complete records for an embedding terrain ({x_std.shape[0]})"
        )
    from weight_atlas.embedding.umap import compute_umap

    seeds = getattr(spec, "seeds", None) or {}
    seed = int(seeds.get("umap", 0))
    grid = int((getattr(spec, "embedding", None) or {}).get("grid", 768))
    proj, umap_meta = compute_umap(x_std.astype(np.float32), n_components=2,
                                   seed=seed, n_neighbors=30, min_dist=0.1)

    fams = np.array([fam(s) for s in slots])
    pad = 2.0
    x0, x1 = float(proj[:, 0].min() - pad), float(proj[:, 0].max() + pad)
    y0, y1 = float(proj[:, 1].min() - pad), float(proj[:, 1].max() + pad)
    ix = np.clip(((proj[:, 0] - x0) / (x1 - x0) * (grid - 1)).astype(int), 0, grid - 1)
    iy = np.clip(((proj[:, 1] - y0) / (y1 - y0) * (grid - 1)).astype(int), 0, grid - 1)

    count = {}
    for f in _FAMILIES:
        m = fams == f
        count[f] = (np.histogram2d(iy[m], ix[m], bins=grid,
                                   range=[[0, grid - 1], [0, grid - 1]])[0]
                    if m.any() else np.zeros((grid, grid)))
    total = sum(count.values())
    height = ndimage.gaussian_filter(np.log1p(total), 2.5)
    lo, hi = np.quantile(height, 0.02), np.quantile(height, 0.995)
    height_grid = (np.clip((height - lo) / (hi - lo), 0, 1)
                   if hi > lo else np.zeros_like(height))
    stack = np.stack([ndimage.gaussian_filter(count[f], 2.0) for f in _FAMILIES])
    dominance = stack.argmax(axis=0)
    present = stack.sum(axis=0) > 0.5

    peaks = []
    for _fi, f in enumerate(_FAMILIES):
        if not (fams == f).any():
            continue
        sm = ndimage.gaussian_filter(count[f], 2.0)
        if sm.max() <= 0:
            continue
        cy, cx = np.unravel_index(sm.argmax(), sm.shape)
        peaks.append({"family": f, "row": int(cy), "col": int(cx)})

    np.savez(out_dir / "embedding_terrain_density.npz",
             Hn=height_grid, dom=dominance, present=present,
             fams=np.array(_FAMILIES))
    np.savez(out_dir / "embedding_umap_records.npz", proj=proj.astype(np.float32))
    (out_dir / "embedding_umap_records_meta.json").write_text(json.dumps(
        {"names": names, "slots": slots,
         "umap": umap_meta, "seed": seed,
         "extent": [x0, x1, y0, y1], "grid": grid}))
    meta = {"n_points": int(x_std.shape[0]), "seed": seed, "grid": grid,
            "umap": umap_meta}
    return height_grid, peaks, meta


# ── renderer plugin ───────────────────────────────────────────────────────


@register_renderer("embedding_terrain")
class EmbeddingTerrainRenderer:
    """UMAP density landscape of per-tensor records (Blender Cycles).

    Camera/light/relief knobs come from the render form (UI) or defaults;
    every knob is deterministic (no randomness anywhere in the pipeline).
    """

    renderer_id = "embedding_terrain"

    def __init__(self) -> None:
        self._done: set[Path] = set()

    def render(self, field: Field2D, spec: AtlasSpec, out: Path,
               *, field_name: str = "height",
               knobs: dict[str, Any] | None = None,
               scan_dir: Path | None = None,
               **_ignored: Any) -> list[Path]:
        # ``out`` is the render/ subdirectory when invoked from the worker;
        # the fingerprint and density artefacts live in the scan root.
        scan_root = scan_dir or (out.parent if out.name == "render" else out)
        out.mkdir(parents=True, exist_ok=True)
        knobs = knobs or {}

        fp_path = scan_root / "fingerprint.json"
        if not fp_path.exists():
            raise FileNotFoundError(f"no fingerprint.json in {scan_root}")
        import json

        tensors = json.loads(fp_path.read_text()).get("tensors", {})
        density_cache = scan_root / "embedding_terrain_density.npz"
        if density_cache.exists():
            # precomputed density (previous render or scan-time artefact):
            # reuse verbatim — UMAP is the expensive step and deterministic
            d = np.load(density_cache)
            height_grid = d["Hn"]
            peaks = json.loads(
                (scan_root / "embedding_terrain_peaks.json").read_text()
            ) if (scan_root / "embedding_terrain_peaks.json").exists() else []
            meta = {"n_points": -1, "seed": 0, "grid": int(height_grid.shape[0]),
                    "umap": {}, "cached": True}
        else:
            height_grid, peaks, meta = build_density_field(tensors, scan_root, spec)

        np.save(out / "embedding_terrain_height.npy", height_grid)
        (out / "embedding_terrain_peaks.json").write_text(json.dumps(peaks, indent=1))

        def kn(name: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(knobs.get(name, default))
            except (TypeError, ValueError):
                return default
            return min(hi, max(lo, v))

        pitch = kn("pitch", 55.0, 5.0, 89.0)
        yaw = kn("yaw", 18.0, 0.0, 360.0)
        dist_factor = kn("dist_factor", 2.1, 0.8, 6.0)
        lens = kn("lens", 42.0, 20.0, 120.0)
        z_scale = kn("z_scale", 7.0, 0.5, 30.0)
        gamma = kn("gamma", 1.35, 0.5, 3.0)
        sun_alt = kn("sun_alt", 14.0, 2.0, 80.0)
        sun_azi = kn("sun_azi", 305.0, 0.0, 360.0)
        resolution = int(kn("resolution", 1800, 400, 4096))
        samples = int(kn("samples", 96, 16, 1024))
        labels = bool(knobs.get("labels", True))

        if out in self._done and (out / "embedding_terrain.png").exists():
            return []  # one artefact per scan — channel iteration is irrelevant

        blender_path = resolve_blender_path()
        env = build_blender_env()
        height_npy = out / "embedding_terrain_height.npy"
        peaks_json = out / "embedding_terrain_peaks.json"
        out_png = out / "embedding_terrain.png"

        cmd = [
            str(blender_path), "-b", "-P", str(_SCRIPT),
            "--",
            "--density", str(height_npy),
            "--peaks", str(peaks_json),
            "--labels", "1" if labels else "0",
            "--pitch", str(pitch),
            "--yaw", str(yaw),
            "--dist-factor", str(dist_factor),
            "--lens", str(lens),
            "--z-scale", str(z_scale),
            "--gamma", str(gamma),
            "--sun-alt", str(sun_alt),
            "--sun-azi", str(sun_azi),
            "--resolution", str(resolution),
            "--samples", str(samples),
            "--out", str(out_png),
        ]
        run_blender_command(cmd, env)
        _strip_png_metadata(str(out_png))
        self._done.add(out)
        self._last_key = (str(out.resolve()), pitch, yaw, dist_factor)
        self._last_produced = [out_png]
        return [out_png]
