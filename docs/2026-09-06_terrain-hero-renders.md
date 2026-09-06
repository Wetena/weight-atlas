# Hero Terrain Renders — Making the Eye Candy Actually Wow (Proposal)

> Status: Proposal | Date: 2026-09-06
> Trigger: operator feedback — terrain renders are "neither appealing nor
> impressive" while being the site's designated visual eye candy.
> Constraint kept: **determinism stays** (same inputs → byte-identical PNGs;
> `_strip_png_metadata` already handles Blender's timestamp stamping).

---

## 1. Why the current renders look flat (root cause, not vibes)

Reading the pipeline (`render/fractal/wrapper.py` → `render/blender/
render_terrain.py`) predicts the exact weaknesses seen in the images:

| Choice today | Visual consequence |
|---|---|
| **Workbench engine** (a *viewport preview* engine) | No real shading response, no specular interest, no atmosphere — matte "clay model" look |
| `color_type = "VERTEX"` with tint → RGB `(t, 0.5t, 1−t)` | A two-hue blue↔yellow ramp for the ENTIRE terrain; no color diversity, no meaningful color language |
| Background `(0.05, 0.05, 0.05)` flat | No gradient, no ground plane, no context — floats in dead space |
| Ortho camera at fixed 18° pitch, dead-center | A flat map that pretends to be 3D; no depth cues, no hero angle |
| Two SUNs (key 1.0 @45°/315° + fill 0.35) | Uniform studio lighting — no long shadows, no golden-hour drama, silhouette read is weak at 18° |
| NaN void renders as dark void, hard boundary | The 75% empty region of sparse models reads as "broken", not "intentional" |
| No post, no fog, no DOF, no water | Nothing that gives scale or atmosphere |

The geometry underneath is genuinely good — per-slot-stat-driven fBm
fractals (octaves/persistence/lacunarity from effective_rank/kurtosis/
sparsity) are a real differentiator. It's purely a *presentation* gap.

## 2. The proposal: a `hero` render mode, additive to the analysis render

Keep the deterministic Workbench terrain for footprint/compare purposes;
add `terrain_hero_*.png` artefacts rendered with real shading. Spec
block (`fractal.hero`) drives it; all knobs deterministic, no hand-tuning
per model (colors derive from stats, not taste).

### 2.1 Engine + material (the big levers)

1. **Switch hero renders to Cycles (CPU, fixed seed 0, denoiser on)** —
   real light transport: soft shadows, reflections, subtle GI. Cycles is
   deterministic for fixed seed on CPU. (EEVEE alternative: faster but
   GPU/driver-dependent bloom/GI can break byte-identity — Cycles is the
   safe choice for the determinism contract.)
2. **Layered terrain material** replacing raw vertex colors:
   - Elevation-based hypsometric bands (deep valley → peak), like
     physical relief maps — instantly readable
   - Per-slot tint from the tint stat mapped through a perceptual
     colormap (**viridis/magma** class, not the 2-hue ramp) as a
     multiply/overlay so slot identity stays visible
   - Micro-detail: deterministic triplanar noise for rock texture
     (breaks the CG-plastic smoothness)
   - Wetness trick: valleys slightly darker + higher specular — reads
     as water-worn, adds instant depth
3. **Sun: golden-hour** — low-altitude warm key (long shadows across the
   fBm ridges give the 3D structure its drama), cool blue fill opposite,
   faint rim from camera. Altitude/azimuth fixed per spec.

### 2.2 Composition

4. **Hero angles, not flat map**: default pitch 32–38°, yaw rotated
   ~20° off-axis (diagonal composition beats axis-aligned), terrain fills
   ~80% of frame with margin computed for the relief extent.
5. **Water plane** at a low percentile height: reflective surface filling
   the NaN/low regions. Two wins: empty voids become *lakes* (sparse
   models go from "broken" to "archipelago"), and reflections double the
   visual information. Level from the same percentile normalisation the
   geometry already uses.
6. **Mesa edge**: where valid cells border NaN void, drop a steep cliff
   skirt (extrude boundary down to the water plane) so the data boundary
   looks like a deliberate landform — this kills the "broken raster"
   read completely.
7. **Volumetric haze** (exponential height fog) + slight aerial
   perspective: distance = depth, and it softens the back edge.

### 2.3 Colour language (derived, never hand-picked)

8. Map the tint stat (currently stable_rank) through the perceptual
   colormap; overlay hue rotation by slot family (attn warm, ssm cool,
   hc neutral) so an informed viewer can *see* architecture structure —
   the thing the flat sheets prove but nobody stares at.
9. Optional "story mode" for paired scans (the decay/repair work): two
   terrains from the same angle, one frame — raw vs repaired checkpoints.
   The RA2/G2 decay signature becomes a *picture*.

### 2.4 Output set (per model, deterministic)

- `terrain_hero_ne.png` — 4K (4096) golden-hour hero, primary artefact
- `terrain_hero_low.png` — low sun, long shadows variant
- `terrain_map.png` — flat top-down, analysis-grade (existing Workbench
  render keeps this role; optionally restyled with the hypsometric ramp)
- `terrain_turntable/` — 72 frames, 360° orbit (deterministic frame
  indices), assembled to a looped webm/mp4 for the landing page hero

## 3. Implementation sketch

- `render/blender/render_hero.py` (new): Cycles scene setup, layered
  material builder, water/mesa/fog constructors — reuses
  `normalise_height`, `resample_bilinear`, `_strip_png_metadata`.
- `render/fractal/wrapper.py`: `hero` mode branch (same fBm field, new
  scene module); spec `fractal.hero` block: `{"engine": "cycles",
  "pitch": 35, "yaw": 20, "sun_alt": 12, "fog": 0.08, "water_level":
  "p05", "colormap": "magma", "resolution": 4096}` — all defaulted.
- Determinism pins: fixed seed, CPU device only, metadata strip (exists),
  and a byte-identity test (two hero renders of the same field compare
  equal) mirroring the existing terrain determinism test.
- Cost estimate: Cycles CPU at 4K ≈ minutes per frame on the 4090 box —
  acceptable for one-time per-model hero artefacts (render is already
  async job infrastructure).

## 4. Milestones

1. **M-H1**: Cycles + hypsometric material + golden-hour lighting, single
   NE hero angle. Exit: operator says "wow" unprompted (the actual test).
2. **M-H2**: water plane + mesa skirt + fog (sparse models become
   archipelagos). Exit: NVFP4 (24% valid) looks intentional.
3. **M-H3**: turntable + paired-story mode + landing-page integration
   (Phase 1 M2's hero imagery comes from here).

## 5. Verification

- Byte-identity: same field + spec → identical PNG (existing pattern).
- Slot-color correctness: a slot's hue in the render must match its
  mapped colormap position (test via a synthetic 3-slot field with known
  stats).
- No regressions: default Workbench terrain renders unchanged (diff the
  existing artefacts before/after the patch).
