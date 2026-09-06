# Terrain WOW Rework — Semantic Landscapes (Research + Proposal v2)

> Status: Proposal | Date: 2026-09-06
> Supersedes the M-H1..M-H3 milestones in `2026-09-06_terrain-hero-renders.md`
> (that proposal's Cycles/material/water work is validated and kept as the
> *rendering* layer; this changes *what* we render).
> Trigger: operator research pointers — Li et al. 2018 loss landscapes,
> UMAP/t-SNE as analysis step vs Blender as presentation step, Mandelbulber
> parameter-injection, multi-octave detail that "holds the eye for hours".

---

## 1. The honest diagnosis of v2 (after the fence fix)

The camera bug is fixed and the images are *credible* — but credible is
not WOW. The operator's three pointers converge on one diagnosis:

**Our terrains are decorative; the WOW images are *semantic*.**

- Li et al. render *actual loss surfaces* — every ridge and basin is a
  real, computed property of the network. The eye keeps looking because
  the shape MEANS something.
- UMAP/t-SNE pipelines separate the analysis step (reduce
  high-dim data → 2D where proximity = similarity) from the presentation
  step. Our current terrain fails at the analysis step: the fBm fractal
  parameters come from per-slot stat *medians*, but the height field
  itself is synthetic noise — pretty dunes that encode one number per
  slot, nothing more.
- Mandelbulber-style injection: stats drive the fractal FORMULA, so
  infinite detail is genuinely derived from the data.

And one empirical fact from our own renders: 77% of the NVFP4 field was
flat floor, three slot-families formed walls. Per-slot-median → fBm is
too coarse: one median per slot strip, then decorated noise. The
instrument we actually built — 149k real per-tensor records with
spectra, ranks, kurtosis — is richer than what the terrain shows.

## 2. Design: three semantic landscape types (v2 roadmap)

### 2.1 — Statistic Landscape (Li-style, cheapest, ships first)

**What:** render a REAL measured statistic as terrain — not fBm noise.
We already produce smoothed `field_height_raw.tif` rasters
(48×87 etc.). The upgrade is fidelity + meaning, not new math:

- **Upsample measured cells to render resolution with bicubic +
  deterministic detail octaves** whose amplitude is proportional to the
  local *measurement confidence* (more/consistent tensors in a cell →
  fine detail allowed; empty cells → smooth plain). Detail now RESOLVES
  as you zoom (each octave is anchored to real data variance), instead
  of standing at one noise scale.
- Annotate the 20 highest peaks with tensor names (hover/caption in the
  gallery; etched into the plinth for print).
- Every ridge, valley and outlier = a real tensor property. The
  "hours of gazing" comes from recognisability: users find THEIR
  model's known weird layers as landforms.

Cost: days. Reuses everything (Cycles pipeline, water, golden hour).

### 2.2 — Embedding Terrain (UMAP islands — the flagship)

**What:** UMAP/t-SNE the per-tensor record vectors (per-tensor metric
vectors: spectral_norm, stable_rank, kurtosis, sparsity, sv_decay,
numel-log, per-slot one-hot, …) → 2D coordinates where **proximity =
statistical similarity**. Then:

- Density of the projected points → height field (KDE, deterministic
  seed): mountains = dense clusters of similar tensors (e.g. "the
  512-equal expert massif", "the norm-vector archipelago", "the
  attention ridge").
- Each point optionally rendered as a small marker/pillar whose height
  is its standout metric; colour = slot family.
- **Cross-model mode (the killer feature):** overlay two scans' point
  clouds (base vs finetuned, raw vs repaired — the ladder work!) with
  paired colours. Model differences become *geography*: clusters that
  moved, split, or vanished are visible at a glance. This is the
  picture that makes people forward it.
- UMAP already exists in-repo (`embedding/umap.py`, `compute_umap`
  with seeded `init`), currently only applied to token embeddings —
  the new step is feeding it the per-tensor RECORD vectors instead.
  ~148k points is standard UMAP scale.

Caveat to document honestly: UMAP is stochastic — fixed seed +
`init=pca` gives reproducibility in practice, and we pin `umap-learn`
version; determinism tests must run same-version only.

Cost: 1–2 weeks. This becomes the registry's signature image.

### 2.3 — Statistic-Driven Fractal Object (Mandelbulber route, flagship #2)

**What:** per-SLOT (or per-cluster) solid fractal objects arranged on a
map, with tensor stats injected into the fractal formula itself — not a
heightmap with fractal texture:

- Mandelbulb power ← kurtosis (spiky distributions → spikier bulbs)
- Iteration depth ← effective_rank; folding symmetry ← head/expert
  structure; surface roughness ← sv_decay
- Rendered via our existing SDF path (`fractal/sdf.py` +
  `surface_nets.py` already does mini-SDFs — extend the formula
  repertoire: Mandelbulb, kaleidoscopic IFS, Menger variants)
- Arrangement: the 2D layout from §2.2 (UMAP of slots) so similar
  slots' objects sit near each other → a "sculpture garden" that is a
  map, not a grid.

The repo already has this seed (`fractal.mode: "sdf"` with Menger/
Mandelbulb per-slot mini-SDFs, spec `fractal.sdf` mapping power/
iterations/scale from stats) — v2 deepens the formula coupling and
removes the rigid 8×8 grid in favour of the UMAP layout.

Cost: 2–3 weeks; infinite-detail property comes free from the fractal
itself (deep zooms keep resolving — the "hours in the ban" property).

## 3. Recommendation (order)

1. **§2.1 first** — small effort, immediately more meaningful; also
   fixes the sparse-model flat-floor problem (confidence-scaled detail
   handles 24%-valid fields gracefully).
2. **§2.2 second** — the flagship; lands the cross-model geography that
   no other tool shows, and directly exploits our existing UMAP +
   query-API infrastructure. Landing-page hero material.
3. **§2.3 third** — the deepest rabbit hole and the most distinctive;
   do it once §2.2 proves the UMAP layout reads well.

All three reuse the validated Cycles/material/water/lighting layer
from the v2 spike (`/tmp/opencode/terrain-lab/hero_scene.py`) — the
rendering layer is solved; we are changing the geometry's *source of
truth* from synthetic noise to measured data and learned projections.

## 4. Verification

- §2.1: zoom invariant — a crop of a peak at 2× render scale must show
  octave detail absent from the wide shot (visual), and detail amplitude
  must be zero where measurement confidence is zero (unit test).
- §2.2: seed+version determinism test; cluster sanity — the 512 expert
  records must form one connected dense massif on every model (known
  ground truth from the ladder scans).
- §2.3: parameter injection unit tests (kurtosis → power exponent
  mapping pinned), render determinism as everywhere.
