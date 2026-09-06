# Grok Review: K3 Embedding Terrain — Parameter-Mass Visualization Assessment

**Date:** 2026-09-06
**Reviewer:** Grok (via operator)
**Subject:** Embedding terrain render of Kimi K3 (`bdh-ladG2` series context; the K3 model is a ~2.78T total / ~104B active MoE from Moonshot AI)
**Verdict:** The visualization makes good conceptual and quantitative sense. The expert massif correctly conveys capacity distribution. Several concrete improvement suggestions below.

---

## Validation

Kimi K3's parameter budget is extremely skewed:

- **Routed experts dominate** (~97%+ of total parameters). There are 896 routed experts per MoE layer across ~92 MoE layers (layer 0 is dense), each operating in a latent dimension (3584) with intermediate size 3072. This produces the vast bulk of the ~2.78T parameters (roughly 2.72T in the routed experts alone). Only 16 are activated per token (+ 2 shared experts).
- Everything else is comparatively tiny: attention projections (69 KDA + 24 Gated MLA layers), shared experts, norms, embeddings + LM head, latent up/down projections, vision components, etc. — in the low tens of billions combined.

The tall central orange "expert massif" correctly conveys that the model's capacity lives almost entirely in the sparse expert pool. The much shorter ground spikes for norms, attention, residual/embedding-related terms, and miscellaneous pieces accurately reflect their far smaller footprints.

## Limitations of the current form

- Primarily a **static parameter-count / magnitude terrain**, not a depiction of compute, activation patterns, routing, or information flow. The mountain is capacity, not the active subgraph for any token.
- Spatial arrangement appears artistic rather than strictly ordered (e.g. not a clean grid by layer × expert index). Labels are partial/cut off in the provided view.
- Height is almost certainly linear in some measure of parameter volume or magnitude; extreme dynamic range can make smaller components hard to see without careful scaling.
- Does not highlight architectural novelties (Kimi Delta Attention recurrence, Attention Residuals across blocks, Stable LatentMoE down-projection to 3584-d, SiTU-GLU, Quantile Balancing, etc.).

## Improvement suggestions

### 1. Make the scale explicit and multi-view

- Primary view: linear height proportional to parameter count (or total magnitude).
- Secondary/log view or inset: log-scale heights so tiny components remain visible and comparable.
- Color by component family (routed experts one hue, shared experts another, KDA vs MLA distinct, norms/embeddings, latent projectors).

### 2. Structure the layout more informatively

- Arrange the massif as a rough grid or cylindrical/layered arrangement ordered by layer index × expert index (or by expert specialization if any analysis is available).
- Or keep the organic "terrain" aesthetic but add a transparent wireframe or base map showing the 93-layer / block structure (groups of 3 KDA + 1 MLA, Attention Residual blocks).
- Separate or annotate the two shared experts per layer distinctly from the 896 routed ones.

### 3. Add interactive or multi-panel depth

- Hover/click reveals exact parameter counts, tensor shapes, precision (MXFP4 for experts), and activation frequency.
- Side-by-side or overlay: "total parameters" mountain vs "active parameters per token" (the active slice is only ~3.7% and far flatter).
- Animate a single-token forward pass: light up the 16 routed + 2 shared experts that fire, dim the rest, show residual/attention pathways.

### 4. Enhance readability and accuracy cues

- Cleaner, fully legible labels with exact numbers or percentages (e.g. "Routed experts ≈ 97.9%").
- A small legend and scale bar (parameters or relative magnitude).
- Optional second terrain for activation norms, gradient magnitudes, or router scores if those data are available — would move beyond pure parameter mass into functional importance.
- Include the vision pathway (MoonViT) as a distinct small feature if the full multimodal model is intended.

### 5. Alternative complementary visualizations

- Stacked bar or treemap of parameter fractions.
- Sankey or flow diagram of information paths (sequence via hybrid KDA/MLA, depth via AttnRes, width via LatentMoE).
- 3D "cityscape" with layers as floors and experts as buildings of height proportional to size.

## Assessment

The current image already succeeds at the core communication goal: it makes the extreme sparsity and expert dominance of a 2.8T-class LatentMoE immediately visceral. Refining the quantitative mapping, adding explicit scales/legends, and optionally pairing it with an "active-subgraph" view would make it both more rigorous and more useful for technical audiences while preserving the striking terrain metaphor.
