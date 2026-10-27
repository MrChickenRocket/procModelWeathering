# GRI rigid characters

How this repo paints the rigid-piece robot characters for the GRI_Dan project. Every character gets two texture sets
on the same OBJs/UVs: **worn** (minimal-tier weathering) and **pristine** (factory-new with Gunpla-style decals). Both
carry the model-kit finish pass.

## Boundaries with GRI

- Input: `$GRI_CHARS/<name>/export/` (GRI writes it): per-piece `<Part>.obj`, `mog.mtl`, `rig.json` (frames, atlases,
  `look` colours, slots), `exposure/<Part>.npz`.
- Output: `$GRI_CHARS/<name>/paint/` only. Never edit GRI source, never upload or publish.
  - `<Part>.obj`, `atlases.json`, `mog_a*.mtl` (UV'd in bind pose; `verify.sh` checks v/vn unchanged)
  - `textures_minimal_2048/a1|a2|a3/` and `renders/minimal_*.png`: the worn set
  - `textures_pristine_2048/a1|a2|a3/` and `renders/pristine_*.png`: the pristine set
  - `DONE.txt` / `PRISTINE_DONE.txt`: written **last**, only once that set is final. GRI's publisher may republish on
    any rewrite, so test runs must not touch them (`NODONE=1`).
- Atlases: a1 torso, neck, shoulders, upper and fore arms; a2 pelvis and legs; a3 head.
- Primers are grey or blue-grey only (no red oxide). White and light topcoats use the dark blue-grey primer
  `[0.05, 0.06, 0.08]`.

## Scripts (`tools/gri/`)

All run from anywhere; `common.sh` sets `GRI_CHARS` (default `/c/Repos/GRI_Dan/tools/out/characters`) and `BLENDER`.
Point `GRI_CHARS` at a scratch tree to test without touching GRI's output.

| script | does |
|---|---|
| `rigid_spec.py` | `build(name, prefix, primer, primer2, ...)`: new spec from the rocketeer template + the rig's `look` colours |
| `rigid.sh <name>` | worn set: re-import export, full bake, sim, downsize, render, copy. No DONE file |
| `pristine.sh <name>` | pristine set on the last worn bake; writes `PRISTINE_DONE.txt` last unless `NODONE=1` |
| `repaint.sh <name>` | re-sim both sets on the existing bake after a sim or spec change; adds missing finish maps. No DONE files |
| `lane.sh <name>...` | `rigid.sh` + verify + `DONE.txt`, then `pristine.sh` |
| `pristine_cfg.py` | worn mog config -> pristine config (livery instead of decals, joint pinch points from rig frames) |
| `downsize.py`, `verify.sh` | 2048 copies (normals renormalised); OBJ geometry check |

New character: `rigid_spec.build(...)`, then `lane.sh <name>`. Changed sim code or spec only: `repaint.sh`, review
the renders, then write the DONE files. A full `rigid.sh` re-bakes every map (minutes per atlas on the GPU).

## Spec (`assets/mogs/specs/<name>_rigid.json`)

Built into `assets/mogs/<name>_rigid.json` by `tools/mog_config.py`. Keys that matter here:

- `tier_set.minimal`: Params for the one tier GRI uses (wear knobs, `ridge`/`cavity`/`rivets`/`seams`, finish pass).
- `ao_per_piece: true`: colour AO baked per piece so joint overlaps don't bake dark.
- `exposure`: `{dir, lo, hi, reach_studs}`, GRI's per-sample exposure; gates wear and dirt to surfaces a player sees.
- `livery_tune`: per-character overrides merged into the pristine livery (see below).
- Near-black gunmetal (sRGB <= 0.25) gets primer `[0.06, 0.06, 0.058]` and weather `ridge 0.6`; `rigid_spec` applies it.

## Model-kit finish pass

`sim/composite.py: _finish`, fields from `sim/finish.py`. Strengths are fractions of **displayed (sRGB) value**,
converted to linear factors with `^2.2`. All default to off in `Params`; GRI's approved values (2026-10-27, set in
every roster spec's `tier_set.minimal`):

| Param | value | effect |
|---|---|---|
| `ao_contact` | 0.2 | darkening from the assembled short-range AO bake, stacked on the per-piece AO (`ao_self` 0.75) |
| `tones` | 1.0 | per-panel tone variants: +-4.5% value, +-3 deg hue |
| `hemi` | 0.07 | up-facing lighter, down-facing darker |
| `lining` | 0.35 | darkening in creases and piece-to-piece seams (Gunpla panel lining) |
| `edge_hi` | 0.05 | lightening on convex chamfers, removed wherever edge wear or layer loss is |
| `satin` | 0.45 | topcoat clean roughness; wear still roughens from there |
| `metallic` / `metal_rough` | 0.5 / 0.33 | the +1 tone variant's topcoat becomes metallic paint |

Satin and metallic apply only to painted topcoats (non-fabric, topcoat metal < 0.5) and never under stickers, chips
or primer.

Bake maps it needs (`tools/bake.py`, baked by default; add to an old bake with `--maps ao_contact,piece`):
- `ao_contact`: AO at 0.05 m with all parts present. Interiors hidden under other pieces read near 0, which is fine
  because they are never seen.
- `piece`: 1-based id of each connected mesh run per atlas (a "panel").

Tone colouring (`sim/finish.py`):
- **Adjacency:** pieces whose texels come within 0.1 stud of each other count as neighbours.
- **Mirror pairing:** each piece is paired with its X-mirror twin (centroid within 0.2 stud, texel count within 30%),
  so left and right always match. GRI's user required this symmetry.
- **Colouring:** greedy, in id order, over the paired nodes. It is stable across repaints while the mesh is
  unchanged. A re-export that renumbers pieces can reshuffle the variants.
- **Scope:** colouring is per atlas only; it doesn't cross atlas borders.

## Pristine livery (`sim/livery.py`)

Caution panels on the largest flat plates (single-facet rule), hazard stripes on straight bottom rims of planar
plates (left/right matched), joint pinch markers, one unit number on the -x side; everything else mirrored.
`livery_tune` fixes for plates the search misses:
- `panel_min_d` (studs, default 0.16): smaller rim distance allowed for panel centres (small faceted plates).
- `anchors`: `[{part, at: [x, y, z] part-space metres on -x, h, cover, tol, plan}]` forces a panel; mirrored. Used on
  Tinkerer's breastplate.

## Checking a test before rollout

`tools/render_mog.py` views include `back` and `torso` (close-up of the chest band) besides front/three_quarter/etc.
Render from 2048 textures when judging lining width, since that is what ships.
