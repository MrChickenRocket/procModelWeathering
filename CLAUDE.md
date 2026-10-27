# procModelWeathering

Bakes and simulates PBR weathering textures for peer projects. Pipeline: `tools/bake.py` (Blender bake maps) ->
`sim/run.py` (wear sim in `sim/wear.py`, shading in `sim/composite.py`) -> per-project delivery scripts.

- Weapon source assets: `CONVENTIONS.md`.
- GRI rigid characters (worn + pristine sets, model-kit finish pass, livery): `docs/gri_characters.md`, scripts in
  `tools/gri/`.
- Mogs (MogCompany): `tools/run_mog.sh`, specs in `assets/mogs/specs/`, built by `tools/mog_config.py`.

Write only into a peer project's `tools/out` dirs; never edit its source or upload.
