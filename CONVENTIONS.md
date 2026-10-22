# Source asset conventions

Every weapon `.blend` entering the pipeline follows this. Foreign meshes get normalized to it.

## Space
- Meters. Barrel along +Y, up +Z, origin at the top of the grip.
- Up +Z is the rest orientation used for gravity-driven effects (rust streaks, dust).

## Collections
- `low`: `<part>_low` meshes. One object per Roblox MeshPart.
- `high`: `<part>_high` meshes, same names, same space. Bake source only.
- `wear`: handling volumes.

## Low-poly parts
- Exactly one material slot, named `MAT_<stack>` (stacks defined in `sim/materials.py`: `MAT_cerakote_steel`, `MAT_painted_steel`, `MAT_painted_polymer`, `MAT_steel_blued`, `MAT_steel_parkerized`, `MAT_steel_bare`, `MAT_anodized_alu`, `MAT_anodized_alu_grey`, `MAT_polymer`). The stack name selects the layer stack in the sim.
- One UV map, non-overlapping, no mirrored islands. Padding 16 px at 4096 (`16/4096` UV units).
- Every hard (sharp) edge must lie on a UV seam, or tangent-space normal bakes show gradients along it. The generator uses one angle (35°) for both.

## Wear volumes
- Cube empties named `WEAR_<zone>`, display size 1; object scale = half extents.
- Custom property `weight` (0–1): relative contact frequency inside the volume.
