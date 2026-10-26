"""Build assets/mogs/<mech>.json from assets/mogs/raptor.json plus a short spec.

python tools/mog_config.py <spec.json> <export_dir> <out.json>

spec: {height, stacks, slots, tint, wear, pinch: [{at, mirror_x, kinds?, count?}], rams: true, decals: {...},
       tier_scale: {<Params field>: multiplier}, tier_set: {<tier>: {<Params field>: value}}, min_feature_px, base_tint, atlases, ao_per_piece, equalize, burns, grime, dust, haze, density, glow_through, soot, stripes, patterns}
glow_through, soot, stripes and patterns are documented at their use in sim/run.py.
base_tint false drops the raptor's tint rules (random red rubber).
tier_scale scales both tiers for every stack (frames and rubber too); a stack's weather scales only that stack.
height is the build-space height in studs. k = height / RAPTOR_HEIGHT scales the sim (m_per_stud 0.04 / k) and
the markings, so every mog weathers like the raptor relative to its own size.
"""
import json
import os
import sys

RAPTOR_HEIGHT = 4.96  # build-space studs of the raptor export (installed at 2x in game)
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    spec_path, src, out = sys.argv[1:4]
    with open(spec_path) as f:
        spec = json.load(f)
    with open(os.path.join(HERE, "..", "assets", "mogs", "raptor.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(src, "rig.json")) as f:
        rig = json.load(f)
    # Build height, never the installed height: the install scale grows the textures with the mesh.
    k = spec["height"] / RAPTOR_HEIGHT
    cfg["m_per_stud"] = round(0.04 / k, 6)
    cfg["stacks"] = spec.get("stacks", {})
    cfg["slots"].update(spec.get("slots", {}))
    cfg["tint"] = (cfg["tint"] if spec.get("base_tint", True) else []) + spec.get("tint", [])
    cfg["wear"] = spec["wear"]
    for key in ("atlases", "density", "glow_through", "soot", "stripes", "patterns", "ao_per_piece", "equalize", "burns", "grime", "dust", "haze"):
        cfg.pop(key, None)
        if key in spec:
            cfg[key] = spec[key]
    for name, tier in cfg["tiers"].items():
        for key, v in spec.get("tier_scale", {}).items():
            tier[key] = round(tier[key] * v, 4)
        tier.update(spec.get("tier_set", {}).get(name, {}))
        if "min_feature_px" in spec:
            tier["min_feature_px"] = spec["min_feature_px"]
    d = cfg["decals"]
    d.update(spec.get("decals", {}))
    d["scale"] = max(k, spec.get("min_decal_scale", 1.0))
    d["targets"] = spec.get("targets", {cfg["slots"]["paint"]: 1.0, "MECH_gunmetal": 0.45})
    points = list(spec["pinch"])
    if spec.get("rams", True):
        for p in rig.get("pistons", []):
            points += [dict(at=p[e]["at"], mirror_x=False, count=1) for e in ("a", "b")]
    d["pinch"]["points"] = points
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[mog_config] {rig['mech']}: k {k:.2f}, m_per_stud {cfg['m_per_stud']}, decal scale {d['scale']:.2f}, "
          f"{len(points)} pinch points")


if __name__ == "__main__":
    main()
