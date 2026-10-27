"""pristine_cfg.py <name> <worn_cfg.json> <out_cfg.json> <rig.json>: worn mog config + Gunpla-style livery (sim/livery.py)
in place of the worn decals. Pinch markers at the elbows, knees and shoulders come from the rig's joint frames (Roblox
build studs). The worn config's livery_tune (from the spec) is merged last, for per-character fixes."""
import json, sys
n, src, out, rig_path = sys.argv[1:5]
cfg = json.load(open(src))
rig = json.load(open(rig_path))
cfg.pop("exposure", None)
slots, stacks = cfg["slots"], cfg["stacks"]
targets, light = {}, []
for slot, w in (("paint", 1.0), ("paint2", 0.6), ("gunmetal", 0.5)):
    st = slots.get(slot)
    if st not in stacks:
        continue
    targets[st] = w
    top = stacks[st].get("colors", {}).get("topcoat") or stacks[st].get("color")
    if top and 0.2126 * top[0] + 0.7152 * top[1] + 0.0722 * top[2] > 0.3:
        light.append(st)
fr = rig["frames"]
def at(part, off=(0, 0, 0)):
    return [round(fr[part][r][3] + off[r], 3) for r in range(3)]
points = []
# Joint pivots sit inside the body; nudge the anchors toward the faces a viewer sees (-z is forward, -x is the L side).
for part, kinds, count, off in (("ShinL", ["warning", "hazard"], 2, (0, 0, -0.45)), ("ForeL", ["warning"], 1, (-0.3, 0, -0.2)),
                                ("UpperL", ["hazard"], 1, (-0.25, 0.2, -0.2))):
    if part in fr:
        points.append({"at": at(part, off), "mirror_x": True, "kinds": kinds, "count": count})
cfg.pop("decals", None)
cfg["livery"] = dict(targets=targets, light_paint=light, pinch=points, panels=3, stripes=4)
cfg["livery"].update(cfg.pop("livery_tune", {}))
json.dump(cfg, open(out, "w"), indent=1)
print(f"[pristine_cfg] {n}: targets {targets}, light {light}, {len(points)} joint points")
