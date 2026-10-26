"""Base materials and coating stacks, keyed by the low-poly material slot name (see CONVENTIONS.md).

Colors are linear and set the diffuse color of each base or layer; a texture set (assets/materials/<id>)
only adds variation around that color plus roughness and height detail. Layers are listed bottom to top.

A stack may set color and rough to override its base's (e.g. a tinted rubber, a matte screen).
A stack may set weave {cell_mm, height_um}: a plain-weave cloth height under the wear (composite._weave).
A stack with overlay ("base" or a layer name) treats that colour as the in-game part Color: composite writes an
overlay.png whose alpha is 0 wherever that colour shows unworn (SurfaceAppearance AlphaMode Overlay).

Layer fields:
  thickness_um  how much abrasion the layer absorbs, in microns of film
  step_um       physical height step at chip edges; defaults to thickness_um. Conversion coats (bluing,
                phosphate, anodize) grow into the metal, so they wear like a thick film but barely step.
  resist        abrasion resistance multiplier
  chip          probability an impact that reaches this layer flakes it off (adhesion)
  profile       chip shape: "ragged" (noisy round chips) or "voronoi" (whole polygonal flakes)
  tex           texture reference or None
Texture reference fields: id, tile_mm (real-world size of one tile), height_um (std of detail height),
  rough_var / color_var (how strongly the scan's roughness / luminance variation applies).
"""
import copy



def tex(id, tile_mm, height_um, rough_var=1.0, color_var=1.0):
    return dict(id=id, tile_mm=tile_mm, height_um=height_um, rough_var=rough_var, color_var=color_var)


# round: how much of the baked large-radius edge rounding (normal_ts_round) shows before any wear; worn
# corners blend the rest of the way.
BASES = {
    "steel": dict(color=(0.56, 0.57, 0.58), metal=1.0, rough=0.38, rusts=True, dent_mm=0.02, round=0.3,
                  tex=tex("Metal009", 80, 1.5, 0.8, 0.6)),
    "aluminum": dict(color=(0.80, 0.81, 0.82), metal=1.0, rough=0.32, rusts=False, dent_mm=0.035, round=0.3,
                     tex=tex("Metal011", 80, 1.5, 0.8, 0.5)),
    "polymer": dict(color=(0.022, 0.022, 0.024), metal=0.0, rough=0.62, rusts=False, dent_mm=0.03, round=1.0,
                    tex=tex("Plastic012A", 50, 3.0, 0.6, 0.4)),
    "rubber": dict(color=(0.02, 0.02, 0.021), metal=0.0, rough=0.82, rusts=False, dent_mm=0.0, round=1.0,
                   tex=tex("Plastic012B", 40, 4.0, 0.5, 0.3)),
    "glass": dict(color=(0.008, 0.011, 0.013), metal=0.0, rough=0.06, rusts=False, dent_mm=0.0, round=0.5, tex=None),
    # Bare copper alloys: corrosion (verdigris) grows in crevices even under the tarnish film.
    "copper": dict(color=(0.93, 0.6, 0.47), metal=1.0, rough=0.28, rusts=True, dent_mm=0.03, round=0.3,
                   tex=tex("Metal009", 80, 1.5, 0.8, 0.4), corrodes_under_layers=True, corrosion_rate=1.3,
                   corrosion=dict(color=(0.10, 0.30, 0.21), metal=0.0, rough=0.85, height_mm=0.01)),
    "brass": dict(color=(0.86, 0.66, 0.30), metal=1.0, rough=0.3, rusts=True, dent_mm=0.03, round=0.3,
                  tex=tex("Metal009", 80, 1.5, 0.8, 0.4), corrodes_under_layers=True, corrosion_rate=1.3,
                  corrosion=dict(color=(0.12, 0.28, 0.2), metal=0.0, rough=0.85, height_mm=0.01)),
    # Woven bodysuit: never chips; it only takes grime, dust and scuffed nap.
    "fabric": dict(color=(0.02, 0.02, 0.022), metal=0.0, rough=0.85, rusts=False, dent_mm=0.0, round=1.0, tex=None),
    "lens": dict(color=(0.12, 0.85, 0.7), metal=0.0, rough=0.12, rusts=False, dent_mm=0.0, round=0.5, tex=None,
                 emissive=True, clean=True),
}


def layer(name, color, metal, rough, thickness_um, resist, chip, profile="ragged", tex=None, step_um=None):
    return dict(name=name, color=color, metal=metal, rough=rough, thickness_um=thickness_um,
                step_um=thickness_um if step_um is None else step_um,
                resist=resist, chip=chip, profile=profile, tex=tex)


CERAKOTE_OD = (0.11, 0.13, 0.095)
PRIMER_GREY = (0.24, 0.24, 0.23)
FIRE_RED = (0.48, 0.02, 0.018)
GUNMETAL = (0.032, 0.034, 0.038)
HAZARD_YELLOW = (0.8, 0.5, 0.02)
FDE = (0.30, 0.22, 0.13)

STACKS = {
    "MAT_cerakote_steel": dict(base="steel", layers=[
        layer("cerakote", CERAKOTE_OD, 0.0, 0.58, 25, 1.0, 1.0, tex=tex("Proc_Topcoat_Cerakote", 40, 3.2, 1.0, 0.5)),
    ]),
    "MAT_painted_steel": dict(base="steel", layers=[
        layer("primer", (0.24, 0.24, 0.23), 0.0, 0.82, 30, 0.8, 0.5, tex=tex("Proc_Primer_Grey", 40, 3.2)),
        layer("topcoat", CERAKOTE_OD, 0.0, 0.3, 40, 0.6, 0.9, tex=tex("Proc_Topcoat_Enamel", 40, 6.0, 1.0, 0.5)),
    ]),
    # Paint bonds poorly to polymer, so it flakes off in sheets rather than chipping.
    "MAT_painted_polymer": dict(base="polymer", layers=[
        layer("primer", (0.24, 0.24, 0.23), 0.0, 0.82, 25, 0.7, 0.8, "voronoi", tex("Proc_Primer_Grey", 40, 3.2)),
        layer("topcoat", FDE, 0.0, 0.55, 35, 0.6, 0.95, "voronoi", tex("Proc_Topcoat_Cerakote", 40, 3.2, 1.0, 0.5)),
    ]),
    "MAT_steel_blued": dict(base="steel", layers=[
        layer("bluing", (0.018, 0.021, 0.03), 1.0, 0.32, 25, 0.45, 0.15, step_um=2),
    ]),
    "MAT_steel_parkerized": dict(base="steel", layers=[
        layer("phosphate", (0.035, 0.037, 0.032), 0.0, 0.78, 25, 0.7, 0.3, tex=tex("Metal046A", 60, 2.0, 0.6, 0.6),
              step_um=4),
    ]),
    # Type III hard anodize: a dielectric oxide, abrasion resistant, but impacts crack it down to bare aluminum.
    "MAT_anodized_alu": dict(base="aluminum", layers=[
        layer("anodize", (0.022, 0.022, 0.024), 0.0, 0.42, 25, 2.0, 0.6, tex=tex("Metal028", 30, 1.0, 0.5, 0.0),
              step_um=2),
    ]),
    "MAT_anodized_alu_grey": dict(base="aluminum", layers=[
        layer("anodize", (0.085, 0.085, 0.08), 0.0, 0.5, 25, 1.6, 0.6, tex=tex("Metal028", 30, 1.0, 0.5, 0.0),
              step_um=2),
    ]),
    "MAT_steel_bare": dict(base="steel", layers=[]),
    "MAT_polymer": dict(base="polymer", layers=[]),
    # Mech stacks (assets/mogs). Same chemistry as the gun stacks; the mech's geometry is scaled so its texel size
    # matches a gun's (m_per_stud), which keeps chip, flake and tile sizes in the range they were tuned for.
    "MECH_paint_red": dict(base="steel", layers=[
        layer("primer", PRIMER_GREY, 0.0, 0.82, 30, 0.8, 0.5, tex=tex("Proc_Primer_Grey", 40, 3.2)),
        layer("topcoat", FIRE_RED, 0.0, 0.38, 40, 0.6, 0.9, tex=tex("Proc_Topcoat_Enamel", 40, 6.0, 1.0, 0.5)),
    ]),
    "MECH_gunmetal": dict(base="steel", layers=[
        layer("primer", PRIMER_GREY, 0.0, 0.82, 30, 0.8, 0.5, tex=tex("Proc_Primer_Grey", 40, 3.2)),
        layer("topcoat", GUNMETAL, 0.12, 0.45, 40, 0.7, 0.85, tex=tex("Proc_Topcoat_Enamel", 40, 6.0, 1.0, 0.4)),
    ]),
    # Hazard-yellow over the frame's gunmetal: wear shows black first, then primer and steel.
    "MECH_paint_yellow": dict(base="steel", layers=[
        layer("primer", PRIMER_GREY, 0.0, 0.82, 30, 0.8, 0.5, tex=tex("Proc_Primer_Grey", 40, 3.2)),
        layer("gunmetal", GUNMETAL, 0.12, 0.45, 40, 0.7, 0.85, tex=tex("Proc_Topcoat_Enamel", 40, 6.0, 1.0, 0.4)),
        layer("topcoat", HAZARD_YELLOW, 0.15, 0.4, 50, 1.2, 0.7, tex=tex("Proc_Topcoat_Enamel", 40, 6.0, 1.0, 0.5)),
    ]),
    # Painted plastic housings (TV cabinets): ragged chips like the mech paint, but the deepest show plastic.
    "MAT_painted_plastic": dict(base="polymer", layers=[
        layer("primer", PRIMER_GREY, 0.0, 0.82, 30, 0.8, 0.5, tex=tex("Proc_Primer_Grey", 40, 3.2)),
        layer("topcoat", FDE, 0.0, 0.45, 40, 0.6, 0.9, tex=tex("Proc_Topcoat_Enamel", 40, 6.0, 1.0, 0.5)),
    ]),
    "MECH_steel": dict(base="steel", layers=[]),
    "MECH_rubber": dict(base="rubber", layers=[]),
    "MECH_rubber_red": dict(base="rubber", color=(0.32, 0.018, 0.014), layers=[]),
    "MECH_glass": dict(base="glass", layers=[]),
    "MECH_eye": dict(base="lens", layers=[]),
    "MAT_fabric": dict(base="fabric", layers=[]),
    # Brown tarnish is a conversion film: it never flakes, only rubs back to bright metal on contact edges.
    "MECH_copper": dict(base="copper", layers=[
        layer("tarnish", (0.30, 0.13, 0.06), 0.7, 0.48, 15, 1.5, 0.0, step_um=0),
    ]),
    "MECH_brass": dict(base="brass", layers=[
        layer("tarnish", (0.32, 0.22, 0.08), 0.75, 0.45, 15, 1.5, 0.0, step_um=0),
    ]),
}

# Handling detail stamped from CC0 masks (assets/masks). tile_mm is the real-world tile size at detail_scale 1;
# raise --detail_scale for large assets. pick: how many masks from the family are layered per part.
DETAIL = {
    "scratches": dict(ids=[f"Scratches00{i}" for i in range(1, 6)], pick=2, tile_mm=100),
    "imperfections": dict(ids=[f"SurfaceImperfections{i:03d}" for i in range(1, 21) if i != 10], pick=2, tile_mm=150),
    "fingerprints": dict(ids=[f"Fingerprints00{i}" for i in range(1, 10) if i != 6], pick=2, tile_mm=90),
    "smears": dict(ids=["Smear005", "Smear006", "Smear007"], pick=1, tile_mm=200),
}

RUST = dict(color=(0.20, 0.065, 0.025), metal=0.0, rough=0.88, height_mm=0.015)
SOOT = dict(color=(0.012, 0.011, 0.010), rough=0.95)
# Temper colours of steel by peak temperature, low to high: straw, bronze, purple, blue.
HEAT_TINT = [(0.55, 0.42, 0.20), (0.40, 0.20, 0.08), (0.22, 0.09, 0.18), (0.08, 0.12, 0.30)]
GRIME = dict(color=(0.018, 0.015, 0.012), rough=0.75)


def register(defs):
    """Per-mog stacks from a config's "stacks": {name: {from, color: base rgb, rough: base roughness, overlay, emissive: mask strength 0..1, colors: {layer: rgb}, layers: {layer: {key: value}},
    weather: {<Params field>: multiplier, flat_impacts, large_edges: multiplier}}}. weather scales the tier's parameters on this
    stack only; light paints show every mark, so they need less wear than dark ones.
    Runs in each sim worker; spawned processes don't see registrations made in the parent."""
    for name, d in defs.items():
        s = copy.deepcopy(STACKS[d["from"]])
        for lay in s["layers"]:
            if lay["name"] in d.get("colors", {}):
                lay["color"] = tuple(d["colors"][lay["name"]])
            lay.update(d.get("layers", {}).get(lay["name"], {}))
        if "color" in d:
            s["color"] = tuple(d["color"])
        if "rough" in d:
            s["rough"] = d["rough"]
        if "overlay" in d:
            s["overlay"] = d["overlay"]
        if "weave" in d:
            s["weave"] = d["weave"]
        if "emissive" in d:
            s["emissive"] = d["emissive"]
        s["weather"] = d.get("weather", {})
        STACKS[name] = s
DUST = dict(color=(0.30, 0.27, 0.22), rough=0.92)
