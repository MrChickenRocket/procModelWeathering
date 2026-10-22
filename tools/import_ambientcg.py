"""Unpack ambientCG 2K-PNG zips into assets/materials/<id>/ (or assets/masks/<id>/ for sets that ship an
Opacity mask, e.g. Scratches, SurfaceImperfections) and refresh the preview copies.

python -I tools/import_ambientcg.py <zip_dir>
"""
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "materials")
MASKS = os.path.join(ROOT, "assets", "masks")
PREVIEW = os.path.join(ROOT, "build", "matpreview")
MAPS = {"Color": "color", "NormalGL": "normal", "Roughness": "roughness", "Metalness": "metalness",
        "Displacement": "height", "Opacity": "mask"}
PREVIEW_SIZE = 1024


def to_8bit(im):
    a = np.asarray(im)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        a = (a.astype(np.float32) / (65535 if a.max() > 255 else 255) * 255).round()
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def write_preview(mat_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    maps = []
    for fn in sorted(os.listdir(mat_dir)):
        name, ext = os.path.splitext(fn)
        if ext != ".png":
            continue
        im = Image.open(os.path.join(mat_dir, fn))
        im = im.convert("RGB") if name in ("color", "normal") else to_8bit(im).convert("L")
        im.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS).save(os.path.join(out_dir, f"{name}.jpg"), quality=92)
        maps.append(name)
    return maps


def unpack(zip_path):
    aid = os.path.basename(zip_path)[:-4]
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        out = os.path.join(MASKS if f"{aid}_2K-PNG_Opacity.png" in names else ASSETS, aid)
        os.makedirs(out, exist_ok=True)
        for n in names:
            m = re.fullmatch(rf"{re.escape(aid)}_2K-PNG_(\w+)\.png", n)
            if m and m.group(1) in MAPS:
                Image.open(io.BytesIO(z.read(n))).save(os.path.join(out, f"{MAPS[m.group(1)]}.png"))
    return aid


def refresh_manifest():
    manifest = []
    for kind, root in (("material", ASSETS), ("mask", MASKS)):
        for aid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, aid)
            if os.path.isdir(d):
                manifest.append({"id": aid, "kind": kind, "maps": write_preview(d, os.path.join(PREVIEW, aid))})
    with open(os.path.join(PREVIEW, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


if __name__ == "__main__":
    for fn in sorted(os.listdir(sys.argv[1])):
        if fn.endswith(".zip"):
            print("[acg]", unpack(os.path.join(sys.argv[1], fn)))
    print("[acg]", len(refresh_manifest()), "materials in preview")
