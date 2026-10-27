"""downsize.py <paint_dir> <size>: textures_<tier>/a<k>/*.png -> textures_<tier>_<size>/a<k>/ (Lanczos; normals renormalized)"""
import os, sys, glob
import numpy as np
from PIL import Image
paint, SZ = sys.argv[1], int(sys.argv[2])
for src in glob.glob(os.path.join(paint, "textures_*", "a*", "*.png")):
    tier_dir = os.path.basename(os.path.dirname(os.path.dirname(src)))
    if tier_dir.split("_")[-1].isdigit():
        continue
    out = os.path.join(paint, f"{tier_dir}_{SZ}", os.path.basename(os.path.dirname(src)), os.path.basename(src))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im = Image.open(src)
    small = im.resize((SZ, SZ), Image.LANCZOS)
    if os.path.basename(src) == "normal.png":
        n = np.asarray(small.convert("RGB")).astype(np.float32) / 127.5 - 1
        n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
        small = Image.fromarray(np.clip((n + 1) * 127.5 + 0.5, 0, 255).astype(np.uint8))
    small.save(out)
print(f"[{SZ}]", paint)
