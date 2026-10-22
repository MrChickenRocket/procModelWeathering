"""Synthesize tileable primer / topcoat material sets at physical scale into assets/materials/Proc_*.

python tools/make_paint_materials.py

Noise is band-limited in the frequency domain, so every map tiles exactly. Height is in millimeters
before encoding; normals are derived from it with the real texel size so slopes are physical.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_ambientcg import ASSETS, refresh_manifest  # noqa: E402

SIZE = 2048
TILE_MM = 40.0
PX_MM = TILE_MM / SIZE


def band_noise(rng, lo_mm, hi_mm, power=0.0):
    """Unit-variance tileable noise with wavelengths between lo_mm and hi_mm."""
    f = np.fft.fftfreq(SIZE, d=PX_MM)
    r = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    lo, hi = 1 / hi_mm, 1 / lo_mm
    band = np.exp(-0.5 * (np.log(np.maximum(r, 1e-9) / np.sqrt(lo * hi)) / (0.5 * np.log(hi / lo))) ** 2)
    band *= np.maximum(r, 1e-9) ** -power
    band[0, 0] = 0
    n = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((SIZE, SIZE))) * band))
    return ((n - n.mean()) / n.std()).astype(np.float32)


def normal_from_height(h_mm):
    dx = (np.roll(h_mm, -1, 1) - np.roll(h_mm, 1, 1)) / (2 * PX_MM)
    dy = (np.roll(h_mm, -1, 0) - np.roll(h_mm, 1, 0)) / (2 * PX_MM)
    # Image rows run top-down, so +v (OpenGL green) is -row.
    n = np.stack([-dx, dy, np.ones_like(dx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def save(name, color_lin, rough, height_mm, metal, meta):
    out = os.path.join(ASSETS, name)
    os.makedirs(out, exist_ok=True)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(u8(srgb(color_lin))).save(os.path.join(out, "color.png"))
    Image.fromarray(u8(rough)).save(os.path.join(out, "roughness.png"))
    Image.fromarray(u8(np.full_like(rough, metal))).save(os.path.join(out, "metalness.png"))
    Image.fromarray(u8(normal_from_height(height_mm) * 0.5 + 0.5)).save(os.path.join(out, "normal.png"))
    span = float(np.abs(height_mm).max())
    h16 = ((height_mm / (2 * span) + 0.5) * 65535).round().astype(np.uint16)
    Image.fromarray(h16).save(os.path.join(out, "height.png"))
    meta = dict(meta, tile_mm=TILE_MM, height_range_mm=[-span, span])
    with open(os.path.join(out, "material.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[paint] {name}: height +-{span * 1000:.1f} um, rough {rough.mean():.2f}")


def primer(rng, base, name):
    grain = band_noise(rng, 0.04, 0.15)
    tooth = band_noise(rng, 0.15, 0.6)
    mottle = band_noise(rng, 3, 15)
    height = 0.0025 * grain + 0.002 * tooth
    color = np.asarray(base, np.float32) * (1 + 0.05 * mottle + 0.04 * grain)[..., None]
    rough = 0.82 + 0.05 * grain - 0.03 * mottle
    save(name, color, rough, height, 0.0, dict(role="primer", note="matte sprayed primer: fine grain, tooth, mottling"))


def topcoat(rng, base, name, rough_mean, peel_um, grain_um, note):
    peel = band_noise(rng, 0.6, 2.5)
    grain = band_noise(rng, 0.03, 0.12)
    mottle = band_noise(rng, 4, 20)
    height = peel_um / 1000 * peel + grain_um / 1000 * grain
    color = np.asarray(base, np.float32) * (1 + 0.025 * mottle)[..., None]
    rough = rough_mean + 0.04 * mottle + 0.03 * grain - 0.02 * peel
    save(name, color, rough, height, 0.0, dict(role="topcoat", note=note))


def main():
    rng = np.random.default_rng(7)
    primer(rng, (0.24, 0.24, 0.23), "Proc_Primer_Grey")
    topcoat(rng, (0.11, 0.13, 0.095), "Proc_Topcoat_Cerakote", 0.55, 2.0, 2.5,
            "ceramic polymer coat: faint orange peel, fine matte grain")
    topcoat(rng, (0.11, 0.13, 0.095), "Proc_Topcoat_Enamel", 0.22, 6.0, 0.6,
            "gloss enamel: visible orange peel, smooth")
    print("[paint]", len(refresh_manifest()), "materials in preview")


main()
