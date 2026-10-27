#!/usr/bin/env bash
# Full mog loop: tools/run_mog.sh <mech>   (config assets/mogs/<mech>.json, export and paint dirs under MOGS)
# NAME overrides the config/blend/build name, for two jobs that share an export folder name (a suit and its helmet).
# Writes paint/textures_<tier>/a<k>/ for every atlas, re-UV'd OBJs, atlases.json, mog_a<k>.mtl, FBX and renders.
set -euo pipefail
cd "$(dirname "$0")/.."
mech=$1
name=${NAME:-$mech}
MOGS=${MOGS:-/c/Repos/MogCompany/tools/out/mogs}
B=${BLENDER:-"/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"}
PY=.venv/Scripts/python
JOBS=${JOBS:-2}
src=$MOGS/$mech/export
paint=$MOGS/$mech/paint
build=build/$name
"$B" -b --python-exit-code 1 -P tools/import_mog.py -- "$src" assets/mogs/$name.json assets/mogs/$name.blend 2>&1 | grep -F '[mog]'
rm -rf $build/bake $build/tex_minimal $build/tex_scarred $build/tex_pristine
"$B" -b assets/mogs/$name.blend -P tools/bake.py -- $build/bake 2>&1 | grep -iE '^\[bake\].*(done|total)|error' || true
# Single-atlas leftovers from earlier runs: the old OBJs, the textures/ alias and flat textures_<tier>/*.png.
mkdir -p "$paint"
rm -rf "$paint/textures" "$paint"/*.obj "$paint/mog.mtl" "$paint/textures_minimal" "$paint/textures_scarred"
for tier in ${TIERS:-minimal scarred}; do
  $PY -m sim.run $build/bake $build/tex_$tier --mog assets/mogs/$name.json --tier $tier --jobs $JOBS 2>&1 \
    | grep -E 'decals\]|parts in|Error|Traceback'
  for d in $build/tex_$tier/*_a*/; do
    k=$(basename "$d"); k=a${k##*_a}
    mkdir -p "$paint/textures_$tier/$k"
    cp "$d"/{color,metalness,roughness,normal,emissive}.png "$paint/textures_$tier/$k/"
    if [ -f "$d/overlay.png" ]; then cp "$d/overlay.png" "$paint/textures_$tier/$k/"; fi
  done
done
"$B" -b assets/mogs/$name.blend -P tools/export_mog.py -- "$paint" 2>&1 | grep -iE '^\[export|error' || true
for tier in ${TIERS:-minimal scarred}; do
  "$B" -b assets/mogs/$name.blend -P tools/render_mog.py -- $build/tex_$tier "$paint/renders" --prefix ${tier}_ \
    --views three_quarter,head,side,legs,pauldron,chest --size 1600 --samples 128 2>&1 | grep -E '^\[render\]' || true
done
