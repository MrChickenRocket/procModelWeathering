#!/usr/bin/env bash
# repaint.sh <name>: re-sim both sets on the existing bake (no re-import), e.g. after a sim or spec change. Bakes the
# finish-pass maps (ao_contact, piece) first if the bake predates them. Writes no DONE files: review, then write them.
set -e
source "$(dirname "$0")/common.sh"
n=$1; nm=${n}_rigid; G=$GRI_CHARS/$n; build=build/$nm; P=build/gri_src/$n/paint
if ! $PY -c "import numpy as np,sys; sys.exit('piece' not in np.load('$build/bake/${n}_a1/maps.npz').files)"; then
  "$BLENDER" -b assets/mogs/$nm.blend -P tools/bake.py -- $build/bake --maps ao_contact,piece 2>&1 | grep -E "pieces|done in|Error|Traceback" || true
fi
$PY tools/mog_config.py assets/mogs/specs/$nm.json build/gri_src/$n/export assets/mogs/$nm.json
rm -rf $build/tex_minimal $P/textures_minimal $P/textures_minimal_2048
$PY -m sim.run $build/bake $build/tex_minimal --mog assets/mogs/$nm.json --tier minimal --jobs 3 2>&1 | grep -E 'finish\]|parts in|Error|Traceback' || true
for d in $build/tex_minimal/*_a*/; do
  k=$(basename "$d"); k=a${k##*_a}; mkdir -p $P/textures_minimal/$k
  cp "$d"/{color,metalness,roughness,normal,emissive}.png $P/textures_minimal/$k/
done
$PY "$H/downsize.py" $P 2048
"$BLENDER" -b assets/mogs/$nm.blend -P tools/render_mog.py -- $build/tex_minimal $P/renders --prefix minimal_ \
  --views three_quarter_rear,front,shoulder,knee --size 1600 --samples 128 2>&1 | grep -cE '^\[render\]' | sed 's/^/[repaint] worn renders: /'
mkdir -p $G/paint; rm -rf $G/paint/textures_minimal_2048; cp -r $P/textures_minimal_2048 $G/paint/
rm -f $G/paint/renders/minimal_*.png; cp $P/renders/minimal_*.png $G/paint/renders/
test -f $G/paint/textures_minimal_2048/a1/color.png
NODONE=1 bash "$H/pristine.sh" $n
echo "[repaint] $n ok"
