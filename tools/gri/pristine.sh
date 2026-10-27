#!/usr/bin/env bash
# pristine.sh <name>: factory-new set on the bake/UVs of the last worn run (rigid.sh): no wear or dirt, livery decals.
# Writes textures_pristine_2048/a*, renders/pristine_*.png, then PRISTINE_DONE.txt last unless NODONE is set.
set -e
source "$(dirname "$0")/common.sh"
n=$1; nm=${n}_rigid; G=$GRI_CHARS/$n; build=build/$nm; P=build/gri_src/$n/paint
test -f $build/bake/wear_volumes.json
$PY "$H/pristine_cfg.py" $n assets/mogs/$nm.json assets/mogs/${nm}_pristine.json "$G/export/rig.json"
rm -rf $build/tex_pristine $P/textures_pristine $P/textures_pristine_2048
$PY -m sim.run $build/bake $build/tex_pristine --mog assets/mogs/${nm}_pristine.json --tier minimal \
  --factory 1 --age 0 --detail 0 --jobs 2 2>&1 | grep -E 'livery]|finish]|parts in|Error|Traceback' || true
for d in $build/tex_pristine/*_a*/; do
  k=$(basename "$d"); k=a${k##*_a}
  mkdir -p $P/textures_pristine/$k
  cp "$d"/{color,metalness,roughness,normal,emissive}.png $P/textures_pristine/$k/
done
$PY "$H/downsize.py" $P 2048
"$BLENDER" -b assets/mogs/$nm.blend -P tools/render_mog.py -- $build/tex_pristine $P/renders --prefix pristine_ \
  --views three_quarter,front,three_quarter_rear,head,shoulder,knee,legs --size 1600 --samples 128 2>&1 | grep -E '^\[render\]' || true
mkdir -p $G/paint; rm -rf $G/paint/textures_pristine_2048; cp -r $P/textures_pristine_2048 $G/paint/
mkdir -p $G/paint/renders; rm -f $G/paint/renders/pristine_*.png; cp $P/renders/pristine_*.png $G/paint/renders/
test -f $G/paint/textures_pristine_2048/a1/color.png
[ -n "$NODONE" ] || printf "%s pristine painted %s: factory-new finish (no wear/dirt), livery decals, model-kit finish.\nSame <Part>.obj/UVs as the worn set; textures_pristine_2048/a1|a2|a3; renders/pristine_*.png.\n" $n "$(date +%F)" > $G/paint/PRISTINE_DONE.txt
echo "[pristine] $n done"
