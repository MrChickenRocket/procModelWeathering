#!/usr/bin/env bash
# rigid.sh <name>: worn set for a GRI rigid-piece character. Re-imports $GRI_CHARS/<name>/export, bakes, sims the
# minimal tier (spec assets/mogs/specs/<name>_rigid.json), writes <Part>.obj, textures_minimal_2048 and
# renders/minimal_*.png to $GRI_CHARS/<name>/paint. Writes no DONE file (lane.sh does, after verify).
set -e
source "$(dirname "$0")/common.sh"
n=$1; nm=${n}_rigid; G=$GRI_CHARS/$n; M=build/gri_src; X=$M/$n/export; P=$M/$n/paint
rm -rf $X; mkdir -p $X; cp $G/export/*.obj $G/export/mog.mtl $X/
# Rigid characters ship frames, not bones: one identity-framed bone per piece keeps the bind pose as exported.
$PY - "$G/export/rig.json" "$X/rig.json" <<'P'
import json, sys
r = json.load(open(sys.argv[1]))
pieces = [p for a in r["atlases"].values() for p in a]
ident = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
json.dump(dict(mech=r["name"], slots=r["slots"], bones=[dict(name=p, parent=None, cframe=ident) for p in pieces]), open(sys.argv[2], "w"))
P
$PY tools/mog_config.py assets/mogs/specs/$nm.json $X assets/mogs/$nm.json
mkdir -p build/$nm
TIERS=minimal NAME=$nm MOGS=$M JOBS=3 bash tools/run_mog.sh $n > build/$nm/run.log 2>&1
$PY "$H/downsize.py" $P 2048
"$BLENDER" -b assets/mogs/$nm.blend -P tools/render_mog.py -- build/$nm/tex_minimal $P/renders --prefix minimal_ \
  --views three_quarter_rear,front,shoulder,knee --size 1600 --samples 128 2>&1 | grep -E '^\[render\]' || true
mkdir -p $G/paint
cp $P/*.obj $P/atlases.json $P/mog_a*.mtl $G/paint/
rm -rf $G/paint/textures_minimal_2048; cp -r $P/textures_minimal_2048 $G/paint/
mkdir -p $G/paint/renders; rm -f $G/paint/renders/minimal_*.png; cp $P/renders/minimal_*.png $G/paint/renders/
test -f $G/paint/textures_minimal_2048/a1/color.png
grep -E "\[mog\]|\[export\]|\[bake\].*total|Traceback|Error" build/$nm/run.log | grep -v HIPEW || true
