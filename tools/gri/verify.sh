#!/usr/bin/env bash
# verify.sh <dir with export/ and paint/>: v/vn identical per OBJ
cd $1
for f in export/*.obj; do n=$(basename $f); for t in v vn; do a=$(grep "^$t " export/$n | awk '{printf "%.4f %.4f %.4f\n",$2,$3,$4}' | md5sum); b=$(grep "^$t " paint/$n | awk '{printf "%.4f %.4f %.4f\n",$2,$3,$4}' | md5sum); [ "$a" = "$b" ] && echo "$n $t same" || echo "$n $t DIFF"; done; done
ls paint/textures_minimal*/*/ | tr '\n' ' '; echo
