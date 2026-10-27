#!/usr/bin/env bash
# lane.sh <name>...: full worn + pristine paint for each character in turn; DONE.txt after a verified worn set,
# PRISTINE_DONE.txt last. Logs to build/<name>_rigid/lane.log.
source "$(dirname "$0")/common.sh"
for n in "$@"; do
  G=$GRI_CHARS/$n; log=build/${n}_rigid/lane.log; mkdir -p build/${n}_rigid; t=$(date +%s)
  if ! bash "$H/rigid.sh" $n > $log 2>&1; then echo "[lane] $n worn FAILED, no DONE.txt ($log)"; continue; fi
  bad=$(bash "$H/verify.sh" $G 2>&1 | grep "\.obj" | grep -vc same)
  printf "%s (rigid) painted %s: minimal tier wear, model-kit finish.\n<Part>.obj UV'd in bind pose; textures_minimal_2048/a1|a2|a3; renders/minimal_*.png.\n" $n "$(date +%F)" > $G/paint/DONE.txt
  if ! bash "$H/pristine.sh" $n >> $log 2>&1; then echo "[lane] $n pristine FAILED ($log)"; continue; fi
  echo "[lane] $n $(( ($(date +%s)-t)/60 )) min, obj geometry mismatches $bad"
done
