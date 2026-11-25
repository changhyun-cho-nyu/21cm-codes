#!/bin/bash
# Build (if needed) and run the floor-box variants one after another on this machine.
#   bash run_local.sh <root> [variants...]      (default: fid nofloor entropy noclamp clean)
# Each run: mpirun -np ${NP:-8} ./Arepo param.txt in <root>/<variant>, log in run.log; skips variants already done.
set -u
ROOT=$1; shift
VARS=${*:-fid nofloor entropy noclamp clean}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/opt/homebrew/bin:$PATH
for v in $VARS; do
  D=$ROOT/$v
  if [ -f "$D/DONE" ]; then echo "$v: already done"; continue; fi
  if [ "$v" = noclamp ]; then
    python3 "$HERE/make_variants.py" --root "$ROOT" --only fid --build -j 8 > /dev/null || exit 1
    python3 "$HERE/make_variants.py" --root "$ROOT" --only noclamp > /dev/null && cp "$ROOT/fid/Arepo" "$D/Arepo"
  else
    python3 "$HERE/make_variants.py" --root "$ROOT" --only "$v" --build -j 8 | tail -1 || exit 1
  fi
  echo "$v: run started $(date)"
  (cd "$D" && rm -rf output && mpirun -np "${NP:-8}" ./Arepo param.txt > run.log 2>&1)
  if grep -q TERMINATE "$D/run.log"; then echo "$v: FAILED $(date)"; grep -m2 TERMINATE "$D/run.log"; continue; fi
  n=$(ls "$D"/output/snap_*.hdf5 2>/dev/null | wc -l)
  echo "$v: finished $(date), $n snapshots"; touch "$D/DONE"
done
