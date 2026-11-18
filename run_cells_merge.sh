#!/bin/bash
#SBATCH --job-name=cellMerge
#SBATCH --output=logs/cellMerge_%j.out
#SBATCH --error=logs/cellMerge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Cell-level delta T_b test, step 2 (test_cells_Tb.py merge): sums the scan tasks (run_cells_Tb.sh), compares the
# whole-box cell level with the production calculation (mean, P(k)), and runs the resolution ladder on the cubes.
#   sbatch --dependency=afterok:<scan job id> run_cells_merge.sh
# Keep the settings below the same as in run_cells_Tb.sh. Writes the 640^3 cell-level and production maps
# (cellmap_<line>_<snap>.hdf5, 2 GB) to OUT_DIR.

LINE=${LINE:-He}
RUN=${RUN:-below}
SNAP=${SNAP:-136}
RSNAP=${RSNAP:-501}
JDIR=${JDIR:-/orcd/data/mvogelsb/004/chcho/Jalpha_test}   # production He J at 501 not yet made (2026-10-15): the test J
JSNAP=${JSNAP:-500}                      # snapshot of the J files (render 500, z = 4.0116, dz = 0.0074; same J on both sides of the ratio)
TR_TAG=${TR_TAG:-tr_sfmask}
OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/cell_test}

SNAP_ABOVE=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots
SNAP_BELOW=/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots
BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
if [ "$RUN" = above ]; then SD=$SNAP_ABOVE; else SD=$SNAP_BELOW; fi
if [ "$RSNAP" -le 428 ]; then REN=$BASE/Flagship_new/postprocessing/ren_640; else REN=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640; fi
if [ "$LINE" = H ]; then J1=$JDIR/Jalpha_H_$JSNAP.hdf5; J2=$JDIR/Jalpha_AGN_H_$JSNAP.hdf5
else J1=$JDIR/Jalpha_He_${JSNAP}_$TR_TAG.hdf5; J2=$JDIR/Jalpha_AGN_He_${JSNAP}_$TR_TAG.hdf5; fi
JARGS="--J $J1 --J $J2"
if [ "$JSNAP" = none ]; then J1=none; J2=; JARGS="--J none"; fi   # JSNAP=none: J = 0 (He above z = 10.01, no J files)
MAX_DZ=${MAX_DZ:-1e-4}                 # largest |z_snapshot - z_render RSNAP| accepted
RSNAP_B=${RSNAP_B:-}                   # second render bracketing the snapshot z: J linear in z between RSNAP and RSNAP_B
JSNAP_B=${JSNAP_B:-$RSNAP_B}           # (full snapshots 11, 17, 24, which fall between renders; user decision 2026-11-17)
JB1=; JB2=
if [ -n "$RSNAP_B" ]; then
    if [ "$JSNAP" != "$RSNAP" ] || [ "$JSNAP_B" != "$RSNAP_B" ]; then echo "RSNAP_B needs J at the renders themselves (JSNAP = RSNAP, JSNAP_B = RSNAP_B)"; exit 1; fi
    if [ $(( RSNAP <= 428 )) != $(( RSNAP_B <= 428 )) ]; then echo "RSNAP and RSNAP_B must be in the same render directory"; exit 1; fi
    if [ "$LINE" = H ]; then JB1=$JDIR/Jalpha_H_$JSNAP_B.hdf5; JB2=$JDIR/Jalpha_AGN_H_$JSNAP_B.hdf5
    else JB1=$JDIR/Jalpha_He_${JSNAP_B}_$TR_TAG.hdf5; JB2=$JDIR/Jalpha_AGN_He_${JSNAP_B}_$TR_TAG.hdf5; fi
    JARGS="$JARGS --J-b $JB1 --J-b $JB2 --render-snap-b $RSNAP_B"
fi
JARGS="$JARGS --max-dz $MAX_DZ"

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "merge: $LINE, $RUN snapshot $SNAP, render $RSNAP${RSNAP_B:+ (J linear in z with $RSNAP_B)}, J $J1 $J2 $JB1 $JB2, $(date), commit $(git rev-parse --short HEAD)"
python -u test_cells_Tb.py merge --snapdir $SD --snap $SNAP --line $LINE $JARGS --render $REN --render-snap $RSNAP \
    --out $OUT_DIR --save-map
echo "done $(date)"
