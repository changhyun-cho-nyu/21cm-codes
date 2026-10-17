#!/bin/bash
#SBATCH --job-name=cellMerge
#SBATCH --output=logs/cellMerge_%j.out
#SBATCH --error=logs/cellMerge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_mvogelsb
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

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "merge: $LINE, $RUN snapshot $SNAP, render $RSNAP, J $J1 $J2, $(date), commit $(git rev-parse --short HEAD)"
python -u test_cells_Tb.py merge --snapdir $SD --snap $SNAP --line $LINE --J $J1 --J $J2 --render $REN --render-snap $RSNAP \
    --out $OUT_DIR --save-map
echo "done $(date)"
