#!/bin/bash
#SBATCH --job-name=cellScan
#SBATCH --output=logs/cellScan_%A_%a.out
#SBATCH --error=logs/cellScan_%A_%a.err
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Cell-level delta T_b test, step 1 (test_cells_Tb.py scan): each task reads all gas fields of 1/NTASK of the
# rows of one full snapshot, adds the cells into 640^3 sums (whole box) and keeps the cells of 64 random cubes.
# Then step 2: sbatch --dependency=afterok:<this job id> run_cells_merge.sh   (same settings, set them there too)
#   mkdir -p logs && sbatch run_cells_Tb.sh
# Reads all fields of the snapshot once (~8.6 TB in total). Disk in OUT_DIR: ~6.3 GB per task (640^3 sums) plus the
# cube cells (~75 GB); the task files can be deleted after the merge.
# A task whose output exists is skipped, so a resubmission redoes only the failed tasks.
# Settings can be given on the command line, e.g. for the slim run at full snapshot 92:
#   RUN=above SNAP=92 RSNAP=304 JDIR=/orcd/data/mvogelsb/004/chcho/Jalpha JSNAP=304 GRIDS=slim NCUBE=16 \
#   OUT_DIR=/orcd/data/mvogelsb/004/chcho/cell_test_92 sbatch run_cells_Tb.sh   (the same variables for run_cells_merge.sh)

LINE=${LINE:-He}
RUN=${RUN:-below}                      # snapshot directory: above | below z = 4.75
SNAP=${SNAP:-136}                       # full snapshot (z = 4.00417)
RSNAP=${RSNAP:-501}                      # render snapshot at the same z (inspect: dz = 0)
JDIR=${JDIR:-/orcd/data/mvogelsb/004/chcho/Jalpha_test}   # production He J at 501 not yet made (2026-10-15): the test J
JSNAP=${JSNAP:-500}                      # snapshot of the J files (render 500, z = 4.0116, dz = 0.0074; same J on both sides of the ratio)
TR_TAG=${TR_TAG:-tr_sfmask}
OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/cell_test}
NTASK=${NTASK:-32}                       # = the --array size
GRIDS=${GRIDS:-full}           # full (render checks, ~7.3 GB per task) or slim (volume and volume x dT_b, 2.1 GB)
NCUBE=${NCUBE:-64}             # cubes kept for the ladder (~1.6 GB each in total)

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
for f in "$J1" "$J2"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
echo "scan task $SLURM_ARRAY_TASK_ID/$NTASK: $LINE, $RUN snapshot $SNAP, render $RSNAP, J $J1 $J2, $(date), commit $(git rev-parse --short HEAD)"
python -u test_cells_Tb.py scan --snapdir $SD --snap $SNAP --line $LINE --J $J1 --J $J2 --render $REN --render-snap $RSNAP \
    --out $OUT_DIR --task $SLURM_ARRAY_TASK_ID --ntask $NTASK --grids $GRIDS --n-cube $NCUBE
echo "done $(date)"
