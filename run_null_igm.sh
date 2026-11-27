#!/bin/bash
#SBATCH --job-name=nullIGM
#SBATCH --output=logs/nullIGM_%A_%a.out
#SBATCH --error=logs/nullIGM_%A_%a.err
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Null test for the IGM-only cross-spectra (null_igm_map.py scan; user, 2026-11-27): the He maps' cut and deposit applied
# to (1+delta)^2, which has no x_HeIII. Full snapshots 116 (z = 5.0), 139 (3.86, bubble peak), 146 (3.5), 158 (3.0);
# NTASK tasks each, array task = snapshot index x NTASK + task. Then (merge, cross-spectra, table):
#   sbatch --dependency=afterok:<this job id> run_null_igm_cross.sh
#   mkdir -p logs && sbatch run_null_igm.sh
# Reads Coordinates, Masses, Density, StarFormationRate (~0.7 TB per task). Disk in OUT_DIR: 6.3 GB per task file
# (~200 GB in all; run_null_igm_cross.sh deletes them after a successful merge with CLEAN=1). A task whose output exists
# is skipped.

NTASK=${NTASK:-8}
OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/null_igm}
#        snap  run    render
TABLE=("116 above  406"
       "139 below  516"
       "146 below  582"
       "158 below  705")

SNAP_ABOVE=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots
SNAP_BELOW=/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots
BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
NARR=$(( ${#TABLE[@]} * NTASK ))
[ -z "$SLURM_ARRAY_TASK_COUNT" ] || [ "$SLURM_ARRAY_TASK_COUNT" = "$NARR" ] || \
    { echo "ERROR: the array has $SLURM_ARRAY_TASK_COUNT tasks, need $NARR (sbatch --array=0-$((NARR - 1)))"; exit 1; }
read -r SNAP RUN RSNAP <<< "${TABLE[$(( SLURM_ARRAY_TASK_ID / NTASK ))]}"
TASK=$(( SLURM_ARRAY_TASK_ID % NTASK ))
if [ "$RUN" = above ]; then SD=$SNAP_ABOVE; else SD=$SNAP_BELOW; fi
if [ "$RSNAP" -le 428 ]; then REN=$BASE/Flagship_new/postprocessing/ren_640; else REN=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640; fi
echo "null task $TASK/$NTASK: snapshot $SNAP ($RUN), render $RSNAP, $(date), commit $(git rev-parse --short HEAD)"
python -u null_igm_map.py scan --snapdir $SD --snap $SNAP --render $REN --render-snap $RSNAP --out $OUT_DIR \
    --task $TASK --ntask $NTASK
echo "done $(date)"
