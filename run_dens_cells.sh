#!/bin/bash
#SBATCH --job-name=densHe
#SBATCH --output=logs/densHe_%A_%a.out
#SBATCH --error=logs/densHe_%A_%a.err
#SBATCH --array=0-95
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Whole-box gas state and 3.46 cm signal by 1+delta, from every gas cell (dens_cells_He.py scan; user, 2026-11-26).
# Full snapshots 116 (z = 5.00), 136 (4.00), 158 (3.00); NTASK tasks each, array task = snapshot index x NTASK + task.
#   mkdir -p logs && sbatch run_dens_cells.sh
# Then, once all tasks are done (seconds each):
#   for s in 116 136 158; do python dens_cells_He.py merge --snap $s --out $OUT_DIR --ntask 32; done
# Reads 8 gas fields of each snapshot once (~3 x 4 TB); writes 96 small npz files (~50 kB). A task whose output exists
# is skipped, so a resubmission redoes only the failed ones.
# J: the He J of the earlier gas-cell scans (cell_test_summary.csv), so the whole-box means can be checked against
# cell_uK there (0.09081 / 0.0780 / 0.0442 uK). 136: J at render 500 (z = 4.0116, as the original scan, 2026-10-16), so
# render 500 is passed (the J file's z must match the render's) with MAX_DZ = 0.01 (snapshot z = 4.0042).

NTASK=${NTASK:-32}
OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/dens_He}
JDIR=${JDIR:-/orcd/data/mvogelsb/004/chcho/Jalpha}       # production He J (= the test J at 406 and 500, JALPHA_NOTES 6.1)
TR_TAG=${TR_TAG:-tr_sfmask}
#        snap  run    render  max_dz
TABLE=("116 above  406 1e-4"
       "136 below  500 0.01"
       "158 below  705 1e-4")

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
read -r SNAP RUN RSNAP MAX_DZ <<< "${TABLE[$(( SLURM_ARRAY_TASK_ID / NTASK ))]}"
TASK=$(( SLURM_ARRAY_TASK_ID % NTASK ))
if [ "$RUN" = above ]; then SD=$SNAP_ABOVE; else SD=$SNAP_BELOW; fi
if [ "$RSNAP" -le 428 ]; then REN=$BASE/Flagship_new/postprocessing/ren_640; else REN=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640; fi
J1=$JDIR/Jalpha_He_${RSNAP}_$TR_TAG.hdf5; J2=$JDIR/Jalpha_AGN_He_${RSNAP}_$TR_TAG.hdf5
for f in "$J1" "$J2" "$SD/snap_$(printf %03d $SNAP).hdf5"; do [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done
echo "dens task $TASK/$NTASK: snapshot $SNAP ($RUN), render $RSNAP, J $J1 $J2, $(date), commit $(git rev-parse --short HEAD)"
python -u dens_cells_He.py scan --snapdir $SD --snap $SNAP --J $J1 --J $J2 --render $REN --render-snap $RSNAP \
    --max-dz $MAX_DZ --out $OUT_DIR --task $TASK --ntask $NTASK
echo "done $(date)"
