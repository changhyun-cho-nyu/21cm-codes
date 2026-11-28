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
#   for s in 116 136 158; do python dens_cells_He.py merge --snap $s --out /orcd/data/mvogelsb/004/chcho/dens_He_Ly --ntask 32; done
# Reads 8 gas fields of each snapshot once (~3 x 4 TB); writes 96 small npz files (~50 kB). A task whose output exists
# is skipped, so a resubmission redoes only the failed ones.
# J: the He J of the earlier gas-cell scans (cell_test_summary.csv), so the whole-box means can be checked against
# cell_uK there (0.09081 / 0.0780 / 0.0442 uK). 136: J at render 500 (z = 4.0116, as the original scan, 2026-10-16) in
# the runs to dens_He/ and dens_He_T/ (render 500, MAX_DZ = 0.01); from 2026-11-27 (dens_He_Ly/) the production J at the
# matching render 501 (dz = 0; user), so the z = 4 mean differs slightly from 0.0780 (J 500).
# From 2026-11-27 the task files also hold the T-class sums and Lumina's He rates (effective He II photoionization rate
# by density; the merge prints it). The first run (job 30624503) is in dens_He/ without them; the new default output is
# dens_He_T/ (z = 3.0 only, job 30724213). From 2026-11-27 they also hold V dT_b with J = 0 (x_alpha = 0), for the
# gas-cell Lyman alpha share of the signal (the merge prints it, whole box and by 1+delta); the new default output is
# dens_He_Ly/ (a task stops if its own output file is an older one; the merge stops on mixed files). All three snapshots: sbatch run_dens_cells.sh;
# only z = 3.0 (158): sbatch --array=64-95 run_dens_cells.sh

NTASK=${NTASK:-32}
OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/dens_He_Ly}
JDIR=${JDIR:-/orcd/data/mvogelsb/004/chcho/Jalpha}       # production He J (= the test J at 406 and 500, JALPHA_NOTES 6.1)
TR_TAG=${TR_TAG:-tr_sfmask}
#        snap  run    render  max_dz
TABLE=("116 above  406 1e-4"
       "136 below  501 1e-4"
       "158 below  705 1e-4")

SNAP_ABOVE=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots
SNAP_BELOW=/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots
BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
NARR=$(( ${#TABLE[@]} * NTASK ))
# a sub-range of the array (one snapshot) is allowed; the merge stops unless all NTASK tasks of a snapshot are there
[ "$NTASK" = 32 ] || [ -n "$ALLOW_NTASK" ] || { echo "ERROR: NTASK=$NTASK: set --array to match and ALLOW_NTASK=1"; exit 1; }
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: not an array task"; exit 1; }
[ "$SLURM_ARRAY_TASK_ID" -lt "$NARR" ] || { echo "ERROR: array task $SLURM_ARRAY_TASK_ID >= $NARR (0-$((NARR - 1)))"; exit 1; }
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
