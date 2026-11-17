#!/bin/bash
#SBATCH --job-name=Jinterp
#SBATCH --output=logs/Jinterp_%A_%a.out
#SBATCH --error=logs/Jinterp_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Test of J_alpha (H band) interpolated in time between the production J snapshots (lightcone_J.cone_J, linear and
# nearest): J computed directly at renders inside the gaps of the run_21cm.sh list, compared over the whole box
# (check_J_interp.py). One array task per test snapshot. Default test renders (all outside the production list):
#   157 (gap 156-159, 3 renders, z ~ 10)   246 (242-250, 8)   603 (596-610, 14, the largest gap)   625 (620-630, 10)
# (610-620 is avoided: the self-shielding step at renders 617/618 lies inside it.)
#
# 1. the test J (stars + AGN), in a separate directory so that cone_J never sees it (8 tasks, ~1.1 GB each):
#      mkdir -p logs
#      export BAND=H ONLY="157 246 603 625" J_OUT=/orcd/data/mvogelsb/004/chcho/Jalpha_interp_test
#      bash run_Jalpha_prod.sh count
#      jid=$(sbatch --parsable -p ou_mki --array=0-7 run_Jalpha_prod.sh)
# 2. the comparison, after all J tasks succeeded (same shell, so J_OUT is still set):
#      sbatch -p ou_mki --dependency=afterok:$jid --array=0-3 run_J_interp_test.sh
# Output: logs/Jinterp_<job>_<task>.out (the tables) and $OUT/J_interp_NNN.npz.
# TEST_SNAPS overrides the list (keep it equal to ONLY of step 1, and --array=0-<n-1>).

SNAPS=(${TEST_SNAPS:-157 246 603 625})
JTEST=${J_OUT:-/orcd/data/mvogelsb/004/chcho/Jalpha_interp_test}
JPROD=/orcd/data/mvogelsb/004/chcho/Jalpha
OUT=${OUT:-$JTEST}
SPIN_MODEL=1                     # as the production 21 cm maps (Tb_maps_prod/H_real_wf)
R=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640
RB=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f check_J_interp.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
[ "$(realpath -m "$JTEST")" != "$(realpath -m "$JPROD")" ] || { echo "J_OUT must not be the production J directory"; exit 1; }
[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch --array=0-$((${#SNAPS[@]} - 1))"; exit 1; }
i=$SLURM_ARRAY_TASK_ID
[ "$i" -lt "${#SNAPS[@]}" ] || { echo "task $i >= ${#SNAPS[@]} test snapshots: nothing to do"; exit 0; }
S=${SNAPS[$i]}
if [ "$S" -le 428 ]; then D=$R; else D=$RB; fi
for f in "$JTEST/Jalpha_H_$S.hdf5" "$JTEST/Jalpha_AGN_H_$S.hdf5"; do
    [ -f "$f" ] || { echo "missing $f (step 1 not done?)"; exit 1; }
done
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "task $i: snapshot $S, renders $D, test J $JTEST, host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
python -u check_J_interp.py --snap "$S" --jtest "$JTEST" --jprod "$JPROD" --dir "$D" --spin-model $SPIN_MODEL --out "$OUT"
echo "done $(date)"
