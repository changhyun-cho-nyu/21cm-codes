#!/bin/bash
#SBATCH --job-name=placeX
#SBATCH --output=logs/placeX_%A_%a.out
#SBATCH --error=logs/placeX_%A_%a.err
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Placement test of the no-X-ray model on the 50 cMpc pair (closure_placement.py): one array task per full snapshot,
# both runs read in the task, six dT_b passes over every cell (no X, X, and the predicted no X with four placements:
# tx = the current model, random, txloc, denv). Default: 048 057 066 080 092 (z = 10.0, 8.9, 8.0, 7.0, 6.3).
# Reads Oliver's snapshots and raw renders and the J_alpha maps under /orcd/data/mvogelsb/004/chcho/xray/<run>/Jalpha
# (read only); writes .../xray/pair_cells/closure_place_NNN.npz.
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_closure_placement.sh
#   a subset:  SNAPS="57" sbatch -p ou_mki --array=0 run_closure_placement.sh
SNAPS=(${SNAPS:-48 57 66 80 92})

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f closure_placement.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}
[ -n "$SNAP" ] || { echo "no snapshot for array task $SLURM_ARRAY_TASK_ID (SNAPS = ${SNAPS[*]})"; exit 1; }
df -h /orcd/data/mvogelsb/004 | tail -1
echo "snapshot $SNAP, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u closure_placement.py --snap $SNAP
echo "done $(date)"
