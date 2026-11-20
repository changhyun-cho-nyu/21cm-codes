#!/bin/bash
#SBATCH --job-name=closX
#SBATCH --output=logs/closX_%A_%a.out
#SBATCH --error=logs/closX_%A_%a.err
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Closure test of the post-processed no-X-ray model on the 50 cMpc pair (closure_noX_xray.py): one array task per full
# snapshot, both runs read in the task (~2.2e8 cells each), four dT_b passes over every cell (no X, X, predicted no X
# trained on all pairs and on the other half of the 16^3 blocks). Default: every full snapshot down to z = 5.5
# (024 036 048 057 066 074 080 086 092 097 102 106). Reads Oliver's snapshots and raw renders and the J_alpha maps
# under /orcd/data/mvogelsb/004/chcho/xray/<run>/Jalpha (read only); writes .../xray/pair_cells/closure_NNN.npz.
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_closure_noX.sh
#   a subset:  SNAPS="57 74" sbatch -p ou_mki --array=0-1 run_closure_noX.sh
SNAPS=(${SNAPS:-24 36 48 57 66 74 80 86 92 97 102 106})

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f closure_noX_xray.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}
[ -n "$SNAP" ] || { echo "no snapshot for array task $SLURM_ARRAY_TASK_ID (SNAPS = ${SNAPS[*]})"; exit 1; }
df -h /orcd/data/mvogelsb/004 | tail -1
echo "snapshot $SNAP, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u closure_noX_xray.py --snap $SNAP
echo "done $(date)"
