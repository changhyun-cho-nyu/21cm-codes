#!/bin/bash
#SBATCH --job-name=cmpTX
#SBATCH --output=logs/cmpTX_%A_%a.out
#SBATCH --error=logs/cmpTX_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# T_X | 1+delta of Lumina's neutral IGM vs the 50 cMpc X-ray run (compare_TX_lumina.py). Two array tasks per snapshot
# in SNAPS: task 2i = Lumina (the 64 cube cell sets of the H gas-cell scans in
# /orcd/data/mvogelsb/004/chcho/cell_test_H_<snap>, read only; ~40 min), task 2i+1 = 50 cMpc (both runs, all cells,
# ~2-10 min). Writes /orcd/data/mvogelsb/004/chcho/xray/TX_compare/*.npz (~20 MB per snapshot).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_compare_TX.sh                                    # 036, 048
#   SNAPS="24 57 66" sbatch -p ou_mki --array=0-5 run_compare_TX.sh
# then:  python compare_TX_lumina.py compare
SNAPS=(${SNAPS:-36 48})

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f compare_TX_lumina.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
[ "$((2 * ${#SNAPS[@]}))" -ge "${SLURM_ARRAY_TASK_COUNT:-1}" ] || { echo "ERROR: ${#SNAPS[@]} snapshots, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
KINDS=(lumina pair)
J=(${KINDS[$((SLURM_ARRAY_TASK_ID % 2))]} ${SNAPS[$((SLURM_ARRAY_TASK_ID / 2))]})
[ -n "${J[1]}" ] || { echo "no snapshot for array task $SLURM_ARRAY_TASK_ID (SNAPS = ${SNAPS[*]})"; exit 1; }
df -h /orcd/data/mvogelsb/004 | tail -1
echo "${J[0]} snapshot ${J[1]}, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u compare_TX_lumina.py ${J[0]} --snap ${J[1]}
echo "done $(date)"
