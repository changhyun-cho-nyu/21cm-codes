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
# T_X | 1+delta of Lumina's neutral IGM vs the 50 cMpc X-ray run (compare_TX_lumina.py). Tasks: Lumina 036, Lumina
# 048 (the 64 cube cell sets of the H gas-cell scans in /orcd/data/mvogelsb/004/chcho/cell_test_H_{36,48}, read only;
# ~1 h each, as decompose_cells_H.py), 50 cMpc 036, 048 (both runs, all cells, ~5-10 min). Writes
# /orcd/data/mvogelsb/004/chcho/xray/TX_compare/*.npz (a few hundred MB in all).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_compare_TX.sh
# then:  python compare_TX_lumina.py compare
JOBS=("lumina 36" "lumina 48" "pair 36" "pair 48")

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f compare_TX_lumina.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
J=(${JOBS[$SLURM_ARRAY_TASK_ID]})
[ -n "${J[0]}" ] || { echo "no job for array task $SLURM_ARRAY_TASK_ID"; exit 1; }
df -h /orcd/data/mvogelsb/004 | tail -1
echo "${J[0]} snapshot ${J[1]}, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u compare_TX_lumina.py ${J[0]} --snap ${J[1]}
echo "done $(date)"
