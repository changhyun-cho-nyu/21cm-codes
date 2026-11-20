#!/bin/bash
#SBATCH --job-name=switchCells
#SBATCH --output=logs/switchCells_%A_%a.out
#SBATCH --error=logs/switchCells_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Gas state by density around the run switch at z = 4.75 (check_switch_cells.py scan), from the cube cells of the He
# scan files: full snapshots 116, 121 (last of the z > 4.75 run), 126 (first of the z < 4.75 run), 131. Then, once:
#   python check_switch_cells.py compare --snaps 116 121 126 131
#   mkdir -p logs && sbatch run_switch_cells.sh
# Other snapshots (user, 2026-11-20: after He II reionization, z = 3.86 -> 3.0): SNAPS from the environment, with an
# array of the same length:
#   SNAPS="139 141 144 146 150 152 155 158" sbatch --array=0-7 run_switch_cells.sh

read -ra SNAPS <<< "${SNAPS:-116 121 126 131}"

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ "$SLURM_ARRAY_TASK_ID" -lt "${#SNAPS[@]}" ] || { echo "array task $SLURM_ARRAY_TASK_ID: only ${#SNAPS[@]} snapshots (${SNAPS[*]})"; exit 1; }
S=${SNAPS[$SLURM_ARRAY_TASK_ID]}
echo "full snapshot $S, $(date), commit $(git rev-parse --short HEAD)"
python -u check_switch_cells.py scan --snap $S
echo "done $(date)"
