#!/bin/bash
#SBATCH --job-name=cellDecH
#SBATCH --output=logs/cellDecH_%A_%a.out
#SBATCH --error=logs/cellDecH_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Which gas makes the gas-cell 21 cm delta T_b (decompose_cells_H.py): reads only the cube cells saved by the scan
# tasks of run_cells_Tb.sh (LINE=H) in cell_test_H_<snap>; no rescan. Read-only, so it can run while a merge of the
# same snapshot is still running. One array task per snapshot:
#   mkdir -p logs && sbatch --array=0-1 run_decompose_cells_H.sh                      # 011 (z = 23), 024 (z = 15)
#   SNAPS="11 24 36 48" sbatch --array=0-3 run_decompose_cells_H.sh                   # also z = 12, 10
# Check in each log: the CHECK line must equal the "cell" line of the CUBES table in that snapshot's merge log.

SNAPS=(${SNAPS:-11 24})
DATA=/orcd/data/mvogelsb/004/chcho
SD=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots      # all H gas-cell snapshots are above z = 4.75

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ "${#SNAPS[@]}" -ge "${SLURM_ARRAY_TASK_COUNT:-1}" ] || { echo "ERROR: ${#SNAPS[@]} snapshots, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
S=${SNAPS[${SLURM_ARRAY_TASK_ID:-0}]}
[ -n "$S" ] || { echo "ERROR: no snapshot for task $SLURM_ARRAY_TASK_ID"; exit 1; }
OUT=$DATA/cell_test_H_$S
echo "decompose H: snapshot $S, $OUT, $(date), commit $(git rev-parse --short HEAD)"
python -u decompose_cells_H.py --snapdir $SD --snap $S --out $OUT
echo "done $(date)"
