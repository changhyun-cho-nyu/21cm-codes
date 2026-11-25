#!/bin/bash
#SBATCH --job-name=He640hz
#SBATCH --output=logs/He640hz_%A_%a.out
#SBATCH --error=logs/He640hz_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# he640_highz.py: volume-mean 3.46 cm dT_b on the 640^3 renders at z = 15.05-11.98 (renders 73-113, user 2026-11-24; J = 0, no He J files
# above z = 10.01), for panel (b) of the Appendix metallicity figure above the last He gas-cell point (full snapshot 36,
# z = 11.93; user, 2026-11-24). Task 0 first checks the calculation against the production He map at render 156
# (with its J files). Every render's whole box; task i does every 8th render of SNAPS.
#     mkdir -p logs && sbatch -p ou_mki run_he640_highz.sh
# Writes /orcd/data/mvogelsb/004/chcho/metallicity/He640/He640_NNN.csv.
OUT=${OUT:-/orcd/data/mvogelsb/004/chcho/metallicity/He640}
SNAPS=(${SNAPS:-$(seq 73 113)})
NTASK=${SLURM_ARRAY_TASK_COUNT:-8}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f he640_highz.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
MINE=()
for ((i = SLURM_ARRAY_TASK_ID; i < ${#SNAPS[@]}; i += NTASK)); do MINE+=(${SNAPS[$i]}); done
echo "He640 high z: task $SLURM_ARRAY_TASK_ID/$NTASK, renders ${MINE[*]}, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    python -u he640_highz.py --check 156 --out "$OUT"
fi
python -u he640_highz.py --snaps ${MINE[*]} --out "$OUT"
echo "done $(date)"
