#!/bin/bash
#SBATCH --job-name=SFR_eff_conv
#SBATCH --output=logs/SFR_eff_conv_%j.out
#SBATCH --error=logs/SFR_eff_conv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Cadence convergence test of the linear-in-time SFR interpolation in make_SFR_eff.py:
# every 2nd and every 4th snapshot on planes 0-63, compared with the full run in $OUT.
# Submit from the repository directory:  mkdir -p logs && sbatch run_SFR_eff_convergence.sh

OUT=/orcd/data/mvogelsb/004/chcho/SFR_eff                 # the finished full run (same as run_SFR_eff.sh)
WEIGHTS=/orcd/data/mvogelsb/004/chcho/cell_by_cell/hydrogen/Z_weights.txt
PLANES="0 64"

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
[ -f "$OUT/SFR_eff_summary.csv" ] || { echo "no finished run in $OUT"; exit 1; }

for S in 2 4; do
    python -u make_SFR_eff.py --out "${OUT}_s$S" --weights "$WEIGHTS" --plane-range $PLANES --stride $S
done
for S in 2 4; do
    echo "===== stride 1 vs $S ====="
    python -u compare_SFR_eff.py "$OUT" "${OUT}_s$S"
done
echo "done $(date)"
