#!/bin/bash
#SBATCH --job-name=partHe
#SBATCH --output=logs/partHe_%j.out
#SBATCH --error=logs/partHe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Where the partially ionized He III sits (check_partial_HeIII.py): whole 640^3 renders, x_HeIII histogram, distance of the
# partially ionized pixels to the nearest ionized and neutral pixels. Default renders: 406, 428 (the two small early He
# sign-flip points) and 452, 476 (where x_HeII and the MFP split agree). ~25-40 GB, a few minutes per render.
#   mkdir -p logs && sbatch -p ou_mki run_partial_HeIII.sh
#   SNAPS="406" sbatch -p ou_mki run_partial_HeIII.sh
U=/orcd/data/mvogelsb/004/chcho
SNAPS=${SNAPS:-"406 428 452 476"}
OUT=${OUT:-$U/partial_HeIII}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "partial He III check, renders $SNAPS, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u check_partial_HeIII.py --mock
python -u check_partial_HeIII.py --snaps $SNAPS --out $OUT
echo "done $(date)"
