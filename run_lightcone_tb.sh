#!/bin/bash
#SBATCH --job-name=coneTb
#SBATCH --output=logs/coneTb_%j.out
#SBATCH --error=logs/coneTb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# delta T_b (21 cm and 3.46 cm) on the whole light cone with the sampled production J (make_lightcone_tb.py; time
# linear, outside hold). Resumes: resubmit with the same OUT_DIR to continue after a time-out.
#   mkdir -p logs && sbatch run_lightcone_tb.sh
#   OUT_DIR=/orcd/data/mvogelsb/004/chcho/lightcone_test EXTRA="--zrange 7.4 7.6" sbatch run_lightcone_tb.sh   # a test band
# Output: OUT_DIR/lightcone_dTb_{H,He}.hdf5, 9.7 GB each (check df -h first).

OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/lightcone}
LINES=${LINES:-"H He"}
EXTRA=${EXTRA:-}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "light cone dT_b: lines $LINES, out $OUT_DIR, extra '$EXTRA', $(date), commit $(git rev-parse --short HEAD)"
python -u make_lightcone_tb.py --out $OUT_DIR --lines $LINES $EXTRA
echo "done $(date)"
