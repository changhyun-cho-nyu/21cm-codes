#!/bin/bash
#SBATCH --job-name=conePS
#SBATCH --output=logs/conePS_%j.out
#SBATCH --error=logs/conePS_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Power spectra of the light-cone delta T_b (make_lightcone_ps.py) against the production maps, bands z = 3.5 ... 15,
# depths 338.3 and 150 cMpc/h. Run only after make_lightcone_tb.py has finished writing CONE_DIR (HDF5 files must not
# be read while another job writes them).
#   mkdir -p logs && sbatch run_lightcone_ps.sh
#   CONE_DIR=/orcd/data/mvogelsb/004/chcho/lightcone_test OUT_DIR=/orcd/data/mvogelsb/004/chcho/lightcone_test \
#     EXTRA="--z 7.5 --depth 60" sbatch run_lightcone_ps.sh                     # the test band z = 7.4-7.6
# Output: OUT_DIR/lightcone_ps_{H,He}.hdf5 (small).

CONE_DIR=${CONE_DIR:-/orcd/data/mvogelsb/004/chcho/lightcone}
OUT_DIR=${OUT_DIR:-$CONE_DIR}
LINES=${LINES:-"H He"}
EXTRA=${EXTRA:-}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "light cone PS: lines $LINES, cone $CONE_DIR, out $OUT_DIR, extra '$EXTRA', $(date), commit $(git rev-parse --short HEAD)"
python -u make_lightcone_ps.py --cone-dir $CONE_DIR --out $OUT_DIR --lines $LINES $EXTRA
echo "done $(date)"
