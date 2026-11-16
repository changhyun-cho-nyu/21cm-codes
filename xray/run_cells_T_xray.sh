#!/bin/bash
#SBATCH --job-name=cellTx
#SBATCH --output=logs/cellTx_%A_%a.out
#SBATCH --error=logs/cellTx_%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Gas-cell temperatures of the X-ray test runs at full snapshot SNAP (default 24, z = 15.02 = render 71) against the
# render pixels (cells_T_xray.py scan): task 0 = withoutXrays, 1 = withXrays. Reads Oliver's snapshots (read only),
# writes /orcd/data/mvogelsb/004/chcho/xray/<run>/cells_T/cells_T_NNN.npz (~1 MB).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_cells_T_xray.sh
# then:  python cells_T_xray.py report --snap 24
SNAP=${SNAP:-24}
RENDER=${RENDER:-71}
RUNS=(withoutXrays withXrays)

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f cells_T_xray.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
echo "run $RUN, snapshot $SNAP, render $RENDER, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u cells_T_xray.py scan --run $RUN --snap $SNAP --render $RENDER
echo "done $(date)"
