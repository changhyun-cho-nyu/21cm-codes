#!/bin/bash
#SBATCH --job-name=cellTbx
#SBATCH --output=logs/cellTbx_%A_%a.out
#SBATCH --error=logs/cellTbx_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# 21 cm delta T_b of every gas cell of the X-ray test runs (cells_Tb_xray.py scan) against the render pixels, tau_GP
# from the cell's and from its pixel's n_HI. Task i: run i / 4 (0 withoutXrays, 1 withXrays), snapshot i % 4 of
#   024 (z = 15.02, render 71), 036 (11.93, 112), 048 (10.01, 154): every full snapshot at z > 10;
#   066 (7.995, render 207): validation in the X-ray run only (T_S >> T_gamma: cells and pixels must agree); in the
#   no-X-ray run the void cells are still cold, so a difference there is physics, not a failure.
# Reads Oliver's snapshots (read only), the assembled renders, J_alpha and the 21 cm maps under $BASE/<run>/;
# writes $BASE/<run>/cells_Tb/cells_Tb_NNN.npz (~40 MB). One task ~1 h (two T_S solves per cell).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_cells_Tb_xray.sh          (one snapshot: --array=0,4 etc.)
# then:  python cells_Tb_xray.py report
SNAPS=(24 36 48 66)
RENDERS=(71 112 154 207)
RUNS=(withoutXrays withXrays)

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f cells_Tb_xray.py ] && [ -f ../tb_physics.py ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
T=$SLURM_ARRAY_TASK_ID
RUN=${RUNS[$((T / 4))]}
SNAP=${SNAPS[$((T % 4))]}
RENDER=${RENDERS[$((T % 4))]}
echo "run $RUN, snapshot $SNAP, render $RENDER, $(date), commit $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- . ../tb_physics.py)" ] || echo ' + local or untracked changes'), host $(hostname)"
python -u cells_Tb_xray.py scan --run $RUN --snap $SNAP --render $RENDER
echo "done $(date)"
