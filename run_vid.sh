#!/bin/bash
#SBATCH --job-name=VID
#SBATCH --output=logs/VID_%A_%a.out
#SBATCH --error=logs/VID_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# VIDs and one-point statistics of the production delta T_b maps (make_vid.py), for VID.ipynb.
# Task i does every NTASK-th snapshot of the map directory; existing outputs are skipped. Then, once:
#   python make_vid.py table --line $LINE --out $OUT
#   mkdir -p logs && LINE=H sbatch -p ou_mki run_vid.sh

# Settings from the command line, e.g.  LINE=H sbatch -p ou_mki run_vid.sh
#   H:  the SPIN_MODEL 1 maps (Tb_maps_prod/H_real_wf, from 2026-11-15) -> VID/H_wf. The SPIN_MODEL 0 VIDs stay in VID/H
#       (MAPS=.../H_real OUT=.../VID/H reproduces them). A new OUT is needed: existing outputs are skipped.
#   He: the gas-cell maps (CELL=1) -> VID/He_cell  (LINE=He CELL=1 sbatch -p ou_mki run_vid.sh). Every 3.46 cm statistic
#       comes from the gas-cell maps (2026-11-18); the 640^3 He maps (Tb_maps_prod/He -> VID/He) only with ALLOW_640=1.
LINE=${LINE:-He}                                                   # H or He
if [ "$LINE" = He ] && [ "${CELL:-0}" != 1 ] && [ "${ALLOW_640:-0}" != 1 ]; then
    echo "ERROR: He VIDs come from the gas-cell maps: LINE=He CELL=1 (ALLOW_640=1 for the old 640^3 He maps)"; exit 1; fi
U=/orcd/data/mvogelsb/004/chcho
if [ "$LINE" = H ]; then MAPS=${MAPS:-$U/Tb_maps_prod/H_real_wf}; OUT=${OUT:-$U/VID/H_wf}
else MAPS=${MAPS:-$U/Tb_maps_prod/He}; OUT=${OUT:-$U/VID/He}; fi
# CELL=1 (2026-11-18): the gas-cell-level maps (cell_test*/RIN_cell_<LINE>_NNN.hdf5, full snapshots) -> VID/<LINE>_cell
if [ "${CELL:-0}" = 1 ]; then SRC="--cell-root $U"; OUT=${OUT_CELL:-$U/VID/${LINE}_cell}; else SRC="--maps $MAPS"; fi
NTASK=8                                                           # = the --array size

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "VID $LINE task $SLURM_ARRAY_TASK_ID/$NTASK, $SRC, $(date), commit $(git rev-parse --short HEAD)"
python -u make_vid.py vid --line $LINE $SRC --out $OUT --task $SLURM_ARRAY_TASK_ID --ntask $NTASK
echo "done $(date)"
