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
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# VIDs and one-point statistics of the production delta T_b maps (make_vid.py), for VID.ipynb.
# Task i does every NTASK-th snapshot of the map directory; existing outputs are skipped. Then, once:
#   python make_vid.py table --line $LINE --out $OUT
#   mkdir -p logs && sbatch run_vid.sh

LINE=H                                                            # H or He
if [ "$LINE" = H ]; then MAPS=/orcd/data/mvogelsb/004/chcho/Tb_maps_prod/H_real
else MAPS=/orcd/data/mvogelsb/004/chcho/Tb_maps_prod/He; fi
OUT=/orcd/data/mvogelsb/004/chcho/VID/$LINE
NTASK=8                                                           # = the --array size

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "VID $LINE task $SLURM_ARRAY_TASK_ID/$NTASK, maps $MAPS, $(date), commit $(git rev-parse --short HEAD)"
python -u make_vid.py vid --line $LINE --maps $MAPS --out $OUT --task $SLURM_ARRAY_TASK_ID --ntask $NTASK
echo "done $(date)"
