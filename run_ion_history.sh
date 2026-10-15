#!/bin/bash
#SBATCH --job-name=ionHist
#SBATCH --output=logs/ionHist_%A_%a.out
#SBATCH --error=logs/ionHist_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Global ionization history of every snapshot (0-708) from the 640^3 renders (check_ionization_history.py):
# header z and x_HII, x_HeIII volume-weighted, mass-weighted and pixel means. Task i does snapshots i, i+16, ...
#   mkdir -p logs && sbatch run_ion_history.sh
# then, when all tasks are done:
#   python check_ionization_history.py report --csv "$OUT_DIR/part_*.csv" --write $OUT_DIR/ionization_history.csv

BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
DIRA=$BASE/Flagship_new/postprocessing/ren_640
DIRB=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640
OUT_DIR=/orcd/data/mvogelsb/004/chcho/ion_history
NTASK=16

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_DIR"
T=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
echo "task $T of $NTASK, $(date), commit $(git rev-parse --short HEAD)"
python -u check_ionization_history.py measure --dirA $DIRA --dirB $DIRB --task $SLURM_ARRAY_TASK_ID --ntask $NTASK \
    --out "$OUT_DIR/part_$T.csv"
echo "done $(date)"
