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
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Global ionization history of every snapshot (0-708) from the 640^3 renders (check_ionization_history.py):
# header z and x_HII, x_HeIII, x_HeII volume-weighted, mass-weighted and pixel means. Task i does snapshots i, i+16, ...
#   mkdir -p logs && sbatch run_ion_history.sh
# then, when all tasks are done:
#   python check_ionization_history.py report --csv "$OUT_DIR/part_*.csv" --write $OUT_DIR/ionization_history.csv
# x_HeII added 2026-11-24: rerun into a new directory and compare before replacing the old history, e.g.
#   OUT_DIR=/orcd/data/mvogelsb/004/chcho/ion_history_heii sbatch -p ou_mki run_ion_history.sh
#   python check_ionization_history.py report --csv "$OUT_DIR/part_*.csv" --write $OUT_DIR/ionization_history.csv \
#       --compare /orcd/data/mvogelsb/004/chcho/ion_history/ionization_history.csv
#   2026-11-24: 9 fields at ~0.73 h per snapshot (shared-filesystem load) -> 44 snapshots per task does not fit 12 h;
#   use NTASK=32 --array=0-31 --time=24:00:00 (~22 snapshots, ~16 h per task)

BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
DIRA=$BASE/Flagship_new/postprocessing/ren_640
DIRB=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640
OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/ion_history}
NTASK=${NTASK:-16}                     # must equal the array size

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_DIR"
[ "$NTASK" = "${SLURM_ARRAY_TASK_COUNT:-$NTASK}" ] || { echo "ERROR: NTASK=$NTASK, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
T=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
echo "task $T of $NTASK, $(date), commit $(git rev-parse --short HEAD)"
python -u check_ionization_history.py measure --dirA $DIRA --dirB $DIRB --task $SLURM_ARRAY_TASK_ID --ntask $NTASK \
    --out "$OUT_DIR/part_$T.csv"
echo "done $(date)"
