#!/bin/bash
#SBATCH --job-name=resTb
#SBATCH --output=logs/resTb_%A_%a.out
#SBATCH --error=logs/resTb_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Pixel-averaging test of the delta T_b maps (test_resolution_Tb.py): delta T_b from ren_1280 averaged
# 2x2x2 vs delta T_b from ren_640 (production), real space, same J_alpha. Writes nothing but the logs.
#   mkdir -p logs && sbatch run_resolution_test.sh
# Tasks 0-4: 21 cm at snapshots 60, 100, 156, 220, 300 (production J); tasks 5-7: 3.46 cm at 300, 406, 500
# (the absorbed test J, Jalpha_test/*_tr_sfmask).

TASKS=("H 60" "H 100" "H 156" "H 220" "H 300" "He 300" "He 406" "He 500")
JH=/orcd/data/mvogelsb/004/chcho/Jalpha
JHE=/orcd/data/mvogelsb/004/chcho/Jalpha_test
TR_TAG=tr_sfmask
BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
A6=$BASE/Flagship_new/postprocessing/ren_640;               A12=$BASE/Flagship_new/postprocessing/ren_1280
B6=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640;   B12=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_1280

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
read LINE S <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [ "$S" -le 428 ]; then D6=$A6; D12=$A12; else D6=$B6; D12=$B12; fi
if [ "$LINE" = H ]; then J1=$JH/Jalpha_H_$S.hdf5; J2=$JH/Jalpha_AGN_H_$S.hdf5
else J1=$JHE/Jalpha_He_${S}_$TR_TAG.hdf5; J2=$JHE/Jalpha_AGN_He_${S}_$TR_TAG.hdf5; fi
echo "line $LINE snap $S, $(date), commit $(git rev-parse --short HEAD)"
python -u test_resolution_Tb.py --line $LINE --dir640 $D6 --dir1280 $D12 --snap $S --J $J1 --J $J2
echo "done $(date)"
