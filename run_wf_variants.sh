#!/bin/bash
#SBATCH --job-name=WFvar
#SBATCH --output=logs/WFvar_%A_%a.out
#SBATCH --error=logs/WFvar_%A_%a.err
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#
# Wouthuysen-Field variants of the 21 cm maps (check_wf_variants.py): global signal and power spectra with
# V0 production, V1 exact tau_GP coefficient, V2 + (1+delta), V3 + Hirata (2006) S~_a and T_c.
# One snapshot per array task, the fiducial maps only.
#   mkdir -p logs && sbatch run_wf_variants.sh
# then:  python check_wf_variants.py --summary $OUT
# A task whose CSVs exist is skipped (resubmit to redo failures).

SNAPS=(0 30 60 80 100 110 120 130 140 150 156 174 200 242 300)   # z = 29.4 ... ~6; = the --array size
INPUT_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640/"
INPUT_NEW_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640/"
J_DIR=/orcd/data/mvogelsb/004/chcho/Jalpha
MAPS=/orcd/data/mvogelsb/004/chcho/Tb_maps_prod/H_real
OUT=/orcd/data/mvogelsb/004/chcho/wf_variants

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}
if [ "$SNAP" -le 428 ]; then DIR=$INPUT_DIR; else DIR=$INPUT_NEW_DIR; fi
J="--J $J_DIR/Jalpha_H_${SNAP}.hdf5 --J $J_DIR/Jalpha_AGN_H_${SNAP}.hdf5"
echo "WF variants, snapshot $SNAP, $(date), commit $(git rev-parse --short HEAD)"
df -h /orcd/data/mvogelsb/004 | tail -1

if [ ! -f "$OUT/wf_global_${SNAP}.csv" ]; then
    RIN=""; [ -f "$MAPS/RIN_${SNAP}.hdf5" ] && RIN="--rin $MAPS/RIN_${SNAP}.hdf5"
    python -u check_wf_variants.py --dir $DIR --snap $SNAP $J $RIN --ps --out $OUT
else echo "done already"; fi
echo "done $(date)"
