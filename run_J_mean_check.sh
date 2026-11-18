#!/bin/bash
#SBATCH --job-name=Jmean
#SBATCH --output=logs/Jmean_%j.out
#SBATCH --error=logs/Jmean_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Which box mean of the H stellar J files is off: J_mean (shell map, clipped) or J_scalar (midpoint over snapshots)?
# check_J_mean.py: (1) box means of the 709 SFR_eff boxes (reads ~0.7 TB, resumable: resubmit after a time-out),
# (2) for every production Jalpha_H file: the C code's J_scalar rule and shell sum re-implemented, and a converged
# integral, against the file's J_mean and J_scalar. Submit from the repository directory:
#   mkdir -p logs && sbatch run_J_mean_check.sh
# Output: logs/Jmean_<job>.out (the table) and $OUT/J_mean_check.csv; the SFR means in $OUT/sfr_means_H.csv.

SFR=/orcd/data/mvogelsb/004/chcho/SFR_eff/H
JDIR=/orcd/data/mvogelsb/004/chcho/Jalpha
SPEC=/orcd/data/mvogelsb/004/ozier/ZOOM_XL/sharedData/bpass_spectra_bin_chab100_v2.2.1.hdf5   # compute_Jalpha.c SPECTRUM_FILE
WEIGHTS=/orcd/data/mvogelsb/004/chcho/cell_by_cell/hydrogen/Z_weights.txt                     # run_Jalpha_test.sh WEIGHTS
OUT=${OUT:-/orcd/data/mvogelsb/004/chcho/J_mean_check}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f check_J_mean.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
mkdir -p "$OUT"
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
python -u check_J_mean.py means --sfr "$SFR" --csv "$OUT/sfr_means_H.csv"
python -u check_J_mean.py compare --csv "$OUT/sfr_means_H.csv" --sfr "$SFR" --jdir "$JDIR" \
    --spectrum "$SPEC" --weights "$WEIGHTS" --out "$OUT/J_mean_check.csv"
echo "done $(date)"
