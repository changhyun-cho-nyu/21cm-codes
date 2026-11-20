#!/bin/bash
#SBATCH --job-name=bispec
#SBATCH --output=logs/bispec_%A_%a.out
#SBATCH --error=logs/bispec_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Bispectra of the production maps with BiFFT (bispec_maps.py), k in h/cMpc (box 338.3 cMpc/h).
# Needs the patched BiFFT (double-precision sums, ~/apps/bifft/src rebuilt 2026-11-19). Check it once first:
#   mkdir -p logs && SELFTEST=1 sbatch -p ou_mki --array=0 run_bispec.sh
# Then, settings from the command line (no defaults for SNAPS / K1: they are your choice), e.g.
#   FIELD=H  CONFIG=iso SNAPS="177 203 220 242" K1="0.1 0.3 0.6" sbatch -p ou_mki run_bispec.sh
#   FIELD=He CONFIG=iso SNAPS="126 131 136 141" K1="0.1 0.3"     sbatch -p ou_mki run_bispec.sh   # FULL snapshots
#   FIELD=xHeII CONFIG=iso SNAPS="450 500" K1="0.1 0.3"         sbatch -p ou_mki run_bispec.sh   # render snapshots
#   FIELD=H  CONFIG=equi SNAPS="242" KFAC="1.1"                  sbatch -p ou_mki run_bispec.sh
# FIELD: H (Tb_maps_prod/H_real_wf, render snapshots), He (gas-cell maps cell_test*/, full snapshots),
#        xHII / xHI (= 1 - xHII) / xHeIII / xHeII (render *_VolumeFraction, He ones / y; B flips sign between x and
#        1 - x: pair xHI with xHeII, or xHII with xHeIII). Task i does every NTASK-th (snapshot, k) job
#        (NTASK = the --array size); existing outputs are skipped, so a resubmission finishes what is missing.
# Memory: ~45 GB per 640^3 BiFFT call; the selftest also builds a float64 reference (~25 GB more).
# NUM_TH: BiFFT threads (the old jobs used 1 to avoid FFTW/OpenMP crashes); raise --cpus-per-task with it.
U=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$U/bispec_prod}
NUM_TH=${NUM_TH:-1}
NTASK=${SLURM_ARRAY_TASK_COUNT:-1}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "bispec task $SLURM_ARRAY_TASK_ID/$NTASK, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
if [ "${SELFTEST:-0}" = 1 ]; then
    python -u bispec_maps.py selftest --n 640 --k1 ${K1:-0.2} --num-th $NUM_TH
    python -u bispec_maps.py selftest --n 640 --k1 ${K1:-0.2} --theta 0.1 --num-th $NUM_TH
    echo "done $(date)"; exit 0
fi
: "${FIELD:?set FIELD (H, He, xHII, xHI, xHeIII, xHeII)}" "${CONFIG:?set CONFIG (iso, gen, equi)}" "${SNAPS:?set SNAPS}"
KARGS=""
if [ "$CONFIG" = equi ]; then KARGS="--kfac ${KFAC:-1.1}"
else : "${K1:?set K1 [h/cMpc]}"; KARGS="--k1 $K1"; [ "$CONFIG" = gen ] && KARGS="$KARGS --k2 ${K2:?set K2 for gen}"; fi
python -u bispec_maps.py run --field $FIELD --config $CONFIG --snaps $SNAPS $KARGS --out $OUT \
    --task $SLURM_ARRAY_TASK_ID --ntask $NTASK --num-th $NUM_TH
echo "done $(date)"
