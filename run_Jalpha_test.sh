#!/bin/bash
#SBATCH --job-name=Jalpha_test
#SBATCH --output=logs/Jtest_%A_%a.out
#SBATCH --error=logs/Jtest_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# J_alpha test runs with the effective SFR: 4 snapshots x 4 codes = 16 array tasks
# (H stars, He stars, AGN-H, AGN-He). From the repository directory:
#   1. bash run_Jalpha_test.sh compile      # once, on the login node (builds bin/)
#   2. mkdir -p logs && sbatch run_Jalpha_test.sh
#   3. grep -A12 "=== Results" logs/Jtest_<jobid>_*.out
#      grep -h "Input:" logs/Jtest_<jobid>_*.out

SNAPS=(156 300 406 500)                 # z ~ 10, 6.35, 5.0, 4.0
CODES=(H He AGN_H AGN_He)

OUT=/orcd/data/mvogelsb/004/chcho/SFR_eff                       # make_SFR_eff.py output
WEIGHTS=/orcd/data/mvogelsb/004/chcho/cell_by_cell/hydrogen/Z_weights.txt   # one file for H and He
J=/orcd/data/mvogelsb/004/chcho/Jalpha_test                     # new output directory
R=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640
RB=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640
SED=agn_sed_full.hdf5                                           # in the repository
MIN_FREE_GB=20                                                  # ~1 GB per output, 16 outputs
# FFTW (single precision, with OpenMP) and HDF5 install prefixes, if no module provides them:
# the directories that contain include/fftw3.h and include/hdf5.h (same as in your Makefile).
FFTW_DIR=
HDF5_DIR=

source ~/.bashrc
# module load ...        # <- the same modules as your usual J_alpha jobs (MPI, FFTW, HDF5)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

INC=""; LIBS="-lfftw3f_omp -lfftw3f -lhdf5 -lm"
for d in "$FFTW_DIR" "$HDF5_DIR"; do
    [ -n "$d" ] || continue
    INC="$INC -I$d/include"
    for l in "$d/lib" "$d/lib64"; do [ -d "$l" ] && LIBS="-L$l -Wl,-rpath,$l $LIBS"; done
done
if [ "$1" = "compile" ]; then
    mkdir -p bin
    # no -march=native: the login node CPU may differ from the compute nodes
    mpicc -O3 -fopenmp $INC compute_Jalpha.c -o bin/Jalpha_H \
        -DZ_WEIGHTS_FILE="\"$WEIGHTS\"" $LIBS
    mpicc -O3 -fopenmp $INC compute_Jalpha_He.c -o bin/Jalpha_He \
        -DZ_WEIGHTS_FILE="\"$WEIGHTS\"" -DSFR_DIR_OLD="\"$OUT/He\"" -DSFR_DIR_NEW="\"$OUT/He\"" $LIBS
    mpicc -O3 -fopenmp $INC compute_Jalpha_AGN.c -o bin/Jalpha_AGN $LIBS
    echo "built: $(ls bin)   commit $(git rev-parse --short HEAD)"
    exit 0
fi

[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch (or run: bash $0 compile)"; exit 1; }
for b in bin/Jalpha_H bin/Jalpha_He bin/Jalpha_AGN; do
    [ -x "$b" ] || { echo "missing $b: run 'bash $0 compile' first"; exit 1; }
done
[ -f "$SED" ] && [ -f "$WEIGHTS" ] && [ -d "$OUT/H/All" ] && [ -d "$OUT/He/StarFormationRate" ] \
    || { echo "check SED, WEIGHTS and OUT paths"; exit 1; }

mkdir -p "$J"
FREE_GB=$(df --output=avail -BG "$J" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || { echo "only ${FREE_GB} GB free on $J"; exit 1; }

i=$SLURM_ARRAY_TASK_ID
SNAP=${SNAPS[$((i / 4))]}
CODE=${CODES[$((i % 4))]}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "task $i: snap $SNAP, code $CODE, host $(hostname), $(date), commit $(git rev-parse --short HEAD)"

case $CODE in
    H)      mpirun -np $SLURM_NTASKS bin/Jalpha_H  $SNAP "$OUT/H" "$OUT/H" "$J/Jalpha_H_${SNAP}.hdf5" ;;
    He)     mpirun -np $SLURM_NTASKS bin/Jalpha_He $SNAP "$J/Jalpha_He_${SNAP}.hdf5" ;;
    AGN_H)  mpirun -np $SLURM_NTASKS bin/Jalpha_AGN $SNAP "$R/LuminosityAGN" "$J/Jalpha_AGN_H_${SNAP}.hdf5" \
                "$SED" "$RB/LuminosityAGN" LuminosityAGN H ;;
    AGN_He) mpirun -np $SLURM_NTASKS bin/Jalpha_AGN $SNAP "$R/LuminosityAGN" "$J/Jalpha_AGN_He_${SNAP}.hdf5" \
                "$SED" "$RB/LuminosityAGN" LuminosityAGN He ;;
esac
echo "done $(date)"
