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
#      only the AGN tasks:  sbatch --array=2,3,6,7,10,11,14,15 run_Jalpha_test.sh
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
CONDA_ENV=jupyter_env      # has hdf5plugin: its filter plugins let the C codes read the compressed renders
TRANS_DIR=                 # He II-band IGM transmission (run_transmission.sh output); empty = no IGM absorption.
                           # When set: compile builds bin/Jalpha_He_tr and bin/Jalpha_AGN_tr, and the He tasks
                           # write *_tr.hdf5 (rerun only the He tasks: sbatch --array=1,3,5,7,9,11,13,15)
TR_TAG=tr                  # suffix of those binaries and outputs; use a distinct one per transmission variant,
                           # e.g. TR_TAG=tr_sfmask with TRANS_DIR=.../transmission_He_sfmask
# FFTW (single precision, with OpenMP) and HDF5 install prefixes: the directories that contain
# include/fftw3.h and include/hdf5.h. Left empty, they are found from the loaded modules
# (FFTW_ROOT/HDF5_ROOT-style variables, pkg-config, CMAKE_PREFIX_PATH, h5cc on PATH).
FFTW_PREFIX=
HDF5_PREFIX=

source ~/.bashrc
# module load ...        # <- the same modules as your usual J_alpha jobs (MPI, FFTW, HDF5)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

find_prefix() {   # $1 header, $2 name pattern, rest: candidate prefixes; prints the first that has include/$1
    local h=$1 pat=$2 c; shift 2
    for c in "$@"; do                                   # explicit candidates first
        [ -n "$c" ] && [ -f "$c/include/$h" ] && { echo "$c"; return; }
    done
    for c in $(echo "${CMAKE_PREFIX_PATH:-}" | tr ':' ' '); do   # then module prefixes named like the library
        case "$c" in *$pat*) [ -f "$c/include/$h" ] && { echo "$c"; return; } ;; esac
    done
}
FFTW_PREFIX=${FFTW_PREFIX:-$(find_prefix fftw3.h fftw "${FFTW_ROOT:-}" "${FFTW_HOME:-}" "${FFTW_DIR:-}" \
                             "$(pkg-config --variable=prefix fftw3f 2>/dev/null)")}
HDF5_PREFIX=${HDF5_PREFIX:-$(find_prefix hdf5.h hdf5 "${HDF5_ROOT:-}" "${HDF5_HOME:-}" "${HDF5_DIR:-}" \
                             "$(command -v h5cc >/dev/null && dirname "$(dirname "$(command -v h5cc)")")")}
INC=""; LIBS="-lfftw3f_omp -lfftw3f -lhdf5 -lm"
for d in "$FFTW_PREFIX" "$HDF5_PREFIX"; do
    [ -n "$d" ] || continue
    INC="$INC -I$d/include"
    for l in "$d/lib" "$d/lib64"; do [ -d "$l" ] && LIBS="-L$l -Wl,-rpath,$l $LIBS"; done
done
if [ "$1" = "compile" ]; then
    echo "FFTW prefix: ${FFTW_PREFIX:-not found}   HDF5 prefix: ${HDF5_PREFIX:-not found}"
    [ -n "$FFTW_PREFIX" ] && [ -n "$HDF5_PREFIX" ] || echo "WARNING: set FFTW_PREFIX / HDF5_PREFIX at the top if the compile fails"
    mkdir -p bin
    # no -march=native: the login node CPU may differ from the compute nodes
    mpicc -O3 -fopenmp $INC compute_Jalpha.c -o bin/Jalpha_H \
        -DZ_WEIGHTS_FILE="\"$WEIGHTS\"" $LIBS
    mpicc -O3 -fopenmp $INC compute_Jalpha_He.c -o bin/Jalpha_He \
        -DZ_WEIGHTS_FILE="\"$WEIGHTS\"" -DSFR_DIR_OLD="\"$OUT/He\"" -DSFR_DIR_NEW="\"$OUT/He\"" $LIBS
    mpicc -O3 -fopenmp $INC compute_Jalpha_AGN.c -o bin/Jalpha_AGN $LIBS
    if [ -n "$TRANS_DIR" ]; then
        mpicc -O3 -fopenmp $INC compute_Jalpha_He.c -o bin/Jalpha_He_$TR_TAG -DTRANSMISSION_DIR="\"$TRANS_DIR\"" \
            -DZ_WEIGHTS_FILE="\"$WEIGHTS\"" -DSFR_DIR_OLD="\"$OUT/He\"" -DSFR_DIR_NEW="\"$OUT/He\"" $LIBS
        mpicc -O3 -fopenmp $INC compute_Jalpha_AGN.c -o bin/Jalpha_AGN_$TR_TAG -DTRANSMISSION_DIR="\"$TRANS_DIR\"" $LIBS
    fi
    echo "built: $(ls bin)   commit $(git rev-parse --short HEAD)"
    exit 0
fi

[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch (or run: bash $0 compile)"; exit 1; }
# The LuminosityAGN renders are compressed with a filter the cluster HDF5 does not have built in:
# point HDF5 at the plugins shipped with the hdf5plugin Python package.
if [ -z "${HDF5_PLUGIN_PATH:-}" ]; then
    conda activate "$CONDA_ENV" 2>/dev/null || true
    HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)' 2>/dev/null) || HDF5_PLUGIN_PATH=
fi
[ -n "$HDF5_PLUGIN_PATH" ] || { echo "HDF5_PLUGIN_PATH not set and hdf5plugin not found in $CONDA_ENV"; exit 1; }
export HDF5_PLUGIN_PATH
echo "HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH"
BINS="bin/Jalpha_H bin/Jalpha_He bin/Jalpha_AGN"; [ -n "$TRANS_DIR" ] && BINS="$BINS bin/Jalpha_He_$TR_TAG bin/Jalpha_AGN_$TR_TAG"
for b in $BINS; do
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

TR=""; [ -n "$TRANS_DIR" ] && TR="_$TR_TAG"      # He tasks only; the H band has no continuum absorption
case $CODE in
    H)      mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH bin/Jalpha_H  $SNAP "$OUT/H" "$OUT/H" "$J/Jalpha_H_${SNAP}.hdf5" ;;
    He)     mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH bin/Jalpha_He$TR $SNAP "$J/Jalpha_He_${SNAP}${TR}.hdf5" ;;
    AGN_H)  mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH bin/Jalpha_AGN $SNAP "$R/LuminosityAGN" "$J/Jalpha_AGN_H_${SNAP}.hdf5" \
                "$SED" "$RB/LuminosityAGN" LuminosityAGN H ;;
    AGN_He) mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH bin/Jalpha_AGN$TR $SNAP "$R/LuminosityAGN" "$J/Jalpha_AGN_He_${SNAP}${TR}.hdf5" \
                "$SED" "$RB/LuminosityAGN" LuminosityAGN He ;;
esac
echo "done $(date)"
