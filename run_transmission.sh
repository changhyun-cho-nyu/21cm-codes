#!/bin/bash
#SBATCH --job-name=transHe
#SBATCH --output=logs/trans_%A_%a.out
#SBATCH --error=logs/trans_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# He II Lyman-band IGM transmission for the J_alpha^He snapshots (needs run_opacity.sh first).
#   1. bash run_transmission.sh compile
#   2. mkdir -p logs && sbatch run_transmission.sh
# Each task also prints the simulation's own 912 A mean free path at its snapshot (mfp912 mode),
# to compare with observations (Worseck+14, Becker+21) before deciding on the LLS term.

SNAPS=(156 300 406 500)
OPA=/orcd/data/mvogelsb/004/chcho/opacity_He       # run_opacity.sh output
TRANS=/orcd/data/mvogelsb/004/chcho/transmission_He
HDF5_PREFIX=                                       # set if hdf5.h is not found (see run_Jalpha_test.sh)
# Modules: load the same MPI + HDF5 modules as for the J_alpha jobs before compiling AND before sbatch
# (sbatch passes the loaded environment to the job; the binary needs the MPI/HDF5 libraries at run time).
LLS_A=0                                            # extra lambda_912 at (1+z)/5 = 1 [proper Mpc]; 0 = off
SF_MASK=0                                          # 1: read ${OPA}_sfmask, write ${TRANS}_sfmask
                                                   # (run_opacity.sh with SF_MASK=1 first)

source ~/.bashrc
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ "$1" = "compile" ]; then
    [ -z "$HDF5_PREFIX" ] && command -v h5cc >/dev/null && HDF5_PREFIX=$(dirname "$(dirname "$(command -v h5cc)")")
    [ -z "$HDF5_PREFIX" ] && [ -n "${HDF5_ROOT:-}" ] && HDF5_PREFIX=$HDF5_ROOT
    INC=""; LIB=""
    if [ -n "$HDF5_PREFIX" ]; then
        INC="-I$HDF5_PREFIX/include"
        for l in "$HDF5_PREFIX/lib" "$HDF5_PREFIX/lib64"; do [ -d "$l" ] && LIB="$LIB -L$l -Wl,-rpath,$l"; done
    fi
    # A parallel HDF5 build (H5_HAVE_PARALLEL) includes mpi.h from hdf5.h and links MPI: compile with mpicc
    # then (the program itself makes no MPI calls and runs without mpirun).
    CC=gcc
    PUBCONF=""; [ -n "$HDF5_PREFIX" ] && PUBCONF="$HDF5_PREFIX/include/H5pubconf.h"
    if [ -n "$PUBCONF" ] && grep -qE '^#define H5_HAVE_PARALLEL 1' "$PUBCONF" 2>/dev/null || ! echo '#include <hdf5.h>' | gcc $INC -fsyntax-only -x c - 2>/dev/null; then
        command -v mpicc >/dev/null || { echo "this HDF5 is an MPI build: load the MPI module first (the one used for the J_alpha jobs)"; exit 1; }
        CC=mpicc
    fi
    mkdir -p bin
    $CC -O3 -fopenmp $INC -DLLS_A=$LLS_A compute_transmission_He.c -o bin/transmission_He $LIB -lhdf5 -lm
    echo "built bin/transmission_He with $CC (HDF5 prefix: ${HDF5_PREFIX:-default}, LLS_A = $LLS_A)"
    exit 0
fi

[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch (or run: bash $0 compile)"; exit 1; }
[ -x bin/transmission_He ] || { echo "run 'bash $0 compile' first"; exit 1; }
if [ -z "${HDF5_PLUGIN_PATH:-}" ]; then     # the opacity files use gzip only; kept for safety
    conda activate jupyter_env 2>/dev/null || true
    HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)' 2>/dev/null) || HDF5_PLUGIN_PATH=
    [ -n "$HDF5_PLUGIN_PATH" ] && export HDF5_PLUGIN_PATH
fi
SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}
[ "$SF_MASK" = 1 ] && { OPA=${OPA}_sfmask; TRANS=${TRANS}_sfmask; }
[ "$LLS_A" != 0 ] && TRANS=${TRANS}_lls                 # keep the LLS variant apart (attribute LLS_A records it)
mkdir -p "$TRANS"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "snap $SNAP  host $(hostname)  $(date)  commit $(git rev-parse --short HEAD)"
# the target snapshot itself must have an opacity file (EVERY grid): use the nearest one if not
if [ ! -f "$OPA/opacity_$(printf %03d $SNAP).hdf5" ]; then
    echo "ERROR: $OPA/opacity_$(printf %03d $SNAP).hdf5 missing (run_opacity.sh with EVERY that includes it)"; exit 1
fi
bin/transmission_He $SNAP "$OPA" mfp912
bin/transmission_He $SNAP "$OPA" "$TRANS/transmission_He_$(printf %03d $SNAP).hdf5"
echo "done $(date)"
