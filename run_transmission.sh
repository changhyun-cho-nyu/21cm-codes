#!/bin/bash
#SBATCH --job-name=transHe
#SBATCH --output=logs/trans_%A_%a.out
#SBATCH --error=logs/trans_%A_%a.err
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
#   2. bash run_transmission.sh count    (prints the sbatch --array line for SNAPS)
#   3. mkdir -p logs && sbatch --array=0-<N-1>%8 run_transmission.sh   (existing outputs are skipped)
# Each task also prints the simulation's own mean free paths at its snapshot: at 912 A (mfp912 mode), to
# compare with observations (Worseck+14, Becker+21), and in the He band (mfpE mode), lambda_E, which sets
# the LLS term: 1/lambda_912,extra = 1/lambda_912,obs - r/lambda_E,sim with r = 1.59 (JALPHA_NOTES 4.5).

PER_TASK=8                 # snapshots per array task, run one after another (keeps the array small for the
                           # job-submit limit; ~2 min per snapshot at z ~ 10, ~30 min at z ~ 4)
SNAPS=()                   # empty: the snapshot list of run_3cm.sh (its uncommented SNAPS block);
                           # or a list, e.g. SNAPS=(156 300 406 500) for the tests
OPA=/orcd/data/mvogelsb/004/chcho/opacity_He       # run_opacity.sh output
TRANS=/orcd/data/mvogelsb/004/chcho/transmission_He
HDF5_PREFIX=                                       # set if hdf5.h is not found (see run_Jalpha_test.sh)
# Modules: load the same MPI + HDF5 modules as for the J_alpha jobs before compiling AND before sbatch
# (sbatch passes the loaded environment to the job; the binary needs the MPI/HDF5 libraries at run time).
LLS_A=0                                            # extra lambda_912 at (1+z)/5 = 1 [proper Mpc]; 0 = off
LLS_SLOPE=-5.4                                     # lambda_912,extra = LLS_A [(1+z)/5]^LLS_SLOPE
LLS_ZMIN=0                                         # applied for LLS_ZMIN <= z < LLS_ZMAX
LLS_ZMAX=5.5
SF_MASK=1                                          # 1 (default): read ${OPA}_sfmask, write ${TRANS}_sfmask
                                                   # (run_opacity.sh with SF_MASK=1 first)

source ~/.bashrc
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ "$1" = "compile" ]; then
    for w in h5pcc h5cc; do                       # parallel builds ship h5pcc
        [ -z "$HDF5_PREFIX" ] && command -v $w >/dev/null && HDF5_PREFIX=$(dirname "$(dirname "$(command -v $w)")")
    done
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
        command -v mpicc >/dev/null || { echo "hdf5.h does not compile with gcc -I${HDF5_PREFIX:-?}/include (an MPI build of HDF5, or a wrong HDF5_PREFIX): load the MPI module, or set HDF5_PREFIX"; exit 1; }
        CC=mpicc
    fi
    mkdir -p bin
    $CC -O3 -fopenmp $INC -DLLS_A=$LLS_A "-DLLS_SLOPE=($LLS_SLOPE)" -DLLS_ZMIN=$LLS_ZMIN -DLLS_ZMAX=$LLS_ZMAX \
        compute_transmission_He.c -o bin/transmission_He $LIB -lhdf5 -lm
    echo "$LLS_A $LLS_SLOPE $LLS_ZMIN $LLS_ZMAX" > bin/transmission_He.LLS_A   # the run step checks these match the binary
    echo "built bin/transmission_He with $CC (HDF5 prefix: ${HDF5_PREFIX:-default}, LLS_A = $LLS_A, slope $LLS_SLOPE, $LLS_ZMIN <= z < $LLS_ZMAX)"
    exit 0
fi

[ ${#SNAPS[@]} -gt 0 ] || SNAPS=($(sed -n '/^SNAPS=(/,/)/p' run_3cm.sh | sed 's/#.*//' | tr -d 'SNAPS=()'))
NTASK=$(( (${#SNAPS[@]} + PER_TASK - 1) / PER_TASK ))
if [ "$1" = "count" ]; then
    echo "${#SNAPS[@]} snapshots (${SNAPS[0]} .. ${SNAPS[${#SNAPS[@]}-1]}), $PER_TASK per task = $NTASK tasks; ~1.5 GB each"
    echo "submit:  mkdir -p logs && sbatch --array=0-$(( NTASK - 1 ))%8 run_transmission.sh"
    exit 0
fi
[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch --array (see 'bash $0 count'), or run: bash $0 compile"; exit 1; }
[ "$SLURM_ARRAY_TASK_ID" -lt $NTASK ] || { echo "task $SLURM_ARRAY_TASK_ID >= $NTASK tasks: nothing to do"; exit 0; }
[ -x bin/transmission_He ] || { echo "run 'bash $0 compile' first"; exit 1; }
[ "$(cat bin/transmission_He.LLS_A 2>/dev/null)" = "$LLS_A $LLS_SLOPE $LLS_ZMIN $LLS_ZMAX" ] || \
    { echo "bin/transmission_He was built with LLS (A slope zmin zmax) = $(cat bin/transmission_He.LLS_A 2>/dev/null || echo '?'), the script says $LLS_A $LLS_SLOPE $LLS_ZMIN $LLS_ZMAX: run 'bash $0 compile' again"; exit 1; }
conda activate jupyter_env                        # h5py for make_opacity_fields.py, hdf5plugin
if [ -z "${HDF5_PLUGIN_PATH:-}" ]; then           # the opacity files use gzip only; kept for safety
    HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)' 2>/dev/null) || HDF5_PLUGIN_PATH=
    [ -n "$HDF5_PLUGIN_PATH" ] && export HDF5_PLUGIN_PATH
fi
[ "$SF_MASK" = 1 ] && { OPA=${OPA}_sfmask; TRANS=${TRANS}_sfmask; }
[ "$LLS_A" != 0 ] && TRANS=${TRANS}_lls                 # keep the LLS variant apart (attribute LLS_A records it)
mkdir -p "$TRANS"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
K0=$(( SLURM_ARRAY_TASK_ID * PER_TASK )); K1=$(( K0 + PER_TASK )); [ $K1 -gt ${#SNAPS[@]} ] && K1=${#SNAPS[@]}
echo "task $SLURM_ARRAY_TASK_ID: snapshots ${SNAPS[@]:$K0:$((K1 - K0))}  host $(hostname)  $(date)  commit $(git rev-parse --short HEAD)"
for ((k = K0; k < K1; k++)); do
    SNAP=${SNAPS[$k]}
    S3=$(printf %03d $SNAP); OUTF="$TRANS/transmission_He_$S3.hdf5"
    echo "=== snap $SNAP  $(date)"
    [ -f "$OUTF" ] && { echo "$OUTF exists, skipped"; continue; }
    # the target snapshot itself needs an opacity file; run_opacity.sh makes every EVERY-th one, so an odd
    # snapshot of the T_b list is made here (same settings), in a private directory and then moved in, so
    # that other tasks scanning $OPA never see a half-written file
    if [ ! -f "$OPA/opacity_$S3.hdf5" ]; then
        FAC=$(python -c "import h5py,glob; print(int(h5py.File(sorted(glob.glob('$OPA/opacity_*.hdf5'))[0],'r').attrs['factor']))")
        echo "making $OPA/opacity_$S3.hdf5 (factor $FAC, sf_mask $SF_MASK)"
        TMP="$OPA/.tmp_$S3"; rm -rf "$TMP"; mkdir -p "$TMP"
        python -u make_opacity_fields.py --out "$TMP" --snaps $SNAP $SNAP --every 1 --factor $FAC $([ "$SF_MASK" = 1 ] && echo --sf-mask)
        mv "$TMP/opacity_$S3.hdf5" "$OPA/opacity_$S3.hdf5" && rmdir "$TMP"
    fi
    bin/transmission_He $SNAP "$OPA" mfp912
    bin/transmission_He $SNAP "$OPA" mfpE        # He-band lambda_E (diagnostic; LLS_A = 0 is the choice made)
    bin/transmission_He $SNAP "$OPA" "$OUTF.part" && mv "$OUTF.part" "$OUTF"
done
echo "done $(date)"
