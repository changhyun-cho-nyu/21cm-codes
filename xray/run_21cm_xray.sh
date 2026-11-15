#!/bin/bash
#SBATCH --job-name=T21xray
#SBATCH --output=logs/T21xray_%A_%a.out
#SBATCH --error=logs/T21xray_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# 21 cm delta T_b maps (RIN_N.hdf5) and power spectra (PS_N.out) of the SonjaXrayPaper runs at all 405 renders,
# with ../PS_21cm_mpi.c and THE SAME compile flags as the fiducial Lumina maps: EXTRA_CFLAGS is read from
# ../run_21cm.sh at compile time (2026-11-15: -DH_REDSHIFT_SPACE=0 -DSPIN_MODEL=1) and recorded in bin/.
# Array tasks 0-3 = withoutXrays, 4-7 = withXrays (task i of a run does renders i, i+4, ...); resumable.
#   0. cd xray
#   1. bash run_21cm_xray.sh compile
#   2. mkdir -p logs && sbatch -p ou_mki run_21cm_xray.sh          (after run_Jalpha_xray.sh has finished)
#   3. global signal of both runs (plot_global_Tb.py reads the per-render mean of each map from the logs):
#      python ../plot_global_Tb.py --log "logs/T21xray_<jobid>_[0-3].out" --label "no X-rays" \
#             --log "logs/T21xray_<jobid>_[4-7].out" --label "with X-rays" --out data/global_Tb_xray.png
# Output: $BASE/<run>/Tb_21cm/RIN_N.hdf5, PS_N.out (not zero-padded, as for Lumina), ~17 MB per render.

BASE=/orcd/data/mvogelsb/004/chcho/xray
RUNS=(withoutXrays withXrays)
NREN=405
NTASK=4

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f ../PS_21cm_mpi.c ] && [ -f assemble_renders.py ] || { echo "run from the xray/ directory of the repository"; exit 1; }
# environment and compile flags of the fiducial script (one source of truth)
eval "$(grep -E '^(FFTW_PREFIX|HDF5_PREFIX|EXTRA_CFLAGS)=' ../run_21cm.sh)"
[ -n "$FFTW_PREFIX" ] && [ -n "$HDF5_PREFIX" ] && [ -n "$EXTRA_CFLAGS" ] || { echo "could not read the settings of ../run_21cm.sh"; exit 1; }
export LD_LIBRARY_PATH=$FFTW_PREFIX/lib:$HDF5_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
SRC="../PS_21cm_mpi.c ../functions.c ../collisions.c"
BIN=bin/PS_21cm_mpi

if [ "$1" = "compile" ]; then
    mkdir -p bin
    INC=""; LIBS="-lfftw3_mpi -lfftw3 -lhdf5 -lm"
    for d in "$FFTW_PREFIX" "$HDF5_PREFIX"; do
        INC="$INC -I$d/include"
        for l in "$d/lib" "$d/lib64"; do [ -d "$l" ] && LIBS="-L$l -Wl,-rpath,$l $LIBS"; done
    done
    mpicc -O3 -fopenmp -DH5_USE_16_API $EXTRA_CFLAGS $INC $SRC -o $BIN.new $LIBS && mv $BIN.new $BIN \
        || { echo "COMPILE FAILED: $BIN was not rebuilt; do not submit"; exit 1; }
    echo "$EXTRA_CFLAGS" > $BIN.flags
    echo "built $BIN (EXTRA_CFLAGS: $EXTRA_CFLAGS, as ../run_21cm.sh)   commit $(git rev-parse --short HEAD)"
    exit 0
fi

[ -x $BIN ] || { echo "run 'bash $0 compile' first"; exit 1; }
[ "$(cat $BIN.flags 2>/dev/null)" = "$EXTRA_CFLAGS" ] || { echo "$BIN was built with other flags than ../run_21cm.sh now has: recompile"; exit 1; }
for f in $SRC ../spin_wf.h; do
    [ $BIN -nt $f ] || { echo "$BIN is older than $f: run 'bash $0 compile'"; exit 1; }
done
[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch (see the header)"; exit 1; }
RUN=${RUNS[$((SLURM_ARRAY_TASK_ID / NTASK))]}
TASK=$((SLURM_ARRAY_TASK_ID % NTASK))
REN=$BASE/$RUN/ren_128
J=$BASE/$RUN/Jalpha
OUT=$BASE/$RUN/Tb_21cm
case "$(realpath -m $OUT)/" in /orcd/data/mvogelsb/004/chcho/*) ;; *) echo "refusing to write to $OUT"; exit 1;; esac
mkdir -p $OUT

nfail=0
echo "run $RUN task $TASK of $NTASK, commit $(git rev-parse --short HEAD), host $(hostname), $(date), flags $EXTRA_CFLAGS"
for ((s = TASK; s < NREN; s += NTASK)); do
    echo "=================================================="
    echo "Snapshot $s"
    JH=$J/Jalpha_H_$s.hdf5; JA=$J/Jalpha_AGN_H_$s.hdf5
    for f in $JH $JA; do [ -f $f ] || { echo "WARNING: $f not found, skipping render $s"; nfail=$((nfail + 1)); continue 2; }; done
    if [ -f $OUT/PS_$s.out ] && [ -f $OUT/RIN_$s.hdf5 ]; then echo "done already, skipped"; continue; fi
    rm -f $OUT/PS_$s.out.part $OUT/RIN_$s.hdf5.part
    if mpirun -np 1 $BIN $s $REN $OUT/PS_$s.out.part $OUT/RIN_$s.hdf5.part 3 $JH $JA; then
        mv $OUT/PS_$s.out.part $OUT/PS_$s.out; mv $OUT/RIN_$s.hdf5.part $OUT/RIN_$s.hdf5
        echo "finished render $s"
    else
        echo "FAILED render $s"; nfail=$((nfail + 1))
    fi
done
echo "done, $nfail failed or skipped, $(date)"
[ $nfail -eq 0 ]
