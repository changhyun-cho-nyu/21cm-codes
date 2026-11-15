#!/bin/bash
#SBATCH --job-name=Jxray
#SBATCH --output=logs/Jxray_%A_%a.out
#SBATCH --error=logs/Jxray_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# J_alpha (H Lyman band) of the SonjaXrayPaper runs at all 405 renders: stars (compute_Jalpha.c, SFR_eff, the run's
# own Z_weights.txt) and AGN (compute_Jalpha_AGN.c, species H). Same codes as Lumina, with N_SNAPS = 405.
#   0. cd xray
#   1. bash run_Jalpha_xray.sh compile          (login node; builds bin/Jalpha_H_xray_<run>, bin/Jalpha_AGN_xray)
#   2. mkdir -p logs && sbatch -p ou_mki run_Jalpha_xray.sh
#      array tasks 0-7 = withoutXrays, 8-15 = withXrays; task i of a run does renders i, i+8, ...;
#      resumable (existing outputs are skipped; each is written as .part and renamed when the code succeeds)
#   3. bash run_Jalpha_xray.sh status
# Output: $BASE/<run>/Jalpha/Jalpha_H_NNN.hdf5, Jalpha_AGN_H_NNN.hdf5 (~8 MB each)
# The AGN code reads <dir>/LuminosityAGN_NNN.hdf5: links to the assembled All_NNN.hdf5 (no copy) are made in
# $BASE/<run>/ren_128/LuminosityAGN/.

BASE=/orcd/data/mvogelsb/004/chcho/xray
RUNS=(withoutXrays withXrays)
NREN=405
NTASK=8
SED=../agn_sed_full.hdf5         # in the repository root (as for Lumina)
FFTW_PREFIX=
HDF5_PREFIX=

source ~/.bashrc
# module load ...        # <- the same modules as your usual J_alpha jobs (MPI, FFTW, HDF5)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f ../compute_Jalpha.c ] && [ -f assemble_renders.py ] || { echo "run from the xray/ directory of the repository"; exit 1; }

find_prefix() {   # as in run_Jalpha_test.sh
    local h=$1 pat=$2 c; shift 2
    for c in "$@"; do [ -n "$c" ] && [ -f "$c/include/$h" ] && { echo "$c"; return; }; done
    for c in $(echo "${CMAKE_PREFIX_PATH:-}" | tr ':' ' '); do
        case "$c" in *$pat*) [ -f "$c/include/$h" ] && { echo "$c"; return; } ;; esac
    done
}

if [ "$1" = "compile" ]; then
    FFTW_PREFIX=${FFTW_PREFIX:-$(find_prefix fftw3.h fftw "${FFTW_ROOT:-}" "${FFTW_HOME:-}" "${FFTW_DIR:-}" \
                                 "$(pkg-config --variable=prefix fftw3f 2>/dev/null)")}
    HDF5_PREFIX=${HDF5_PREFIX:-$(find_prefix hdf5.h hdf5 "${HDF5_ROOT:-}" "${HDF5_HOME:-}" "${HDF5_DIR:-}" \
                                 "$(command -v h5cc >/dev/null && dirname "$(dirname "$(command -v h5cc)")")")}
    echo "FFTW prefix: ${FFTW_PREFIX:-not found}   HDF5 prefix: ${HDF5_PREFIX:-not found}"
    INC=""; LIBS="-lfftw3f_omp -lfftw3f -lhdf5 -lm"
    for d in "$FFTW_PREFIX" "$HDF5_PREFIX"; do
        [ -n "$d" ] || continue
        INC="$INC -I$d/include"
        for l in "$d/lib" "$d/lib64"; do [ -d "$l" ] && LIBS="-L$l -Wl,-rpath,$l $LIBS"; done
    done
    mkdir -p bin
    for RUN in "${RUNS[@]}"; do
        W=$BASE/$RUN/Z/Z_weights.txt
        [ -f "$W" ] || { echo "missing $W (run_Z_xray.sh first)"; exit 1; }
        mpicc -O3 -fopenmp $INC ../compute_Jalpha.c -o bin/Jalpha_H_xray_$RUN.new \
            -DN_SNAPS=$NREN -DZ_WEIGHTS_FILE="\"$W\"" $LIBS && mv bin/Jalpha_H_xray_$RUN.new bin/Jalpha_H_xray_$RUN
    done
    mpicc -O3 -fopenmp $INC ../compute_Jalpha_AGN.c -o bin/Jalpha_AGN_xray.new -DN_SNAPS=$NREN $LIBS \
        && mv bin/Jalpha_AGN_xray.new bin/Jalpha_AGN_xray
    echo "built: $(ls bin | grep xray | tr '\n' ' ')  commit $(git rev-parse --short HEAD)"
    exit 0
fi

if [ "$1" = "status" ]; then
    for RUN in "${RUNS[@]}"; do
        J=$BASE/$RUN/Jalpha
        echo "$RUN: $(ls $J/Jalpha_H_*.hdf5 2>/dev/null | wc -l) stars, $(ls $J/Jalpha_AGN_H_*.hdf5 2>/dev/null | wc -l) AGN of $NREN"
    done
    exit 0
fi

[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch (see the header)"; exit 1; }
RUN=${RUNS[$((SLURM_ARRAY_TASK_ID / NTASK))]}
TASK=$((SLURM_ARRAY_TASK_ID % NTASK))
REN=$BASE/$RUN/ren_128
SFR=$BASE/$RUN/SFR_eff/H
J=$BASE/$RUN/Jalpha
BH=bin/Jalpha_H_xray_$RUN; BA=bin/Jalpha_AGN_xray
case "$(realpath -m $J)/" in /orcd/data/mvogelsb/004/chcho/*) ;; *) echo "refusing to write to $J"; exit 1;; esac
for b in $BH $BA; do [ -x $b ] || { echo "missing $b: run 'bash $0 compile' first"; exit 1; }; done
[ $BH -nt ../compute_Jalpha.c ] && [ $BA -nt ../compute_Jalpha_AGN.c ] || { echo "binaries older than the sources: recompile"; exit 1; }
[ "$(ls $SFR/All/All_*.hdf5 2>/dev/null | wc -l)" -eq "$NREN" ] || { echo "SFR_eff incomplete in $SFR (run_SFR_eff_xray.sh)"; exit 1; }
[ -f "$SED" ] || { echo "missing $SED"; exit 1; }

# links for the AGN code (idempotent; several tasks may run this at once)
mkdir -p $REN/LuminosityAGN $J
for f in $REN/All/All_*.hdf5; do
    n=$(basename $f .hdf5); n=${n#All_}
    [ -e $REN/LuminosityAGN/LuminosityAGN_$n.hdf5 ] || ln -sf ../All/All_$n.hdf5 $REN/LuminosityAGN/LuminosityAGN_$n.hdf5 2>/dev/null || true
done

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "run $RUN task $TASK of $NTASK, host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
fail=0
for ((s = TASK; s < NREN; s += NTASK)); do
    for code in H AGN_H; do
        F=$J/Jalpha_${code}_$s.hdf5
        [ -f "$F" ] && continue
        P=$F.part; rm -f "$P"
        rc=0
        if [ $code = H ]; then
            mpirun -np $SLURM_NTASKS $BH $s $SFR $SFR "$P" > $J/log_${code}_$s.txt 2>&1 || rc=$?
        else
            mpirun -np $SLURM_NTASKS $BA $s $REN/LuminosityAGN "$P" $SED none LuminosityAGN H > $J/log_${code}_$s.txt 2>&1 || rc=$?
        fi
        if [ $rc -eq 0 ] && [ -s "$P" ]; then mv "$P" "$F"; echo "render $s $code done"
        else rm -f "$P"; echo "render $s $code FAILED (exit $rc, see $J/log_${code}_$s.txt)"; fail=$((fail + 1)); fi
    done
done
echo "done $(date), $fail failed"
