#!/bin/bash
#SBATCH --job-name=mfpVol
#SBATCH --output=logs/mfpVol_%A_%a.out
#SBATCH --error=logs/mfpVol_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# MFP bubble sizes from the VOLUME-weighted ionized fractions (user decision 2026-11-19), with 64-bit histograms:
# the snapshots of the bubble-size figure (BSD.ipynb) that have no volume-weighted output yet. Outputs go next to the
# existing volume-weighted ones (made with the same field; their 32-bit counters never overflowed):
#   $OUT_ROOT/LUMINA_hydrogen_Vol/mfp_NNN.hdf5,  $OUT_ROOT/LUMINA_helium_Vol/mfp_NNN.hdf5
# Existing outputs are skipped. Build first (once): bash mfp/build_mfp.sh. Then, from the 21cm-codes directory:
#   mkdir -p logs
#   LINE=H  sbatch --array=0-7%8   mfp/run_mfp.sh
#   LINE=He sbatch --array=0-46%12 mfp/run_mfp.sh
# Run time grows with the bubble size: minutes at z >~ 8, about a day for the latest He snapshots (660, 695); the
# 2026-05 runs used 32 threads.

LINE=${LINE:?set LINE=H or LINE=He}
OUT_ROOT=${OUT_ROOT:-/orcd/data/mvogelsb/004/chcho/mfp-bubbles-thesan/output}
R=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
if [ "$LINE" = H ]; then
    SNAPS=(244 266 285 302 320 340 352 360)
    OUT=$OUT_ROOT/LUMINA_hydrogen_Vol
else
    SNAPS=(112 132 154 165 179 190 206 212 223 233 244 255 266 277 285 300 318 338 343 349 352 360 364 366 368 370
           374 378 380 387 394 403 406 445 452 469 473 489 502 513 516 525 558 598 624 660 695)
    OUT=$OUT_ROOT/LUMINA_helium_Vol
fi

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ "${#SNAPS[@]}" -ge "${SLURM_ARRAY_TASK_COUNT:-0}" ] || { echo "ERROR: ${#SNAPS[@]} snapshots, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
EXE=mfp/lumina-mfp-$LINE
[ -x "$EXE" ] || { echo "ERROR: no $EXE: run bash mfp/build_mfp.sh first"; exit 1; }
[ "$EXE" -nt mfp/lumina-mfp-hist-grid.cc ] || { echo "ERROR: $EXE is older than the source: rebuild (bash mfp/build_mfp.sh)"; exit 1; }
module load gcc openmpi/4.1.4 hdf5 >/dev/null 2>&1 || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export HDF5_PLUGIN_PATH=$(python -c "import hdf5plugin; print(hdf5plugin.PLUGINS_PATH)")${HDF5_PLUGIN_PATH:+:$HDF5_PLUGIN_PATH}

S=${SNAPS[$SLURM_ARRAY_TASK_ID]}
if [ "$S" -le 428 ]; then REN=$R/Flagship_new/postprocessing/ren_640; else REN=$R/Flagship_below_z_4p75_new/postprocessing/ren_640; fi
mkdir -p "$OUT"
F=$OUT/mfp_$(printf %03d $S).hdf5
echo "MFP $LINE render $S ($REN) -> $F, $(date), commit $(git rev-parse --short HEAD), $OMP_NUM_THREADS threads"
[ -f "$F" ] && { echo "$F exists: skipped"; exit 0; }
./$EXE "$REN" "$OUT" $S
echo "done $(date)"
