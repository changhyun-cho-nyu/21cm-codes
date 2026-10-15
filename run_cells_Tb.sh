#!/bin/bash
#SBATCH --job-name=cellTb
#SBATCH --output=logs/cellTb_%A_%a.out
#SBATCH --error=logs/cellTb_%A_%a.err
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Cell-level delta T_b test (test_cells_Tb.py run): Voronoi cells of a full snapshot vs the same gas grouped in
# 320^3 ... 10240^3 pixels, and the 640^3 render (production inputs) at the same pixels. Writes only the logs.
# One task per line of TASKS: "<line> <above|below> <full snapshot> <matching render snapshot>"
# (from test_cells_Tb.py inspect). Set --array=0-(number of TASKS - 1), then
#   mkdir -p logs && sbatch run_cells_Tb.sh

TASKS=("He below 0 500")              # placeholder: fill in from the inspect output
BLOCKS=32
BLOCK_SIZE=40000000
SNAP_ABOVE=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots
SNAP_BELOW=/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots
BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
REN_A=$BASE/Flagship_new/postprocessing/ren_640
REN_B=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640
JDIR=/orcd/data/mvogelsb/004/chcho/Jalpha
TR_TAG=tr_sfmask

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
read LINE RUN S RS <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
if [ "$RUN" = above ]; then SD=$SNAP_ABOVE; else SD=$SNAP_BELOW; fi
if [ "$RS" -le 428 ]; then REN=$REN_A; else REN=$REN_B; fi
if [ "$LINE" = H ]; then J1=$JDIR/Jalpha_H_$RS.hdf5; J2=$JDIR/Jalpha_AGN_H_$RS.hdf5
else J1=$JDIR/Jalpha_He_${RS}_$TR_TAG.hdf5; J2=$JDIR/Jalpha_AGN_He_${RS}_$TR_TAG.hdf5; fi
for f in "$J1" "$J2"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
echo "line $LINE, $RUN snapshot $S, render snapshot $RS, $(date), commit $(git rev-parse --short HEAD)"
python -u test_cells_Tb.py run --snapdir $SD --snap $S --line $LINE --J $J1 --J $J2 --render $REN --render-snap $RS \
    --blocks $BLOCKS --block-size $BLOCK_SIZE
echo "done $(date)"
