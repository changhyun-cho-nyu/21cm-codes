#!/bin/bash
#SBATCH --job-name=cellDec
#SBATCH --output=logs/cellDec_%j.out
#SBATCH --error=logs/cellDec_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Which gas makes the cell-level delta T_b (decompose_cells_He.py): reads the cube cells saved by the scan
# tasks of run_cells_Tb.sh (keep the scan_*.hdf5 files until this has run). Same settings as run_cells_Tb.sh.
#   mkdir -p logs && sbatch run_decompose_cells.sh
# Settings from the command line, e.g.  RUN=above SNAP=92 OUT_DIR=/orcd/data/mvogelsb/004/chcho/cell_test_92 IGM=1 sbatch run_decompose_cells.sh

LINE=${LINE:-He}
RUN=${RUN:-below}
SNAP=${SNAP:-136}
OUT_DIR=${OUT_DIR:-/orcd/data/mvogelsb/004/chcho/cell_test}
IGM=${IGM:-0}              # 1: the IGM-component test instead (decompose_cells_He.py --igm; needs cellmap_<line>_<snap>.hdf5)
if [ "$RUN" = above ]; then SD=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots
else SD=/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots; fi

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "decompose: $LINE, $RUN snapshot $SNAP, $(date), commit $(git rev-parse --short HEAD)"
EXTRA=""; [ "$IGM" = 1 ] && EXTRA="--igm"
python -u decompose_cells_He.py --snapdir $SD --snap $SNAP --line $LINE --out $OUT_DIR $EXTRA
echo "done $(date)"
