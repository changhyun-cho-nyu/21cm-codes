#!/bin/bash
#SBATCH --job-name=SFR_eff
#SBATCH --output=logs/SFR_eff_%j.out
#SBATCH --error=logs/SFR_eff_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Effective SFR boxes (make_SFR_eff.py) for compute_Jalpha.c and compute_Jalpha_He.c.
# Serial job: one Python process. Submit from the repository directory:
#     mkdir -p logs && sbatch run_SFR_eff.sh
# Memory: sparse buffers of the past snapshots inside the 300 Myr window, a slab of
# PLANES planes each; lower PLANES (multiple of 16) if the job runs out of memory.

OUT=/orcd/data/mvogelsb/004/chcho/SFR_eff
WEIGHTS=/orcd/data/mvogelsb/004/chcho/cell_by_cell/hydrogen/Z_weights.txt
PLANES=32
MIN_FREE_GB=600      # refuse to start with less free space than this on $OUT (expected ~0.2-0.4 TB)

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
[ -f make_SFR_eff.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
[ -f "$WEIGHTS" ]      || { echo "missing $WEIGHTS (run make_Z_snapshots.py first)"; exit 1; }
python -c "import h5py, hdf5plugin" || { echo "h5py/hdf5plugin not importable"; exit 1; }

mkdir -p "$OUT"
echo "host $(hostname), start $(date), commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
df -h "$OUT" | tail -1
FREE_GB=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "ERROR: only ${FREE_GB} GB free on $OUT (MIN_FREE_GB = $MIN_FREE_GB); not starting"; exit 1
fi

python -u make_SFR_eff.py --out "$OUT" --weights "$WEIGHTS" --planes "$PLANES"

echo "done $(date)"
du -sh "$OUT"/H "$OUT"/He
