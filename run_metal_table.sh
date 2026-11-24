#!/bin/bash
#SBATCH --job-name=metalTab
#SBATCH --output=logs/metalTab_%j.out
#SBATCH --error=logs/metalTab_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Appendix sec:metallicity, step 4: merge the cube task files of every snapshot (metallicity_global.py merge; whole-box
# gas-cell fiducial from the scan files + the cubes' change for each Z, jackknife errors) and collect them with the
# 640^3 rows into $OUT/metallicity_global.{csv,png}. Minutes.
#     sbatch -p ou_mki run_metal_table.sh          (after run_metal_cells.sh for H and He, and run_metal_H640.sh)
D=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$D/metallicity}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f metallicity_global.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
echo "table: $(date), commit $(git rev-parse --short HEAD)"
for LINE in H He; do
    for f in $(ls "$OUT/cells_$LINE"/cubes_*_000.npz 2>/dev/null); do
        SNAP=$(basename $f | cut -d_ -f2 | sed 's/^0*//')
        if [ "$LINE" = H ]; then SCAN=$D/cell_test_H_$SNAP; else SCAN=$D/cell_test_$SNAP; fi
        python -u metallicity_global.py merge --line $LINE --snap $SNAP --scan-dir $SCAN --out "$OUT"
    done
done
python -u metallicity_global.py table --out "$OUT" --png
echo "done $(date)"
