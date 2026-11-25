#!/bin/bash
#SBATCH --job-name=metalR
#SBATCH --output=logs/metalR_%j.out
#SBATCH --error=logs/metalR_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Appendix sec:metallicity, step 1 (metallicity_global.py R): R_j(z) = <J_star>(all stars at BPASS Z_j) /
# <J_star>(production young-star Z mix), H and He bands, at every render with a stellar J file in JDIRS.
# Box means only (production SFR_eff_summary.csv + BPASS + weights): no boxes are read; minutes.
#     mkdir -p logs && sbatch -p ou_mki run_metal_R.sh
# Then run_metal_H640.sh and run_metal_box.sh (both read $OUT/R_Z.csv).
# A render that only a test J directory has (box stops with "no R for ..."): rerun with
#     JDIRS="$D/Jalpha $D/Jalpha_fullsnap $D/Jalpha_test" sbatch -p ou_mki run_metal_R.sh
D=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$D/metallicity}
JDIRS=${JDIRS:-"$D/Jalpha $D/Jalpha_fullsnap"}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f metallicity_global.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
mkdir -p "$OUT"
echo "R: $(date), commit $(git rev-parse --short HEAD), host $(hostname); JDIRS = $JDIRS"
python -u metallicity_global.py R --sfr-summary $D/SFR_eff/SFR_eff_summary.csv \
    --weights $D/cell_by_cell/hydrogen/Z_weights.txt --weights-he $D/cell_by_cell/helium/Z_weights.txt \
    --jdir $JDIRS --unit-from $D/SFR_eff/H --out "$OUT/R_Z.csv"
echo "done $(date)"
