#!/bin/bash
#SBATCH --job-name=metalH640
#SBATCH --output=logs/metalH640_%A_%a.out
#SBATCH --error=logs/metalH640_%A_%a.err
#SBATCH --array=0-30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Appendix sec:metallicity, step 2 (metallicity_global.py map640): 21 cm global <dT_b> on the 640^3 render pixels
# for the 13 BPASS metallicities, J = R_j(z) J_star + J_AGN (production J files), spin model 1, real space,
# mass-weighted fields, as run_21cm.sh; the fiducial mean is checked against the production map (H_real_wf).
# One render per array task (~12-13 h on ORCD, job 30279062: 2.6e8 pixels x 14 variants). Low z only (z <= 7.45; z >= 8 from the gas cells,
# run_metal_cells.sh). The default list: about every 0.15 in z from 7.45 to 3.1, plus 620 (just after the
# self-shielding step at z = 3.35).
#     mkdir -p logs && sbatch -p ou_mki run_metal_H640.sh        (after run_metal_R.sh)
# Writes $OUT/H640_NNN.csv (one row each).
D=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$D/metallicity}
SNAPS=(${SNAPS:-227 237 242 255 264 273 283 295 309 318 328 340 353 364 376 389 403 414 427 443 456 471 487 501 518 535 569 610 620 640 680})
STRIDE=${STRIDE:-1}
JDIR=$D/Jalpha
MAPS=$D/Tb_maps_prod/H_real_wf
BASE=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f metallicity_global.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
[ "${#SNAPS[@]}" = "${SLURM_ARRAY_TASK_COUNT:-${#SNAPS[@]}}" ] || \
    { echo "ERROR: ${#SNAPS[@]} snapshots, array of $SLURM_ARRAY_TASK_COUNT tasks (sbatch --array=0-$(( ${#SNAPS[@]} - 1 )))"; exit 1; }
[ -f "$OUT/R_Z.csv" ] || { echo "no $OUT/R_Z.csv: run run_metal_R.sh first"; exit 1; }
SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}
if [ "$SNAP" -le 428 ]; then REN=$BASE/Flagship_new/postprocessing/ren_640; else REN=$BASE/Flagship_below_z_4p75_new/postprocessing/ren_640; fi
echo "map640: render $SNAP, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u metallicity_global.py map640 --snap $SNAP --dir $REN --rcsv "$OUT/R_Z.csv" --jdir $JDIR \
    --rin $MAPS/RIN_$SNAP.hdf5 --stride $STRIDE --planes 16 --out "$OUT"
echo "done $(date)"
