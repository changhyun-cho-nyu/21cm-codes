#!/bin/bash
#SBATCH --job-name=metalCell
#SBATCH --output=logs/metalCell_%A_%a.out
#SBATCH --error=logs/metalCell_%A_%a.err
#SBATCH --array=0-55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Appendix sec:metallicity, step 3 (metallicity_global.py cubes): gas-cell <dT_b> for the 13 BPASS metallicities
# from the cube cells stored by the gas-cell scans (64 cubes, ~1.7e9 cells per snapshot; scan files read only):
# J of each cell = stored J + (R_j - 1) J_star at its pixel. Before anything, the stored J is reproduced bit for bit
# from the J files (which files, J render and axis order are checked or found that way); the fiducial dT_b is
# recomputed and compared with the stored one. NTASK tasks per snapshot (cubes q with q % NTASK = task),
# array = (number of SNAPS) x NTASK.
#   H  (z >= 8; 7 snapshots x 8 = 56 tasks, ~2-4 h each):
#       mkdir -p logs && LINE=H sbatch -p ou_mki run_metal_cells.sh
#   He (z = 10.0-3.0; 23 snapshots x 8 = 184 tasks, faster: no T_S solve):
#       LINE=He sbatch -p ou_mki --array=0-183 run_metal_cells.sh
#   (full snapshot 136, z = 4.0, has no scan files left: not in the He list)
# Then run_metal_table.sh (merges every snapshot and writes the table).
# Writes $OUT/cells_<LINE>/cubes_NNN_TTT.npz (small).
D=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$D/metallicity}
LINE=${LINE:-H}
if [ "$LINE" = H ]; then
    SNAPS=(${SNAPS:-11 17 24 36 48 57 66})
else
    SNAPS=(${SNAPS:-48 57 66 74 80 86 92 97 102 106 111 116 121 126 131 139 141 144 146 150 152 155 158})
fi
NTASK=${NTASK:-8}
JDIRS=${JDIRS:-"$D/Jalpha $D/Jalpha_fullsnap $D/Jalpha_test"}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f metallicity_global.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
[ "$(( ${#SNAPS[@]} * NTASK ))" = "${SLURM_ARRAY_TASK_COUNT:-$(( ${#SNAPS[@]} * NTASK ))}" ] || \
    { echo "ERROR: ${#SNAPS[@]} snapshots x NTASK=$NTASK, array of $SLURM_ARRAY_TASK_COUNT (sbatch --array=0-$(( ${#SNAPS[@]} * NTASK - 1 )))"; exit 1; }
[ -f "$OUT/R_Z.csv" ] || { echo "no $OUT/R_Z.csv: run run_metal_R.sh first"; exit 1; }
SNAP=${SNAPS[$((SLURM_ARRAY_TASK_ID / NTASK))]}
TASK=$((SLURM_ARRAY_TASK_ID % NTASK))
if [ "$SNAP" -le 121 ]; then SD=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots
else SD=/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots; fi
if [ "$LINE" = H ]; then SCAN=$D/cell_test_H_$SNAP; else SCAN=$D/cell_test_$SNAP; fi
echo "cubes: $LINE snapshot $SNAP task $TASK/$NTASK, scan files $SCAN, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u metallicity_global.py cubes --line $LINE --snap $SNAP --snapdir $SD --scan-dir $SCAN \
    --rcsv "$OUT/R_Z.csv" --jdir $JDIRS --task $TASK --ntask $NTASK --out "$OUT"
echo "done $(date)"
