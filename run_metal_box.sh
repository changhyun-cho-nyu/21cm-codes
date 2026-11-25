#!/bin/bash
#SBATCH --job-name=metalBox
#SBATCH --output=logs/metalBox_%A_%a.out
#SBATCH --error=logs/metalBox_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Appendix sec:metallicity, step 3 (metallicity_global.py box): gas-cell <dT_b> for the 13 BPASS metallicities over
# EVERY gas cell of the full snapshot (whole box; user 2026-11-24: no cubes or samples). Each of the gas-cell scan's 32
# row blocks is split into SPLIT tasks: array = (number of SNAPS) x 32 x SPLIT. J of each cell = the scan's J + (R_j - 1)
# J_star; the J files are identified from the scan, and box-merge checks every block's fiducial sums against it.
#   1. timing (one task, MAXROWS rows, nothing written; prints us per cell and the task-hours of the whole snapshot):
#        mkdir -p logs && MAXROWS=4000000 LINE=H  SNAPS=48  sbatch -p ou_mki --array=0 --time=02:00:00 run_metal_box.sh
#        MAXROWS=4000000 LINE=He SNAPS=116 sbatch -p ou_mki --array=0 --time=02:00:00 run_metal_box.sh
#   2. production, SPLIT from the timing (per-task hours = block hours / SPLIT, keep < 24 h); e.g.
#        LINE=He SPLIT=2 SNAPS="48 57 66 74 80 86 92" sbatch -p ou_mki --array=0-447 run_metal_box.sh
#      (keep each submission within the account's submit limit)
#   (full snapshot 136, z = 4.0: no gas-cell scan files left, so no row blocks or J identification: not in the He list)
# Then run_metal_table.sh (box-merge of every snapshot + the table). Writes $OUT/box_<LINE>/box_NNN_TTTT.npz (small).
D=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$D/metallicity}
LINE=${LINE:-H}
if [ "$LINE" = H ]; then
    SNAPS=(${SNAPS:-11 17 24 36 48 57 66})
else
    SNAPS=(${SNAPS:-48 57 66 74 80 86 92 97 102 106 111 116 121 126 131 139 141 144 146 150 152 155 158})
fi
NBLOCK=32
SPLIT=${SPLIT:-1}
MAXROWS=${MAXROWS:-}
JDIRS=${JDIRS:-"$D/Jalpha $D/Jalpha_fullsnap $D/Jalpha_test"}
PER=$(( NBLOCK * SPLIT ))

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f metallicity_global.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
[ -f "$OUT/R_Z.csv" ] || { echo "no $OUT/R_Z.csv: run run_metal_R.sh first"; exit 1; }
if [ -z "$MAXROWS" ]; then
    [ "$(( ${#SNAPS[@]} * PER ))" = "${SLURM_ARRAY_TASK_COUNT:-$(( ${#SNAPS[@]} * PER ))}" ] || \
        { echo "ERROR: ${#SNAPS[@]} snapshots x 32 blocks x SPLIT=$SPLIT, array of $SLURM_ARRAY_TASK_COUNT (sbatch --array=0-$(( ${#SNAPS[@]} * PER - 1 )))"; exit 1; }
fi
SNAP=${SNAPS[$((SLURM_ARRAY_TASK_ID / PER))]}
TASK=$((SLURM_ARRAY_TASK_ID % PER))
if [ "$SNAP" -le 121 ]; then SD=/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots
else SD=/orcd/data/mvogelsb/005/Lumina/Lumina_below_z_4p75/snapshots; fi
if [ "$LINE" = H ]; then SCAN=$D/cell_test_H_$SNAP; else SCAN=$D/cell_test_$SNAP; fi
echo "box: $LINE snapshot $SNAP task $TASK/$PER (split $SPLIT), $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u metallicity_global.py box --line $LINE --snap $SNAP --snapdir $SD --scan-dir $SCAN --rcsv "$OUT/R_Z.csv" \
    --jdir $JDIRS --task $TASK --split $SPLIT --out "$OUT" ${MAXROWS:+--max-rows $MAXROWS}
echo "done $(date)"
