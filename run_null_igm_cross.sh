#!/bin/bash
#SBATCH --job-name=nullX
#SBATCH --output=logs/nullX_%j.out
#SBATCH --error=logs/nullX_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# After run_null_igm.sh: merge the null maps (null_igm_map.py merge, checked against cellrho), cross each with x_HeIII
# and the gas-cell density (cross_spectrum.py, as run_cross_he_igm.sh), and print the table
# (igm_cross_summary.py --null; compare with the He table: igm_cross_summary.py without --null).
#   sbatch --dependency=afterok:<run_null_igm.sh job id> run_null_igm_cross.sh
# CLEAN=1 deletes a snapshot's task files after its merge succeeded (~50 GB each).

U=${U:-/orcd/data/mvogelsb/004/chcho}
R=${R:-/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL}
NULL_DIR=${NULL_DIR:-$U/null_igm}
OUT=${OUT:-$U/cross_he_igm}               # next to the He cross-spectra (cross_x_rho_NNN.npz is reused)
NTASK=${NTASK:-8}
BOX=${BOX:-338300}
PAIRS=${PAIRS:-"116:406 139:516 146:582 158:705"}
MAPS="null_all null_lt100 null_lt30 null_lt10 null_lt3"

[ -n "$SKIP_ENV" ] || { source ~/.bashrc; conda activate jupyter_env; }
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f cross_spectrum.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD 2>/dev/null)"
rdir() { if [ "$1" -le 428 ]; then echo $R/Flagship_new/postprocessing/ren_640; else echo $R/Flagship_below_z_4p75_new/postprocessing/ren_640; fi; }
field() { local p="$1/$2/$2_$3.hdf5"; [ -f "$p" ] && echo "$p:$2" || echo "$1/All/All_$3.hdf5:$2"; }
one() {
    local m; m=$(ls $1 2>/dev/null || true)
    [ -n "$m" ] && [ "$(echo "$m" | wc -l)" -eq 1 ] || { echo "STOP: expected one file for $1, found: '${m}'"; exit 1; }
    echo "$m"
}
for pr in $PAIRS; do
    s=${pr%%:*}; r=${pr##*:}; S=$(printf %03d $s); RS=$(printf %03d $r); D=$(rdir $r)
    rho=$(one "$U/cell_test*/cellrho_$S.hdf5")
    x=$(field $D HeIII_VolumeFraction $RS)
    nm=$NULL_DIR/nullmap_$S.hdf5
    echo; echo "===== full snapshot $s: merge"
    [ -f "$nm" ] || python -u null_igm_map.py merge --snap $s --out $NULL_DIR --ntask $NTASK --cellrho $rho
    [ -n "$CLEAN" ] && rm -f $NULL_DIR/null_${S}_[0-9][0-9][0-9].hdf5
    if [ ! -f "$OUT/cross_x_rho_$S.npz" ]; then
        python -u cross_spectrum.py --a "$x" --b "$rho:rho" --box $BOX --out "$OUT/cross_x_rho_$S.npz"
    fi
    for m in $MAPS; do
        o="$OUT/cross_igm_${m}_$S.npz"
        [ -f "$o" ] && { echo "$(basename $o) exists"; continue; }
        echo; echo "===== $m x (x_HeIII, gas-cell density)"
        python -u cross_spectrum.py --a "$nm:$m" --b "$x" --b "$rho:rho" --box $BOX --out "$o"
    done
done
echo; echo "===== null table"
python igm_cross_summary.py --dir $OUT --null
echo "done $(date)"
