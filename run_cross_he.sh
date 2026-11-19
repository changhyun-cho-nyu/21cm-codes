#!/bin/bash
#SBATCH --job-name=crossHe
#SBATCH --output=logs/crossHe_%j.out
#SBATCH --error=logs/crossHe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Does the gas-cell delta T_b^He map carry the He III bubbles? cross_spectrum.py at the nine full snapshots of the He
# power-spectrum figure (z = 5.0-3.5): A = the gas-cell map (cell_test*/RIN_cell_He_NNN.hdf5; 136: cellmap:dTb_cell),
# and for comparison A = the 640^3 production map stored with it (cellmap:dTb_production); B = the render fields at the
# matched render: HeIII_VolumeFraction, HeII_VolumeFraction, Density. Output: logs/crossHe_<job>.out (tables) and
# $OUT/cross_{cell,prod}_He_NNN.npz.
#   mkdir -p logs && sbatch run_cross_he.sh

U=/orcd/data/mvogelsb/004/chcho
R=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
OUT=${OUT:-$U/cross_he}
PAIRS=${PAIRS:-"116:406 121:428 126:452 131:476 136:501 139:516 141:528 144:552 146:582"}   # full snapshot : render

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f cross_spectrum.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
mkdir -p "$OUT"
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
rdir() { if [ "$1" -le 428 ]; then echo $R/Flagship_new/postprocessing/ren_640; else echo $R/Flagship_below_z_4p75_new/postprocessing/ren_640; fi; }
field() { local p="$1/$2/$2_$3.hdf5"; [ -f "$p" ] && echo "$p:$2" || echo "$1/All/All_$3.hdf5:$2"; }
for pr in $PAIRS; do
    s=${pr%%:*}; r=${pr##*:}; S=$(printf %03d $s); RS=$(printf %03d $r); D=$(rdir $r)
    cm=$(ls $U/cell_test*/cellmap_He_$S.hdf5)
    rin=$(ls $U/cell_test*/RIN_cell_He_$S.hdf5 2>/dev/null || true)
    A_cell=${rin:+$rin:rin}; A_cell=${A_cell:-$cm:dTb_cell}
    B="--b $(field $D HeIII_VolumeFraction $RS) --b $(field $D HeII_VolumeFraction $RS) --b $(field $D Density $RS)"
    echo; echo "===== full snapshot $s, render $r: gas-cell map"
    python -u cross_spectrum.py --a "$A_cell" $B --out "$OUT/cross_cell_He_$S.npz"
    echo; echo "===== full snapshot $s, render $r: 640^3 production map (for comparison)"
    python -u cross_spectrum.py --a "$cm:dTb_production" $B --out "$OUT/cross_prod_He_$S.npz"
done
echo "done $(date)"
