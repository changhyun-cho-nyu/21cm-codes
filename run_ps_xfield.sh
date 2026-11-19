#!/bin/bash
#SBATCH --job-name=psX
#SBATCH --output=logs/psX_%j.out
#SBATCH --error=logs/psX_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Power spectra of the ionized-fraction fields of the 640^3 renders, in the PS_NNN.out format (ps_from_map.py), to
# replace cell_by_cell/hydrogen/f_HII and helium/f_HeIII (April 2026, no code in the repo; their stored mean, line 3 /
# line 4, is 0.025 at every z, not x-bar, so their amplitude is not reliable):
#   $OUT/PS_xHII_NNN.out   : x_HII   = HII_VolumeFraction            (volume-weighted, as the maps' x labels)
#   $OUT/PS_xHeIII_NNN.out : x_HeIII = HeIII_VolumeFraction / y      (the field is per H nucleus; y = 0.0789)
# NNN = render snapshot. z and BoxSize from the render's All_NNN.hdf5 header.
#   mkdir -p logs && sbatch run_ps_xfield.sh

U=/orcd/data/mvogelsb/004/chcho
R=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
OUT=${OUT:-$U/xfield_ps}
# x_HII: the snapshots of visual_PS cells 3-5 (and PS2Bubble); x_HeIII: visual_PS cell 10's renders and the renders of
# the full snapshots used for the gas-cell delta T_b^He spectra (116 ... 146 -> 406, 428, 452, 476, 501, 516, 528, 552, 582)
H_SNAPS=${H_SNAPS:-"177 203 220 242 264 283 300 318 338"}
HE_SNAPS=${HE_SNAPS:-"406 428 450 452 471 476 487 500 501 514 516 528 529 552 558 582 596 614"}
Y_INV=12.666666666666666          # 1/y, y = (1 - 0.76) / (4 x 0.76)

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f ps_from_map.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
mkdir -p "$OUT"
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
rdir() { if [ "$1" -le 428 ]; then echo $R/Flagship_new/postprocessing/ren_640; else echo $R/Flagship_below_z_4p75_new/postprocessing/ren_640; fi; }
field() {   # $1 render dir, $2 field, $3 snapshot (3 digits): the per-field file, else All
    local p="$1/$2/$2_$3.hdf5"; [ -f "$p" ] && echo "$p" || echo "$1/All/All_$3.hdf5"
}
for s in $H_SNAPS; do
    S=$(printf %03d $s); D=$(rdir $s)
    python -u ps_from_map.py --map "$(field $D HII_VolumeFraction $S)" --dset HII_VolumeFraction \
        --header "$D/All/All_$S.hdf5" --out "$OUT/PS_xHII_$s.out"
done
for s in $HE_SNAPS; do
    S=$(printf %03d $s); D=$(rdir $s)
    python -u ps_from_map.py --map "$(field $D HeIII_VolumeFraction $S)" --dset HeIII_VolumeFraction --scale $Y_INV \
        --header "$D/All/All_$S.hdf5" --out "$OUT/PS_xHeIII_$s.out"
done
echo "done $(date)"
