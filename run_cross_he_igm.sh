#!/bin/bash
#SBATCH --job-name=crossIGM
#SBATCH --output=logs/crossIGM_%A_%a.out
#SBATCH --error=logs/crossIGM_%A_%a.err
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Is the He III bubble imprint on the 3.46 cm signal carried by the IGM, the gas the simulation resolves well, or by the
# dense halo gas (user, 2026-11-27)? cross_spectrum.py on the whole-box IGM-only gas-cell maps that test_cells_Tb.py merge
# wrote with --igm-cuts (cell_test*/cellmap_He_NNN.hdf5: dTb_cell = all cells, dTb_cell_igm_lt{100,30,10,3} = only the
# cells with 1+delta below the cut; 136 has none), one array task per full snapshot:
#   1. A = x_HeIII (render HeIII_VolumeFraction), B0 = gas-cell density (cellrho_NNN.hdf5:rho)  -> cross_x_rho_NNN.npz
#   2. A = each He map, B0 = HeIII_VolumeFraction, B1 = gas-cell density                       -> cross_igm_<map>_NNN.npz
# Then the table (1 - r^2 with density, r with x_HeIII, partial r(He, x_HeIII | density), share of the mean):
#   python igm_cross_summary.py --dir /orcd/data/mvogelsb/004/chcho/cross_he_igm
#   mkdir -p logs && sbatch run_cross_he_igm.sh
# ~2-4 min per cross_spectrum.py call at 640^3, 6 calls per snapshot. A finished snapshot (all 6 npz) is skipped.

U=${U:-/orcd/data/mvogelsb/004/chcho}
R=${R:-/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL}
OUT=${OUT:-$U/cross_he_igm}
BOX=${BOX:-338300}
read -ra PAIRS <<< "${PAIRS:-116:406 121:428 126:452 131:476 139:516 141:528 144:552 146:582 150:614 152:641 155:672 158:705}"
MAPS="dTb_cell dTb_cell_igm_lt100 dTb_cell_igm_lt30 dTb_cell_igm_lt10 dTb_cell_igm_lt3"

[ -n "$SKIP_ENV" ] || { source ~/.bashrc; conda activate jupyter_env; }
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f cross_spectrum.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
ID=${SLURM_ARRAY_TASK_ID:?not an array task}
[ "$ID" -lt "${#PAIRS[@]}" ] || { echo "array task $ID: only ${#PAIRS[@]} snapshots"; exit 1; }
mkdir -p "$OUT"
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD 2>/dev/null)$([ -z "$(git status --porcelain -- cross_spectrum.py run_cross_he_igm.sh 2>/dev/null)" ] || echo ' + local or untracked changes')"
rdir() { if [ "$1" -le 428 ]; then echo $R/Flagship_new/postprocessing/ren_640; else echo $R/Flagship_below_z_4p75_new/postprocessing/ren_640; fi; }
field() { local p="$1/$2/$2_$3.hdf5"; [ -f "$p" ] && echo "$p:$2" || echo "$1/All/All_$3.hdf5:$2"; }
one() {   # exactly one file matching the pattern, or stop
    local m; m=$(ls $1 2>/dev/null || true)
    [ -n "$m" ] && [ "$(echo "$m" | wc -l)" -eq 1 ] || { echo "STOP: expected one file for $1, found: '${m}'"; exit 1; }
    echo "$m"
}
pr=${PAIRS[$ID]}; s=${pr%%:*}; r=${pr##*:}; S=$(printf %03d $s); RS=$(printf %03d $r); D=$(rdir $r)
cm=$(one "$U/cell_test*/cellmap_He_$S.hdf5")
rho=$(one "$U/cell_test*/cellrho_$S.hdf5")
x=$(field $D HeIII_VolumeFraction $RS)
python -c "import h5py,sys; f=h5py.File(sys.argv[1],'r'); m=[k for k in sys.argv[2:] if k not in f]; sys.exit(f'STOP: {sys.argv[1]} has no {m}' if m else 0)" "$cm" $MAPS
echo "full snapshot $s, render $r: $cm, $rho, $x"
if [ -f "$OUT/cross_x_rho_$S.npz" ]; then echo "cross_x_rho_$S.npz exists"; else
    echo; echo "===== x_HeIII x gas-cell density"
    python -u cross_spectrum.py --a "$x" --b "$rho:rho" --box $BOX --out "$OUT/cross_x_rho_$S.npz"
fi
for m in $MAPS; do
    o="$OUT/cross_igm_${m}_$S.npz"
    if [ -f "$o" ]; then echo "$(basename $o) exists"; continue; fi
    echo; echo "===== $m x (x_HeIII, gas-cell density)"
    python -u cross_spectrum.py --a "$cm:$m" --b "$x" --b "$rho:rho" --box $BOX --out "$o"
done
echo "done $(date)"
