#!/bin/bash
#SBATCH --job-name=crossHeRho
#SBATCH --output=logs/crossHeRho_%j.out
#SBATCH --error=logs/crossHeRho_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Test for visual_PS cell 12, panel (c) (user, 2026-11-20). cross_he/ (run_cross_he.sh, job 29593843) crosses the
# gas-cell delta T_b^He map (cells deposited by nearest pixel) with the 640^3 RENDER Density (cells deposited as cubes),
# so part of 1 - r^2 could be the deposit difference, not physics. Here, at the nine full snapshots with a gas-cell
# density map (cell_test*/cellrho_NNN.hdf5:rho = sum m / sum V per pixel, cell_density_map.py; 136 has none):
#   1. A = gas-cell He map, B0 = gas-cell density (same deposit as A), B1 = render Density  -> cross_cellHe_rho_NNN.npz
#   2. A = gas-cell density, B0 = render Density (the deposit difference alone)             -> cross_rho_render_NNN.npz
# and the maps' means (cross_spectrum.py prints them; mean of the He map = the <dT_b> that b_rel divides by, to compare
# with cell_test_summary.csv cell_uK). A table of 1 - r^2 at k = 0.1, 0.2, 0.5 and the means follows at the end.
#   mkdir -p logs && sbatch run_cross_he_rho.sh
# ~2 min per cross_spectrum.py call at 640^3 (two calls per snapshot).

U=${U:-/orcd/data/mvogelsb/004/chcho}
R=${R:-/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL}
OUT=${OUT:-$U/cross_he_rho}
PAIRS=${PAIRS:-"116:406 121:428 126:452 131:476 139:516 141:528 144:552 146:582 150:614"}   # full snapshot : render

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f cross_spectrum.py ] && [ -f cell_test_summary.csv ] || { echo "submit from the 21cm-codes directory"; exit 1; }
mkdir -p "$OUT"
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- cross_spectrum.py run_cross_he_rho.sh)" ] || echo ' + local or untracked changes')"
rdir() { if [ "$1" -le 428 ]; then echo $R/Flagship_new/postprocessing/ren_640; else echo $R/Flagship_below_z_4p75_new/postprocessing/ren_640; fi; }
field() { local p="$1/$2/$2_$3.hdf5"; [ -f "$p" ] && echo "$p:$2" || echo "$1/All/All_$3.hdf5:$2"; }
one() {   # exactly one file matching the pattern, or stop
    local m; m=$(ls $1 2>/dev/null || true)
    [ -n "$m" ] && [ "$(echo "$m" | wc -l)" -eq 1 ] || { echo "STOP: expected one file for $1, found: '${m}'"; exit 1; }
    echo "$m"
}
for pr in $PAIRS; do
    s=${pr%%:*}; r=${pr##*:}; S=$(printf %03d $s); RS=$(printf %03d $r); D=$(rdir $r)
    he=$(one "$U/cell_test*/RIN_cell_He_$S.hdf5")
    rho=$(one "$U/cell_test*/cellrho_$S.hdf5")
    den=$(field $D Density $RS)
    echo; echo "===== full snapshot $s, render $r: gas-cell He x (gas-cell density, render density)"
    python -u cross_spectrum.py --a "$he:rin" --b "$rho:rho" --b "$den" --out "$OUT/cross_cellHe_rho_$S.npz"
    echo; echo "===== full snapshot $s, render $r: gas-cell density x render density (deposit only)"
    python -u cross_spectrum.py --a "$rho:rho" --b "$den" --out "$OUT/cross_rho_render_$S.npz"
done

echo; echo "===== summary"
python - "$OUT" "$PAIRS" <<'EOF'
import csv, sys
import numpy as np
out, pairs = sys.argv[1], sys.argv[2].split()
cell = {int(r["snap"]): float(r["cell_uK"]) for r in csv.DictReader(l for l in open("cell_test_summary.csv") if not l.startswith("#"))}
def at(d, key, k0):
    return np.interp(np.log(k0), np.log(d["k"]), d[key])
print("1 - r^2 at k = 0.1 / 0.2 / 0.5 h/cMpc:")
print(" snap  He x gas-cell rho         He x render rho           gas-cell rho x render rho   <He map> / cell_uK")
for p in pairs:
    s = int(p.split(":")[0])
    a = np.load(f"{out}/cross_cellHe_rho_{s:03d}.npz"); b = np.load(f"{out}/cross_rho_render_{s:03d}.npz")
    f = lambda d, key: " ".join(f"{1 - at(d, key, k0)**2:.4f}" for k0 in (0.1, 0.2, 0.5))
    print(f" {s:4d}  {f(a, 'r_0')}    {f(a, 'r_1')}    {f(b, 'r_0')}      {float(a['mean_A']) / cell[s]:.6f}")
EOF
echo "done $(date)"
