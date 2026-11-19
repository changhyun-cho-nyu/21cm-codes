#!/bin/bash
#SBATCH --job-name=psCell
#SBATCH --output=logs/psCell_%j.out
#SBATCH --error=logs/psCell_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Power spectra of the gas-cell-level 3.46 cm maps (test_cells_Tb.py merge, cell_test*/) in the PS_NNN.out format of
# the production maps (ps_from_map.py), for every full snapshot that has one:
#   $OUT/PS_cell_He_NNN.out : the gas-cell-level map (RIN_cell_He_NNN.hdf5, else cellmap_He_NNN.hdf5:dTb_cell)
#   $OUT/PS_prod_He_NNN.out : the 640^3 production map stored with it (cellmap_He_NNN.hdf5:dTb_production), same z
# NNN = full snapshot. The second is checked against Tb_maps_prod/He/PS_<render>.out where that file exists.
#   mkdir -p logs && sbatch run_ps_cell.sh

U=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$U/cell_test_ps}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f ps_from_map.py ] || { echo "submit from the 21cm-codes directory"; exit 1; }
mkdir -p "$OUT"
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
dup=$(ls $U/cell_test*/cellmap_He_*.hdf5 | sed 's#.*/cellmap_He_##' | sort | uniq -d)
[ -z "$dup" ] || { echo "ERROR: full snapshot(s) $dup have cellmap_He files in more than one cell_test* directory"; exit 1; }
n=0
for cm in $U/cell_test*/cellmap_He_*.hdf5; do
    s=$(basename "$cm" .hdf5 | sed 's/cellmap_He_//')
    d=$(dirname "$cm")
    if [ -f "$d/RIN_cell_He_$s.hdf5" ]; then
        python -u ps_from_map.py --map "$d/RIN_cell_He_$s.hdf5" --out "$OUT/PS_cell_He_$s.out"
    else
        python -u ps_from_map.py --map "$cm" --dset dTb_cell --out "$OUT/PS_cell_He_$s.out"
    fi
    r=$(python -c "import h5py,sys; f=h5py.File(sys.argv[1]); print(int(f.attrs['render_snap']) if 'render_snap' in f.attrs else '')" \
        "$d/RIN_cell_He_$s.hdf5" 2>/dev/null || true)
    chk=""
    [ -n "$r" ] && [ -f "$U/Tb_maps_prod/He/PS_$r.out" ] && chk="--check $U/Tb_maps_prod/He/PS_$r.out"
    python -u ps_from_map.py --map "$cm" --dset dTb_production --out "$OUT/PS_prod_He_$s.out" $chk \
        || { rc=$?; [ $rc = 3 ] || exit $rc; echo "  NOTE: production map of full snapshot $s differs from Tb_maps_prod/He/PS_$r.out (see above)"; }
    n=$((n + 1))
done
echo "$n full snapshots done $(date)"
