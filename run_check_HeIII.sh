#!/bin/bash
#SBATCH --job-name=chkHeIII
#SBATCH --output=logs/chkHeIII_%j.out
#SBATCH --error=logs/chkHeIII_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=ou_mki
#
# Gas-cell vs render x_HeIII (and x_HII, x_HeII as controls), pixel by pixel in the scans' saved cubes
# (check_cells_HeIII.py), at the full snapshots of the He power-spectrum figure that still have scan files
# (136 has none). Output: logs/chkHeIII_<job>.out.
#   mkdir -p logs && sbatch run_check_HeIII.sh

U=/orcd/data/mvogelsb/004/chcho
R=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL
SNAPS=${SNAPS:-"116 126 146"}
NCUBES=${NCUBES:-8}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
for s in $SNAPS; do
    d=$U/cell_test_$s
    r=$(python -c "import h5py,glob,sys; print(int(h5py.File(sorted(glob.glob(sys.argv[1]))[0]).attrs['render_snap']))" "$d/scan_He_${s}_*.hdf5")
    if [ "$r" -le 428 ]; then REN=$R/Flagship_new/postprocessing/ren_640; else REN=$R/Flagship_below_z_4p75_new/postprocessing/ren_640; fi
    echo; echo "===== full snapshot $s (render $r)"
    python -u check_cells_HeIII.py --scandir "$d" --snap $s --render "$REN" --n-cubes $NCUBES
done
echo "done $(date)"
