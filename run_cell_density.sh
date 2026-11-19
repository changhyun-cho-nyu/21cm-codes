#!/bin/bash
#SBATCH --job-name=cellRho
#SBATCH --output=logs/cellRho_%A_%a.out
#SBATCH --error=logs/cellRho_%A_%a.err
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Gas density at the gas-cell level (cell_density_map.py: sum m / sum V per 640^3 pixel from the He scan task files,
# the cells and pixels of the gas-cell 3.46 cm maps) and its power spectrum (ps_from_map.py --overdensity):
#   cell_test_NNN/cellrho_NNN.hdf5           (~1 GB each)
#   $OUT/PS_cell_delta_NNN.out               (Delta^2_delta, dimensionless; NNN = full snapshot, as PS_cell_He_NNN.out)
# One array task per full snapshot of the helium power-spectrum figure (visual_PS cell 10: 116 ... 150, z = 5.0-3.37;
# user, 2026-11-19), except 136, whose task files are gone. Others: SNAPS_D="..." and --array=0-<n-1>.
#   mkdir -p logs && sbatch run_cell_density.sh

U=/orcd/data/mvogelsb/004/chcho
OUT=${OUT:-$U/cell_test_ps}
SNAPS=(${SNAPS_D:-116 121 126 131 139 141 144 146 150})

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ "${#SNAPS[@]}" = "${SLURM_ARRAY_TASK_COUNT:-${#SNAPS[@]}}" ] || { echo "ERROR: ${#SNAPS[@]} snapshots, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
S=${SNAPS[$SLURM_ARRAY_TASK_ID]}
D=$U/cell_test_$S
mkdir -p "$OUT"
echo "full snapshot $S, $D, $(date), commit $(git rev-parse --short HEAD)"
MAP=$D/cellrho_$(printf %03d $S).hdf5
[ -f "$MAP" ] || python -u cell_density_map.py --scandir "$D" --snap $S --out "$MAP"
python -u ps_from_map.py --map "$MAP" --dset rho --overdensity --out "$OUT/PS_cell_delta_$(printf %03d $S).out"
echo "done $(date)"
