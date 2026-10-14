#!/bin/bash
#SBATCH --job-name=rsdHe
#SBATCH --output=logs/remapQ_%A_%a.out
#SBATCH --error=logs/remapQ_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Size of the redshift-space effect on He II (remap_rsd.py --rsd; the He T_b maps are real space):
# P_rs/P_real for gas and H I (exact, from the render) and He II, with the He II in redshift space from
#   the q method  He II_s = [remap(He II) / remap(H II)] x (0.76 SpectralDensity - SpectralDensityHI),
#   the plain remap and the shortcut; plus the q-method checks (--qtest). Snapshots 220, 300, 406, 500.
# Writes nothing but the logs.
#   mkdir -p logs && sbatch run_remap_qtest.sh

RA=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640
RB=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640
SNAPS=(220 300 406 500)

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
s=${SNAPS[$SLURM_ARRAY_TASK_ID]}
if [ "$s" -le 428 ]; then D=$RA; else D=$RB; fi
echo "snap $s from $D, $(date), commit $(git rev-parse --short HEAD)"
python -u remap_rsd.py --dir "$D" --snap $s --qtest --rsd --conv sqrt_a
echo "done $(date)"
