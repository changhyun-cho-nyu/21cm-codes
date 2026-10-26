#!/bin/bash
#SBATCH --job-name=covJ
#SBATCH --output=logs/covJ_%A_%a.out
#SBATCH --error=logs/covJ_%A_%a.err
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Angular-covariance test of the He-band J_alpha with IGM transmission (check_angular_covariance.py):
# J_dir (direction by direction) / J_avg (shell-averaged emissivity x direction-averaged transmission,
# what compute_Jalpha_He.c / compute_Jalpha_AGN.c do), for stars and AGN, at snapshots 300, 406, 500.
# Writes nothing but the logs.   mkdir -p logs && sbatch run_covariance_test.sh

SNAPS=(160 200 250 340 380)
OPA=/orcd/data/mvogelsb/004/chcho/opacity_He_sfmask          # the opacity used for the transmission
SFR=/orcd/data/mvogelsb/004/chcho/SFR_eff/He
ZCSV=/orcd/data/mvogelsb/004/chcho/SFR_eff/SFR_eff_summary.csv
RA=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640
RB=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
S=${SNAPS[$SLURM_ARRAY_TASK_ID]}
echo "snap $S, $(date), commit $(git rev-parse --short HEAD)"
python -u check_angular_covariance.py --snap $S --opa $OPA --sfr $SFR --render $RA $RB --zcsv $ZCSV \
    --n-points 1000 --ndir 192 --sources stars agn
echo "done $(date)"
