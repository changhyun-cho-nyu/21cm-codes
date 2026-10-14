#!/bin/bash
#SBATCH --job-name=remapHe
#SBATCH --output=logs/remapHe_%A_%a.out
#SBATCH --error=logs/remapHe_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Redshift-space He II density (SpectralDensityHeII) for every snapshot of run_3cm.sh, with
# remap_rsd.py (velocity convention sqrt_a, validated against SpectralDensity / SpectralDensityHI).
#   mkdir -p logs && sbatch run_remap_He.sh
# Then set HE_RS_DIR=$OUT in run_3cm.sh, compile, and run it.

OUT=/orcd/data/mvogelsb/004/chcho/RSD_He          # -> $OUT/SpectralDensityHeII/SpectralDensityHeII_NNN.hdf5
RA=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640
RB=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640
MIN_FREE_GB=60                                    # ~1 GB per snapshot

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
# the snapshot list of run_3cm.sh (its uncommented SNAPS=( ... ) block)
SNAPS=($(sed -n '/^SNAPS=(/,/)/p' run_3cm.sh | sed 's/#.*//' | tr -d 'SNAPS=()'))
mkdir -p "$OUT"
FREE_GB=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || { echo "only ${FREE_GB} GB free on $OUT"; exit 1; }

NT=8; i=$SLURM_ARRAY_TASK_ID
echo "task $i of $NT: ${#SNAPS[@]} snapshots in run_3cm.sh, $(date), commit $(git rev-parse --short HEAD)"
for ((k = i; k < ${#SNAPS[@]}; k += NT)); do
    s=${SNAPS[$k]}
    if [ -f "$OUT/SpectralDensityHeII/SpectralDensityHeII_$(printf %03d $s).hdf5" ]; then echo "snap $s done already"; continue; fi
    if [ "$s" -le 428 ]; then D=$RA; else D=$RB; fi
    [ -d "$D/Velocities" ] || { echo "ERROR: no Velocities render in $D"; exit 1; }
    python -u remap_rsd.py --dir "$D" --snap $s --out "$OUT" --conv sqrt_a
done
echo "done $(date)"
