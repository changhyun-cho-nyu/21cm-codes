#!/bin/bash
#SBATCH --job-name=opacity
#SBATCH --output=logs/opacity_%A_%a.out
#SBATCH --error=logs/opacity_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Absorber densities (n_HI, n_HeI) on a 320^3 grid for every EVERY-th render snapshot,
# split over 8 array tasks. Input of compute_transmission_He.c.
#   mkdir -p logs && sbatch run_opacity.sh

OPA=/orcd/data/mvogelsb/004/chcho/opacity_He       # output directory
EVERY=2                                            # every 2nd snapshot (~5 Myr apart)
FACTOR=2                                           # 640^3 -> 320^3
FIRST=0; LAST=708

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OPA"
NT=${SLURM_ARRAY_TASK_COUNT:-8}; i=$SLURM_ARRAY_TASK_ID
n=$(( (LAST - FIRST) / EVERY + 1 ))                # snapshots on the EVERY grid
per=$(( (n + NT - 1) / NT ))
a=$(( FIRST + i * per * EVERY )); b=$(( FIRST + ((i + 1) * per - 1) * EVERY ))
[ $b -gt $LAST ] && b=$LAST
[ $a -gt $LAST ] && { echo "nothing to do"; exit 0; }
echo "task $i: snapshots $a..$b every $EVERY  $(date)"
python -u make_opacity_fields.py --out "$OPA" --snaps $a $b --every $EVERY --factor $FACTOR
echo "done $(date)"
