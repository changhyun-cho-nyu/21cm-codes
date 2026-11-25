#!/bin/bash
#SBATCH --job-name=jeansF
#SBATCH --output=logs/jeansF_%A_%a.out
#SBATCH --error=logs/jeansF_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Jeans pressure floor check (jeans_floor_check.py scan): gas cells of one X-ray test run followed by ParticleIDs
# between consecutive full snapshots, whole box (~2.2e8 cells per snapshot). One array task per snapshot pair;
# default the no-X-ray run, pairs 024->036, 036->048, 048->057, 057->066 (z = 15.0 -> 12 -> 10 -> 8.9 -> 8.0).
# Reads Oliver's snapshots (read only), writes /orcd/data/mvogelsb/004/chcho/xray/<run>/jeans_floor/jeans_S1_S2.npz.
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_jeans_floor.sh
#   with X-rays:  RUN=withXrays sbatch -p ou_mki run_jeans_floor.sh
# then:  python jeans_floor_check.py report --run withoutXrays
RUN=${RUN:-withoutXrays}
PAIRS=(${PAIRS:-24:36 36:48 48:57 57:66})

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f jeans_floor_check.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
P=${PAIRS[$SLURM_ARRAY_TASK_ID]}
[ -n "$P" ] || { echo "no snapshot pair for array task $SLURM_ARRAY_TASK_ID (PAIRS = ${PAIRS[*]})"; exit 1; }
S1=${P%%:*}; S2=${P##*:}
df -h /orcd/data/mvogelsb/004 | tail -1
echo "run $RUN, snapshots $S1 -> $S2, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u jeans_floor_check.py scan --run $RUN --snaps $S1 $S2
echo "done $(date)"
