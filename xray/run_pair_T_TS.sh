#!/bin/bash
#SBATCH --job-name=pairTTS
#SBATCH --output=logs/pairTTS_%A_%a.out
#SBATCH --error=logs/pairTTS_%A_%a.err
#SBATCH --array=0-21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Whole-box H I-weighted <1/T>^-1 and <1/T_S>^-1 (and box <dT_b>) of every gas cell of the 50 cMpc pair, the same sums
# as the Lumina whole-box no-X scan (pair_T_TS.py; user, 2026-11-24, figure in global_noX.ipynb). Array = SNAPS x the
# two runs (even task: withoutXrays, odd: withXrays). Reads Oliver's snapshots and renders and the pair's J
# (<out>/<run>/Jalpha, read only); writes .../xray/<run>/cells_TTS/cells_TTS_NNN.npz (a few kB).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_pair_T_TS.sh          # 11 snapshots x 2 runs = 22 tasks
# then:  python lumina_noX.py summary   (adds the pair_T_* / pair_TS_* columns)
SNAPS=(${SNAPS:-24 36 48 57 66 74 80 86 92 97 102})
RUNS=(withoutXrays withXrays)

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f pair_T_TS.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
[ "$(( ${#SNAPS[@]} * 2 ))" = "${SLURM_ARRAY_TASK_COUNT:-$(( ${#SNAPS[@]} * 2 ))}" ] || \
    { echo "ERROR: ${#SNAPS[@]} snapshots x 2 runs, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
SNAP=${SNAPS[$((SLURM_ARRAY_TASK_ID / 2))]}
RUN=${RUNS[$((SLURM_ARRAY_TASK_ID % 2))]}
echo "run $RUN, snapshot $SNAP, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u pair_T_TS.py --run $RUN --snap $SNAP
echo "done $(date)"
