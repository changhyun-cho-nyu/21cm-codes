#!/bin/bash
#SBATCH --job-name=pairX
#SBATCH --output=logs/pairX_%A_%a.out
#SBATCH --error=logs/pairX_%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Cell-by-cell pairing of the two X-ray test runs (pair_cells_xray.py scan): one array task per full snapshot,
# both runs read in the same task (~2.2e8 cells each, ~40 GB peak expected). Default snapshots 36 and 48
# (z = 11.93, 10.01; renders 112, 154). Reads Oliver's snapshots and raw renders (read only), writes
# /orcd/data/mvogelsb/004/chcho/xray/pair_cells/pair_cells_NNN.npz (~100 MB with the 2e6-pair subsample).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_pair_cells_xray.sh
#   other snapshots:  SNAPS="24 57 66" sbatch -p ou_mki --array=0-2 run_pair_cells_xray.sh
# then:  python pair_cells_xray.py report
SNAPS=(${SNAPS:-36 48})

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f pair_cells_xray.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
SNAP=${SNAPS[$SLURM_ARRAY_TASK_ID]}
[ -n "$SNAP" ] || { echo "no snapshot for array task $SLURM_ARRAY_TASK_ID (SNAPS = ${SNAPS[*]})"; exit 1; }
df -h /orcd/data/mvogelsb/004 | tail -1
echo "snapshot $SNAP, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u pair_cells_xray.py scan --snap $SNAP
echo "done $(date)"
