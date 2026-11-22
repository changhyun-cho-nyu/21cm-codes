#!/bin/bash
#SBATCH --job-name=noXcube
#SBATCH --output=logs/noXcube_%A_%a.out
#SBATCH --error=logs/noXcube_%A_%a.err
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Lumina without X-ray heating, option 3, global mean (lumina_noX.py cubes): the stored cube cells of the H gas-cell
# scans (cell_test_H_<snap>, 64 cubes, ~1.7e9 cells per snapshot, read only) get T drawn from the 50 cMpc no-X-ray
# tables (xray/pair_cells/closure_NNN.npz); dT_b with and without X-rays. NTASK tasks per snapshot (cubes q with
# q % NTASK = task), array = (number of SNAPS) x NTASK. Writes .../xray/lumina_noX/noX_cubes_NNN_TTT.npz (small).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_lumina_noX_cubes.sh            # 024 036 048 057 066, 8 tasks each
# then on the login node, per snapshot:  python lumina_noX.py merge --snap 24   ...   and  python lumina_noX.py summary
SNAPS=(${SNAPS:-24 36 48 57 66})
NTASK=${NTASK:-8}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f lumina_noX.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
[ "$(( ${#SNAPS[@]} * NTASK ))" = "${SLURM_ARRAY_TASK_COUNT:-$(( ${#SNAPS[@]} * NTASK ))}" ] || \
    { echo "ERROR: ${#SNAPS[@]} snapshots x NTASK=$NTASK, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
SNAP=${SNAPS[$((SLURM_ARRAY_TASK_ID / NTASK))]}
TASK=$((SLURM_ARRAY_TASK_ID % NTASK))
echo "cubes: snapshot $SNAP task $TASK/$NTASK, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u lumina_noX.py cubes --snap $SNAP --task $TASK --ntask $NTASK
echo "done $(date)"
