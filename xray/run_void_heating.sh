#!/bin/bash
#SBATCH --job-name=voidX
#SBATCH --output=logs/voidX_%A_%a.out
#SBATCH --error=logs/voidX_%A_%a.err
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Are Lumina's warmer neutral voids at z = 15 / 12 X-ray heated? (void_heating_test.py). Two array tasks per snapshot in
# SNAPS: task 2i = Lumina (the 64 cubes of cell_test_H_<snap>, split into 512 units of 8.46 cMpc/h, + the X-ray photon
# density of every 640^3 render up to the snapshot over those units; read only), task 2i+1 = 50 cMpc (both runs, 64 units,
# the X run's 128^3 renders up to the snapshot). Writes /orcd/data/mvogelsb/004/chcho/xray/void_heat/*.npz (small).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_void_heating.sh                     # 024 036 048
#   SNAPS="24" sbatch -p ou_mki --array=0-1 run_void_heating.sh
# then:  python void_heating_test.py compare
SNAPS=(${SNAPS:-24 36 48})

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f void_heating_test.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
[ "$((2 * ${#SNAPS[@]}))" -ge "${SLURM_ARRAY_TASK_COUNT:-1}" ] || { echo "ERROR: ${#SNAPS[@]} snapshots, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
KINDS=(lumina pair)
J=(${KINDS[$((SLURM_ARRAY_TASK_ID % 2))]} ${SNAPS[$((SLURM_ARRAY_TASK_ID / 2))]})
[ -n "${J[1]}" ] || { echo "no snapshot for array task $SLURM_ARRAY_TASK_ID (SNAPS = ${SNAPS[*]})"; exit 1; }
df -h /orcd/data/mvogelsb/004 | tail -1
echo "${J[0]} snapshot ${J[1]}, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u void_heating_test.py ${J[0]} --snap ${J[1]}
echo "done $(date)"
