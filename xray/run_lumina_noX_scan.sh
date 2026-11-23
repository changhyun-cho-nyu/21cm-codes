#!/bin/bash
#SBATCH --job-name=noXscan
#SBATCH --output=logs/noXscan_%A_%a.out
#SBATCH --error=logs/noXscan_%A_%a.err
#SBATCH --array=0-223%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Lumina without X-ray heating, option 3, global mean, for the full snapshots without an H gas-cell scan
# (lumina_noX.py scan): each task reads 1/NTASK of the rows of one full snapshot (read only), keeps only the cells of
# the same 64 cubes as the H scans (seed 12345, 16^3 of the 320^3 pixels), and computes dT_b with and without X-rays
# for them (J_alpha of the render at the snapshot's z, production J in JDIR). Array = (number of SNAPS) x NTASK,
# at most 32 running. Writes .../xray/lumina_noX/noX_scan_NNN_TTT.npz (small). A task whose output exists is skipped.
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_lumina_noX_scan.sh     # 074 080 086 092 097 102 106, 32 tasks each
#   one snapshot: SNAPS="74" sbatch -p ou_mki --array=0-31 run_lumina_noX_scan.sh
#   EXTRA="..." is passed on to lumina_noX.py scan, e.g. J linear in z between two renders that have J (production J
#   exists only for some renders: 074/080/086/097 fall on renders 226/256/280/326, which have none):
#   SNAPS="74" EXTRA="--render-snap 224 --render-snap-b 227 --J-b <JDIR>/Jalpha_H_227.hdf5 --J-b <JDIR>/Jalpha_AGN_H_227.hdf5 --max-dz 0.05"
# then:  python lumina_noX.py merge --snap 74   ...   and  python lumina_noX.py summary
SNAPS=(${SNAPS:-74 80 86 92 97 102 106})
NTASK=${NTASK:-32}
JDIR=${JDIR:-/orcd/data/mvogelsb/004/chcho/Jalpha}
EXTRA=${EXTRA:-}
REN=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f lumina_noX.py ] && [ -f ../compute_Jalpha.c ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
[ "$(( ${#SNAPS[@]} * NTASK ))" = "${SLURM_ARRAY_TASK_COUNT:-$(( ${#SNAPS[@]} * NTASK ))}" ] || \
    { echo "ERROR: ${#SNAPS[@]} snapshots x NTASK=$NTASK, array of $SLURM_ARRAY_TASK_COUNT"; exit 1; }
SNAP=${SNAPS[$((SLURM_ARRAY_TASK_ID / NTASK))]}
TASK=$((SLURM_ARRAY_TASK_ID % NTASK))
df -h /orcd/data/mvogelsb/004 | tail -1
echo "scan: snapshot $SNAP task $TASK/$NTASK, J from $JDIR, extra args: ${EXTRA:-none}, $(date), commit $(git rev-parse --short HEAD), host $(hostname)"
python -u lumina_noX.py scan --snap $SNAP --task $TASK --ntask $NTASK --jdir $JDIR --render $REN $EXTRA
echo "done $(date)"
