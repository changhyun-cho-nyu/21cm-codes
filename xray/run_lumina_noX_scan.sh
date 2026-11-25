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
#   A snapshot whose render has no production J: compute J at that render itself (user, 2026-11-23: use the render
#   value, not an interpolation in z), into a directory of its own (not the production J directory: lightcone_J.py
#   reads every file there), with task/merged files in a subdirectory of their own:
#   (repo root) BAND=H ONLY="226 256 280 326" J_OUT=/orcd/data/mvogelsb/004/chcho/Jalpha_fullsnap sbatch -p ou_mki --array=0-7 run_Jalpha_prod.sh
#   (xray/)     SNAPS="74 80 86 97" JDIR=/orcd/data/mvogelsb/004/chcho/Jalpha_fullsnap EXTRA="--sub lumina_noX_Jexact" \
#                   sbatch -p ou_mki --array=0-127%32 run_lumina_noX_scan.sh
#   merge with --sub lumina_noX_Jexact; summary uses lumina_noX_Jexact first by default.
#   For comparison only (EXTRA is passed on to lumina_noX.py scan): J linear in z between the neighbouring renders that
#   have production J, into the default subdirectory, e.g.
#   SNAPS="74" EXTRA="--render-snap 224 --render-snap-b 227 --J-b <JDIR>/Jalpha_H_227.hdf5 --J-b <JDIR>/Jalpha_AGN_H_227.hdf5 --max-dz 0.05"
#   (074/080/086/097: the two agree to <= 3e-5 in the cube sums, NOTES.md step 8).
#   WHOLE BOX (user, 2026-11-24): EXTRA="--whole-box --sub lumina_noX_box" draws T_noX and sums dT_b for every cell of
#   the snapshot (task files noX_box_*, cube sums kept for comparison); J as for the H scans (production J at the render,
#   Jalpha_fullsnap at 226/256/280/326, 024 linear in z between renders 73 and 74). Three submissions (one JDIR each):
#   (NTASK=64, see Cost below; the array must be (number of SNAPS) x 64)
#   NTASK=64 SNAPS="36 48 57 66 92 102" EXTRA="--whole-box --sub lumina_noX_box" sbatch -p ou_mki --mem=48G --array=0-383%32 run_lumina_noX_scan.sh
#   NTASK=64 SNAPS="74 80 86 97" JDIR=/orcd/data/mvogelsb/004/chcho/Jalpha_fullsnap EXTRA="--whole-box --sub lumina_noX_box" \
#       sbatch -p ou_mki --mem=48G --array=0-255%32 run_lumina_noX_scan.sh
#   NTASK=64 SNAPS="24" EXTRA="--whole-box --sub lumina_noX_box --render-snap 73 --render-snap-b 74 --J-b <JDIR>/Jalpha_H_74.hdf5
#       --J-b <JDIR>/Jalpha_AGN_H_74.hdf5 --max-dz 0.05" sbatch -p ou_mki --mem=48G --array=0-63 run_lumina_noX_scan.sh
#   merge with --sub lumina_noX_box; summary reads lumina_noX_box first by default
#   Cost (2026-11-24): with the bisection T_S solver the first whole-box tasks projected ~54 h (over the 48 h limit;
#   cancelled). The scan now uses tb_physics.solve_ts_wf_fast (--solver fast, default; same T_S to ~1e-15, 6-8x faster
#   in dT_b); expected ~14 h per task at NTASK=32, so use NTASK=64 (~7 h): arrays 0-383 / 0-255 / 0-63.
#   T and T_S (2026-11-24, figure in global_noX.ipynb): the task files also hold H I-weighted sums of 1/T and
#   1 - T_g/T_S for the X and no-X states (box_ts / ts); merge STOPs on a mix of task files with and without them
#   (delete the old ones). The pair's matching values: run_pair_T_TS.sh.
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
