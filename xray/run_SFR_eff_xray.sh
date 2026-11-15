#!/bin/bash
#SBATCH --job-name=SFRxray
#SBATCH --output=logs/SFRxray_%A_%a.out
#SBATCH --error=logs/SFRxray_%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Effective SFR (H band, 300 Myr window) of the SonjaXrayPaper runs from their assembled 128^3 renders
# (make_SFR_eff.py, as for Lumina): task 0 = withoutXrays, 1 = withXrays, each with its own Z_weights.txt.
# Needs all 405 renders assembled (run_assemble_xray.sh) and the weights (run_Z_xray.sh).
#   cd xray && mkdir -p logs && sbatch -p ou_mki run_SFR_eff_xray.sh
# Output: $BASE/<run>/SFR_eff/H/All/All_NNN.hdf5 (~8 MB each).

BASE=/orcd/data/mvogelsb/004/chcho/xray
RUNS=(withoutXrays withXrays)
NREN=405

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
[ -f ../compute_Jalpha.c ] && [ -f assemble_renders.py ] || { echo "submit from the xray/ directory of the repository"; exit 1; }
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
REN=$BASE/$RUN/ren_128
OUT=$BASE/$RUN/SFR_eff
W=$BASE/$RUN/Z/Z_weights.txt
case "$(realpath -m $OUT)/" in /orcd/data/mvogelsb/004/chcho/*) ;; *) echo "refusing to write to $OUT"; exit 1;; esac
[ -f "$W" ] || { echo "missing $W (run_Z_xray.sh first)"; exit 1; }
N=$(ls $REN/All/All_*.hdf5 2>/dev/null | wc -l)
[ "$N" -eq "$NREN" ] || { echo "$N of $NREN renders assembled in $REN: finish run_assemble_xray.sh first"; exit 1; }
mkdir -p "$OUT"
echo "run $RUN, host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
python -u ../make_SFR_eff.py --dirs $REN --n-snaps $NREN --bands H --weights $W --out $OUT --planes 32
echo "done $(date)"
du -sh $OUT/H
