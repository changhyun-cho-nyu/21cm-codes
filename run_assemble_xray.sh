#!/bin/bash
#SBATCH --job-name=asmXray
#SBATCH --output=logs/asmXray_%A_%a.out
#SBATCH --error=logs/asmXray_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# 128^3 render cubes of the SonjaXrayPaper runs in the pipeline layout (assemble_renders.py):
# array tasks 0-7 = withoutXrays, 8-15 = withXrays; task i of a run does renders i, i+8, ... (405 renders per run).
# Reads Oliver's directory only; writes ~130 MB per render (~105 GB for both runs) under OUT_BASE.
#   mkdir -p logs && sbatch run_assemble_xray.sh        (resumable: existing All files are skipped)
#   test first:  sbatch --array=0,8 --export=ALL,SNAPS=309 run_assemble_xray.sh
# then, when all tasks are done:
#   for r in withoutXrays withXrays; do python assemble_renders.py report --out $OUT_BASE/$r/ren_128; done

SRC_BASE=/nfs/mvogelsblab002/Users/ozier/SonjaXrayPaper
OUT_BASE=/orcd/data/mvogelsb/004/chcho/xray
RUNS=(withoutXrays withXrays)
NTASK=8

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN=${RUNS[$((SLURM_ARRAY_TASK_ID / NTASK))]}
TASK=$((SLURM_ARRAY_TASK_ID % NTASK))
OUT=$OUT_BASE/$RUN/ren_128
df -h /orcd/data/mvogelsb/004 | tail -1
echo "run $RUN task $TASK of $NTASK, $(date), commit $(git rev-parse --short HEAD)"
python -u assemble_renders.py --src $SRC_BASE/$RUN/output/movie/renderings --out $OUT \
    --task $TASK --ntask $NTASK ${SNAPS:+--snaps $SNAPS}
echo "done $(date)"
