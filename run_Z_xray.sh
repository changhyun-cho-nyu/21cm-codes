#!/bin/bash
#SBATCH --job-name=Zxray
#SBATCH --output=logs/Zxray_%A_%a.out
#SBATCH --error=logs/Zxray_%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Young-star metallicity weights of the SonjaXrayPaper runs (make_Z_snapshots.py) from their 14 full snapshots
# (024 ... 116, z = 15 ... 5): task 0 = withoutXrays, 1 = withXrays; one weights file per run, read by that run's
# make_SFR_eff.py and compute_Jalpha.c. Reads the snapshots only; writes under OUT_BASE/<run>/Z.
#   mkdir -p logs && sbatch run_Z_xray.sh

SRC_BASE=/nfs/mvogelsblab002/Users/ozier/SonjaXrayPaper
OUT_BASE=/orcd/data/mvogelsb/004/chcho/xray
RUNS=(withoutXrays withXrays)

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
OUT=$OUT_BASE/$RUN/Z
case "$(realpath -m $OUT)/" in /orcd/data/mvogelsb/004/chcho/*) ;; *) echo "refusing to write to $OUT"; exit 1;; esac
mkdir -p "$OUT"
echo "run $RUN, $(date), commit $(git rev-parse --short HEAD)"
python -u make_Z_snapshots.py --runs $RUN=$SRC_BASE/$RUN/output \
    --out-csv $OUT/Z_snapshots.csv --out-weights $OUT/Z_weights.txt --out-history $OUT/Z_history.txt
echo "done $(date)"
