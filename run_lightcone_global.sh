#!/bin/bash
#SBATCH --job-name=coneGlob
#SBATCH --output=logs/coneGlob_%j.out
#SBATCH --error=logs/coneGlob_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Global 21 cm history along the light cone, per slice (lightcone_global.py: <dT_b>, rms, T_S, T_K, x_HII, x_HeIII,
# x_alpha, x_c, J), recomputed with the settings of CONE_DIR/lightcone_dTb_H.hdf5 (SPIN_MODEL 1) and checked against
# it; then the figures (plot_lightcone_global.py; the PS panel only if PS_FILE exists). Resumes: resubmit with the
# same OUT_DIR after a time-out (the SPIN_MODEL 1 cone itself took 10.6 h, job 29356253).
#   mkdir -p logs && sbatch run_lightcone_global.sh
#   EXTRA="--zrange 7.4 7.6" OUT_DIR=/orcd/data/mvogelsb/004/chcho/lightcone_wf/global_test sbatch run_lightcone_global.sh   # a test band
# Power-spectrum evolution (contiguous bands over the whole cone, cone only), before or after this job:
#   CONE_DIR=/orcd/data/mvogelsb/004/chcho/lightcone_wf OUT_DIR=/orcd/data/mvogelsb/004/chcho/lightcone_wf/ps_tiles \
#     LINES=H EXTRA="--tile --no-maps" sbatch -p ou_mki run_lightcone_ps.sh
# Output: OUT_DIR/lightcone_global_H.csv (5896 rows), OUT_DIR/figs/lightcone_global.png, lightcone_ps_evolution.{png,csv}.

DATA=/orcd/data/mvogelsb/004/chcho
CONE_DIR=${CONE_DIR:-$DATA/lightcone_wf}
OUT_DIR=${OUT_DIR:-$CONE_DIR/global}
PS_FILE=${PS_FILE:-$CONE_DIR/ps_tiles/lightcone_ps_H.hdf5}
EXTRA=${EXTRA:-}

source ~/.bashrc
conda activate jupyter_env
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p "$OUT_DIR"
echo "light-cone global history: cone $CONE_DIR, out $OUT_DIR, extra '$EXTRA', $(date), commit $(git rev-parse --short HEAD)"
python -u lightcone_global.py --cone-file $CONE_DIR/lightcone_dTb_H.hdf5 --out $OUT_DIR/lightcone_global_H.csv $EXTRA
PS_ARG=""
[ -f "$PS_FILE" ] && PS_ARG="--ps $PS_FILE"
python -u plot_lightcone_global.py --global-csv $OUT_DIR/lightcone_global_H.csv $PS_ARG \
    --ion $DATA/ion_history/ionization_history.csv --outdir $OUT_DIR/figs
echo "done $(date)"
