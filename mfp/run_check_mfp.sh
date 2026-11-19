#!/bin/bash
#SBATCH --job-name=mfpCheck
#SBATCH --output=logs/mfpCheck_%j.out
#SBATCH --error=logs/mfpCheck_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=ou_mki
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# check_mfp_outputs.py on both volume-weighted MFP directories: which field made each file (ray count = 192 x
# N(x > 0.5) exactly), wrapped 32-bit bins, medians; with RERUN set, also bin-by-bin against the 64-bit reruns there.
#   mkdir -p logs && sbatch mfp/run_check_mfp.sh
#   RERUN=/orcd/data/mvogelsb/004/chcho/mfp_rerun sbatch mfp/run_check_mfp.sh
O=${O:-/orcd/data/mvogelsb/004/chcho/mfp-bubbles-thesan/output}
source ~/.bashrc
conda activate jupyter_env
cd "$SLURM_SUBMIT_DIR"
echo "$(date), commit $(git rev-parse --short HEAD)"
st=0
for L in H He; do
    D=$O/LUMINA_$([ $L = H ] && echo hydrogen || echo helium)_Vol
    C=${RERUN:+--compare-dir $RERUN/$(basename $D)}
    echo; echo "=== $L: $D ${C}"
    python -u mfp/check_mfp_outputs.py --line $L --dir $D $C || st=1
done
echo "done $(date), status $st"
exit $st
