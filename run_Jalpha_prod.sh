#!/bin/bash
#SBATCH --job-name=Jalpha
#SBATCH --output=logs/Jprod_%A_%a.out
#SBATCH --error=logs/Jprod_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=FAIL
#
# Production J_alpha for the T_b snapshot lists, two codes per snapshot:
#   BAND=H : H stars + AGN-H   for the SNAPS of run_21cm.sh -> $J/Jalpha_H_NNN.hdf5, Jalpha_AGN_H_NNN.hdf5
#   BAND=He: He stars + AGN-He for the SNAPS of run_3cm.sh  -> $J/Jalpha_He_NNN[_TR_TAG].hdf5, Jalpha_AGN_He_...
# Uses the binaries of 'bash run_Jalpha_test.sh compile' (for He with absorption, compile there with
# TRANS_DIR/TR_TAG set). Outputs that exist are skipped; each is written as .part and renamed when the
# code succeeds, so a killed task never leaves a file that looks finished. From the repository directory:
#   1. bash run_Jalpha_prod.sh count                       # prints the sbatch line for the current BAND
#   2. mkdir -p logs && sbatch --array=0-<N-1>%10 run_Jalpha_prod.sh
#      (%10: at most 10 tasks at once; resubmitting the same line later redoes only the missing outputs)
#   3. bash run_Jalpha_prod.sh status                      # which outputs exist / are missing
# Then set J_DIR=$J in run_21cm.sh (and run_3cm.sh, with its TR_TAG = the TR_TAG here).

#BAND=H                     # H or He
#TR_TAG=                    # BAND=He only: e.g. tr_sfmask (bin/Jalpha_He_<TR_TAG>); empty = no IGM absorption
BAND=He
TR_TAG=tr_sfmask

OUT=/orcd/data/mvogelsb/004/chcho/SFR_eff                       # make_SFR_eff.py output
J=/orcd/data/mvogelsb/004/chcho/Jalpha                          # production output directory
R=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640
RB=/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640
SED=agn_sed_full.hdf5                                           # in the repository
MIN_FREE_GB=5                                                   # each output is ~1.1 GB (640^3 float32)
CONDA_ENV=jupyter_env      # has hdf5plugin: its filter plugins let the C codes read the compressed renders

source ~/.bashrc
# module load ...        # <- the same modules as your usual J_alpha jobs (MPI, FFTW, HDF5)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

case $BAND in
    H)  LIST=run_21cm.sh; CODES=(H AGN_H);   TR="" ;;
    He) LIST=run_3cm.sh;  CODES=(He AGN_He); TR=${TR_TAG:+_$TR_TAG} ;;
    *)  echo "BAND must be H or He"; exit 1 ;;
esac
# the uncommented SNAPS=( ... ) block of the T_b script
SNAPS=($(sed -n '/^SNAPS=(/,/)/p' $LIST | sed 's/#.*//' | tr -d 'SNAPS=()'))
NT=$(( ${#SNAPS[@]} * ${#CODES[@]} ))
outfile() {   # $1 snap, $2 code
    case $2 in
        H)      echo "$J/Jalpha_H_$1.hdf5" ;;
        AGN_H)  echo "$J/Jalpha_AGN_H_$1.hdf5" ;;
        He)     echo "$J/Jalpha_He_$1$TR.hdf5" ;;
        AGN_He) echo "$J/Jalpha_AGN_He_$1$TR.hdf5" ;;
    esac
}

if [ "$1" = "count" ] || [ "$1" = "status" ]; then
    echo "BAND=$BAND${TR:+ ($TR_TAG)}: ${#SNAPS[@]} snapshots of $LIST (${SNAPS[0]} .. ${SNAPS[${#SNAPS[@]}-1]}) x ${#CODES[@]} codes = $NT tasks"
    if [ "$1" = "count" ]; then
        echo "submit:  mkdir -p logs && sbatch --array=0-$((NT - 1))%10 run_Jalpha_prod.sh"
        echo "disk:    ~$(( NT * 11 / 10 )) GB in $J;  $(df -h --output=avail "$(dirname "$J")" | tail -1 | tr -d ' ') free"
    else
        miss=0
        for ((i = 0; i < NT; i++)); do
            f=$(outfile ${SNAPS[$((i / ${#CODES[@]}))]} ${CODES[$((i % ${#CODES[@]}))]})
            [ -f "$f" ] || { echo "missing (task $i): $f"; miss=$((miss + 1)); }
        done
        echo "$((NT - miss)) of $NT outputs exist"
    fi
    exit 0
fi

[ -n "$SLURM_ARRAY_TASK_ID" ] || { echo "submit with sbatch --array (see 'bash $0 count')"; exit 1; }
i=$SLURM_ARRAY_TASK_ID
[ "$i" -lt "$NT" ] || { echo "task $i >= $NT tasks: nothing to do"; exit 0; }
SNAP=${SNAPS[$((i / ${#CODES[@]}))]}
CODE=${CODES[$((i % ${#CODES[@]}))]}
F=$(outfile $SNAP $CODE)
echo "task $i: snap $SNAP, code $CODE -> $F, host $(hostname), $(date), commit $(git rev-parse --short HEAD)"
[ -f "$F" ] && { echo "exists, skipped"; exit 0; }

if [ -z "${HDF5_PLUGIN_PATH:-}" ]; then
    conda activate "$CONDA_ENV" 2>/dev/null || true
    HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)' 2>/dev/null) || HDF5_PLUGIN_PATH=
fi
[ -n "$HDF5_PLUGIN_PATH" ] || { echo "HDF5_PLUGIN_PATH not set and hdf5plugin not found in $CONDA_ENV"; exit 1; }
export HDF5_PLUGIN_PATH
case $CODE in
    H) B=bin/Jalpha_H ;;  AGN_H) B=bin/Jalpha_AGN ;;  He) B=bin/Jalpha_He$TR ;;  AGN_He) B=bin/Jalpha_AGN$TR ;;
esac
[ -x "$B" ] || { echo "missing $B: run 'bash run_Jalpha_test.sh compile' first${TR:+ (with TRANS_DIR set and TR_TAG=$TR_TAG)}"; exit 1; }
[ -f "$SED" ] && [ -d "$OUT/H/All" ] && [ -d "$OUT/He/StarFormationRate" ] || { echo "check SED and OUT paths"; exit 1; }
mkdir -p "$J"
FREE_GB=$(df --output=avail -BG "$J" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || { echo "only ${FREE_GB} GB free on $J"; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
P="$F.part"; rm -f "$P"
case $CODE in
    H)      mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH $B $SNAP "$OUT/H" "$OUT/H" "$P" ;;
    He)     mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH $B $SNAP "$P" ;;
    AGN_H)  mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH $B $SNAP "$R/LuminosityAGN" "$P" "$SED" "$RB/LuminosityAGN" LuminosityAGN H ;;
    AGN_He) mpirun -np $SLURM_NTASKS -x HDF5_PLUGIN_PATH $B $SNAP "$R/LuminosityAGN" "$P" "$SED" "$RB/LuminosityAGN" LuminosityAGN He ;;
esac
mv "$P" "$F"
echo "done $(date)"
