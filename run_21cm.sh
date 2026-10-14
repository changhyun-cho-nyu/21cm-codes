#!/bin/bash
#SBATCH --job-name=PS_21cm
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/PS_21cm_%j.out
#SBATCH --error=logs/PS_21cm_%j.err
#
# 21 cm delta T_b maps (RIN_NNN.hdf5) and power spectra (PS_NNN.out)
#   1. bash run_21cm.sh compile     (needs the MPI + FFTW + HDF5 modules: mpicc, fftw3-mpi.h, parallel hdf5.h)
#   2. mkdir -p logs && sbatch run_21cm.sh   (resumable: resubmit to finish snapshots a 24 h job did not reach)
# Spin-temperature inputs and the signal convention are compile-time options of PS_21cm_mpi.c:
#   EXTRA_CFLAGS="-DTS_VOLUME_WEIGHTED=1" (volume-weighted T, fractions)
#   EXTRA_CFLAGS="-DH_REDSHIFT_SPACE=0"   (real-space H I signal: the choice made here, 2026-10-14, so that every
#     factor of delta T_b belongs to the same gas and H matches the real-space He maps. Empty EXTRA_CFLAGS gives
#     the code default: SpectralDensityHI (redshift space) x the spin factor of the real-space pixel)

# -------------------------
# Environment
# -------------------------
FFTW_PREFIX=/orcd/software/core/001/spack/pkg/fftw/3.3.10/2qziucy
HDF5_PREFIX=/orcd/software/community/001/spack/pkg/hdf5/1.14.3/suffu3v
EXTRA_CFLAGS="-DH_REDSHIFT_SPACE=0"
export LD_LIBRARY_PATH=$FFTW_PREFIX/lib:$HDF5_PREFIX/lib:$LD_LIBRARY_PATH
export HDF5_PLUGIN_PATH=/home/chcho/.local/lib/python3.12/site-packages/hdf5plugin/plugins
export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ "$1" = "compile" ]; then
    mkdir -p bin
    INC=""; LIBS="-lfftw3_mpi -lfftw3 -lhdf5 -lm"
    for d in "$FFTW_PREFIX" "$HDF5_PREFIX"; do
        INC="$INC -I$d/include"
        for l in "$d/lib" "$d/lib64"; do [ -d "$l" ] && LIBS="-L$l -Wl,-rpath,$l $LIBS"; done
    done
    # the code uses the HDF5 1.6 API names (H5Dopen/H5Gopen with 2 arguments)
    mpicc -O3 -fopenmp -DH5_USE_16_API $EXTRA_CFLAGS $INC PS_21cm_mpi.c functions.c collisions.c -o bin/PS_21cm_mpi $LIBS
    echo "built bin/PS_21cm_mpi (EXTRA_CFLAGS: ${EXTRA_CFLAGS:-none})   commit $(git rev-parse --short HEAD)"
    exit 0
fi
[ -x bin/PS_21cm_mpi ] || { echo "run 'bash $0 compile' first"; exit 1; }

# -------------------------
# Paths
# -------------------------
INPUT_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640/"
INPUT_NEW_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640/"
J_DIR=/orcd/data/mvogelsb/004/chcho/Jalpha            # run_Jalpha_prod.sh outputs (BAND=H)
OUT_DIR=/orcd/data/mvogelsb/004/chcho/Tb_maps_prod/H_real # production maps, real space (Tb_maps/: test-J maps;
                                                          # Tb_maps_prod/H: earlier redshift-space maps, if any)
mkdir -p "$OUT_DIR"

# -------------------------
# Snapshots List
# -------------------------
SNAPS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
       24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
       48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
       72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
       96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
       120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
       144 145 146 147 148 149 150 151 152 153 154 155
       156 159 162 165 168 171 174 177 181 184 187 190 193 196 200 203 209 212 215 218 220 224 227 230
       234 237 242 250 255 260 264 269 273 278 283 289 295 300 304 309 314 318 320 321 323 325 328 331
       332 334 337 338 340 343 346 349 350 353 356 359 361 364 366 370 373 376 379 382 385 389 392 395
       398 400 403 405 406 408 411 414 417 420 423 427 430 431 433 434 436 437 440 443 444 446 448 450
       452 456 460 463 464 467 471 475 476 478 481 482 484 487 490 491 493 496 500 501 504 507 510 514
       518 525 529 535 540 545 549 554 558 565 569 574 578 582 585 589 592 596 610 620 630 640 650 660
       670 680 690 700 705)

nfail=0
echo "commit $(git rev-parse --short HEAD), host $(hostname), $(date)"
for SNAP in "${SNAPS[@]}"; do
    echo "=================================================="
    echo "Snapshot $SNAP"
    if [ "$SNAP" -le 428 ]; then DIR=$INPUT_DIR; else DIR=$INPUT_NEW_DIR; fi
    JALPHA_FILE="$J_DIR/Jalpha_H_${SNAP}.hdf5"; JALPHA_AGN_FILE="$J_DIR/Jalpha_AGN_H_${SNAP}.hdf5"
    for f in "$JALPHA_FILE" "$JALPHA_AGN_FILE"; do
        [ -f "$f" ] || { echo "WARNING: $f not found, skipping snapshot $SNAP"; continue 2; }
    done
    # resumable: a snapshot with both outputs is skipped; outputs are written as .part and renamed on success
    if [ -f "$OUT_DIR/PS_${SNAP}.out" ] && [ -f "$OUT_DIR/RIN_${SNAP}.hdf5" ]; then echo "done already, skipped"; continue; fi
    rm -f "$OUT_DIR/PS_${SNAP}.out.part" "$OUT_DIR/RIN_${SNAP}.hdf5.part"
    if mpirun -np 1 bin/PS_21cm_mpi $SNAP $DIR $OUT_DIR/PS_${SNAP}.out.part $OUT_DIR/RIN_${SNAP}.hdf5.part 3 "$JALPHA_FILE" "$JALPHA_AGN_FILE"; then
        mv "$OUT_DIR/PS_${SNAP}.out.part" "$OUT_DIR/PS_${SNAP}.out"; mv "$OUT_DIR/RIN_${SNAP}.hdf5.part" "$OUT_DIR/RIN_${SNAP}.hdf5"
        echo "finished snapshot $SNAP"
    else
        echo "FAILED snapshot $SNAP"; nfail=$((nfail + 1))
    fi
done
echo "done, $nfail failed, $(date)"
[ $nfail -eq 0 ]
