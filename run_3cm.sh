#!/bin/bash
#SBATCH --job-name=PS_3cm
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/PS_3cm_%j.out
#SBATCH --error=logs/PS_3cm_%j.err
#
# 3.46 cm (He II) delta T_b maps and power spectra
#   1. bash run_3cm.sh compile     (needs the MPI + FFTW + HDF5 modules: mpicc, fftw3-mpi.h, parallel hdf5.h)
#   2. mkdir -p logs && sbatch run_3cm.sh   (resumable: resubmit to finish snapshots a 24 h job did not reach)
# Spin-temperature inputs and (He) the signal convention are compile-time options of PS_3cm_mpi.c:
#   EXTRA_CFLAGS="-DTS_VOLUME_WEIGHTED=1" (volume-weighted T, fractions)
#   EXTRA_CFLAGS="-DHE_REDSHIFT_SPACE=0" (real-space He signal: the choice made here; the renders have no
#     redshift-space He II. Empty EXTRA_CFLAGS gives the code default, HeII_Fraction x SpectralDensity)
# HE_RS_DIR: directory of remap_rsd.py output (SpectralDensityHeII); when set, the He signal is that
#   approximate redshift-space He II density (overrides HE_REDSHIFT_SPACE). Recompile after changing it.

# -------------------------
# Environment
# -------------------------
FFTW_PREFIX=/orcd/software/core/001/spack/pkg/fftw/3.3.10/2qziucy
HDF5_PREFIX=/orcd/software/community/001/spack/pkg/hdf5/1.14.3/suffu3v
EXTRA_CFLAGS="-DHE_REDSHIFT_SPACE=0"
HE_RS_DIR=""               # e.g. /orcd/data/mvogelsb/004/chcho/SpectralDensityHeII_root
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
    [ -n "$HE_RS_DIR" ] && EXTRA_CFLAGS="$EXTRA_CFLAGS -DHE_RS_DIR=\"$HE_RS_DIR\""
    mpicc -O3 -fopenmp -DH5_USE_16_API $EXTRA_CFLAGS $INC PS_3cm_mpi.c functions.c collisions.c -o bin/PS_3cm_mpi $LIBS
    echo "built bin/PS_3cm_mpi (EXTRA_CFLAGS: ${EXTRA_CFLAGS:-none})   commit $(git rev-parse --short HEAD)"
    exit 0
fi
[ -x bin/PS_3cm_mpi ] || { echo "run 'bash $0 compile' first"; exit 1; }

# -------------------------
# Paths
# -------------------------
INPUT_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640/"
INPUT_NEW_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640/"
J_DIR=/orcd/data/mvogelsb/004/chcho/Jalpha            # run_Jalpha_prod.sh outputs (BAND=He)
TR_TAG=tr_sfmask           # attenuated maps Jalpha_*He_NNN_<TR_TAG>.hdf5 (run_Jalpha_test.sh TR_TAG); empty = no absorption
OUT_DIR=/orcd/data/mvogelsb/004/chcho/Tb_maps_prod/He     # production maps (Tb_maps/ holds the test-J maps)
mkdir -p "$OUT_DIR"

# -------------------------
# Snapshots List
# -------------------------
SNAPS=(156 159 162 165 168 171 174 177 181 184 187 190 193 196 200 203 209 212 215 218 220 224 227 230
       234 237 242 250 255 260 264 269 273 278 283 289 295 300 304 309 314 318 320 321 323 325 328 331
       332 334 337 338 340 343 346 349 350 353 356 359 361 364 366 370 373 376 379 382 385 389 392 395
       398 400 403 405 406 408 411 414 417 420 423 425 427 430 431 433 434 436 437 440 443 444 446 448
       450 452 454 456 458 460 463 464 467 469 471 473 475 476 478 481 482 484 487 490 491 493 496 498
       500 501 504 507 510 512 514 516 518 520 523 525 527 529 532 535 538 540 542 545 547 549 552 554
       556 558 560 563 565 567 569 572 574 576 578 580 582 585 587 589 592 594 596 599 602 604 607 610
       612 615 618 620 622 625 628 630 632 635 638 640 642 645 648 650 652 655 658 660 662 665 668 670
       672 675 678 680 682 685 688 690 692 695 698 700 705)
#(156 159 162 165 168 171 174 177 181 184 187 190 193 196 200 203 209 255 283 289 295 300 304 309
#       314 318 320 321 323 325 328 331 332 334 337 338 340 343 346 349 350 353 356 359 361 364 366 370
#       373 376 379 382 385 389 392 395 398 400 403 405 406 408 411 414 417 420 423 427 430 431 433 434
#       436 437 440 443 444 446 448 450 452 456 460 463 464 467 471 475 476 478 481 482 484 487 490 491
#       493 496 500 501 504 507 510 514 518 525 529 535 540 545 549 554 558 565 569 574 578 582 585 589
#       592 596 610 620 630 640 650 660 670 680 690 700 705)

nfail=0
echo "commit $(git rev-parse --short HEAD), host $(hostname), $(date)"
for SNAP in "${SNAPS[@]}"; do
    echo "=================================================="
    echo "Snapshot $SNAP"
    if [ "$SNAP" -le 428 ]; then DIR=$INPUT_DIR; else DIR=$INPUT_NEW_DIR; fi
    S=""; [ -n "$TR_TAG" ] && S="_$TR_TAG"
    JALPHA_FILE="$J_DIR/Jalpha_He_${SNAP}${S}.hdf5"; JALPHA_AGN_FILE="$J_DIR/Jalpha_AGN_He_${SNAP}${S}.hdf5"
    for f in "$JALPHA_FILE" "$JALPHA_AGN_FILE"; do
        [ -f "$f" ] || { echo "WARNING: $f not found, skipping snapshot $SNAP"; continue 2; }
    done
    # resumable: a snapshot with both outputs is skipped; outputs are written as .part and renamed on success
    if [ -f "$OUT_DIR/PS_${SNAP}.out" ] && [ -f "$OUT_DIR/RIN_${SNAP}.hdf5" ]; then echo "done already, skipped"; continue; fi
    rm -f "$OUT_DIR/PS_${SNAP}.out.part" "$OUT_DIR/RIN_${SNAP}.hdf5.part"
    if mpirun -np 1 bin/PS_3cm_mpi $SNAP $DIR $OUT_DIR/PS_${SNAP}.out.part $OUT_DIR/RIN_${SNAP}.hdf5.part 3 "$JALPHA_FILE" "$JALPHA_AGN_FILE"; then
        mv "$OUT_DIR/PS_${SNAP}.out.part" "$OUT_DIR/PS_${SNAP}.out"; mv "$OUT_DIR/RIN_${SNAP}.hdf5.part" "$OUT_DIR/RIN_${SNAP}.hdf5"
        echo "finished snapshot $SNAP"
    else
        echo "FAILED snapshot $SNAP"; nfail=$((nfail + 1))
    fi
done
echo "done, $nfail failed, $(date)"
[ $nfail -eq 0 ]
