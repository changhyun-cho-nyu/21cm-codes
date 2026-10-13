#!/bin/bash
#SBATCH --job-name=PS_3cm_He
#SBATCH --partition=sched_mit_mvogelsb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=PS_3cm_He_%j.out
#SBATCH --error=PS_3cm_He_%j.err

# -------------------------
# Environment
# -------------------------
export LD_LIBRARY_PATH=/home/chcho/libs/sundials/lib64:/home/chcho/libs/gmp/lib:/orcd/software/community/001/spack/pkg/gsl/2.7.1/y74b3m5/lib:/orcd/software/core/001/spack/pkg/fftw/3.3.10/2qziucy/lib:/orcd/software/community/001/spack/pkg/hdf5/1.14.3/suffu3v/lib:$LD_LIBRARY_PATH
export HDF5_PLUGIN_PATH=/home/chcho/.local/lib/python3.12/site-packages/hdf5plugin/plugins
echo "HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH"

export OMP_NUM_THREADS=1

# -------------------------
# Move to working directory
# -------------------------
cd /orcd/data/mvogelsb/004/chcho/cell_by_cell/helium

# -------------------------
# Paths
# -------------------------
INPUT_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640/"
INPUT_NEW_DIR="/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_below_z_4p75_new/postprocessing/ren_640/"
OUT_DIR="/orcd/data/mvogelsb/004/chcho/cell_by_cell/helium/Tb_He"
JALPHA_DIR="/orcd/data/mvogelsb/004/chcho/cell_by_cell/helium/J_alpha_He"
JALPHA_AGN_DIR="/orcd/data/mvogelsb/004/chcho/compute_J_alpha_H/J_alpha_AGN"

mkdir -p $OUT_DIR

# -------------------------
# Snapshots List
# -------------------------
SNAPS=(425 454 458 469 473 498 512 516 520 523 527 532 538 542 547 552 556 560 563 567 572 576 580 587 594 599 602 604 607 612 615 618 622 625 628 632 635 638 642 645 648 652 655 658 662 665 668 672 675 678 682 685 688 692 695 698)
#(156 159 162 165 168 171 174 177 181 184 187 190 193 196 200 203 209 255 283 289 295 300 304 309
#       314 318 320 321 323 325 328 331 332 334 337 338 340 343 346 349 350 353 356 359 361 364 366 370
#       373 376 379 382 385 389 392 395 398 400 403 405 406 408 411 414 417 420 423 427 430 431 433 434
#       436 437 440 443 444 446 448 450 452 456 460 463 464 467 471 475 476 478 481 482 484 487 490 491
#       493 496 500 501 504 507 510 514 518 525 529 535 540 545 549 554 558 565 569 574 578 582 585 589
#       592 596 610 620 630 640 650 660 670 680 690 700 705)

# -------------------------
# Sequential Loop Execution
# -------------------------
for SNAP in "${SNAPS[@]}"; do
    echo "=================================================="
    echo "Starting processing for Snapshot: $SNAP"
    echo "=================================================="

    if [ "$SNAP" -le 428 ]; then
        DIR=$INPUT_DIR
    else
        DIR=$INPUT_NEW_DIR
    fi

    JALPHA_FILE="$JALPHA_DIR/Jalpha_He_${SNAP}.hdf5"
    JALPHA_AGN_FILE="$JALPHA_AGN_DIR/Jalpha_AGN_He_${SNAP}.hdf5"

    if [ ! -f "$JALPHA_FILE" ]; then
        echo "WARNING: Jalpha file not found, skipping snapshot $SNAP: $JALPHA_FILE"
        continue
    fi
    if [ ! -f "$JALPHA_AGN_FILE" ]; then
        echo "WARNING: Jalpha AGN file not found, skipping snapshot $SNAP: $JALPHA_AGN_FILE"
        continue
    fi

    echo "Running snapshot $SNAP | dir $DIR"
    echo "Jalpha file:     $JALPHA_FILE"
    echo "Jalpha AGN file: $JALPHA_AGN_FILE"

    mpirun -np 1 ./PS_3cm_mpi \
        $SNAP \
        $DIR \
        $OUT_DIR/PS_${SNAP}.out \
        $OUT_DIR/RIN_${SNAP}.hdf5 \
        3 \
        $JALPHA_FILE \
        $JALPHA_AGN_FILE

    echo "Finished processing Snapshot: $SNAP"
    echo ""
done

echo "Loop complete! All snapshots processed sequentially."
