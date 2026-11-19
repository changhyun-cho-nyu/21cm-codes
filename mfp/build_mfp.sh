#!/bin/bash
# Build the two MFP executables (volume-weighted, 64-bit histograms) from mfp/lumina-mfp-hist-grid.cc, once, on the
# login node, before submitting run_mfp.sh:
#   bash mfp/build_mfp.sh
#   -> mfp/lumina-mfp-H   (x_HII   = HII_VolumeFraction)
#   -> mfp/lumina-mfp-He  (x_HeIII = HeIII_VolumeFraction / y)
set -eo pipefail
cd "$(dirname "$0")"
module load gcc openmpi/4.1.4 hdf5 >/dev/null 2>&1 || true
# the MPI wrapper: the hdf5 module is an MPI build (H5public.h includes mpi.h). Not $CXX, which the gcc module sets to
# plain g++ (that gave "mpi.h: No such file or directory", 2026-11-19); override with MPICXX=... if needed.
CXX=${MPICXX:-mpicxx}
command -v $CXX >/dev/null || { echo "no $CXX: run 'module load gcc openmpi/4.1.4 hdf5' first"; exit 1; }
echo "compiler: $(command -v $CXX)"
HDF5=${HDF5:-/orcd/software/community/001/spack/pkg/hdf5/1.14.3/suffu3v}
$CXX -fopenmp -std=c++14 -O3 -I$HDF5/include lumina-mfp-hist-grid.cc -o lumina-mfp-H -L$HDF5/lib -Wl,-rpath,$HDF5/lib -lhdf5 -lz
$CXX -fopenmp -std=c++14 -O3 -DLINE_HE -I$HDF5/include lumina-mfp-hist-grid.cc -o lumina-mfp-He -L$HDF5/lib -Wl,-rpath,$HDF5/lib -lhdf5 -lz
echo "built $(pwd)/lumina-mfp-H and lumina-mfp-He from commit $(git rev-parse --short HEAD)"
