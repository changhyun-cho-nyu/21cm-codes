#!/bin/bash
#SBATCH --job-name=floorbox
#SBATCH --output=logs/floorbox_%A_%a.out
#SBATCH --error=logs/floorbox_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=orcd_ug_pi_mki_mvogelsb_all
#SBATCH --mail-user=chcho@mit.edu
#SBATCH --mail-type=END,FAIL
#
# Floor-box tests built from the AREPO source of the 50 cMpc X-ray runs (Oliver's tree, read only), one variant per
# array task (prepare_orcd.py: Lumina's own Config minus RT / galaxy physics / output flags, the exact parameter
# tags this Config requires with the run's values, Zel'dovich ICs at z = 49 shared by all variants):
#   0 fid  1 nofloor  2 noclamp  3 entropy  4 clean  5 fid_hr  6 clean_hr
# Each task: copy the source (no .git, build or binary) to <W>/<variant>/arepo, prepare, build with SYSTYPE
# orcd-floorbox (added to the copy), run mpirun -np 32 ./Arepo param.txt in <W>/<variant>.
#   cd xray/floor_box && mkdir -p logs
#   smoke test first:  sbatch -p ou_mki --array=0 --time=01:00:00 run_floor_box_orcd.sh
#   then:              sbatch -p ou_mki --array=1-6 run_floor_box_orcd.sh   (0 again only if the test was cut short)
#   analysis:          python analyze_floor_box.py --root /orcd/data/mvogelsb/004/chcho/xray/floor_box
VARIANTS=(fid nofloor noclamp entropy clean fid_hr clean_hr)
SRC=/nfs/mvogelsblab002/Users/ozier/SonjaXrayPaper/withXrays/arepo
PARAMS=/nfs/mvogelsblab002/Users/ozier/SonjaXrayPaper/withoutXrays/output/parameters-usedvalues
W=${W:-/orcd/data/mvogelsb/004/chcho/xray/floor_box}
FFTW_PREFIX=/orcd/software/core/001/spack/pkg/fftw/3.3.10/2qziucy
HDF5_PREFIX=/orcd/software/community/001/spack/pkg/hdf5/1.14.3/suffu3v

source ~/.bashrc
conda activate jupyter_env
module load openmpi/4.1.4 gsl/2.7.1 hdf5/1.14.3 fftw/3.3.10
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f prepare_orcd.py ] && [ -f ../../compute_Jalpha.c ] || { echo "submit from xray/floor_box of the repository"; exit 1; }
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
[ -n "$V" ] || { echo "no variant for array task $SLURM_ARRAY_TASK_ID"; exit 1; }
GSL_PREFIX=${GSL_ROOT_DIR:?gsl module did not set GSL_ROOT_DIR}
# GMP (voronoi.h includes gmp.h): no module on ORCD and no system headers (only /usr/lib64/libgmp.so.10), so use the
# spack install (job 30631701 failed with "gmp.h: No such file or directory", 2026-11-26)
GMP_PREFIX=${GMP_PREFIX:-/orcd/software/core/001/spack/pkg/gmp/6.2.1/vb6uboo}
[ -f "$GMP_PREFIX/include/gmp.h" ] || { echo "no gmp.h in $GMP_PREFIX/include (set GMP_PREFIX)"; exit 1; }
D=$W/$V
df -h /orcd/data/mvogelsb/004 | tail -1
echo "variant $V -> $D, $(date), commit $(git rev-parse --short HEAD), host $(hostname), $SLURM_NTASKS MPI tasks"

# 1. private copy of the run's source, with a system type for ORCD's modules
mkdir -p "$D"
rsync -a --delete --exclude .git --exclude build --exclude Arepo "$SRC/" "$D/arepo/"
chmod u+x "$D/arepo/git_version.sh"    # not executable in the run's source: version.cc was never made (job 30723794)
echo 'SYSTYPE="orcd-floorbox"' > "$D/arepo/Makefile.systype"       # the copy's file named Frontier
# Source patches for building the run's (Frontier) source without RT and GPUs, applied to the private copy; each
# replacement must match exactly once, else the task stops:
#  - proto.h: forcetree.cc, do_gravity_hydro.cc, run.cc and cooling/cooling.cc call the RT debug printer mpi_printf_rt
#    without #ifdef MRT; it is declared only in MRT/RT.h (included only with MRT; job 30722651): a no-op in the #else
#    of proto.h's own MRT include, inside proto.h's include guard.
#  - main.cc: #include <hip/hip_runtime_api.h> and hipInit(0) without #ifdef (Frontier's AMD GPUs; job 30723410):
#    put under the run's own flag AMD_HIP, which prepare_orcd.py removes.
#  - backward.cc, system.cc: BACKWARD_HAS_BFD 1 needs binutils' bfd.h (none on ORCD; job 30723410): 0, so the crash
#    backtrace (BACKTRACE) uses backward.hpp's unwind + backtrace_symbols backends (no source line numbers).
#  - voronoi_makeimage.cc: proto.h declares open_image_files with a 25th argument FILE **h1 that the definition lacks,
#    so its callers link to nothing (the run never compiled this file: it is under #if !defined(MTNG), and
#    prepare_orcd.py removes MTNG); the definition gets the (unused) h1. No image flag is set, so it is never called.
python3 - "$D/arepo/src" <<'PY'
import os, sys
src = sys.argv[1]
patches = [
    ("proto.h", '#ifdef MRT\n#include "MRT/RT.h"\n#endif\n',
     '#ifdef MRT\n#include "MRT/RT.h"\n#else\ninline void mpi_printf_rt(const int, const char *, ...) {}  /* floor_box: RT off */\n#endif\n'),
    ("main.cc", "#include <hip/hip_runtime_api.h>\n", "#ifdef AMD_HIP\n#include <hip/hip_runtime_api.h>\n#endif\n"),
    ("main.cc", "hipInit(0);\n", "#ifdef AMD_HIP\nhipInit(0);\n#endif\n"),
    ("backward.cc", "#define BACKWARD_HAS_BFD 1\n", "#define BACKWARD_HAS_BFD 0  /* floor_box: no bfd.h on ORCD */\n"),
    ("system.cc", "#define BACKWARD_HAS_BFD 1\n", "#define BACKWARD_HAS_BFD 0  /* floor_box: no bfd.h on ORCD */\n"),
    ("voronoi_makeimage.cc", "FILE **hrih, FILE **sxrates)\n", "FILE **hrih, FILE **sxrates, FILE **h1)\n"),
]
for f, old, new in patches:
    p = os.path.join(src, f); s = open(p).read()
    if s.count(old) != 1:
        sys.exit(f"ERROR: {old!r} found {s.count(old)} times in {p}")
    open(p, "w").write(s.replace(old, new))
print(f"source patches: {len(patches)} applied")
PY
cat >> "$D/arepo/makefiles/systypes.make" <<EOF

ifeq (\$(SYSTYPE), "orcd-floorbox")
  CC        = mpicxx
  CPPC      = mpicxx
  OPTIMIZE  = -g -std=c++11 -O3 -march=native -Wall
  GSL_INCL  = -I$GSL_PREFIX/include
  GSL_LIBS  = -L$GSL_PREFIX/lib -Wl,-rpath,$GSL_PREFIX/lib
  FFTW_INCL = -I$FFTW_PREFIX/include
  FFTW_LIBS = -L$FFTW_PREFIX/lib -Wl,-rpath,$FFTW_PREFIX/lib
  HDF5INCL  = -I$HDF5_PREFIX/include -DH5_USE_16_API=1
  HDF5LIB   = -L$HDF5_PREFIX/lib -Wl,-rpath,$HDF5_PREFIX/lib -lhdf5 -lz
  GMP_INCL  = -I$GMP_PREFIX/include
  GMP_LIBS  = -L$GMP_PREFIX/lib -Wl,-rpath,$GMP_PREFIX/lib
  LINKER    = mpicxx
  MPICHLIB  =
endif
EOF

# 2. Config, parameter file, output list, initial conditions
python -u prepare_orcd.py variant --variant "$V" --arepo "$D/arepo" --config "$SRC/Config.sh" --params "$PARAMS" \
    --out "$D" --maxmem 2500

# 3. build into <D> (Config.sh, build/, Arepo); -k: compile every file, so a failed build lists all its errors
( cd "$D/arepo" && SYSTYPE=orcd-floorbox make -k build -j "$SLURM_NTASKS" TARGET_DIR="$D" ) > "$D/build.log" 2>&1 \
    || { echo "BUILD FAILED"; grep "error:" "$D/build.log" | sed -E 's/:[0-9]+:[0-9]+:/:/' | sort | uniq -c | sort -rn | head -40
         tail -20 "$D/build.log"; exit 1; }
echo "build OK $(date)"

# 4. run
cd "$D"
rm -rf output
mpirun -np "$SLURM_NTASKS" ./Arepo param.txt > run.log 2>&1 || true
if grep -q "TERMINATE\|Error in file" run.log; then
    echo "RUN FAILED"; grep -m5 "TERMINATE\|Error in file" run.log; exit 1
fi
echo "done $(date): $(ls output/snap_*.hdf5 2>/dev/null | wc -l) snapshots; last: $(grep Sync-Point output/info.txt | tail -1 | cut -c1-80)"
