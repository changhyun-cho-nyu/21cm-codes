"""
make_variants.py
================
Small cosmological boxes at Lumina's gas-cell mass (3.6e6 Msun, 56.38 ckpc/h mean spacing) from z = 49, same ICs
(internal N-GenIC, 2LPT, fixed seed), differing only in the hydro treatment of cold gas (user, 2026-11-25: "test all
to get the whole picture"). Each variant gets <root>/<name>/{Config.sh, param.txt, output_list.txt} and, with
--build, its own AREPO binary (make TARGET_DIR=<root>/<name>; the AREPO checkout is not modified).

Variants (N = gas cells per dimension):
  fid              floor + _EEOS, MinGasTemp 5 K                 (Lumina's setting)
  nofloor          no floor,      MinGasTemp 5 K                 -> the floor's effect
  entropy          floor + entropy switch (Mach > 1.1 uses energy) -> the solver's cold-flow error
  noclamp          floor,         MinGasTemp 0.01 K              -> the clamp's effect
  clean            no floor + entropy switch + MinGasTemp 0.01 K  -> adiabatic + shocks only (closest to physical)
  fid_hr / clean_hr  the same at N x 2 (cell mass / 8; floor T_J / 4) -> resolution dependence

  python make_variants.py --root <dir> --arepo ~/GitHub/arepo [--build] [--only fid,nofloor] [--n 64]
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPACING = 56.38            # ckpc/h, Lumina's mean gas-cell spacing (33830 / 600)
ZOUT = (45, 40, 36, 33, 30, 27, 24, 21, 18, 15, 12)
FLOOR = ["ENFORCE_JEANS_STABILITY_OF_CELLS", "ENFORCE_JEANS_STABILITY_OF_CELLS_EEOS"]
ENTROPY = ["USE_ENTROPY_FOR_COLD_FLOWS", "ENTROPY_MACH_THRESHOLD=1.1"]
VARIANTS = {
    "fid": (FLOOR, 5.0, 1),
    "nofloor": ([], 5.0, 1),
    "entropy": (FLOOR + ENTROPY, 5.0, 1),
    "noclamp": (FLOOR, 0.01, 1),
    "clean": (ENTROPY, 0.01, 1),
    "fid_hr": (FLOOR, 5.0, 2),
    "clean_hr": (ENTROPY, 0.01, 2),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--arepo", default=os.path.expanduser("~/GitHub/arepo"))
    p.add_argument("--n", type=int, default=64, help="gas cells per dimension of the base variants")
    p.add_argument("--zend", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--maxmem", type=int, default=1500, help="MB per MPI rank")
    p.add_argument("--only", default=None)
    p.add_argument("--build", action="store_true")
    p.add_argument("--systype", default="MacOShomebrew")
    p.add_argument("-j", type=int, default=8)
    a = p.parse_args()
    base = open(os.path.join(HERE, "Config_base.sh")).read()
    tmpl = open(os.path.join(HERE, "param_template.txt")).read()
    names = a.only.split(",") if a.only else list(VARIANTS)
    for name in names:
        if name not in VARIANTS:
            sys.exit(f"unknown variant {name}")
        flags, tmin, f = VARIANTS[name]
        n = a.n * f
        box = SPACING * a.n                      # the same box for all variants; hr has 8x lighter cells
        d = os.path.join(os.path.abspath(a.root), name)
        os.makedirs(d, exist_ok=True)
        cfg = base + "\n".join(flags + [f"NGENIC={n}", f"PMGRID={2 * n}"]) + "\n"
        open(os.path.join(d, "Config.sh"), "w").write(cfg)
        open(os.path.join(d, "param.txt"), "w").write(tmpl.format(BOX=f"{box:.4f}", GRID=n, MINGASTEMP=tmin,
                                                                   SEED=a.seed, MAXMEM=a.maxmem,
                                                                   TIMEMAX=f"{1.0 / (1.0 + a.zend):.8f}"))
        with open(os.path.join(d, "output_list.txt"), "w") as fo:
            for z in ZOUT:
                if z >= a.zend - 1e-9:
                    fo.write(f"{1.0 / (1.0 + z):.8f} 1\n")
        print(f"{name}: N = {n}, box {box:.1f} ckpc/h, cell mass x{1 / f**3:g}, MinGasTemp {tmin}, flags {flags}")
        if a.build:
            # 'make build' skips AREPO's macro-name check, which fails on the checkout's own systypes.make
            # SYSTYPE through the environment: on the make command line it would override the Makefile's quoting
            env = dict(os.environ, SYSTYPE=a.systype)
            r = subprocess.run(["make", "build", f"-j{a.j}", f"TARGET_DIR={d}"], cwd=a.arepo, env=env,
                               capture_output=True, text=True)
            log = os.path.join(d, "build.log")
            open(log, "w").write(r.stdout + r.stderr)
            ok = r.returncode == 0 and os.path.exists(os.path.join(d, "Arepo"))
            print(f"   build {'OK' if ok else 'FAILED'} ({log})")
            if not ok:
                print("\n".join((r.stdout + r.stderr).splitlines()[-25:]))
                sys.exit(1)


if __name__ == "__main__":
    main()
