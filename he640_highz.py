"""
he640_highz.py
==============
Volume-mean 3.46 cm delta T_b on the 640^3 render pixels at z > 10 (above the production He maps, which start at render
156, z = 10.01), for the high-redshift end of the Appendix sec:metallicity figure (user, 2026-11-24: "I know it's not
good but I think I need it"; the He signal statistics are otherwise gas-cell only, and the 640^3 He maps are about half
the gas-cell value at z = 10, 0.497).

Physics: tb_physics.Line("He") as check_Tb_maps.py (which reproduces the C He maps cell by cell): real space,
mass-weighted fields (run_3cm.sh: -DHE_REDSHIFT_SPACE=0), sig = (HeII_Fraction / y)(1 + delta) with the box-mean Density,
x_c^He from n_e, J = 0 above z = 10.01 (no He J files there; the Lyman-alpha share of the He coupling is <= 0.4% there,
JALPHA_NOTES 4.7), so every stellar metallicity gives the same value. Every pixel of the box.

Check first (--check): at a render that has He J files and a production map, the same calculation with those J files
must reproduce the map's mean (Tb_maps_prod/He/RIN_N.hdf5) to --tol.

Usage (ORCD; job run_he640_highz.sh)
  python he640_highz.py --check 156 --out .../metallicity/He640     (J files + production map at render 156)
  python he640_highz.py --snaps 0 10 20 ... 155 --out .../metallicity/He640
Writes <out>/He640_NNN.csv (snap, z, mean_dTb_uK, J, map_mean).
"""
import argparse
import csv
import os
import sys
import time

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from check_Tb_maps import field_path
from tb_physics import Line, X_H, Y_HE, M_H, OMEGA_L_DEFAULT

BASE = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
REN = {"above": f"{BASE}/Flagship_new/postprocessing/ren_640", "below": f"{BASE}/Flagship_below_z_4p75_new/postprocessing/ren_640"}
D = "/orcd/data/mvogelsb/004/chcho"


def first_3d(f, prefer=None):
    names = [k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3]
    return f[prefer] if prefer in names else f[names[0]]


def mean_tb(d, s, J_files, planes):
    with h5py.File(os.path.join(d, "All", f"All_{s:03d}.hdf5"), "r") as f:
        hd = dict(f["Header"].attrs)
    z = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
    h = float(hd["HubbleParam"])
    om, ob = float(hd.get("Omega0", 0.30964144)), float(hd.get("OmegaBaryon", 0.048975))
    ol = float(hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    rho_unit = float(hd["UnitMass_in_g"]) * h**2 / float(hd["UnitLength_in_cm"])**3
    files = {}

    def fld(name):
        if name not in files:
            files[name] = h5py.File(field_path(d, name, s), "r")
        return files[name][name]
    N = fld("Density").shape[0]
    rho_mean = 0.0
    for i0 in range(0, N, planes):
        rho_mean += fld("Density")[i0:i0 + planes].astype(np.float64).sum()
    rho_mean /= float(N)**3
    L = Line("He", z, h, om, ob, ol)
    Jd = []
    for fn in J_files:
        f = h5py.File(fn, "r")
        dd = first_3d(f)
        zj = dd.attrs.get("Redshift")
        if zj is not None and abs(float(zj) - z) > 1e-4:
            sys.exit(f"STOP: {fn} is at z = {float(zj)}, render {s} at z = {z}")
        Jd.append(dd)
    tot = 0.0
    for i0 in range(0, N, planes):
        sl = slice(i0, min(N, i0 + planes))
        rho = fld("Density")[sl].astype(np.float64)
        T = fld("Temperature")[sl].astype(np.float64)
        xHII = fld("HII_Fraction")[sl].astype(np.float64)
        fII = fld("HeII_Fraction")[sl].astype(np.float64)
        fIII = fld("HeIII_Fraction")[sl].astype(np.float64)
        J = sum(dd[sl].astype(np.float64) for dd in Jd) if Jd else np.zeros_like(rho)
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H
        sig = (fII / Y_HE) * rho / rho_mean
        tb = L.tb(sig, nH, T, xHII, fII, fIII, J)[0]
        if not np.all(np.isfinite(tb)):
            sys.exit(f"STOP: non-finite dT_b in render {s}, planes {sl.start}-{sl.stop - 1}")
        tot += tb.sum()
    return z, tot / float(N)**3, N


def write(out, s, z, m, jlab, mmap):
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"He640_{s:03d}.csv")
    with open(fn, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["snap", "z", "mean_dTb_uK", "J", "map_mean"])
        w.writerow([s, repr(z), repr(float(m)), jlab, repr(float(mmap))])
    return fn


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snaps", type=int, nargs="*", default=[], help="renders, J = 0 (z > 10.01)")
    p.add_argument("--check", type=int, help="render with He J files and a production He map: reproduce its mean")
    p.add_argument("--jdir", default=f"{D}/Jalpha")
    p.add_argument("--tr-tag", default="tr_sfmask")
    p.add_argument("--maps", default=f"{D}/Tb_maps_prod/He")
    p.add_argument("--dir", help="render directory for every render (default: the Lumina ren_640 of each render)")
    p.add_argument("--planes", type=int, default=32)
    p.add_argument("--tol", type=float, default=1e-3, help="check: max |here/map - 1| (map constants: 1.4e-4)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.check is not None:
        s = a.check
        d = a.dir or (REN["above"] if s <= 428 else REN["below"])
        Jf = [os.path.join(a.jdir, f"Jalpha_He_{s}_{a.tr_tag}.hdf5"), os.path.join(a.jdir, f"Jalpha_AGN_He_{s}_{a.tr_tag}.hdf5")]
        t0 = time.time()
        z, m, N = mean_tb(d, s, Jf, a.planes)
        with h5py.File(os.path.join(a.maps, f"RIN_{s}.hdf5"), "r") as f:
            dd = first_3d(f, "rin")
            mmap = sum(dd[i0:i0 + a.planes].astype(np.float64).sum() for i0 in range(0, N, a.planes)) / float(N)**3
        r = m / mmap
        print(f"check, render {s} (z = {z:.4f}), J {Jf}: here {m:.7e} uK, production map {mmap:.7e} uK, ratio {r:.6f} "
              f"({time.time() - t0:.0f} s)")
        fn = write(a.out, s, z, m, "check:" + ";".join(os.path.basename(x) for x in Jf), mmap)
        print(f"wrote {fn}")
        if abs(r - 1.0) > a.tol:
            sys.exit(f"STOP: the calculation does not reproduce the production He map (tolerance {a.tol:g})")
    for s in a.snaps:
        d = a.dir or (REN["above"] if s <= 428 else REN["below"])
        t0 = time.time()
        z, m, _ = mean_tb(d, s, [], a.planes)
        if z < 10.01:
            sys.exit(f"STOP: render {s} is at z = {z:.4f}; J = 0 only above z = 10.01 (no He J files there)")
        fn = write(a.out, s, z, m, "none", np.nan)
        print(f"render {s:3d} z = {z:7.4f}: <dT_b^He> = {m:.6e} uK (J = 0)  {fn} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
