"""
test_resolution_Tb.py
=====================
How much does computing delta T_b on 640^3 pixel-averaged inputs bias the maps, and does it converge?

delta T_b is nonlinear in its inputs: with coupling, the spin factor depends on <1/T_K> of the H I but a
pixel holds its mass-weighted T (dominated by the hottest gas); and the weak collisional 3He+ coupling
makes dT_b^He ~ n_e n_HeII f(T), a two-body (clumping-sensitive) quantity. So the map made from pixel
averages differs from the average of a map made at higher resolution.

For one snapshot and one line, with the physics of tb_physics.py (as check_Tb_maps.py and the C codes),
delta T_b is computed at every available resolution and put on the 640^3 grid:
  320   : the 640^3 fields averaged 2x2x2 (mass-weighted fractions and temperature), map repeated 2x2x2
  640   : ren_640, as production
  1280  : ren_1280, all fields, map averaged 2x2x2
  2560  : ren_2560 (--dir2560), which has Density and HII_Fraction (and below z = 4.75 HeII/HeIII_Fraction);
          the fields it lacks (Temperature always) are taken from the 1280^3 parent cell. Map averaged 4x4x4.
J_alpha is the 640^3 file (repeated onto finer cells, averaged onto 320^3): smooth on these scales.
All maps are real space (as production). First it checks that each coarser render is the mass-weighted
average of the finer one. Then for each level: <dT_b>, P(k)/P_finest and r(k), and the successive
differences of <dT_b> with their ratio q; if 0 < q < 1 the geometric extrapolation
X_inf = X_finest + Delta_last q / (1 - q) is printed (a rough continuum estimate, only if q is steady).

Usage
-----
  python test_resolution_Tb.py --line H --dir640 <ren_640> --dir1280 <ren_1280> [--dir2560 <ren_2560>] \\
         --snap 156 --J Jalpha_H_156.hdf5 --J Jalpha_AGN_H_156.hdf5 [--tk-ref 1e-3] [--chunk 4]
Memory with --dir2560 and --chunk 4: ~15 GB per step plus ~5 GB for the maps and ~12 GB for P(k).
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from tb_physics import SRC, X_H, Y_HE, M_H, OMEGA_L_DEFAULT, Line
from check_tb_choices import spectra

FIELDS = ("Density", "Temperature", "HII_Fraction", "HeII_Fraction", "HeIII_Fraction")


class Render:
    def __init__(self, d, s, required=FIELDS):
        self.d, self.s, self.f = d, s, {}
        with h5py.File(os.path.join(d, "All", f"All_{s:03d}.hdf5"), "r") as f:
            self.hd = dict(f["Header"].attrs)
            all_keys = set(f.keys())
        self.have = {k for k in FIELDS
                     if os.path.exists(os.path.join(d, k, f"{k}_{s:03d}.hdf5")) or k in all_keys}
        missing = [k for k in required if k not in self.have]
        if missing:
            sys.exit(f"ERROR: {d} lacks {missing} at snapshot {s}")
        self.N = self.ds("Density").shape[0]
        for k in self.have:
            if self.ds(k).shape != (self.N,) * 3:
                sys.exit(f"ERROR: {k} in {d} has shape {self.ds(k).shape}, expected {(self.N,) * 3}")

    def ds(self, name):
        if name not in self.f:
            p = os.path.join(self.d, name, f"{name}_{self.s:03d}.hdf5")
            if not os.path.exists(p):
                p = os.path.join(self.d, "All", f"All_{self.s:03d}.hdf5")
            self.f[name] = h5py.File(p, "r")
        return self.f[name][name]

    def slab(self, i0, i1, names):
        return {k: self.ds(k)[i0:i1].astype(np.float64) for k in names}

    def mean_density(self, step=32):
        m = 0.0
        for i0 in range(0, self.N, step):
            m += self.ds("Density")[i0:i0 + step].astype(np.float64).sum()
        return m / float(self.N)**3


def blk(a, f):
    """f x f x f mean"""
    n0, n1, n2 = a.shape
    return a.reshape(n0 // f, f, n1 // f, f, n2 // f, f).mean(axis=(1, 3, 5))


def rep(a, f):
    return np.repeat(np.repeat(np.repeat(a, f, axis=0), f, axis=1), f, axis=2)


def coarsen(F, f):
    """mass-weighted f x f x f average of a field set (temperature and fractions mass-weighted)"""
    m = blk(F["Density"], f)
    out = {"Density": m}
    for k in FIELDS[1:]:
        out[k] = blk(F["Density"] * F[k], f) / np.maximum(m, 1e-300)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--line", choices=["H", "He"], required=True)
    p.add_argument("--dir640", required=True)
    p.add_argument("--dir1280", required=True)
    p.add_argument("--dir2560", help="ren_2560 (Density, HII_Fraction, [HeII/HeIII_Fraction]); optional")
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--J", action="append", required=True, help="640^3 J_alpha files of the line (stars, AGN), summed")
    p.add_argument("--tk-ref", type=float, metavar="THRESH", help="H: the no-heating reference (as -DTK_REFERENCE)")
    p.add_argument("--chunk", type=int, default=4, help="640^3 planes per step (even)")
    a = p.parse_args()
    if a.chunk % 2:
        sys.exit("--chunk must be even (the 320^3 level)")

    R6, R12 = Render(a.dir640, a.snap), Render(a.dir1280, a.snap)
    R25 = Render(a.dir2560, a.snap, required=("Density", "HII_Fraction")) if a.dir2560 else None
    N = R6.N
    if R12.N != 2 * N or (R25 and R25.N != 4 * N):
        sys.exit("ERROR: grids are not 640 : 1280 : 2560")
    hd = R6.hd
    z = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
    for R in (R12, R25):
        if R and abs(float(R.hd.get("Redshift", z)) - z) > 1e-4:
            sys.exit(f"ERROR: redshift of {R.d} differs")
    h = float(hd["HubbleParam"])
    om, ob = float(hd.get("Omega0", 0.30964144)), float(hd.get("OmegaBaryon", 0.048975))
    ol = float(hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    rho_unit = float(hd["UnitMass_in_g"]) * h**2 / float(hd["UnitLength_in_cm"])**3
    box = float(hd["BoxSize"]) / 1000.0                                   # cMpc/h
    L = Line(a.line, z, h, om, ob, ol)
    n25 = sorted(R25.have) if R25 else []
    print(f"constants: {SRC}; snap {a.snap}, z = {z:.4f}, line {a.line}, prefactor {L.P:.6e} {L.unit}"
          f"{', no-heating reference x_HII < %g' % a.tk_ref if a.tk_ref is not None else ''}")
    if R25:
        print(f"  2560^3 native fields: {n25}; from the 1280^3 parent: {[k for k in FIELDS if k not in R25.have]}")
    t0 = time.time()
    m6 = R6.mean_density()
    print(f"  <Density> 640^3 = {m6:.6e}")

    Jf = [h5py.File(fn, "r") for fn in a.J]
    Jd = [f[next(k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3)] for f in Jf]
    for dd in Jd:
        if dd.shape != (N,) * 3:
            sys.exit(f"ERROR: J file has shape {dd.shape}")

    levels = ["320", "640", "1280"] + (["2560"] if R25 else [])
    maps = {lv: np.empty((N, N, N), np.float32) for lv in levels}
    chk = {"640<-1280": [0.0, 0.0], "1280<-2560": [0.0, 0.0]}

    def tb_of(F, Jx):
        rho, T = F["Density"], F["Temperature"]
        xHII, fII, fIII = F["HII_Fraction"], F["HeII_Fraction"], F["HeIII_Fraction"]
        if a.tk_ref is not None:
            T = np.where(xHII < a.tk_ref, L.t_ref(rho, m6), T)
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H
        sig = rho * (1.0 - xHII) / m6 if a.line == "H" else (fII / Y_HE) * rho / m6
        return L.tb(sig, nH, T, xHII, fII, fIII, Jx)[0]

    def compare(tag, coarse, fine, f):
        c = coarsen(fine, f)
        chk[tag][0] = max(chk[tag][0], float(np.max(np.abs(c["Density"] / coarse["Density"] - 1.0))))
        chk[tag][1] = max(chk[tag][1], float(np.max(np.abs(c["HII_Fraction"] - coarse["HII_Fraction"]))))

    for i0 in range(0, N, a.chunk):
        i1 = min(N, i0 + a.chunk)
        J6 = sum(dd[i0:i1].astype(np.float64) for dd in Jd)
        F6 = R6.slab(i0, i1, FIELDS)
        F12 = R12.slab(2 * i0, 2 * i1, FIELDS)
        compare("640<-1280", F6, F12, 2)
        F3 = coarsen(F6, 2)
        maps["320"][i0:i1] = rep(tb_of(F3, blk(J6, 2)), 2)
        maps["640"][i0:i1] = tb_of(F6, J6)
        maps["1280"][i0:i1] = blk(tb_of(F12, rep(J6, 2)), 2)
        if R25:
            F25 = R25.slab(4 * i0, 4 * i1, n25)
            for k in FIELDS:
                if k not in F25:
                    F25[k] = rep(F12[k], 2)
            compare("1280<-2560", {k: F12[k] for k in ("Density", "HII_Fraction")}, F25, 2)
            maps["2560"][i0:i1] = blk(tb_of(F25, rep(J6, 4)), 4)
            del F25
        if i0 == 0:
            print(f"  ~{(time.time() - t0) / (i1 - i0) * N / 60:.0f} min for the maps", flush=True)
    for tag, (dr, dx) in chk.items():
        if tag == "1280<-2560" and not R25:
            continue
        print(f"  {tag}: max |rho ratio - 1| = {dr:.2e}, max |x_HII diff| = {dx:.2e}  (small = the coarser render is that average)")

    fin = levels[-1]
    means = {lv: float(maps[lv].astype(np.float64).mean()) for lv in levels}
    print(f"\n  <dT_b> [{L.unit}]:  " + "  ".join(f"{lv}: {means[lv]:+.5e}" for lv in levels))
    dif = [means[levels[i + 1]] - means[levels[i]] for i in range(len(levels) - 1)]
    print("  successive differences:  " + "  ".join(f"{levels[i]}->{levels[i + 1]}: {dif[i]:+.4e}" for i in range(len(dif))))
    qs = [dif[i + 1] / dif[i] for i in range(len(dif) - 1) if dif[i] != 0]
    if qs:
        print("  ratios q of successive differences: " + "  ".join(f"{q:.3f}" for q in qs))
        q = qs[-1]
        if 0 < q < 1:
            print(f"  geometric extrapolation of <dT_b> (if q stays {q:.2f}): {means[fin] + dif[-1] * q / (1 - q):+.5e} {L.unit}")
        else:
            print("  q not in (0, 1): no extrapolation (not converging geometrically)")
    ref = maps[fin].astype(np.float32)
    for lv in levels[:-1]:
        kc, cnt, P, X = spectra([ref, maps[lv]], box)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = X[0] / np.sqrt(P[0] * P[1])
        sel = np.flatnonzero((cnt > 0) & (P[0] > 0))
        pick = sorted({int(sel[np.argmin(abs(kc[sel] - qq))]) for qq in (0.03, 0.05, 0.1, 0.3, 1.0, 2.0)})
        print(f"  P_{lv}/P_{fin} / r(k):  " + "  ".join(f"k={kc[i]:.2f}: {P[1][i] / P[0][i]:.4f}/{r[i]:.4f}" for i in pick))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
