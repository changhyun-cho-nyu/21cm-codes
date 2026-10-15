"""
test_resolution_Tb.py
=====================
How much does computing delta T_b on 640^3 pixel-averaged inputs bias the maps?

delta T_b is nonlinear in its inputs (x_HI, T_K and the density enter T_S through x_c and S_alpha),
so the 640^3 map made from 640^3 mass-weighted fields (what PS_21cm_mpi.c / PS_3cm_mpi.c do) differs
from the 640^3 average of a map made at 1280^3. The difference is the sub-pixel (pixel-averaging)
error, largest where the spin factor varies inside a pixel: at ionization and heating fronts, where a
pixel's mass-weighted temperature mixes hot ionized and cold neutral gas.

For one snapshot and one line this script computes, with the physics of tb_physics.py (the same as
check_Tb_maps.py and the C codes):
  map_640  : delta T_b from the ren_640 fields                        (as production)
  map_1280 : delta T_b from the ren_1280 fields, averaged 2x2x2 onto 640^3
J_alpha is the 640^3 file repeated onto the 2x2x2 sub-cells (smooth on these scales), so the test
isolates the nonlinearity in the render fields. Both maps are real space (as production).
It first checks that ren_640 is the mass-conserving 2x2x2 average of ren_1280 (Density, and the
mass-weighted HII_Fraction), then reports: <dT_b> of each (the global-signal bias), the cell-level
difference and its share in partly ionized pixels (0.01 < x_HII < 0.99), and P_1280/P_640 and r(k).

Usage
-----
  python test_resolution_Tb.py --line H  --dir640 <ren_640> --dir1280 <ren_1280> --snap 156 \\
         --J Jalpha_H_156.hdf5 --J Jalpha_AGN_H_156.hdf5 [--tk-ref 1e-3] [--chunk 8]
  python test_resolution_Tb.py --line He --dir640 ... --dir1280 ... --snap 500 \\
         --J Jalpha_He_500_tr_sfmask.hdf5 --J Jalpha_AGN_He_500_tr_sfmask.hdf5
Memory: two 640^3 float32 maps plus one slab of 1280^3 fields (~ 2 GB per 8 planes); the P(k) step
needs ~ 12 GB.
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
    def __init__(self, d, s):
        self.d, self.s, self.f = d, s, {}
        with h5py.File(os.path.join(d, "All", f"All_{s:03d}.hdf5"), "r") as f:
            self.hd = dict(f["Header"].attrs)
        self.N = self.ds("Density").shape[0]
        for k in FIELDS:
            if self.ds(k).shape != (self.N,) * 3:
                sys.exit(f"ERROR: {k} in {d} has shape {self.ds(k).shape}, expected {(self.N,) * 3} (sub-cell layout?)")

    def ds(self, name):
        if name not in self.f:
            p = os.path.join(self.d, name, f"{name}_{self.s:03d}.hdf5")
            if not os.path.exists(p):
                p = os.path.join(self.d, "All", f"All_{self.s:03d}.hdf5")
            self.f[name] = h5py.File(p, "r")
        return self.f[name][name]

    def slab(self, i0, i1):
        return {k: self.ds(k)[i0:i1].astype(np.float64) for k in FIELDS}

    def mean_density(self, step=32):
        m = 0.0
        for i0 in range(0, self.N, step):
            m += self.ds("Density")[i0:i0 + step].astype(np.float64).sum()
        return m / float(self.N)**3


def block2(a):
    """2x2x2 mean (n, 2m, 2m) -> (n/2, m, m)"""
    n0, n1, n2 = a.shape
    return a.reshape(n0 // 2, 2, n1 // 2, 2, n2 // 2, 2).mean(axis=(1, 3, 5))


def up2(a):
    return np.repeat(np.repeat(np.repeat(a, 2, axis=0), 2, axis=1), 2, axis=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--line", choices=["H", "He"], required=True)
    p.add_argument("--dir640", required=True)
    p.add_argument("--dir1280", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--J", action="append", required=True, help="640^3 J_alpha files of the line (stars, AGN), summed")
    p.add_argument("--tk-ref", type=float, metavar="THRESH", help="H: the no-heating reference (as -DTK_REFERENCE)")
    p.add_argument("--chunk", type=int, default=8, help="640^3 planes per step")
    a = p.parse_args()

    R6, R12 = Render(a.dir640, a.snap), Render(a.dir1280, a.snap)
    if R12.N != 2 * R6.N:
        sys.exit(f"ERROR: grids {R6.N}^3 and {R12.N}^3 are not 1:2")
    hd = R6.hd
    z = float(hd["Redshift"]) if "Redshift" in hd else 1.0 / float(hd["Time"]) - 1.0
    z12 = float(R12.hd.get("Redshift", 1.0 / float(R12.hd["Time"]) - 1.0))
    if abs(z - z12) > 1e-4:
        sys.exit(f"ERROR: redshifts differ, {z} vs {z12}")
    h = float(hd["HubbleParam"])
    om, ob = float(hd.get("Omega0", 0.30964144)), float(hd.get("OmegaBaryon", 0.048975))
    ol = float(hd.get("OmegaLambda", OMEGA_L_DEFAULT))
    rho_unit = float(hd["UnitMass_in_g"]) * h**2 / float(hd["UnitLength_in_cm"])**3
    box = float(hd["BoxSize"]) / 1000.0                                   # cMpc/h
    L = Line(a.line, z, h, om, ob, ol)
    N = R6.N
    print(f"constants: {SRC}; snap {a.snap}, z = {z:.4f}, line {a.line}, prefactor {L.P:.6e} {L.unit}"
          f"{', no-heating reference x_HII < %g' % a.tk_ref if a.tk_ref is not None else ''}")
    t0 = time.time()
    m6, m12 = R6.mean_density(), R12.mean_density()
    print(f"  <Density>: 640^3 {m6:.6e}, 1280^3 {m12:.6e} (ratio {m12 / m6:.7f})")

    Jf = [h5py.File(fn, "r") for fn in a.J]
    Jd = [f[next(k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3)] for f in Jf]
    for dd in Jd:
        if dd.shape != (N,) * 3:
            sys.exit(f"ERROR: J file has shape {dd.shape}")

    T640 = np.empty((N, N, N), np.float32)
    T1280 = np.empty((N, N, N), np.float32)
    part = np.empty((N, N, N), bool)
    dmax_rho = dmax_x = 0.0

    def tb_of(F, Jx, rho_mean):
        rho, T = F["Density"], F["Temperature"]
        xHII, fII, fIII = F["HII_Fraction"], F["HeII_Fraction"], F["HeIII_Fraction"]
        if a.tk_ref is not None:
            T = np.where(xHII < a.tk_ref, L.t_ref(rho, rho_mean), T)
        nH = X_H * rho * rho_unit * (1.0 + z)**3 / M_H
        sig = rho * (1.0 - xHII) / rho_mean if a.line == "H" else (fII / Y_HE) * rho / rho_mean
        return L.tb(sig, nH, T, xHII, fII, fIII, Jx)[0]

    for i0 in range(0, N, a.chunk):
        i1 = min(N, i0 + a.chunk)
        F6, F12 = R6.slab(i0, i1), R12.slab(2 * i0, 2 * i1)
        J6 = sum(dd[i0:i1].astype(np.float64) for dd in Jd)
        # ren_640 = mass-conserving 2x2x2 average of ren_1280?
        r12 = block2(F12["Density"])
        x12 = block2(F12["Density"] * F12["HII_Fraction"]) / np.maximum(r12, 1e-300)
        dmax_rho = max(dmax_rho, float(np.max(np.abs(r12 / F6["Density"] - 1.0))))
        dmax_x = max(dmax_x, float(np.max(np.abs(x12 - F6["HII_Fraction"]))))
        T640[i0:i1] = tb_of(F6, J6, m6)
        T1280[i0:i1] = block2(tb_of(F12, up2(J6), m12))
        part[i0:i1] = (F6["HII_Fraction"] > 0.01) & (F6["HII_Fraction"] < 0.99)
        if i0 == 0:
            print(f"  ~{(time.time() - t0) / (i1 - i0) * N / 60:.0f} min for the maps", flush=True)
    print(f"  ren_640 vs 2x2x2 mean of ren_1280: max |rho ratio - 1| = {dmax_rho:.2e}, "
          f"max |x_HII (mass-weighted) diff| = {dmax_x:.2e}   (small = ren_640 is that average)")

    a6, a12 = T640.astype(np.float64), T1280.astype(np.float64)
    d = a12 - a6
    rms6 = np.sqrt(np.mean((a6 - a6.mean())**2))
    sh = (d[part]**2).sum() / max((d**2).sum(), 1e-300)
    print(f"\n  <dT_b>: 640 {a6.mean():+.6e} {L.unit}, 1280->640 {a12.mean():+.6e} {L.unit}, "
          f"ratio {a12.mean() / a6.mean():.5f}  (global-signal bias of the production maps: {a6.mean() / a12.mean() - 1:+.2%})")
    print(f"  cell rms of the difference / rms of the 640 map = {np.sqrt(np.mean(d**2)) / rms6:.4f}; "
          f"{part.mean():.3f} of the pixels are partly ionized (0.01 < x_HII < 0.99) and hold {sh:.3f} of the squared difference")
    kc, cnt, P, X = spectra([a6.astype(np.float32), a12.astype(np.float32)], box)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = X[0] / np.sqrt(P[0] * P[1])
    sel = np.flatnonzero((cnt > 0) & (P[0] > 0))
    pick = sorted({int(sel[np.argmin(abs(kc[sel] - q))]) for q in (0.03, 0.05, 0.1, 0.3, 1.0, 2.0, 3.0)})
    print("  P_1280/P_640 / r(k):  " + "  ".join(f"k={kc[i]:.2f}: {P[1][i] / P[0][i]:.4f}/{r[i]:.4f}" for i in pick))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
