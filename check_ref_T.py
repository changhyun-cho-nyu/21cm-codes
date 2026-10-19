"""
check_ref_T.py
==============
What temperature does the fiducial run give the gas that the no-heating reference replaces? For the 640^3 pixels
with x_HII < threshold (TK_REF_XTHRESH) at render snapshot --snap, compares Lumina's T (fiducial) with the
reference T_ref = T(p, z0) [(1+z)/(1+z0)]^2 [rho(p)/rho(p, z0)]^(2/3) (TK_REF_ANCHOR=1, anchor --snap0), and both
with T_CMB. All statistics are H I-mass-weighted (Density x (1 - HII_Fraction)), overall and by 1 + delta.

Usage
-----
  python check_ref_T.py --dir <ren_640> --snap 156 [--snap0 0] [--xthresh 1e-3]
"""
import argparse
import os

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def field(d, name, snap):
    p = os.path.join(d, name, f"{name}_{snap:03d}.hdf5")
    with h5py.File(p if os.path.exists(p) else os.path.join(d, "All", f"All_{snap:03d}.hdf5"), "r") as f:
        return f[name][()].astype(np.float32)


def redshift(d, snap):
    with h5py.File(os.path.join(d, "All", f"All_{snap:03d}.hdf5"), "r") as f:
        return float(f["Header"].attrs["Redshift"])


def wpct(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o].astype(np.float64))
    return [float(x[o][np.searchsorted(c, qq / 100.0 * c[-1])]) for qq in q]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--snap0", type=int, default=0)
    p.add_argument("--xthresh", type=float, default=1e-3)
    a = p.parse_args()

    z, z0 = redshift(a.dir, a.snap), redshift(a.dir, a.snap0)
    Tg = 2.7255 * (1.0 + z)
    rho, T, x = field(a.dir, "Density", a.snap), field(a.dir, "Temperature", a.snap), field(a.dir, "HII_Fraction", a.snap)
    sel = x < a.xthresh
    whi_all = rho * (1.0 - x)
    print(f"snapshot {a.snap}, z = {z:.4f}; anchor {a.snap0}, z0 = {z0:.4f}; T_CMB = {Tg:.2f} K")
    print(f"  pixels with x_HII < {a.xthresh:g}: {sel.mean():.4f} of the cells, {whi_all[sel].sum() / whi_all.sum():.4f} of the H I mass")
    w = whi_all[sel]
    dl = rho[sel] / rho.mean()
    Tf = T[sel]
    del T, x, whi_all
    rho0, T0 = field(a.dir, "Density", a.snap0)[sel], field(a.dir, "Temperature", a.snap0)[sel]
    Tr = T0 * ((1.0 + z) / (1.0 + z0))**2 * (rho[sel] / rho0)**(2.0 / 3.0)
    del rho, rho0, T0

    def report(lab, m):
        ww = w[m]
        f, r = wpct(Tf[m], ww, (10, 50, 90)), wpct(Tr[m], ww, (10, 50, 90))
        q = wpct(Tf[m] / Tr[m], ww, (10, 50, 90))
        print(f"  {lab:>14}: H I share {ww.sum() / w.sum():.4f};  T_fid 10/50/90% {f[0]:9.2f} {f[1]:9.2f} {f[2]:9.2f} K;"
              f"  T_ref {r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} K;  T_fid/T_ref {q[0]:7.2f} {q[1]:7.2f} {q[2]:7.2f};"
              f"  H I with T > T_CMB: fid {ww[Tf[m] > Tg].sum() / ww.sum():.3f}, ref {ww[Tr[m] > Tg].sum() / ww.sum():.3f}")

    print("  H I-mass-weighted, replaced pixels only:")
    report("all", np.ones(len(w), bool))
    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, 100), (100, np.inf)):
        m = (dl >= lo) & (dl < hi)
        if m.any():
            report(f"1+d {lo:g}-{hi:g}", m)


if __name__ == "__main__":
    main()
