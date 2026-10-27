"""
check_cone_T.py
===============
Why is the light-cone <delta T_b^H> above the production maps at z >~ 11 (check_lightcone_tb.py, 2026-10-27:
cone / maps 1.10, 1.44, 1.47 at z = 11, 13, 15; <= 1% at z = 6.5-9.5)? At z >~ 10 the 640^3 delta T_b depends on
the pixel size through temperature mixing in neutral pixels (test_resolution_Tb.py: 320^3 / 640^3 = 1.73 at
z = 12.7), and the cone pixels are coarser (~0.66 cMpc/h at z = 11-15 against 0.53 for the maps).

For bands of full light-cone slices (weight chi^2 d chi per slice) and for the render snapshot nearest in z, at 640^3
and averaged to 320^3 (2 x 2 x 2, mass-weighted T and x_HII, as the renders are), prints the H I-mass-weighted
temperature statistics that enter delta T_b:
  <T>_m      mass-weighted T (conserved by mass-weighted averaging: the same gas should give the same value);
  <T>_HI     H I mass-weighted T;  T_harm = <1/T>_HI^-1 (falls towards <T>_HI as pixels mix hot and cold gas);
  <1 - T_gamma/T>_HI (the spin factor for full coupling, T_S = T_K);  <rho (1 - x_HII)> / (Omega_b rho_crit) (renders: / their own gas mean).
If the cone lies between 640^3 and 320^3 in T_harm and the spin factor, its excess is the pixel size; if <T>_m
differs, the cone's T is not the same quantity as the renders'.

Usage
-----
  python check_cone_T.py [--z 7.5 9.5 11 13 15] [--dz 0.1]
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

from lightcone_tb import read_cone, units

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

BASE = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
CONES = [f"{BASE}/Flagship_new/postprocessing/rlc_640", f"{BASE}/Flagship_below_z_4p75_new/postprocessing/rlc_640"]
RENDERS = (f"{BASE}/Flagship_new/postprocessing/ren_640", f"{BASE}/Flagship_below_z_4p75_new/postprocessing/ren_640")
ION = "/orcd/data/mvogelsb/004/chcho/ion_history/ionization_history.csv"
T_CMB0 = 2.7255
FIELDS = ("Density", "Temperature", "HII_Fraction")


def stats(rho, T, x, Tg, w=None):
    """weighted sums -> (<T>_m, <T>_HI, T_harm, <1 - Tg/T>_HI, <rho (1-x)>); w: per-pixel volume weight (broadcast)"""
    w = 1.0 if w is None else w
    m = rho * (1.0 - x)
    s = lambda a: float(np.sum(w * a, dtype=np.float64))
    sw = s(np.ones_like(rho))
    return (s(rho * T) / s(rho), s(m * T) / s(m), s(m) / s(m / T), s(m * (1.0 - Tg / T)) / s(m), s(m) / sw)


def render_fields(snap):
    d = RENDERS[0] if snap <= 428 else RENDERS[1]
    out = {}
    for k in FIELDS:
        p = os.path.join(d, k, f"{k}_{snap:03d}.hdf5")
        with h5py.File(p if os.path.exists(p) else os.path.join(d, "All", f"All_{snap:03d}.hdf5"), "r") as f:
            out[k] = f[k][()].astype(np.float64)
    with h5py.File(os.path.join(d, "All", f"All_{snap:03d}.hdf5"), "r") as f:
        z = float(f["Header"].attrs["Redshift"])
    return out, z


def half(f):
    """2 x 2 x 2 average: rho by volume, T and x_HII mass-weighted (as ren_640 is made from ren_1280)"""
    b = lambda a: a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2] // 2, 2).sum(axis=(1, 3, 5))
    r = b(f["Density"])
    return {"Density": r / 8.0, "Temperature": b(f["Density"] * f["Temperature"]) / r,
            "HII_Fraction": b(f["Density"] * f["HII_Fraction"]) / r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=float, nargs="+", default=[7.5, 9.5, 11.0, 13.0, 15.0])
    ap.add_argument("--dz", type=float, default=0.1)
    ap.add_argument("--cones", nargs="+", default=CONES)
    ap.add_argument("--ion", default=ION)
    a = ap.parse_args()
    ion = pd.read_csv(a.ion, comment="#")

    print(f"  {'':24} {'<T>_m':>9} {'<T>_HI':>9} {'T_harm':>9} {'<1-Tg/T>_HI':>12} {'<rho(1-x)>/rho_b':>17}")
    for zt in a.z:
        cone = read_cone(a.cones, fields=FIELDS, sel=np.s_[:, :], zmin=zt - a.dz, zmax=zt + a.dz)
        rho_b = units(cone["header"])[1]
        chi = 0.5 * (cone["chi_edges"][:-1] + cone["chi_edges"][1:])
        w = (chi**2 * np.abs(np.diff(cone["chi_edges"])))[None, None, :]
        zc = cone["z"]
        zeff = float(np.sum(w[0, 0] * zc) / w.sum())
        pix = cone["header"]["OpeningAngle"] * chi.mean() / cone["Density"].shape[0]
        depth = (chi.max() - chi.min()) / max(len(chi) - 1, 1)
        c = stats(cone["Density"].astype(np.float64) / rho_b, cone["Temperature"].astype(np.float64),
                  cone["HII_Fraction"].astype(np.float64), T_CMB0 * (1.0 + zc)[None, None, :], w)
        snap = int(ion.loc[(ion["z"] - zeff).abs().idxmin(), "snap"])
        f, zr = render_fields(snap)
        rho_mean = f["Density"].mean()                            # the render's gas mean (units need not match the cone's)
        print(f"\nband z = {zc.min():.3f} - {zc.max():.3f}, mean z {zeff:.4f}; cone pixel {pix:.3f} x {depth:.3f} cMpc/h; "
              f"render {snap} (z = {zr:.4f}), 640^3 pixel {338.3 / 640:.3f}, 320^3 {338.3 / 320:.3f}; "
              f"render <rho> in cone units / (Omega_b rho_crit) = {rho_mean / rho_b:.4f}")
        print(f"  {'cone':24} {c[0]:9.4g} {c[1]:9.4g} {c[2]:9.4g} {c[3]:12.5f} {c[4]:17.4f}")
        Tg = T_CMB0 * (1.0 + zr)
        for lab, g in (("render 640^3", f), ("render 320^3", half(f))):
            r = stats(g["Density"] / rho_mean, g["Temperature"], g["HII_Fraction"], Tg)
            print(f"  {lab:24} {r[0]:9.4g} {r[1]:9.4g} {r[2]:9.4g} {r[3]:12.5f} {r[4]:17.4f}")
        del cone, f


if __name__ == "__main__":
    main()
