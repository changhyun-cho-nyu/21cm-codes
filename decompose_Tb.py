"""
decompose_Tb.py
===============
What makes up the mean and the fluctuations of a delta T_b map?

For one RIN_NNN.hdf5 map (PS_21cm_mpi.c or PS_3cm_mpi.c) and its 640^3 render, every pixel is classed
  - by whether it contains star-forming gas (StarFormationRate > 0): Lumina sets that gas fully neutral
    by convention and its Temperature is the effective equation-of-state value, not a physical one;
  - by overdensity 1 + delta = Density / <Density>;
  - by ionized fraction x_HII (mass-weighted HII_Fraction);
  - jointly: pixels without star formation and neutral (x_HII < 0.01), by overdensity.
For each class it reports the volume fraction, the contribution to <dT_b> (in map units and as a share of
the total), the mean dT_b in the class, the share of the variance sum (dT_b - <dT_b>)^2, and, for 21 cm,
the share of the H I mass. It also gives <dT_b> with the star-forming pixels set to zero.

Pixels are classed as a whole: a pixel with star formation also holds IGM gas, and its mass-weighted
temperature mixes both, so "without star-forming pixels" is an upper bound on how much those pixels add,
not an exact IGM-only signal.

Usage
-----
  python decompose_Tb.py --rin Tb_maps_prod/H_real/RIN_156.hdf5 --dir <ren_640> --snap 156 [--line H]
"""
import argparse
import os

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def field(d, name, s):
    p = os.path.join(d, name, f"{name}_{s:03d}.hdf5")
    if not os.path.exists(p):
        p = os.path.join(d, "All", f"All_{s:03d}.hdf5")
    return h5py.File(p, "r")[name]


class Acc:
    def __init__(self, labels):
        self.labels = labels
        n = len(labels)
        self.cnt, self.s1, self.hi = np.zeros(n), np.zeros(n), np.zeros(n)
        self.s2 = np.zeros(n)

    def add(self, idx, tb, hi):
        n = len(self.labels)
        ok = idx >= 0
        self.cnt += np.bincount(idx[ok], minlength=n)
        self.s1 += np.bincount(idx[ok], tb[ok], minlength=n)
        self.hi += np.bincount(idx[ok], hi[ok], minlength=n)

    def add2(self, idx, dev2):
        n = len(self.labels)
        ok = idx >= 0
        self.s2 += np.bincount(idx[ok], dev2[ok], minlength=n)

    def report(self, title, ntot, s1tot, s2tot, hitot, unit, line):
        print(f"\n  {title}")
        print(f"    {'class':>22} {'volume':>8} {'<dT_b> contribution':>22} {'share':>8} {'mean in class':>14} {'var share':>9}"
              + (f" {'H I mass':>9}" if line == "H" else ""))
        for i, lab in enumerate(self.labels):
            if self.cnt[i] == 0:
                continue
            print(f"    {lab:>22} {self.cnt[i] / ntot:8.4f} {self.s1[i] / ntot:+18.4e} {unit:>3} "
                  f"{self.s1[i] / s1tot if s1tot else np.nan:+8.3f} {self.s1[i] / self.cnt[i]:+14.4e} "
                  f"{self.s2[i] / s2tot:9.3f}" + (f" {self.hi[i] / hitot:9.3f}" if line == "H" else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rin", required=True)
    p.add_argument("--dir", required=True, help="640^3 render directory (Density, HII_Fraction, StarFormationRate)")
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--line", choices=["H", "He"], default="H")
    p.add_argument("--planes", type=int, default=32)
    a = p.parse_args()

    rin = h5py.File(a.rin, "r")["rin"]
    rho_d, x_d, sfr_d = field(a.dir, "Density", a.snap), field(a.dir, "HII_Fraction", a.snap), field(a.dir, "StarFormationRate", a.snap)
    N = rin.shape[0]
    unit = "mK" if a.line == "H" else "uK"
    with h5py.File(os.path.join(a.dir, "All", f"All_{a.snap:03d}.hdf5"), "r") as f:
        z = float(f["Header"].attrs["Redshift"])

    rho_mean, tb_mean, ntot = 0.0, 0.0, float(N)**3
    for i0 in range(0, N, a.planes):
        rho_mean += rho_d[i0:i0 + a.planes].astype(np.float64).sum()
        tb_mean += rin[i0:i0 + a.planes].astype(np.float64).sum()
    rho_mean /= ntot; tb_mean /= ntot

    d_edges = [0, 0.5, 1, 2, 5, 20, 100, np.inf]
    d_lab = [f"1+d {d_edges[i]:g}-{d_edges[i + 1]:g}" for i in range(len(d_edges) - 1)]
    x_edges = [-np.inf, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, np.inf]
    x_lab = ["x_HII < 1e-3", "1e-3 - 0.01", "0.01 - 0.1", "0.1 - 0.5", "0.5 - 0.9", "0.9 - 0.99", "> 0.99"]
    A_sf, A_d, A_x, A_j = Acc(["no star formation", "star-forming gas"]), Acc(d_lab), Acc(x_lab), Acc(d_lab)
    s2tot, hitot, tb_nosf = 0.0, 0.0, 0.0
    for i0 in range(0, N, a.planes):
        sl = slice(i0, min(N, i0 + a.planes))
        tb = rin[sl].astype(np.float64).ravel()
        dlt = (rho_d[sl].astype(np.float64) / rho_mean).ravel()
        x = x_d[sl].astype(np.float64).ravel()
        sf = (sfr_d[sl] > 0).ravel()
        hi = dlt * (1.0 - x)
        dev2 = (tb - tb_mean)**2
        s2tot += dev2.sum(); hitot += hi.sum(); tb_nosf += tb[~sf].sum()
        isf = sf.astype(np.int64)
        idd = np.clip(np.digitize(dlt, d_edges) - 1, 0, len(d_lab) - 1)
        ixx = np.clip(np.digitize(x, x_edges) - 1, 0, len(x_lab) - 1)
        ij = np.where((~sf) & (x < 0.01), idd, -1)
        for A, idx in ((A_sf, isf), (A_d, idd), (A_x, ixx), (A_j, ij)):
            A.add(idx, tb, hi); A.add2(idx, dev2)
    s1tot = tb_mean * ntot
    print(f"snap {a.snap}, z = {z:.4f}, line {a.line}: <dT_b> = {tb_mean:+.5e} {unit}, rms = {np.sqrt(s2tot / ntot):.4e} {unit}")
    print(f"  <dT_b> with the star-forming pixels set to 0: {tb_nosf / ntot:+.5e} {unit} "
          f"({tb_nosf / s1tot if s1tot else np.nan:.3f} of the total)")
    A_sf.report("by star formation in the pixel", ntot, s1tot, s2tot, hitot, unit, a.line)
    A_d.report("by overdensity 1 + delta", ntot, s1tot, s2tot, hitot, unit, a.line)
    A_x.report("by ionized fraction x_HII (mass-weighted)", ntot, s1tot, s2tot, hitot, unit, a.line)
    A_j.report("pixels without star formation and neutral (x_HII < 0.01), by overdensity "
               "(volume and shares relative to the whole map)", ntot, s1tot, s2tot, hitot, unit, a.line)


if __name__ == "__main__":
    main()
