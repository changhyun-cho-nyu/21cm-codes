"""
check_cone_outside.py
=====================
How much does the choice of J_alpha outside the J snapshots' z range matter on the light cone (z 30 -> 2.98669)?
H J: z 2.9981 - 29.4285; He J: z 2.9981 - 10.0140. lightcone_J.cone_J needs outside="zero" (J = 0) or "hold" (the
end snapshot's J at the slice's cells) for the slices beyond. For each end it prints the number of cone slices
beyond, and, for the last band of slices inside the range (width --dz), <delta T_b> with the sampled J and with
J = 0. "hold" beyond the end gives about the sampled-J value there, "zero" the J = 0 value, so the difference is
the size of the choice at the end; further out it shrinks if J (and its share of the coupling) falls.

Usage
-----
  python check_cone_outside.py [--dz 0.05]
"""
import argparse

import h5py
import numpy as np

from lightcone_J import JDIR, cone_J, j_files
from lightcone_tb import cone_tb, read_cone

BASE = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
CONES = [f"{BASE}/Flagship_new/postprocessing/rlc_640", f"{BASE}/Flagship_below_z_4p75_new/postprocessing/rlc_640"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dz", type=float, default=0.05)
    ap.add_argument("--cones", nargs="+", default=CONES)
    ap.add_argument("--jdir", default=JDIR)
    a = ap.parse_args()

    zc_all = []
    for d in a.cones:
        with h5py.File(f"{d}/All.hdf5", "r") as f:
            e = f["Redshifts"][()]
        zc_all.append(0.5 * (e[:-1] + e[1:]))
    zc_all = np.concatenate(zc_all)
    ze = zc_all
    print(f"cone slice centres z {ze.max():.4f} - {ze.min():.5f}, {len(ze)} slices")
    for line, unit in (("H", "mK"), ("He", "uK")):
        zs = np.sort([f[2] for f in j_files(line, a.jdir).values()])
        for end, zlo, zhi, beyond in (("low", zs[0], zs[0] + a.dz, zc_all < zs[0]),
                                      ("high", zs[-1] - a.dz, zs[-1], zc_all > zs[-1])):
            nb = int(beyond.sum())
            zr = (zc_all[beyond].min(), zc_all[beyond].max()) if nb else None
            cone = read_cone(a.cones, sel=np.s_[:, :], zmin=zlo, zmax=zhi)
            keep = (cone["z"] >= zs[0]) & (cone["z"] <= zs[-1])
            if keep.sum() < 1:
                print(f"  {line} {end} end: no slices inside z {zlo:.4f} - {zhi:.4f}")
                continue
            sub = {k: (v[..., keep] if isinstance(v, np.ndarray) and v.ndim == 3 else v) for k, v in cone.items()}
            sub["z"] = cone["z"][keep]
            sub["chi_edges"] = np.concatenate([cone["chi_edges"][:-1][keep], cone["chi_edges"][1:][keep][-1:]])
            w = sub["chi_edges"]
            w = 0.5 * (w[:-1] + w[1:])**2 * np.abs(np.diff(w))
            J, _ = cone_J(sub, line, np.s_[:, :], verbose=False, jdir=a.jdir)
            mean = lambda t: float(np.sum(w * t.mean(axis=(0, 1), dtype=np.float64)) / w.sum())
            t1, t0 = mean(cone_tb(sub, line, J)), mean(cone_tb(sub, line, np.zeros_like(J)))
            print(f"  {line} {end} end (J z {zs[0]:.4f} - {zs[-1]:.4f}): {nb} slices beyond"
                  + (f" (z {zr[0]:.4f} - {zr[1]:.4f})" if nb else "")
                  + f"; last band inside z {sub['z'].min():.4f} - {sub['z'].max():.4f} ({keep.sum()} slices): "
                  f"<dT_b> sampled J {t1:.5g}, J = 0 {t0:.5g} {unit}, (J=0)/J - 1 = {t0 / t1 - 1:+.3%}")
            del cone, sub, J


if __name__ == "__main__":
    main()
