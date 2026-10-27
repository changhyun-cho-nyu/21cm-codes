"""
check_lightcone_tb.py
=====================
Does the light-cone delta T_b of lightcone_tb.py agree with the production 640^3 maps? For bands of full light-cone
slices (all 640 x 640 pixels) around each target z, compares the volume-weighted band means with the production
global means (volume mean of RIN_NNN.hdf5), interpolated linearly in z between the two production snapshots that
bracket the band's mean z:
  - <rho> / (Omega_b rho_crit) and <x_HII>_V (HII_VolumeFraction; production: ion_history/ionization_history.csv);
  - <delta T_b^H> (and <delta T_b^He> where He maps exist) with the production J_alpha (stars + AGN) sampled onto
    the cone (lightcone_J.cone_J, time linear by default), and, for comparison, J_box(z) x (1 + delta) with the box
    mean of the production J (J_mean attributes). The old table LUMINA_ALL.csv (pre-production J) is not used.
    Expected cone / maps = 1 with the sampled J; differences also from the pixel size (cone 0.43 cMpc/h at z = 3,
    0.64 at z = 10, 0.73 at z = 20; maps 0.53), which matters for 3.46 cm at all z and 21 cm at z >~ 10.
Light-cone pixels are angular, so each slice is weighted by chi^2 d chi. Errors: the scatter of 8 blocks (2 x 2
across the sky x 2 in depth) / sqrt(8), a rough sample-variance estimate. Expected cone / maps = 1 within the
errors and the normalisation difference (cone: Omega_b rho_crit; maps: the map's gas mean; -0.4% to +0.7%).

Usage
-----
  python check_lightcone_tb.py [--z 3.5 4.5 5.5 6.5 7.5 8.5 9.5 11 13 15] [--dz 0.1] [--time linear]
Bands outside the J snapshots' z range of a line (He: z > 10.014) are skipped for that line.
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
import pandas as pd

from lightcone_J import JDIR, box_mean, cone_J, j_files
from lightcone_tb import FIELDS, cone_tb, read_cone, units

BASE = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
CONES = [f"{BASE}/Flagship_new/postprocessing/rlc_640", f"{BASE}/Flagship_below_z_4p75_new/postprocessing/rlc_640"]
USER = "/orcd/data/mvogelsb/004/chcho"


def map_list(d, ion):
    """{snapshot: z} of the RIN_NNN.hdf5 maps in d (z from the map's Redshift attribute, else the ionization history)"""
    out = {}
    for p in glob.glob(os.path.join(d, "RIN_*.hdf5")):
        m = re.fullmatch(r"RIN_(\d+)\.hdf5", os.path.basename(p))
        if not m:
            continue
        s = int(m.group(1))
        with h5py.File(p, "r") as f:
            z = f["rin"].attrs.get("Redshift")
        if z is None:
            if s not in ion.index:
                continue
            z = ion.loc[s, "z"]
        out[s] = float(z)
    return out


def map_mean(p):
    with h5py.File(p, "r") as f:
        d = f["rin"]
        n = d.shape[0]
        return sum(float(np.sum(d[i:i + 64], dtype=np.float64)) for i in range(0, n, 64)) / d.size


def bracket(maps, z):
    lo = [(z0, s) for s, z0 in maps.items() if z0 <= z]
    hi = [(z0, s) for s, z0 in maps.items() if z0 >= z]
    if not lo or not hi:
        return None
    return max(lo), min(hi)                                  # (z, snap) just below and just above


def interp(z, a, b):
    (za, va), (zb, vb) = a, b
    return va if zb == za else va + (vb - va) * (z - za) / (zb - za)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", type=float, nargs="+", default=[3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11.0, 13.0, 15.0])
    p.add_argument("--dz", type=float, default=0.1, help="half width of each band in z")
    p.add_argument("--cones", nargs="+", default=CONES)
    p.add_argument("--maps-H", default=f"{USER}/Tb_maps_prod/H_real")
    p.add_argument("--maps-He", default=f"{USER}/Tb_maps_prod/He")
    p.add_argument("--ion", default=f"{USER}/ion_history/ionization_history.csv")
    p.add_argument("--jdir", default=JDIR)
    p.add_argument("--time", choices=("linear", "nearest"), default="linear", help="J between J snapshots (cone_J)")
    a = p.parse_args()

    ion = pd.read_csv(a.ion, comment="#").set_index("snap")
    jf = {line: j_files(line, a.jdir) for line in ("H", "He")}
    bm = {line: box_mean(jf[line], line) for line in ("H", "He")}
    for line in ("H", "He"):
        print(f"production J {line}: {len(jf[line])} snapshots, z {bm[line][0][0]:.4f} - {bm[line][0][-1]:.4f} ({a.jdir}); "
              f"time {a.time}")
    maps = {"H": map_list(a.maps_H, ion), "He": map_list(a.maps_He, ion)}
    print(f"production maps: H {len(maps['H'])} ({a.maps_H}), He {len(maps['He'])} ({a.maps_He})")
    zi = ion["z"].values
    xi = ion["xHII_V"].values

    for zt in a.z:
        cone = read_cone(a.cones, fields=FIELDS + ("HII_VolumeFraction",), sel=np.s_[:, :], zmin=zt - a.dz, zmax=zt + a.dz)
        rho_mean = units(cone["header"])[1]
        chi = 0.5 * (cone["chi_edges"][:-1] + cone["chi_edges"][1:])
        w = chi**2 * np.abs(np.diff(cone["chi_edges"]))          # angular pixels: volume ~ chi^2 d chi
        zc = cone["z"]
        zeff = np.sum(w * zc) / w.sum()
        nx, ny, nz = cone["Density"].shape
        if nz < 4:
            raise SystemExit(f"only {nz} slices in the band around z = {zt}: increase --dz")
        blocks = [(np.s_[i * nx // 2:(i + 1) * nx // 2], np.s_[j * ny // 2:(j + 1) * ny // 2], np.s_[k * nz // 2:(k + 1) * nz // 2])
                  for i in range(2) for j in range(2) for k in range(2)]

        def stat(f):
            def wm(x, sl):
                m = x[sl[0], sl[1], sl[2]].mean(axis=(0, 1), dtype=np.float64)
                ww = w[sl[2]]
                return np.sum(ww * m) / ww.sum()
            allm = wm(f, (np.s_[:], np.s_[:], np.s_[:]))
            b = np.array([wm(f, sl) for sl in blocks])
            return allm, b.std(ddof=1) / np.sqrt(len(b))

        delta = cone["Density"] / rho_mean
        r, re_ = stat(delta)
        x, xe = stat(cone["HII_VolumeFraction"])
        xprod = np.interp(zeff, zi[np.argsort(zi)], xi[np.argsort(zi)])
        print(f"\nband z = {zc.min():.3f} - {zc.max():.3f} ({nz} slices, {chi.max() - chi.min():.1f} cMpc/h deep, "
              f"{cone['header']['OpeningAngle'] * chi.mean():.1f} cMpc/h wide), mean z {zeff:.4f}")
        print(f"  <rho>/(Omega_b rho_crit) = {r:.4f} +- {re_:.4f};  <x_HII>_V cone {x:.4f} +- {xe:.4f}, renders {xprod:.4f}")
        for line, unit in (("H", "mK"), ("He", "uK")):
            br = bracket(maps[line], zeff)
            if br is None:
                print(f"  {line}: no production maps bracket z = {zeff:.3f}")
                continue
            zb, Jb = bm[line]
            if zc.min() < zb[0] or zc.max() > zb[-1]:
                print(f"  {line}: band outside the J snapshots (z {zb[0]:.4f} - {zb[-1]:.4f})")
                continue
            dirm = a.maps_H if line == "H" else a.maps_He
            vals = [(z0, map_mean(os.path.join(dirm, f"RIN_{s}.hdf5"))) for z0, s in br]
            prod = interp(zeff, vals[0], vals[1])
            res = []
            for lab in ("J cone", "J_box(z)(1+delta)"):
                if lab == "J cone":
                    J = cone_J(cone, line, np.s_[:, :], time=a.time, jdir=a.jdir, verbose=False)[0]
                else:
                    J = np.interp(zc, zb, Jb).astype(np.float32)[None, None, :] * delta
                tb = cone_tb(cone, line, J)
                del J
                m, e = stat(tb)
                res.append(f"{lab}: {m:.5g} +- {e:.2g} ({m / prod:.4f})")
                del tb
            print(f"  <dT_b^{line}> [{unit}] maps {prod:.5g} (snapshots {br[0][1]} z = {br[0][0]:.3f}: {vals[0][1]:.5g}, "
                  f"{br[1][1]} z = {br[1][0]:.3f}: {vals[1][1]:.5g});  cone " + ";  ".join(res))
        del cone, delta


if __name__ == "__main__":
    main()
