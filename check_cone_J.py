"""
check_cone_J.py
===============
Step 2 of J_alpha option (c): the production J sampled onto one row of the light cone (the fan diagram) with
lightcone_J.cone_J, in bands of z:
  1. the row's mean J against the box mean of the J snapshots (the J files' J_mean attributes, stars + AGN,
     interpolated in z): ~1 within the row's sample variance if the sampling is right; and against the global
     table (LUMINA_ALL.csv) used by the fan notebook so far;
  2. time="nearest" against time="linear": per-pixel |J_nearest / J_linear - 1| (median, 99th percentile), and the
     change in delta T_b (mean, and rms of the difference / rms of delta T_b);
  3. delta T_b with the sampled J (linear) against the notebook's J(z) (1 + delta) with J(z) from the table, and
     against J_box(z) (1 + delta) with the box mean of the production J: mean and rms of the difference. The table
     column mixes the table's normalisation with the missing spatial structure of J; the J_box column is the
     structure alone.
Only slices inside the J snapshots' z range are used (no choice of J outside it is made here).

Usage
-----
  python check_cone_J.py [--row 64 320 576] [--zmax 20]
  python check_cone_J.py --stride 32          # every 32nd row (20 rows across the cone) in one table
Rows sample nearly one x plane of the box each (x ~ O_x + t_x p mod BoxSize): rows 64 and 576 see nearly the same
plane at z ~ 10-20 (512 pixels ~ BoxSize at z ~ 12) and row 320 the plane x = O_x at all z. --stride spreads the
rows over the box for the <J>row/<J>box test.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import h5py
from lightcone_J import DSETS, JDIR, cone_J, j_files
from lightcone_tb import cone_tb, read_cone, units

BASE = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
CONES = [f"{BASE}/Flagship_new/postprocessing/rlc_640", f"{BASE}/Flagship_below_z_4p75_new/postprocessing/rlc_640"]
BANDS = (3, 4, 4.75, 5, 6, 7, 8, 10, 12, 15, 20)


def box_mean(files, band):
    """z (increasing) and the box-mean J (stars + AGN) of the J snapshots, from the J_mean attributes"""
    rows = []
    for s, (p1, p2, z) in files.items():
        with h5py.File(p1, "r") as f1, h5py.File(p2, "r") as f2:
            m1, m2 = f1[DSETS[band][0]].attrs.get("J_mean"), f2[DSETS[band][1]].attrs.get("J_mean")
        if m1 is not None and m2 is not None:
            rows.append((z, float(m1) + float(m2)))
    rows.sort()
    return np.array(rows).T if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", type=int, nargs="+", default=[64],
                    help="ix of the row(s) (the fan notebook: int(0.1 * 640) = 64); several rows: one table each")
    ap.add_argument("--stride", type=int, default=None,
                    help="instead of --row: every stride-th row (ix = 0, stride, ...) in one table")
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--cones", nargs="+", default=CONES)
    ap.add_argument("--jdir", default=JDIR)
    ap.add_argument("--jcsv", default="/orcd/data/mvogelsb/004/chcho/LUMINA_ALL.csv")
    a = ap.parse_args()
    jt = pd.read_csv(a.jcsv).sort_values("redshift")
    sels = [(f"ix = 0:640:{a.stride}", np.s_[::a.stride, :])] if a.stride else [(f"ix = {r}", np.s_[r, :]) for r in a.row]

    for band, col, unit in (("H", "J_alpha", "mK"), ("He", "J_alpha_He", "uK")):
        files = j_files(band, a.jdir)
        zj = np.array(sorted(f[2] for f in files.values()))
        zmin, zmax = zj[0], min(zj[-1], a.zmax)
        bm = box_mean(files, band)
        for label, sel in sels:
            print(f"\n===== {band}: {len(files)} J snapshots, z {zj[0]:.4f} - {zj[-1]:.4f}; cone rows {label}, "
                  f"slices within z {zmin:.4f} - {zmax:.4f}", flush=True)
            cone = read_cone(a.cones, sel=sel, zmin=zmin, zmax=zmax)
            zc = cone["z"]
            Jl, info = cone_J(cone, band, sel, time="linear", jdir=a.jdir)
            Jn, _ = cone_J(cone, band, sel, time="nearest", jdir=a.jdir, verbose=False)
            delta = cone["Density"] / units(cone["header"])[1]
            Jtab = interp1d(jt["redshift"], jt[col], bounds_error=False, fill_value=np.nan)(zc)
            Jmod = Jtab[None, :] * delta
            Jbox = np.interp(zc, bm[0], bm[1]) if bm is not None else np.full(zc.shape, np.nan)
            tl, tn, tm = cone_tb(cone, band, Jl), cone_tb(cone, band, Jn), cone_tb(cone, band, Jmod)
            tb = cone_tb(cone, band, Jbox[None, :] * delta)
            print(f"  J table {a.jcsv} column {col}: z {jt['redshift'].min():.3f} - {jt['redshift'].max():.3f}")
            print(f"  {'z band':>12} {'slices':>6} | {'<J>row/<J>box':>13} {'<J>row/table':>12} | {'nearest vs linear: |dJ/J| med, 99%':>34} "
                  f"{'d<Tb>':>9} {'rms dTb/rms Tb':>14} | {'<Tb> J lin':>11} {'<Tb> Jtab(1+d)':>14} {'rms diff/rms':>12} | "
                  f"{'<Tb> Jbox(1+d)':>14} {'rms diff/rms':>12}")
            for lo, hi in zip(BANDS[:-1], BANDS[1:]):
                m = (zc >= lo) & (zc < hi)
                if m.sum() < 2:
                    continue
                pos = Jl[..., m] > 0                                   # He J can be exactly 0 (fully absorbed)
                rel = np.abs(Jn[..., m][pos] / Jl[..., m][pos] - 1.0)
                dt = tn[..., m] - tl[..., m]
                sd = tl[..., m].std()
                print(f"  {lo:5.2f}-{hi:5.2f} {m.sum():6d} | {Jl[..., m].mean() / Jbox[m].mean():13.4f} {Jl[..., m].mean() / np.nanmean(Jtab[m]):12.4f} | "
                      f"{np.median(rel):17.2e} {np.percentile(rel, 99):16.2e} {dt.mean():9.3g} {dt.std() / sd:14.3g} | "
                      f"{tl[..., m].mean():11.4g} {tm[..., m].mean():14.4g} {(tm[..., m] - tl[..., m]).std() / sd:12.3g} | "
                      f"{tb[..., m].mean():14.4g} {(tb[..., m] - tl[..., m]).std() / sd:12.3g}")
            print(f"  ({unit}; J snapshots used: {len({s for w in info['weights'] for s in w})}; pixels with J = 0: "
                  f"{np.mean(Jl == 0):.2%}, left out of |dJ/J|)")
            del cone, Jl, Jn, tl, tn, tm, tb, delta, Jmod

if __name__ == "__main__":
    main()
