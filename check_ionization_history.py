"""
check_ionization_history.py
===========================
The global ionization history of Lumina from the 640^3 renders, and a check of the snapshot <-> z <-> x tables
that the notebooks hard-code (xhii_vals / xheiii_vals / z_match / z_map).

For each snapshot: z from the render header, and four global averages of x_HII and x_HeIII (= HeIII / y; the
render He fractions are per H nucleus):
  V     sum(Volume * HII_VolumeFraction) / sum(Volume)   volume-weighted over the gas cells (as Zier+26 Figs 16, 21
                                                          and the LIM paper Fig. 2: <x>_V); exact for any pixel size
  Vpix  pixel mean of HII_VolumeFraction                 = V if every pixel holds the same gas volume
  M     sum(Density * HII_Fraction) / sum(Density)       mass-weighted
  pix   pixel mean of HII_Fraction                        neither (a mass-weighted field averaged by volume)
The render sets star-forming gas neutral (MassHI = Mass, He II = He III = 0), so M is biased low by its mass
share; V hardly at all.

Measure (on the cluster; one CSV per task, all snapshots by default):
  python check_ionization_history.py measure --dirA <ren_640 <= 428> --dirB <ren_640 > 428> --task 0 --ntask 16 --out part_00.csv
Report (merges the CSVs, finds where each definition crosses 0.1 ... 0.9, compares with the notebook tables and
Zier+26, and writes the merged history):
  python check_ionization_history.py report --csv 'part_*.csv' --write ionization_history.csv
"""
import argparse
import glob
import os

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

X_H = 0.76
Y_HE = (1.0 - X_H) / (4.0 * X_H)
SNAP_SWITCH = 428                      # snapshots > 428 are in the second render directory
DEFS = ("V", "Vpix", "M", "pix")
COLS = ["snap", "z"] + [f"xHII_{d}" for d in DEFS] + [f"xHeIII_{d}" for d in DEFS]

# What the notebooks use (PS_redshift, visual_PS, visual_maps, PS2Bubble, one-stat, subvol_VID, bispec_H/He):
# x -> (z_match, snapshot of bispec z_map)
NB_H = {0.1: (9.170129, 177), 0.2: (8.207879, 203), 0.3: (7.624910, 220), 0.4: (7.204283, 242), 0.5: (6.872631, 264),
        0.6: (6.592783, 283), 0.7: (6.342011, 300), 0.8: (6.099902, 318), 0.9: (5.826679, 338)}
NB_HE = {0.1: (4.999257, 406), 0.2: (4.517370, 450), 0.3: (4.299100, 471), 0.4: (4.144300, 487), 0.5: (4.011606, 500),
         0.6: (3.880495, 514), 0.7: (3.746897, 529), 0.8: (3.606240, 558), 0.9: (3.439090, 596)}
# snapshot -> rounded z label (z_map in bispec_*, z_labels in PS_redshift / visual_PS / visual_maps)
NB_ZLABEL = {156: 10.0, 181: 9.0, 209: 8.0, 255: 7.0, 289: 6.5, 325: 6.0, 364: 5.5, 406: 5.0, 501: 4.0, 582: 3.5,
             705: 3.0}
# Zier et al. 2026, Figs 16 and 21 (panel labels, <x>_V = 0.1, 0.5, 0.9); LIM paper: midpoint z ~ 6.9
ZIER_H = {0.1: 9.16, 0.5: 7.13, 0.9: 5.83}
ZIER_HE = {0.1: 5.00, 0.5: 4.01, 0.9: 3.44}


def open_field(d, name, s):
    """the dataset `name` of snapshot s: per-field file if present, else All/All_NNN.hdf5; None if absent"""
    for p in (os.path.join(d, name, f"{name}_{s:03d}.hdf5"), os.path.join(d, "All", f"All_{s:03d}.hdf5")):
        if os.path.exists(p):
            f = h5py.File(p, "r")
            if name in f:
                return f, f[name]
            f.close()
    return None, None


def measure_snap(d, s, planes):
    zs = []
    for p in (os.path.join(d, "All", f"All_{s:03d}.hdf5"), os.path.join(d, "Density", f"Density_{s:03d}.hdf5")):
        if os.path.exists(p):
            with h5py.File(p, "r") as f:
                if "Header" in f and "Redshift" in f["Header"].attrs:
                    zs.append(float(f["Header"].attrs["Redshift"]))
    if not zs:
        return None
    if max(zs) - min(zs) > 1e-6:
        print(f"  WARNING snap {s}: header redshifts differ between files: {zs}")
    names = {"rho": "Density", "xM": "HII_Fraction", "xV": "HII_VolumeFraction", "vol": "Volume",
             "hM": "HeIII_Fraction", "hV": "HeIII_VolumeFraction"}
    files, ds = {}, {}
    for k, n in names.items():
        files[k], ds[k] = open_field(d, n, s)
    if ds["rho"] is None or ds["xM"] is None:
        print(f"  snap {s}: no Density / HII_Fraction in {d}, skipped")
        return None
    N = ds["rho"].shape[0]
    S = dict.fromkeys(["rho", "rho_xM", "xM", "xV", "vol", "vol_xV", "rho_hM", "hM", "hV", "vol_hV"], 0.0)
    for i0 in range(0, N, planes):
        sl = slice(i0, min(N, i0 + planes))
        g = {k: (v[sl].astype(np.float64) if v is not None else None) for k, v in ds.items()}
        S["rho"] += g["rho"].sum(); S["rho_xM"] += (g["rho"] * g["xM"]).sum(); S["xM"] += g["xM"].sum()
        if g["hM"] is not None:
            S["rho_hM"] += (g["rho"] * g["hM"]).sum(); S["hM"] += g["hM"].sum()
        if g["xV"] is not None:
            S["xV"] += g["xV"].sum()
            if g["vol"] is not None:
                S["vol"] += g["vol"].sum(); S["vol_xV"] += (g["vol"] * g["xV"]).sum()
        if g["hV"] is not None:
            S["hV"] += g["hV"].sum()
            if g["vol"] is not None:
                S["vol_hV"] += (g["vol"] * g["hV"]).sum()
    for f in files.values():
        if f is not None:
            f.close()
    n3, nan = float(N)**3, float("nan")
    have = {k: ds[k] is not None for k in ds}
    row = {"snap": s, "z": zs[0],
           "xHII_V": S["vol_xV"] / S["vol"] if have["xV"] and have["vol"] else nan,
           "xHII_Vpix": S["xV"] / n3 if have["xV"] else nan,
           "xHII_M": S["rho_xM"] / S["rho"],
           "xHII_pix": S["xM"] / n3,
           "xHeIII_V": S["vol_hV"] / S["vol"] / Y_HE if have["hV"] and have["vol"] else nan,
           "xHeIII_Vpix": S["hV"] / n3 / Y_HE if have["hV"] else nan,
           "xHeIII_M": S["rho_hM"] / S["rho"] / Y_HE if have["hM"] else nan,
           "xHeIII_pix": S["hM"] / n3 / Y_HE if have["hM"] else nan}
    missing = [names[k] for k in names if not have[k]]
    if missing:
        print(f"  snap {s}: missing {missing}")
    return row


def measure(a):
    snaps = [s for s in range(a.smin, a.smax + 1)][a.task::a.ntask]
    with open(a.out + ".part", "w") as fo:
        fo.write(",".join(COLS) + "\n")
        for s in snaps:
            row = measure_snap(a.dirA if s <= SNAP_SWITCH else a.dirB, s, a.planes)
            if row is None:
                continue
            fo.write(",".join(f"{row[c]:d}" if c == "snap" else f"{row[c]:.8g}" for c in COLS) + "\n")
            fo.flush()
            print(f"snap {s:3d} z {row['z']:.4f}  x_HII V {row['xHII_V']:.4f} M {row['xHII_M']:.4f} "
                  f"pix {row['xHII_pix']:.4f}  x_HeIII V {row['xHeIII_V']:.4f} M {row['xHeIII_M']:.4f}", flush=True)
    os.replace(a.out + ".part", a.out)


def crossing(z, x, t):
    """first redshift (going forward in time) at which x reaches t, linear in x between snapshots; nearest snap"""
    ok = np.isfinite(x)
    z, x, idx = z[ok], x[ok], np.nonzero(ok)[0]
    above = np.nonzero(x >= t)[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan, None
    i = above[0]
    zc = z[i - 1] + (t - x[i - 1]) * (z[i] - z[i - 1]) / (x[i] - x[i - 1])
    j = i if abs(x[i] - t) <= abs(x[i - 1] - t) else i - 1
    return zc, idx[j]


def report(a):
    rows = []
    for p in sorted(glob.glob(a.csv)):
        t = np.genfromtxt(p, delimiter=",", names=True)
        if t.size:
            rows.append(np.atleast_1d(t))
    if not rows:
        raise SystemExit(f"no rows in {a.csv}")
    t = np.concatenate(rows)
    t = t[np.argsort(t["snap"])]
    _, first = np.unique(t["snap"], return_index=True)
    t = t[first]
    snap, z = t["snap"].astype(int), t["z"]
    print(f"{len(snap)} snapshots, {snap.min()}-{snap.max()}, z {z.max():.3f}-{z.min():.3f}")
    gaps = np.setdiff1d(np.arange(snap.min(), snap.max() + 1), snap)
    if len(gaps):
        print(f"  missing snapshots: {gaps.tolist()[:40]}{' ...' if len(gaps) > 40 else ''}")
    if np.any(np.diff(z) >= 0):
        print(f"  WARNING: z not decreasing with snapshot number at {snap[1:][np.diff(z) >= 0].tolist()}")
    at = {s: i for i, s in enumerate(snap)}

    print("\nSnapshot redshift labels used in the notebooks (z_map / z_labels):")
    for s, zl in NB_ZLABEL.items():
        if s in at:
            i = at[s]
            print(f"  snap {s:3d}: header z = {z[i]:.4f}, label {zl:4.1f}, diff {z[i] - zl:+.4f}   "
                  f"x_HII_V {t['xHII_V'][i]:.4f}  x_HeIII_V {t['xHeIII_V'][i]:.4f}")

    for sp, nbt, zier in (("HII", NB_H, ZIER_H), ("HeIII", NB_HE, ZIER_HE)):
        print(f"\n=== x_{sp}: notebook table (z_match, snapshot) against the renders ===")
        print("  the notebook snapshot: its header z vs z_match, and its x in each definition")
        print(f"  {'x':>4} {'snap':>5} {'z_match':>9} {'z(hdr)':>9} " + " ".join(f"{'x_' + d:>8}" for d in DEFS))
        for x0, (zm, s) in nbt.items():
            if s not in at:
                print(f"  {x0:4.1f} {s:5d} {zm:9.5f}   (not measured)"); continue
            i = at[s]
            print(f"  {x0:4.1f} {s:5d} {zm:9.5f} {z[i]:9.5f} " + " ".join(f"{t[f'x{sp}_{d}'][i]:8.4f}" for d in DEFS))
        print("  where each definition crosses x (z interpolated between snapshots; nearest snapshot and its x)")
        print(f"  {'x':>4} " + " ".join(f"{d + ': z, snap (x)':>26}" for d in DEFS) + f" {'notebook':>9} {'Zier+26':>8}")
        for x0, (zm, s) in nbt.items():
            cells = []
            for d in DEFS:
                xx = t[f"x{sp}_{d}"]
                zc, j = crossing(z, xx, x0)
                cells.append(f"{zc:6.3f}, {snap[j]:3d} ({xx[j]:.3f})" if j is not None else f"{'-':>20}")
            print(f"  {x0:4.1f} " + " ".join(f"{c:>26}" for c in cells) + f" {zm:9.3f} "
                  + (f"{zier[x0]:8.2f}" if x0 in zier else f"{'':8}"))

    if a.write:
        with open(a.write, "w") as fo:
            fo.write("# Lumina global ionization history from the 640^3 renders (check_ionization_history.py)\n")
            fo.write("# V: volume-weighted (Volume x *_VolumeFraction); Vpix: pixel mean of *_VolumeFraction; "
                     "M: mass-weighted (Density x *_Fraction); pix: pixel mean of *_Fraction; x_HeIII = HeIII/y\n")
            fo.write(",".join(COLS) + "\n")
            for r in t:
                fo.write(",".join(f"{int(r[c]):d}" if c == "snap" else f"{r[c]:.8g}" for c in COLS) + "\n")
        print(f"\nwrote {a.write}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--dirA", required=True, help="ren_640 for snapshots <= 428")
    m.add_argument("--dirB", required=True, help="ren_640 for snapshots > 428")
    m.add_argument("--smin", type=int, default=0)
    m.add_argument("--smax", type=int, default=708)
    m.add_argument("--task", type=int, default=0)
    m.add_argument("--ntask", type=int, default=1)
    m.add_argument("--planes", type=int, default=32)
    m.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--csv", required=True, help="glob of the measure CSVs (quote it)")
    r.add_argument("--write", default="", help="merged history CSV")
    a = p.parse_args()
    measure(a) if a.mode == "measure" else report(a)


if __name__ == "__main__":
    main()
