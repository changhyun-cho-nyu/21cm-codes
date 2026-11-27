"""
analyze_floor_box.py
====================
Compares the floor-box variants (make_variants.py): what the Jeans floor, the 5 K clamp, the entropy switch and the
resolution each do to the temperature of cold neutral gas (no sources, no RT: all gas neutral, mu = 4/(1+3X)).

Per variant and snapshot:
  - box means: mass- and volume-weighted <T>, against T_ad = 54.17 ((1+z)/50)^2 K (Lumina's ICs: 54.17 K at z = 49,
    then adiabatic; for adiabatic gas <T>_V <= T_ad since (1+delta)^(2/3) is concave);
  - by 1+delta (cell density / mean gas density): mass share, median T, mass-weighted harmonic mean <1/T>^-1 (what a
    coupled spin temperature follows), share below 5.5 K, share below 0.8 x its own adiabatic T_ad (1+delta)^(2/3)
    (entropy lost), and the adiabatic value;
  - S = <1 - T_CMB/T>_M: the 21 cm spin factor of fully coupled gas, i.e. the absorption depth the gas could give
    (S < 0 absorption).
Writes <out>/floor_box_summary.csv and floor_box.png (median T vs 1+delta at four redshifts; <T>_M and S vs z).

  python analyze_floor_box.py --root <runs dir> [--variants fid,nofloor,...] [--out ../data]
"""
import argparse
import glob
import os

import h5py
import numpy as np

X_H, K_B, M_P, GAMMA_M1 = 0.76, 1.380649e-16, 1.67262192e-24, 2.0 / 3.0
MU = 4.0 / (1.0 + 3.0 * X_H)
T0, Z0, TCMB0 = 54.17, 49.0, 2.7255
DBINS = (0.0, 0.5, 1.0, 2.0, 3.0, 10.0, np.inf)
ORDER = ("fid", "fid_exact", "nofloor", "noclamp", "entropy", "clean", "fid_hr", "clean_hr")
COLORS = {"fid": "#1f1f1e", "nofloor": "#2a78d6", "noclamp": "#2e9e5b", "entropy": "#eb6834", "clean": "#8a55c7",
          "fid_hr": "#6b6b68", "clean_hr": "#c23b54", "fid_exact": "#b08a1e"}


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(np.asarray(q) * c[-1], c, x[o])


def read(fn):
    with h5py.File(fn, "r") as f:
        h = dict(f["Header"].attrs)
        g = f["PartType0"]
        m = g["Masses"][()].astype(np.float64)
        rho = g["Density"][()].astype(np.float64)
        u = g["InternalEnergy"][()].astype(np.float64)
    z = float(h["Redshift"])
    T = GAMMA_M1 * u * 1e10 * MU * M_P / K_B                 # u in (km/s)^2
    one = rho / (m.sum() / float(h["BoxSize"])**3)
    return z, m, m / rho, T, one


def analyse(fn):
    z, m, V, T, one = read(fn)
    Tad = T0 * ((1.0 + z) / (1.0 + Z0))**2
    Tg = TCMB0 * (1.0 + z)
    row = dict(z=z, n=len(m), T_M=np.average(T, weights=m), T_V=np.average(T, weights=V), T_ad=Tad,
               Thm=m.sum() / np.sum(m / T), S=np.average(1.0 - Tg / T, weights=m),
               below55=m[T < 5.5].sum() / m.sum(),
               lost=m[T < 0.8 * Tad * one**(2.0 / 3.0)].sum() / m.sum())
    for lo, hi in zip(DBINS[:-1], DBINS[1:]):
        s = (one >= lo) & (one < hi)
        k = f"d{lo:g}-{hi:g}"
        if s.sum() < 50:
            continue
        ms, Ts, os_ = m[s], T[s], one[s]
        row[k + "_share"] = ms.sum() / m.sum()
        row[k + "_Tmed"] = wq(Ts, ms, 0.5)
        row[k + "_Thm"] = ms.sum() / np.sum(ms / Ts)
        row[k + "_below55"] = ms[Ts < 5.5].sum() / ms.sum()
        row[k + "_lost"] = ms[Ts < 0.8 * Tad * os_**(2.0 / 3.0)].sum() / ms.sum()
        row[k + "_Tad"] = Tad * wq(os_, ms, 0.5)**(2.0 / 3.0)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--variants", default=None)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    a = p.parse_args()
    names = a.variants.split(",") if a.variants else [v for v in ORDER if os.path.isdir(os.path.join(a.root, v))]
    res = {}
    for v in names:
        fs = sorted(glob.glob(os.path.join(a.root, v, "output", "snap_*.hdf5")))
        if not fs:
            print(f"{v}: no snapshots yet")
            continue
        res[v] = [analyse(f) for f in fs]
        print(f"{v}: {len(fs)} snapshots, z = {res[v][0]['z']:.1f} ... {res[v][-1]['z']:.1f}, {res[v][0]['n']:,} cells")
    if not res:
        return

    # tables: one block per redshift present in the reference variant
    ref = "fid" if "fid" in res else next(iter(res))
    keys_d = [f"d{lo:g}-{hi:g}" for lo, hi in zip(DBINS[:-1], DBINS[1:])]
    for r0 in res[ref]:
        z = r0["z"]
        print(f"\nz = {z:.1f}: T_ad(mean density) = {r0['T_ad']:.1f} K, T_CMB = {TCMB0 * (1 + z):.1f} K")
        print(f"  {'variant':9s} {'<T>_M':>7} {'<T>_V':>7} {'<1/T>^-1':>8} {'S=<1-Tg/T>':>10} {'<5.5K':>6} {'lost':>6} | "
              + " ".join(f"{k[1:]:>11}" for k in keys_d[:5]) + "   (median T [K] by 1+delta)")
        for v, rows in res.items():
            rr = [x for x in rows if abs(x["z"] - z) < 0.05]
            if not rr:
                continue
            x = rr[0]
            meds = " ".join(f"{x.get(k + '_Tmed', np.nan):11.1f}" for k in keys_d[:5])
            print(f"  {v:9s} {x['T_M']:7.1f} {x['T_V']:7.1f} {x['Thm']:8.1f} {x['S']:10.3f} {x['below55']:6.3f} "
                  f"{x['lost']:6.3f} | {meds}")
        print(f"  {'adiabatic':9s} {'':7s} {r0['T_ad']:7.1f}  " + " " * 37 + "| "
              + " ".join(f"{r0.get(k + '_Tad', np.nan):11.1f}" for k in keys_d[:5]))

    os.makedirs(a.out, exist_ok=True)
    cols = sorted({k for rows in res.values() for x in rows for k in x})
    fn = os.path.join(a.out, "floor_box_summary.csv")
    with open(fn, "w") as fo:
        fo.write("variant," + ",".join(cols) + "\n")
        for v, rows in res.items():
            for x in rows:
                fo.write(v + "," + ",".join(f"{x.get(c, np.nan):.6g}" for c in cols) + "\n")
    print(f"\nwrote {fn}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    centres = [0.35, 0.75, 1.5, 2.5, 5.5]
    for v, rows in res.items():
        c = COLORS.get(v, "k")
        zs = [x["z"] for x in rows]
        ax[0].plot(zs, [x["T_M"] for x in rows], color=c, lw=1.8, label=v)
        ax[1].plot(zs, [x["S"] for x in rows], color=c, lw=1.8, label=v)
        last = rows[-1]
        ax[2].plot(centres, [last.get(k + "_Tmed", np.nan) for k in keys_d[:5]], color=c, marker="o", lw=1.5,
                   label=f"{v} (z = {last['z']:.1f})")
    zz = np.linspace(min(x["z"] for r in res.values() for x in r), 49, 100)
    ax[0].plot(zz, T0 * ((1 + zz) / 50)**2, color="k", ls=":", lw=1, label="adiabatic, mean density")
    ax[0].set(xlabel="z", ylabel="<T>_M [K]", yscale="log")
    ax[1].set(xlabel="z", ylabel="S = <1 - T_CMB/T>_M (fully coupled)")
    lastz = res[ref][-1]["z"]
    tad = T0 * ((1 + lastz) / 50)**2
    ax[2].plot(centres, [tad * c**(2 / 3) for c in centres], color="k", ls=":", lw=1, label="adiabatic")
    ax[2].set(xlabel="1+delta", ylabel="median T [K]", xscale="log", yscale="log")
    for x_ in ax:
        x_.grid(alpha=0.3)
        x_.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    png = os.path.join(a.out, "floor_box.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
