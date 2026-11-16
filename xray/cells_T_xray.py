"""
cells_T_xray.py
===============
Gas-cell temperatures of the X-ray test runs at a full snapshot, against the 128^3 render pixels at the same z.
Question (2026-11-16): the H I in neutral, underdense render pixels of the no-X-ray run stays at 17-20 K from z = 29
to 13, where adiabatic gas would cool to ~4 K. Is that the gas cells themselves (heating in the simulation), or warmer
cells averaged into the pixel (sub-pixel mixing)?

scan   (one run; a batch job, reads all gas cells of the snapshot, ~2e8):
  - T = (gamma-1) u mu m_p / k_B, mu = 4/(1 + 3X + 4X Ne), Ne = ElectronAbundance or HII + HeII + 2 HeIII (AREPO-RT);
    H I mass per cell X m (1 - HII_Fraction); star-forming cells (SFR > 0) excluded from the IGM statistics.
  - render axis order from the data (corr of the cell's log density with the log render Density at its pixel, all
    six orders); checks: cell mass deposited at the pixels (nearest pixel) against the render Density, and its
    mass-weighted T against the render Temperature (the render deposits cubes, so these agree only roughly).
  - by the cell's own 1+delta: H I share, neutral (x_HII < 1e-3) share, H I-weighted T percentiles of the neutral
    cells, share below 5.5 K (MinGasTemp = 5 K), and the adiabatic expectation T0(z) (1+delta)^(2/3),
    T0 = 2.7255 (1+z)^2 / 125.8 (adiabatic IGM, the law of the first render at z = 29: 20 K in voids).
  - the sub-pixel test: neutral non-star-forming cells inside neutral pixels (pixel x_HII < 1e-3), grouped by the
    pixel's 1+delta: the cells' H I-weighted T percentiles and harmonic mean against the same pixels' render T.
  - H I-weighted 2-D histogram of (log 1+delta, log T) of neutral non-star-forming cells.
  Output: <out>/<run>/cells_T/cells_T_<snap>.npz and the tables on stdout.
report (both runs, after the scans): tables side by side and data/cells_T_xray.png.

  python cells_T_xray.py scan --run withoutXrays --snap 24 --render 71
  python cells_T_xray.py report --snap 24
Inputs are read only (Oliver's runs); outputs only under /orcd/data/mvogelsb/004/chcho.
"""
import argparse
import glob
import itertools
import os
import re
import sys

import h5py
import numpy as np

SRC = "/nfs/mvogelsblab002/Users/ozier/SonjaXrayPaper"
OUT = "/orcd/data/mvogelsb/004/chcho/xray"
RUNS = ("withoutXrays", "withXrays")
LABELS = {"withoutXrays": "no X-rays", "withXrays": "with X-rays"}
COLORS = {"withoutXrays": "#2a78d6", "withXrays": "#eb6834"}
INK, INK2, GRID = "#1f1f1e", "#6b6b68", "#e6e6e3"
X_H, GAMMA_M1, K_B, M_P = 0.76, 2.0 / 3.0, 1.380649e-16, 1.67262192e-24
XNEU = 1e-3
CBINS = (0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, np.inf)      # cell 1+delta
PBINS = (0.0, 0.5, 1.0, 3.0)                                      # pixel 1+delta (sub-pixel test)
LD_EDGES = np.linspace(-1.5, 3.0, 91)                             # log10(1+delta)
LT_EDGES = np.linspace(0.0, 6.0, 121)                             # log10 T [K]


def wpct(x, w, q):
    if len(x) == 0 or w.sum() <= 0:
        return np.full(len(q), np.nan)
    o = np.argsort(x)
    c = np.cumsum(w[o].astype(np.float64))
    return np.interp(np.asarray(q) / 100.0 * c[-1], c, x[o])


def snap_files(src, run, snap):
    fs = glob.glob(os.path.join(src, run, "output", f"snapdir_{snap:03d}", f"snap_{snap:03d}.*.hdf5"))
    fs = sorted(fs, key=lambda p: int(re.search(r"\.(\d+)\.hdf5$", p).group(1)))
    if not fs:
        sys.exit(f"no full snapshot {snap} for {run} under {src} (snapdir_{snap:03d}/snap_{snap:03d}.<i>.hdf5)")
    return fs


def pick(g, names):
    for n in names:
        if n in g:
            return n
    return None


ALLOWED = "/orcd/data/mvogelsb/004/chcho"


def check_out(path, allow_any):
    rp = os.path.realpath(path)
    if not allow_any and os.path.commonpath([rp, ALLOWED]) != ALLOWED:
        sys.exit(f"refusing to write outside {ALLOWED}: {rp}")


def scan(a):
    od = os.path.join(a.out, a.run, "cells_T")
    check_out(od, a.allow_any_out)                                 # before the long read
    fs = snap_files(a.src, a.run, a.snap)
    with h5py.File(fs[0], "r") as f:
        hd = dict(f["Header"].attrs)
        par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
    z, box = float(hd["Redshift"]), float(hd["BoxSize"])
    nfiles = int(hd.get("NumFilesPerSnapshot", len(fs)))
    ntot = int(hd["NumPart_Total"][0]) + (int(hd["NumPart_Total_HighWord"][0]) << 32 if "NumPart_Total_HighWord" in hd else 0)
    uv = float(hd.get("UnitVelocity_in_cm_per_s", par.get("UnitVelocity_in_cm_per_s", 1e5)))
    print(f"{a.run} snapshot {a.snap}: z = {z:.5f}, BoxSize {box}, {len(fs)} files (header: {nfiles}), "
          f"{ntot:,} gas cells, UnitVelocity {uv:g}")
    if len(fs) != nfiles:
        sys.exit(f"STOP: {len(fs)} files found, the header says {nfiles}")

    ren = os.path.join(a.out, a.run, "ren_128", "All", f"All_{a.render:03d}.hdf5")
    with h5py.File(ren, "r") as f:
        zr = float(f["Header"].attrs["Redshift"])
        R = {k: f[k][()].astype(np.float64) for k in ("Density", "Temperature", "HII_Fraction")}
    N = R["Density"].shape[0]
    print(f"render {a.render}: z = {zr:.5f} (dz = {z - zr:+.1e}), N = {N}")
    if abs(z - zr) > 1e-3:
        sys.exit("STOP: the render is not at the snapshot's redshift")

    cols = {k: [] for k in ("lrho", "T", "w", "x", "sf", "i0", "i1", "i2", "m")}
    names, nread = None, 0
    for p in fs:
        with h5py.File(p, "r") as f:
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            if names is None:
                names = dict(x=pick(g, ("HII_Fraction",)), fII=pick(g, ("HeII_Fraction",)),
                             fIII=pick(g, ("HeIII_Fraction",)), ne=pick(g, ("ElectronAbundance",)),
                             sfr=pick(g, ("StarFormationRate",)))
                need = ("Coordinates", "Masses", "Density", "InternalEnergy")
                if any(k not in g for k in need) or names["x"] is None:
                    sys.exit(f"STOP: missing gas fields; available: {sorted(g.keys())}")
                print(f"fields: {sorted(g.keys())}\n  Ne from {'ElectronAbundance' if names['ne'] else 'HII + HeII + 2 HeIII'}; "
                      f"He fractions {'present' if names['fII'] else 'MISSING (taken as 0)'}; "
                      f"SFR {'present' if names['sfr'] else 'MISSING (no star-forming cut)'}")
            n = g["Masses"].shape[0]
            if n == 0:
                continue
            pos = g["Coordinates"][()].astype(np.float64)
            m = g["Masses"][()].astype(np.float64)
            rho = g["Density"][()].astype(np.float64)
            u = g["InternalEnergy"][()].astype(np.float64)
            x = g[names["x"]][()].astype(np.float64)
            fII = g[names["fII"]][()].astype(np.float64) if names["fII"] else 0.0
            fIII = g[names["fIII"]][()].astype(np.float64) if names["fIII"] else 0.0
            ne = g[names["ne"]][()].astype(np.float64) if names["ne"] else x + fII + 2.0 * fIII
            sf = (g[names["sfr"]][()] > 0) if names["sfr"] else np.zeros(n, bool)
        mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * ne) * M_P
        T = GAMMA_M1 * u * uv**2 * mu / K_B
        idx = np.floor(pos / box * N).astype(np.int64) % N
        cols["lrho"].append(np.log10(rho).astype(np.float32)); cols["T"].append(T.astype(np.float32))
        cols["w"].append((X_H * m * (1.0 - x)).astype(np.float32)); cols["x"].append(x.astype(np.float32))
        cols["sf"].append(sf); cols["m"].append(m.astype(np.float32))
        for k in range(3):
            cols[f"i{k}"].append(idx[:, k].astype(np.uint16))
        nread += n
    C = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"read {nread:,} cells" + ("" if nread == ntot else f"  STOP: header says {ntot:,}"))
    if nread != ntot:
        sys.exit(1)

    vpix = (box / N)**3
    rho_mean = C["m"].astype(np.float64).sum() / box**3
    print(f"gas mass: cells / render = {C['m'].astype(np.float64).sum() / (R['Density'].sum() * vpix):.5f}")

    # render axis order from the data
    rng = np.random.default_rng(1)
    s = rng.choice(nread, min(nread, 3_000_000), replace=False)
    lr = C["lrho"][s].astype(np.float64)
    best = []
    for o in itertools.permutations(range(3)):
        rd = R["Density"][C[f"i{o[0]}"][s], C[f"i{o[1]}"][s], C[f"i{o[2]}"][s]]
        cc = np.corrcoef(lr, np.log10(np.maximum(rd, 1e-30)))[0, 1]
        best.append((cc, o))
        print(f"  render[i0,i1,i2] = coords {o}: corr(log rho_cell, log render Density) = {cc:.4f}")
    best.sort(reverse=True)
    if best[0][0] - best[1][0] < 0.05:
        sys.exit("STOP: the render axis order is not clear from the data")
    o = best[0][1]
    I = (C[f"i{o[0]}"].astype(np.int64), C[f"i{o[1]}"].astype(np.int64), C[f"i{o[2]}"].astype(np.int64))
    flat = np.ravel_multi_index(I, (N, N, N))
    del I
    for k in range(3):
        del C[f"i{k}"]
    print(f"axis order {o}")

    # checks: nearest-pixel deposit of the cells against the render
    mdep = np.bincount(flat, C["m"], minlength=N**3)
    mTdep = np.bincount(flat, C["m"].astype(np.float64) * C["T"], minlength=N**3)
    ok = mdep > 0
    cm = np.corrcoef(np.log10(mdep[ok]), np.log10(R["Density"].ravel()[ok] * vpix))[0, 1]
    Tdep = mTdep[ok] / mdep[ok]
    rT = Tdep / R["Temperature"].ravel()[ok]
    cT = np.corrcoef(np.log10(Tdep), np.log10(R["Temperature"].ravel()[ok]))[0, 1]
    print(f"check: pixels with cells {ok.mean():.4f}; corr(log deposited mass, log render mass) = {cm:.4f}; "
          f"corr(log deposited m-w T, log render T) = {cT:.4f}, median ratio {np.median(rT):.4f} (nearest-pixel vs cube deposit)")

    # statistics of the IGM (non-star-forming cells)
    igm = ~C["sf"]
    one = 10.0**C["lrho"].astype(np.float64) / rho_mean
    T, w, x = C["T"].astype(np.float64), C["w"].astype(np.float64), C["x"].astype(np.float64)
    neu = igm & (x < XNEU)
    Wall = w[igm].sum()
    T0 = 2.7255 * (1.0 + z)**2 / 125.8
    Tg = 2.7255 * (1.0 + z)
    print(f"\nT_gamma = {Tg:.1f} K; adiabatic IGM T0 = {T0:.2f} K (x (1+delta)^2/3); star-forming cells hold "
          f"{w[C['sf']].sum() / w.sum():.4f} of the H I; min T of neutral IGM cells "
          f"{(f'{T[neu].min():.2f} K' if neu.any() else 'none (no neutral IGM cells)')}")
    print("cells by their own 1+delta (non-star-forming): H I share | neutral share of it | neutral cells: T10 T50 T90 "
          "[K] | <1/T>^-1 | f(T<5.5K) f(T<T_gam) | T_ad at the bin's median 1+delta")
    tab = []
    for lo, hi in zip(CBINS[:-1], CBINS[1:]):
        b = igm & (one >= lo) & (one < hi)
        bn = b & neu
        wb, wn = w[b].sum(), w[bn].sum()
        t10, t50, t90 = wpct(T[bn], w[bn], (10, 50, 90))
        dmed = wpct(one[bn], w[bn], (50,))[0]
        harm = wn / np.sum(w[bn] / T[bn]) if wn > 0 else np.nan
        f55 = w[bn & (T < 5.5)].sum() / wn if wn > 0 else np.nan
        fTg = w[bn & (T < Tg)].sum() / wn if wn > 0 else np.nan
        tad = T0 * dmed**(2.0 / 3.0) if np.isfinite(dmed) else np.nan
        tab.append([lo, hi, wb / Wall, wn / wb if wb > 0 else np.nan, t10, t50, t90, harm, f55, fTg, tad, dmed])
        print(f"  {lo:5g}-{hi:<5g} | {wb / Wall:6.4f} | {tab[-1][3]:6.4f} | {t10:7.1f} {t50:7.1f} {t90:8.1f} | {harm:7.1f} | "
              f"{f55:5.3f} {fTg:5.3f} | {tad:6.2f}")

    # sub-pixel test: neutral cells in neutral pixels, by the pixel's 1+delta
    pd = (R["Density"] / R["Density"].mean()).ravel()
    px = R["HII_Fraction"].ravel()
    pT = R["Temperature"].ravel()
    pw = (R["Density"] * (1.0 - R["HII_Fraction"])).ravel()
    print("\nsub-pixel test: neutral IGM cells inside neutral pixels (pixel x_HII < 1e-3), by the pixel's 1+delta:\n"
          "  pixel 1+delta | H I share | cells: T10 T50 T90 <1/T>^-1 f(T<5.5K) | pixels (render T): T10 T50 T90 <1/T>^-1")
    sub = []
    for lo, hi in zip(PBINS[:-1], PBINS[1:]):
        pm = (pd >= lo) & (pd < hi) & (px < XNEU)
        cm_ = neu & pm[flat]
        wc = w[cm_]
        c10, c50, c90 = wpct(T[cm_], wc, (10, 50, 90))
        ch = wc.sum() / np.sum(wc / T[cm_]) if wc.sum() > 0 else np.nan
        c55 = wc[T[cm_] < 5.5].sum() / wc.sum() if wc.sum() > 0 else np.nan
        p10, p50, p90 = wpct(pT[pm], pw[pm], (10, 50, 90))
        ph = pw[pm].sum() / np.sum(pw[pm] / pT[pm]) if pm.any() else np.nan
        sub.append([lo, hi, wc.sum() / Wall, c10, c50, c90, ch, c55, p10, p50, p90, ph])
        print(f"  {lo:g}-{hi:g}{'':8s} | {wc.sum() / Wall:6.4f} | {c10:6.1f} {c50:6.1f} {c90:7.1f} {ch:7.1f} {c55:5.3f} | "
              f"{p10:6.1f} {p50:6.1f} {p90:7.1f} {ph:7.1f}")

    h2 = np.histogram2d(np.log10(np.maximum(one[neu], 1e-10)), np.log10(np.maximum(T[neu], 1.0)),
                        [LD_EDGES, LT_EDGES], weights=w[neu] / Wall)[0]
    os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, f"cells_T_{a.snap:03d}.npz"), z=z, render=a.render, T0=T0, Tgamma=Tg, axis_order=o,
             cbins=np.array(CBINS), table=np.array(tab), pbins=np.array(PBINS), sub=np.array(sub), h2=h2,
             ld_edges=LD_EDGES, lt_edges=LT_EDGES, checks=np.array([cm, cT, np.median(rT)]))
    print(f"\nwrote {od}/cells_T_{a.snap:03d}.npz")


def report(a):
    D = {}
    for run in RUNS:
        p = os.path.join(a.out, run, "cells_T", f"cells_T_{a.snap:03d}.npz")
        if os.path.exists(p):
            D[run] = np.load(p)
    if not D:
        sys.exit("no scan outputs found")
    for run, d in D.items():
        print(f"\n{LABELS[run]} (z = {float(d['z']):.3f}, render {int(d['render'])}): neutral IGM cells by 1+delta, "
              f"T50 [K] / adiabatic; sub-pixel test (cells vs pixels, T50 and <1/T>^-1):")
        for r in d["table"]:
            print(f"  1+delta {r[0]:5g}-{r[1]:<5g}: H I share {r[2]:.4f}, T50 {r[5]:7.1f} K, adiabatic {r[10]:6.2f} K, "
                  f"f(T<5.5K) {r[8]:.3f}")
        for r in d["sub"]:
            print(f"  pixel 1+delta {r[0]:g}-{r[1]:g}: cells T50 {r[4]:6.1f} K, <1/T>^-1 {r[6]:6.1f} K | "
                  f"pixels T50 {r[9]:6.1f} K, <1/T>^-1 {r[11]:6.1f} K")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    runs = [r for r in RUNS if r in D]
    fig, ax = plt.subplots(1, len(runs) + 1, figsize=(3.6 * (len(runs) + 1), 3.6))
    ax = np.atleast_1d(ax)
    d0 = D[runs[0]]
    xd = 0.5 * (d0["ld_edges"][1:] + d0["ld_edges"][:-1])
    for c, run in enumerate(runs):
        d = D[run]
        h = d["h2"].T
        ax[c].pcolormesh(d["ld_edges"], d["lt_edges"], np.log10(np.where(h > 0, h, np.nan)), cmap="Blues",
                         vmin=np.log10(np.nanmax(h)) - 4, vmax=np.log10(np.nanmax(h)), shading="flat")
        ax[c].plot(xd, np.log10(float(d["T0"]) * (10**xd)**(2 / 3)), color=INK, lw=1.2, ls="--", label="adiabatic")
        ax[c].axhline(np.log10(5.0), color=INK2, lw=1, ls=":", label="5 K floor")
        ax[c].axhline(np.log10(float(d["Tgamma"])), color=INK2, lw=1, ls="-.", label=r"$T_\gamma$")
        ax[c].set_title(f"{LABELS[run]}, neutral cells, z = {float(d['z']):.2f}", color=INK, fontsize=9.5)
        ax[c].set_xlabel(r"$\log_{10}(1+\delta)$ of the cell")
        ax[c].set_ylim(0.3, 4.5)
    ax[0].set_ylabel(r"$\log_{10} T$ [K] (H I-weighted)")
    ax[0].legend(loc="upper left", fontsize=8)
    axm = ax[-1]
    for run in runs:                                              # like for like: the same neutral pixels
        s = D[run]["sub"]
        xb = np.sqrt(np.maximum(s[:, 0], 0.25) * s[:, 1])         # bin centre in pixel 1+delta (0-0.5 drawn at 0.35)
        axm.plot(xb, s[:, 4], color=COLORS[run], lw=2, marker="o", ms=5, label=f"{LABELS[run]}: cells in the pixels")
        axm.plot(xb, s[:, 9], color=COLORS[run], lw=1, ls="--", marker="s", ms=6, mfc="none", mew=1.5,
                 label=f"{LABELS[run]}: the pixels' T")
    xx = np.geomspace(0.1, 1000, 50)
    axm.plot(xx, float(d0["T0"]) * xx**(2 / 3), color=INK, lw=1.2, ls="--", label="adiabatic")
    axm.axhline(5.0, color=INK2, lw=1, ls=":")
    axm.set_xscale("log"); axm.set_yscale("log")
    axm.set_xlabel(r"pixel $1+\delta$"); axm.set_ylabel("median T of neutral H I [K]")
    axm.set_title("neutral pixels: their cells vs the pixel T", color=INK, fontsize=9.5)
    axm.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    os.makedirs(a.data, exist_ok=True)
    fig.savefig(os.path.join(a.data, f"cells_T_xray_{a.snap:03d}.png"), dpi=200)
    print(f"\nwrote {a.data}/cells_T_xray_{a.snap:03d}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--run", choices=RUNS, required=True)
    s.add_argument("--snap", type=int, default=24)
    s.add_argument("--render", type=int, default=71)
    s.add_argument("--src", default=SRC)
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)   # mock tests only
    r = sp.add_parser("report")
    r.add_argument("--snap", type=int, default=24)
    r.add_argument("--out", default=os.environ.get("BASE", OUT))
    r.add_argument("--data", default="data")
    a = p.parse_args()
    scan(a) if a.cmd == "scan" else report(a)


if __name__ == "__main__":
    main()
