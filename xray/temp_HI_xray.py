"""
temp_HI_xray.py
===============
H I temperature distribution of the X-ray test runs, from the assembled 128^3 renders (ren_128/All/All_NNN.hdf5).
Every statistic is weighted by the H I mass of the pixel, Density x (1 - HII_Fraction) (pixels have equal volume),
with the pixel temperature `Temperature` (mass-weighted within the pixel, as the 21 cm maps use) and, for comparison,
`TemperatureVolumeWeighted`. These are pixel temperatures (0.26 cMpc/h), not gas-cell temperatures: a pixel mixing
hot and cold gas reports the mass-weighted T, biased high for the cold neutral IGM (JALPHA_NOTES §7).

Per render and run:
  H I-weighted percentiles 1/10/50/90/99 % of T, the harmonic mean <1/T>^-1 (what a coupled T_S responds to),
  the H I share below 2 K (edge of Hirata's fit), below T_gamma = 2.7255 (1+z) and below 10 K,
  the adiabatic reference at mean density T_ad = 2.7255 (1+z)^2 / 125.8 (the adiabatic IGM; it matches the H I at
  1+delta < 1 at the first render, 20 K at z = 29),
  and the H I share and median T in bins of 1 + delta = rho/<rho>: < 1, 1-3, 3-10, > 10;
  by the pixel's ionized fraction x_HII (HII_Fraction, mass-weighted; bins < 1e-3, 1e-3-0.01, 0.01-0.1, 0.1-0.5, > 0.5):
  the H I share, median T, harmonic mean and share below T_gamma, and for neutral pixels (x_HII < 1e-3) the H I share
  and median T by 1 + delta (separates heating of neutral gas from pixels mixing ionized and neutral gas).
Outputs (--out, default data/): temp_HI_xray.csv (one row per render, run), temp_HI_xray.npz (H I-weighted
histograms in log T), temp_HI_xray.png (distributions at selected z, both runs).
Run from xray/:  python temp_HI_xray.py        (BASE=<dir> overrides /orcd/data/mvogelsb/004/chcho/xray)
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

RUNS = ("withoutXrays", "withXrays")
LABELS = {"withoutXrays": "no X-rays", "withXrays": "with X-rays"}
COLORS = {"withoutXrays": "#2a78d6", "withXrays": "#eb6834"}     # categorical slots 1, 2 (validated pair)
INK, INK2, GRID = "#1f1f1e", "#6b6b68", "#e6e6e3"
DBINS = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, np.inf))
XBINS = ((0.0, 1e-3), (1e-3, 1e-2), (1e-2, 0.1), (0.1, 0.5), (0.5, 1.01))   # pixel HII_Fraction (mass-weighted)
XLAB = ("x<1e-3", "1e-3-0.01", "0.01-0.1", "0.1-0.5", ">0.5")
LOGT_EDGES = np.linspace(0.0, 7.0, 141)                           # log10 T [K], 0.05 dex


def wpct(x, w, q):
    """weighted percentiles q [0-100] of x with weights w"""
    if w.sum() <= 0:
        return np.full(len(q), np.nan)
    o = np.argsort(x)
    c = np.cumsum(w[o].astype(np.float64))
    return np.interp(np.asarray(q) / 100.0 * c[-1], c, x[o])


def render_list(base, run):
    fs = glob.glob(f"{base}/{run}/ren_128/All/All_*.hdf5")
    return sorted(int(os.path.basename(f)[4:-5]) for f in fs)


def redshift(base, run, s):
    with h5py.File(f"{base}/{run}/ren_128/All/All_{s:03d}.hdf5", "r") as h:
        return float(h["Header"].attrs["Redshift"])


def load(base, run, s):
    with h5py.File(f"{base}/{run}/ren_128/All/All_{s:03d}.hdf5", "r") as h:
        return {k: h[k][()].astype(np.float64).ravel() for k in
                ("Density", "HII_Fraction", "Temperature", "TemperatureVolumeWeighted")}


def stats(d, z, T_ad):
    rho, xH, T, Tv = d["Density"], d["HII_Fraction"], d["Temperature"], d["TemperatureVolumeWeighted"]
    w = rho * (1.0 - xH)                                          # H I mass per pixel (equal volumes)
    W = w.sum()
    Tg = 2.7255 * (1.0 + z)
    one = rho / rho.mean()
    out = dict(z=z, T_gamma=Tg, T_ad=T_ad, HI_mass_frac=W / rho.sum())
    for q, v in zip((1, 10, 50, 90, 99), wpct(T, w, (1, 10, 50, 90, 99))):
        out[f"T{q:02d}"] = v
    out["T50_vol"] = wpct(Tv, w, (50,))[0]
    out["T_harm"] = W / np.sum(w / T)
    out["f_lt2K"] = w[T < 2.0].sum() / W
    out["f_ltTg"] = w[T < Tg].sum() / W
    out["f_lt10K"] = w[T < 10.0].sum() / W
    for lo, hi in DBINS:
        m = (one >= lo) & (one < hi)
        tag = f"d{lo:g}-{hi:g}".replace("inf", "")
        out[f"share_{tag}"] = w[m].sum() / W
        out[f"T50_{tag}"] = wpct(T[m], w[m], (50,))[0] if m.any() else np.nan
    for i, (lo, hi) in enumerate(XBINS):                          # by the pixel's ionized fraction
        m = (xH >= lo) & (xH < hi)
        wm = w[m]
        out[f"xb{i}_share"] = wm.sum() / W
        out[f"xb{i}_T50"] = wpct(T[m], wm, (50,))[0] if wm.sum() > 0 else np.nan
        out[f"xb{i}_Tharm"] = wm.sum() / np.sum(wm / T[m]) if wm.sum() > 0 else np.nan
        out[f"xb{i}_fltTg"] = w[m & (T < Tg)].sum() / wm.sum() if wm.sum() > 0 else np.nan
    neu = xH < XBINS[0][1]                                        # neutral pixels, by 1 + delta
    for lo, hi in DBINS[:3]:
        m = neu & (one >= lo) & (one < hi)
        tag = f"{lo:g}-{hi:g}"
        out[f"xneu_d{tag}_share"] = w[m].sum() / W
        out[f"xneu_d{tag}_T50"] = wpct(T[m], w[m], (50,))[0] if w[m].sum() > 0 else np.nan
    hist = np.histogram(np.log10(np.clip(T, 1.0001, 9.99e6)), LOGT_EDGES, weights=w / W)[0]
    return out, hist


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=os.environ.get("BASE", "/orcd/data/mvogelsb/004/chcho/xray"))
    p.add_argument("--out", default="data")
    p.add_argument("--z", type=float, nargs="+", default=[29, 25, 20, 17, 15, 13, 12, 11, 10, 9, 8, 7, 6, 5],
                   help="redshifts of the table (nearest render)")
    p.add_argument("--zfig", type=float, nargs="+", default=[20, 15, 13, 11, 9, 7],
                   help="redshifts of the figure panels")
    a = p.parse_args()

    ren = render_list(a.base, RUNS[0])
    if ren != render_list(a.base, RUNS[1]):
        raise SystemExit("the two runs have different render lists")
    zr = np.array([redshift(a.base, RUNS[0], s) for s in ren])
    pick = sorted({int(np.argmin(abs(zr - zq))) for zq in a.z + a.zfig})
    rows, hists = [], {}
    for run in RUNS:
        for j in pick:
            s, z = ren[j], zr[j]
            st, h = stats(load(a.base, run, s), z, 2.7255 * (1.0 + z)**2 / 125.8)
            st.update(run=run, render=s)
            rows.append(st); hists[(run, s)] = h

    os.makedirs(a.out, exist_ok=True)
    keys = ["run", "render", "z", "T_gamma", "T_ad", "HI_mass_frac", "T01", "T10", "T50", "T90", "T99", "T50_vol",
            "T_harm", "f_lt2K", "f_ltTg", "f_lt10K"] + [k for k in rows[0] if k.startswith(("share_", "T50_d", "xb", "xneu"))]
    with open(os.path.join(a.out, "temp_HI_xray.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    np.savez(os.path.join(a.out, "temp_HI_xray.npz"), logT_edges=LOGT_EDGES, renders=np.array([ren[j] for j in pick]),
             z=np.array([zr[j] for j in pick]),
             **{f"hist_{run}": np.array([hists[(run, ren[j])] for j in pick]) for run in RUNS})

    by = {(r["run"], r["render"]): r for r in rows}
    zt = sorted({int(np.argmin(abs(zr - zq))) for zq in a.z})
    for run in RUNS:
        print(f"\n{LABELS[run]}: H I-weighted pixel temperature [K]")
        print("     z   T_gam   T_ad |   T1%    T10%    T50%    T90%    T99%  T50(vol)  <1/T>^-1 |"
              " f<2K  f<Tgam  f<10K | H I share / T50 by 1+delta: <1, 1-3, 3-10, >10")
        for j in sorted(zt, key=lambda j: -zr[j]):
            r = by[(run, ren[j])]
            dd = "  ".join(f"{r[k]:.2f}/{r[k.replace('share_', 'T50_')]:.0f}" for k in r if k.startswith("share_"))
            print(f"  {r['z']:5.2f} {r['T_gamma']:6.1f} {r['T_ad']:6.1f} | {r['T01']:6.1f} {r['T10']:7.1f} {r['T50']:7.1f} "
                  f"{r['T90']:7.0f} {r['T99']:7.0f} {r['T50_vol']:8.1f}  {r['T_harm']:8.1f} | {r['f_lt2K']:.3f} "
                  f"{r['f_ltTg']:6.3f} {r['f_lt10K']:6.3f} | {dd}")

    for run in RUNS:
        print(f"\n{LABELS[run]}: H I by the pixel's ionized fraction x_HII (HII_Fraction): "
              "H I share / median T / <1/T>^-1 [K] / share of it below T_gamma")
        print("     z | " + " | ".join(f"{l:^26s}" for l in XLAB) + " | neutral (x<1e-3) by 1+delta: share/T50 <1, 1-3, 3-10")
        for j in sorted(zt, key=lambda j: -zr[j]):
            r = by[(run, ren[j])]
            cells = " | ".join(f"{r[f'xb{i}_share']:.3f} {r[f'xb{i}_T50']:6.0f} {r[f'xb{i}_Tharm']:6.0f} {r[f'xb{i}_fltTg']:5.2f}"
                               for i in range(len(XBINS)))
            nd = "  ".join(f"{r[k]:.3f}/{r[k.replace('_share', '_T50')]:.0f}" for k in r if k.startswith("xneu") and k.endswith("_share"))
            print(f"  {r['z']:5.2f} | {cells} | {nd}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    jf = sorted({int(np.argmin(abs(zr - zq))) for zq in a.zfig}, key=lambda j: -zr[j])
    nc = 3
    nr = int(np.ceil(len(jf) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.6 * nr), sharex=True, sharey=True, squeeze=False)
    xc = 0.5 * (LOGT_EDGES[1:] + LOGT_EDGES[:-1])
    dlog = LOGT_EDGES[1] - LOGT_EDGES[0]
    for n, j in enumerate(jf):
        axx = ax[n // nc, n % nc]
        for run in RUNS:
            axx.plot(10**xc, hists[(run, ren[j])] / dlog, color=COLORS[run], lw=2, label=LABELS[run])
        Tg = 2.7255 * (1 + zr[j])
        axx.axvline(Tg, color=INK2, lw=1, ls="--")
        axx.text(Tg * 1.1, 0.97, r"$T_\gamma$", transform=axx.get_xaxis_transform(), color=INK2, va="top", fontsize=8)
        axx.set_xscale("log")
        axx.set_title(f"z = {zr[j]:.2f}", color=INK, fontsize=10)
    for axx in ax[-1]:
        axx.set_xlabel(r"pixel $T$ [K]")
    for axx in ax[:, 0]:
        axx.set_ylabel(r"H I mass fraction per dex")
    for k in range(len(jf), nr * nc):
        ax[k // nc, k % nc].set_visible(False)
        if k // nc >= 1:
            above = ax[k // nc - 1, k % nc]
            above.xaxis.set_tick_params(labelbottom=True)
            above.set_xlabel(r"pixel $T$ [K]")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2)
    fig.suptitle("H I temperature distribution, X-ray test runs (128³ pixels)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.out, "temp_HI_xray.png"), dpi=200)
    print(f"\nwrote {a.out}/temp_HI_xray.csv, temp_HI_xray.npz, temp_HI_xray.png")


if __name__ == "__main__":
    main()
