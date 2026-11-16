"""
compare_xray.py
===============
With vs without X-rays: the global 21 cm signal and what makes the difference, render by render, from the assembled
128^3 renders (ren_128/All/All_NNN.hdf5) and the 21 cm maps (Tb_21cm/RIN_N.hdf5, SPIN_MODEL 1, real space) of both runs.

The maps are dT_b = P(z) x_HI (1+delta) (1 - T_gamma/T_S) x_CMB pixel by pixel (P = tb_physics.Line("H").P, the
code's prefactor; 1+delta = Density / <Density>, as the code's global_gas). So, exactly,
  <dT_b> = P <sig> S_eff,   sig = x_HI (1+delta),   S_eff = <sig (1 - T_gamma/T_S) x_CMB> / <sig>
S_eff is the H I-weighted spin factor; T_S,eff = T_gamma / (1 - S_eff) is the spin temperature it corresponds to.
Its fully coupled limit (T_S = T_K) is S_K = 1 - T_gamma <1/T_K>_HI (H I-weighted harmonic mean of the pixel T).
The difference of the two runs splits exactly into a heating part and an H I part:
  dT_X - dT_noX = P [ sigbar (S_X - S_noX) + Sbar (sig_X - sig_noX) ],  bars = mean of the two runs.
Per render and run it also stores the volume-weighted <x_HII>_V (Volume x HII_VolumeFraction) and the H I-weighted
median pixel T. The pixel T is biased high for cold neutral gas (gas-cell test, xray/NOTES.md: <1/T>^-1 x2-7 too high
without X-rays at z = 10-15, x1.2-1.9 with X-rays at z = 10), so S_K and the heating part are pixel values.

Outputs (--out, default data/): compare_xray.csv (one row per render: z and both runs' quantities) and
compare_xray.png (six panels). Optional: --lumina-log logs of the Lumina SPIN_MODEL 1 production (plot_global_Tb
read_log) for panel (a); data/ps_xray.npz (ps_xray.py), if present, for panel (f).
Run from xray/:  python compare_xray.py [--step 2] [--lumina-log ../logs/PS_21cm_29237390*.out]
                 (BASE=<dir> overrides /orcd/data/mvogelsb/004/chcho/xray)
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tb_physics as tbp  # noqa: E402

RUNS = ("withoutXrays", "withXrays")
LABELS = {"withoutXrays": "no X-rays", "withXrays": "with X-rays"}
COLORS = {"withoutXrays": "#2a78d6", "withXrays": "#eb6834"}
INK, INK2, GRID = "#1f1f1e", "#6b6b68", "#e6e6e3"
OUT_BASE = "/orcd/data/mvogelsb/004/chcho/xray"
COSMO_KEYS = ("HubbleParam", "Omega0", "OmegaBaryon", "OmegaLambda")
COSMO_LUMINA = (0.6766, 0.30964144, 0.048975, 0.6902672)


def renders(base, run):
    a = {int(os.path.basename(f)[4:-5]) for f in glob.glob(f"{base}/{run}/ren_128/All/All_*.hdf5")}
    b = {int(os.path.basename(f)[4:-5]) for f in glob.glob(f"{base}/{run}/Tb_21cm/RIN_*.hdf5")}
    return a & b


def one(base, run, s):
    """the quantities of one render of one run"""
    with h5py.File(f"{base}/{run}/ren_128/All/All_{s:03d}.hdf5", "r") as h:
        hd = h["Header"].attrs
        z = float(hd["Redshift"])
        cos = tuple(float(hd[k]) for k in COSMO_KEYS) if all(k in hd for k in COSMO_KEYS) else COSMO_LUMINA
        rho = h["Density"][()].astype(np.float64).ravel()
        xm = h["HII_Fraction"][()].astype(np.float64).ravel()
        T = h["Temperature"][()].astype(np.float64).ravel()
        xv = h["HII_VolumeFraction"][()].astype(np.float64).ravel() if "HII_VolumeFraction" in h else None
        V = h["Volume"][()].astype(np.float64).ravel() if "Volume" in h else None
    with h5py.File(f"{base}/{run}/Tb_21cm/RIN_{s}.hdf5", "r") as h:
        tb = h["rin"][()].astype(np.float64).ravel()
    if tb.size != rho.size:
        raise ValueError(f"{run} render {s}: map {tb.size} pixels, render {rho.size}")
    P = tbp.Line("H", z, *cos, spin_model=1).P
    Tg = 2.7255 * (1.0 + z)
    sig = (1.0 - xm) * rho / rho.mean()
    sbar = sig.mean()
    Tb = tb.mean()
    S = Tb / (P * sbar) if sbar > 0 else np.nan
    w = sig
    Th = w.sum() / np.sum(w / T)
    o = np.argsort(T)
    c = np.cumsum(w[o])
    T50 = np.interp(0.5 * c[-1], c, T[o])
    if xv is None:
        xV = np.nan
    elif V is None:
        xV = xv.mean()
    else:
        xV = np.sum(V * xv) / V.sum()
    return dict(z=z, P=P, Tg=Tg, Tb=Tb, sig=sbar, S=S, Ts=Tg / (1.0 - S) if S < 1 else np.inf,
                T_harm=Th, S_K=1.0 - Tg / Th, T50=T50, xHII_V=xV, xHII_M=np.sum(rho * xm) / rho.sum())


def table(a):
    common = sorted(renders(a.base, RUNS[0]) & renders(a.base, RUNS[1]))
    if not common:
        sys.exit(f"no renders with maps in both runs under {a.base}")
    common = common[::a.step]
    rows = []
    for j, s in enumerate(common):
        r = {"render": s}
        for run in RUNS:
            q = one(a.base, run, s)
            tag = "noX" if run == RUNS[0] else "X"
            if "z" in r and abs(q["z"] - r["z"]) > 1e-4:
                raise ValueError(f"render {s}: z differs between the runs ({r['z']} vs {q['z']})")
            r["z"], r["P"], r["Tg"] = q["z"], q["P"], q["Tg"]
            for k in ("Tb", "sig", "S", "Ts", "T_harm", "S_K", "T50", "xHII_V", "xHII_M"):
                r[f"{k}_{tag}"] = q[k]
        sb, Sb = 0.5 * (r["sig_noX"] + r["sig_X"]), 0.5 * (r["S_noX"] + r["S_X"])
        r["dTb"] = r["Tb_X"] - r["Tb_noX"]
        r["dTb_heat"] = r["P"] * sb * (r["S_X"] - r["S_noX"])
        r["dTb_HI"] = r["P"] * Sb * (r["sig_X"] - r["sig_noX"])
        rows.append(r)
        if j % 25 == 0 or j == len(common) - 1:
            print(f"  render {s:3d} z = {r['z']:6.3f}: <dT_b> {r['Tb_noX']:+8.3f} / {r['Tb_X']:+8.3f} mK, "
                  f"S_eff {r['S_noX']:+7.3f} / {r['S_X']:+7.3f}, <x_HI(1+d)> {r['sig_noX']:.4f} / {r['sig_X']:.4f}, "
                  f"<1/T_K>^-1 {r['T_harm_noX']:7.1f} / {r['T_harm_X']:7.1f} K", flush=True)
    keys = list(rows[0])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "compare_xray.csv")
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.8g}" if k != "render" else str(r[k]) for k in keys) + "\n")
    print(f"wrote {path}: {len(rows)} renders, z = {rows[0]['z']:.3f} - {rows[-1]['z']:.3f}")
    return rows


def lumina_global(logs):
    if not logs:
        return None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from plot_global_Tb import read_log
    zs, vs = [], []
    for p in logs:
        z, v = read_log(p)
        zs += list(z); vs += list(v[:, 0])
    if not zs:
        return None
    o = np.argsort(zs)
    return np.array(zs)[o], np.array(vs)[o]


def figure(a, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    z = R["z"]
    m = (z <= a.zmax) & (z >= a.zmin)
    fig, ax = plt.subplots(3, 2, figsize=(11.5, 11), sharex=True)
    ax = ax.ravel()
    for x in ax:
        x.grid(color=GRID, lw=0.6)
        x.axhline(0, color=INK2, lw=0.6)
        for s in x.spines.values():
            s.set_color(INK2)
    for run, tag in zip(RUNS, ("noX", "X")):
        c, lab = COLORS[run], LABELS[run]
        ax[0].plot(z[m], R[f"Tb_{tag}"][m], color=c, lw=2, label=lab)
        ax[2].plot(z[m], R[f"T_harm_{tag}"][m], color=c, lw=2, label=rf"$\langle 1/T_K\rangle^{{-1}}_{{\rm HI}}$, {lab}")
        Ts = np.where(R[f"S_{tag}"] < 0.999, R[f"Ts_{tag}"], np.nan)
        ax[2].plot(z[m], Ts[m], color=c, lw=1.6, ls="--", label=rf"$T_{{S,\rm eff}}$, {lab}")
        ax[3].plot(z[m], R[f"S_{tag}"][m], color=c, lw=2, label=rf"$S_{{\rm eff}}$, {lab}")
        ax[3].plot(z[m], R[f"S_K_{tag}"][m], color=c, lw=1.2, ls=":", label=rf"$S_K$ (fully coupled), {lab}")
        ax[4].plot(z[m], R[f"sig_{tag}"][m], color=c, lw=2, label=rf"$\langle x_{{\rm HI}}(1+\delta)\rangle$, {lab}")
        ax[4].plot(z[m], 1 - R[f"xHII_V_{tag}"][m], color=c, lw=1.2, ls="--", label=rf"$1-\bar x_{{\rm HII,V}}$, {lab}")
    L = lumina_global(a.lumina_log)
    if L is not None:
        k = (L[0] <= a.zmax) & (L[0] >= a.zmin)
        ax[0].plot(L[0][k], L[1][k], color=INK2, lw=1.2, ls="--", label="Lumina (500 cMpc)")
    ax[0].set_ylabel(r"$\langle\delta T_b\rangle$ [mK]")
    ax[0].set_title("(a) global 21 cm signal", loc="left", fontsize=11)
    ax[1].plot(z[m], R["dTb"][m], color=INK, lw=2.2, label="with − without X-rays")
    ax[1].plot(z[m], R["dTb_heat"][m], color="#c0392b", lw=1.5, ls="--", label=r"from the spin factor (heating): $P\,\bar\sigma\,\Delta S$")
    ax[1].plot(z[m], R["dTb_HI"][m], color="#27855f", lw=1.5, ls=":", label=r"from the H I content: $P\,\bar S\,\Delta\sigma$")
    ax[1].set_ylabel(r"$\Delta\langle\delta T_b\rangle$ [mK]")
    ax[1].set_title("(b) what makes the difference", loc="left", fontsize=11)
    ax[2].plot(z[m], R["Tg"][m], color=INK, lw=1.2, label=r"$T_\gamma$")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("temperature [K]")
    ax[2].set_title("(c) H I gas and spin temperatures (128³ pixels)", loc="left", fontsize=11)
    ax[3].set_ylabel(r"spin factor $\langle 1-T_\gamma/T_S\rangle_{\rm HI}$")
    ax[3].set_title("(d) H I-weighted spin factor", loc="left", fontsize=11)
    lo = np.nanmin([R["S_noX"][m].min(), R["S_X"][m].min()])
    ax[3].set_ylim(max(lo * 1.1, -15), 1.05)
    ax[4].set_ylabel("neutral fraction")
    ax[4].set_title("(e) H I content", loc="left", fontsize=11)
    ps = os.path.join(a.out, "ps_xray.npz")
    if os.path.exists(ps):
        d = np.load(ps)
        i = int(np.argmin(np.abs(d["k"] - a.k)))
        zp = d["z"]
        mp = (zp <= a.zmax) & (zp >= a.zmin)
        ax[5].plot(zp[mp], d["D2_noX"][mp, i], color=COLORS[RUNS[0]], lw=2, label=LABELS[RUNS[0]])
        ax[5].plot(zp[mp], d["D2_X"][mp, i], color=COLORS[RUNS[1]], lw=2, label=LABELS[RUNS[1]])
        ax[5].set_yscale("log")
        ax[5].set_ylabel(r"$\Delta^2_{21}$ [mK$^2$]")
        ax[5].set_title(rf"(f) power at $k$ = {d['k'][i]:.2f} $h$/cMpc", loc="left", fontsize=11)
    else:
        ax[5].text(0.5, 0.5, "no data/ps_xray.npz\n(run ps_xray.py)", ha="center", va="center",
                   transform=ax[5].transAxes, color=INK2)
    for x in ax:
        x.set_xlim(a.zmax, a.zmin)
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=8, frameon=False)
    ax[4].set_xlabel("z")
    ax[5].set_xlabel("z")
    fig.suptitle("X-ray test runs (50 cMpc, same ICs, differ only in MRT_XRAY): 21 cm signal and its factors",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(a.out, "compare_xray.png")
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=os.environ.get("BASE", OUT_BASE))
    p.add_argument("--out", default="data")
    p.add_argument("--step", type=int, default=1, help="every step-th common render")
    p.add_argument("--zmin", type=float, default=5.0)
    p.add_argument("--zmax", type=float, default=25.0)
    p.add_argument("--k", type=float, default=0.84, help="k [h/cMpc] of panel (f), nearest bin")
    p.add_argument("--lumina-log", nargs="*", default=[], help="Lumina SPIN_MODEL 1 21 cm job logs (optional)")
    p.add_argument("--plot-only", action="store_true", help="redraw from data/compare_xray.csv")
    a = p.parse_args()
    if a.plot_only:
        import csv
        with open(os.path.join(a.out, "compare_xray.csv")) as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    else:
        rows = table(a)
    figure(a, rows)


if __name__ == "__main__":
    main()
