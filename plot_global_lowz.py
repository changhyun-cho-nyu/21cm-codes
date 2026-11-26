"""
plot_global_lowz.py
===================
Lumina's global 21 cm signal for z <= 10 only (user, 2026-11-26): at higher z the neutral-gas temperature is not
reliable (non-adiabatic heating of cold gas in the hydro; xray/NOTES.md), and the signal depends on it directly.

Central curve: the volume mean of the 640^3 SPIN_MODEL 1 maps (VID/H_wf/stats_H.csv), divided by the measured
640^3 / gas-cell ratio r(z) (whole-box gas cells at z = 10.01, 8.90, 8.00: xray/data/lumina_noX_global.csv, columns
X_640 / X_box), r linear in z between those points and tapering to 1 at the first redshift where the global values
come from the maps themselves (z = 7.46; <= 1% expected there).
Band (one-sided systematics, both push the true signal below the maps):
  upper edge = the uncorrected 640^3 maps (resolution: the maps sit above the gas cells);
  lower edge = central x (1 - d_T), d_T = the fractional drop of (1 - T_CMB/T_S) if the neutral gas were 20 K colder,
               T_S = the H I-weighted <1/T_S>^-1 of the 50 cMpc X-ray run's gas cells (pair_TS_X_all), log-interpolated
               in z between its snapshots (z = 10.01 ... 5.73; d_T = 0 below). 20 K bounds the spurious heating: it is
               the whole non-X-ray thermal state of the neutral gas (no-X run, T_S = 16-23 K at z = 10-7.5).
Points: the whole-box gas-cell values at z = 10.01, 8.90, 8.00.
Writes <out>.pdf, <out>.png and <out>.csv (z, maps, r, central, lower, upper, d_T).

  python plot_global_lowz.py [--stats /orcd/data/mvogelsb/004/chcho/VID/H_wf/stats_H.csv] [--out global_lowz]
"""
import argparse
import csv

import numpy as np

TCMB0 = 2.7255
DT_SPURIOUS = 20.0                      # K
BLUE, BAND, INK, INK2, GRID = "#2a78d6", "#cde2fb", "#1f1f1e", "#6b6b68", "#e6e6e3"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def lowz_band(stats, cells_path, zmax=10.02, verbose=True):
    """The z <= zmax global signal and its band (see the module docstring). Returns a dict with z, maps, r, central,
    lower, upper, d_T (on the map redshifts) and z_cells, Tb_cells (whole-box gas-cell values with z <= zmax)."""
    st = read_csv(stats)
    zm = np.array([fnum(r["z"]) for r in st])
    tm = np.array([fnum(r["mean"]) for r in st])
    o = np.argsort(zm)
    zm, tm = zm[o], tm[o]
    keep = zm <= zmax
    zm, tm = zm[keep], tm[keep]

    cells = read_csv(cells_path)
    zc = np.array([fnum(r["z"]) for r in cells])
    xbox = np.array([fnum(r["X_box"]) for r in cells])
    x640 = np.array([fnum(r["X_640"]) for r in cells])
    src = [r["X_box_src"] for r in cells]
    tsx = np.array([fnum(r["pair_TS_X_all"]) for r in cells])

    # resolution ratio at the gas-cell snapshots, then 1 where the maps are the reference
    gc = np.array([s == "cells" and np.isfinite(x) and np.isfinite(y) and z <= zmax
                   for s, x, y, z in zip(src, x640, xbox, zc)])
    zr = list(zc[gc])
    rr = list(x640[gc] / xbox[gc])
    z640 = zc[np.array([s == "640" for s in src])]
    if len(z640):
        zr.append(z640.max())
        rr.append(1.0)
    o = np.argsort(zr)
    zr, rr = np.array(zr)[o], np.array(rr)[o]
    r = np.interp(zm, zr, rr, left=1.0, right=rr[-1])
    if verbose:
        print("640^3 / gas cells:", ", ".join(f"z = {z:.2f}: {x:.3f}" for z, x in zip(zr, rr)))
    for z, x in zip(zc[gc], x640[gc]):               # consistency: the CSV's 640^3 value vs this table
        i = np.argmin(np.abs(zm - z))
        if verbose:
            print(f"  check z = {z:.2f}: stats_H mean {tm[i]:+.3f} (z = {zm[i]:.3f}), lumina_noX_global X_640 {x:+.3f}")

    # temperature systematic
    ok = np.isfinite(tsx)
    zt, lt = zc[ok], np.log(tsx[ok])
    o = np.argsort(zt)
    zt, lt = zt[o], lt[o]
    TS = np.exp(np.interp(zm, zt, lt))
    Tg = TCMB0 * (1.0 + zm)
    dT = 1.0 - (1.0 - Tg / (TS - DT_SPURIOUS)) / (1.0 - Tg / TS)
    dT = np.where(zm < zt.min(), 0.0, np.clip(dT, 0.0, 1.0))

    central = tm / r
    return dict(z=zm, maps=tm, r=r, central=central, lower=central * (1.0 - dT), upper=tm, d_T=dT,
                z_cells=zc[gc], Tb_cells=xbox[gc])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", default="/orcd/data/mvogelsb/004/chcho/VID/H_wf/stats_H.csv")
    p.add_argument("--cells", default="xray/data/lumina_noX_global.csv")
    p.add_argument("--zmax", type=float, default=10.02, help="highest z shown (the gas-cell point is at 10.01)")
    p.add_argument("--out", default="global_lowz")
    a = p.parse_args()

    B = lowz_band(a.stats, a.cells, a.zmax)
    zm, tm, r, central, lower, upper, dT = (B[k] for k in ("z", "maps", "r", "central", "lower", "upper", "d_T"))
    with open(a.out + ".csv", "w") as f:
        f.write("# plot_global_lowz.py: z, maps = 640^3 mean, r = 640^3/gas cells, central = maps/r, lower = central (1-d_T), "
                "upper = maps, d_T = temperature systematic (20 K)\nz,maps,r,central,lower,upper,d_T\n")
        for row in zip(zm, tm, r, central, lower, upper, dT):
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")
    print(f"\n{'z':>6} {'maps':>8} {'r':>6} {'central':>8} {'lower':>8} {'upper':>8} {'band/central':>13}")
    for zq in (10.0, 9.5, 9.2, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5):
        i = np.argmin(np.abs(zm - zq))
        print(f"{zm[i]:6.2f} {tm[i]:+8.3f} {r[i]:6.3f} {central[i]:+8.3f} {lower[i]:+8.3f} {upper[i]:+8.3f} "
              f"{(upper[i] - lower[i]) / central[i]:12.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.fill_between(zm, lower, upper, color=BAND, lw=0, label="systematic range (gas temperature, resolution)")
    ax.plot(zm, central, color=BLUE, lw=2.0, label="Lumina, 640$^3$ maps / resolution ratio")
    ax.plot(B["z_cells"], B["Tb_cells"], ls="none", marker="o", ms=7, mfc=BLUE, mec="white", mew=1.5, label="whole-box gas cells")
    ax.axhline(0, color=INK2, lw=0.8)
    for zl, txt in ((4.75, "run restart"), (3.35, "self-shielding")):
        if zm.min() <= zl <= zm.max():
            ax.axvline(zl, color=INK2, lw=0.8, ls=":")
            ax.text(zl + 0.04, 1.2, txt, color=INK2, fontsize=8, va="bottom", ha="right", rotation=90)
    ax.set_xlim(a.zmax, zm.min())
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\langle \delta T_b \rangle$ [mK]")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
    print(f"\nwrote {a.out}.pdf, {a.out}.png, {a.out}.csv")


if __name__ == "__main__":
    main()
