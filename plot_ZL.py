"""
plot_ZL.py
==========
Stellar metallicity evolution for the Lyman-band emissivity (from compute_ZL.py and make_Z_snapshots.py outputs):
  (1) all star particles, mass-weighted (the old J_alpha code),
  (2) our way: the metallicity distribution of the stars younger than 20 Myr (initial-mass weights w_j on the BPASS
      grid; shaded 10-90 per cent, line = mean), which the J_alpha codes sum spectrum by spectrum,
  (3) luminosity-weighted Z_L in the H and He II Lyman bands (light emitted now by all populations),
plus Z_eff (the single Z whose band output equals the mix). Lower panels: band photon output of a single metallicity
divided by that of the mix (1 = our way).

  python plot_ZL.py --zsnap Z_snapshots.csv --birth ZL/ZL_birth.csv --now ZL/ZL_now.csv --out ZL_evolution.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ZGRID = np.array([1e-5, 1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.4e-2, 2e-2, 3e-2, 4e-2])
C = {"old": "#7a7a72", "ours": "#2a78d6", "H": "#eb6834", "He": "#1baf7a"}


def percentile_logZ(w, q):
    """q-quantile of log Z for mass w_j at the grid points (CDF linear in log Z between grid points)"""
    c = np.cumsum(w) / w.sum()
    lz = np.log10(ZGRID)
    return 10**np.interp(q, np.concatenate([[0.0], c]), np.concatenate([[lz[0]], lz]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zsnap", default="Z_snapshots.csv")
    p.add_argument("--birth", default="ZL/ZL_birth.csv")
    p.add_argument("--now", default="ZL/ZL_now.csv")
    p.add_argument("--out", default="ZL_evolution.png")
    a = p.parse_args()
    zs = pd.read_csv(a.zsnap).sort_values("redshift")
    zs = zs[zs["Z_young_20"] > 0]          # z = 22.97: one young star at Z = 0 (held at the lowest BPASS Z in the codes)
    b = pd.read_csv(a.birth).sort_values("redshift")
    nw = pd.read_csv(a.now).dropna(subset=["ZL_H"]).sort_values("redshift")
    W = zs[[f"w_{i:02d}" for i in range(13)]].to_numpy()
    lo = np.array([percentile_logZ(w, 0.1) for w in W])
    hi = np.array([percentile_logZ(w, 0.9) for w in W])

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12})
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.32, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(zs["redshift"], lo, hi, color=C["ours"], alpha=0.15, lw=0,
                    label="(2) ours: stars < 20 Myr, 10–90% of mass")
    ax.plot(zs["redshift"], zs["Z_young_20"], color=C["ours"], lw=2, label="(2) ours: mean of stars < 20 Myr")
    ax.plot(zs["redshift"], zs["Z_mean_mw"], color=C["old"], lw=2, ls="--", label="(1) all stars, mass-weighted (old)")
    ax.plot(nw["redshift"], nw["ZL_H"], color=C["H"], lw=2, label=r"(3) $Z_L$, H I Lyman band")
    ax.plot(nw["redshift"], nw["ZL_He"], color=C["He"], lw=2, label=r"(3) $Z_L$, He II Lyman band")
    ax.plot(b["redshift"], b["Zeff_H"], color=C["H"], lw=1.2, ls=":", label=r"$Z_{\rm eff}$, H (reproduces our output)")
    ax.plot(b["redshift"], b["Zeff_He"], color=C["He"], lw=1.2, ls=":", label=r"$Z_{\rm eff}$, He (reproduces our output)")
    ax.set_yscale("log")
    ax.set_xlim(18.5, 3)
    ax.set_ylim(1e-5, 5e-2)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel("Stellar metallicity $Z$")
    ax.grid(True, which="major", color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=9.5, ncol=2, frameon=False, loc="upper left")

    for k, band in enumerate(("H", "He")):
        bx = fig.add_subplot(gs[1, k])
        bx.axhline(1.0, color=C["ours"], lw=2, label="(2) ours (mix of spectra)")
        bx.plot(b["redshift"], b[f"single_Z_mean_mw_{band}"], color=C["old"], lw=2, ls="--",
                label="(1) all stars, mass-weighted")
        bx.plot(b["redshift"], b[f"single_Z_young_20_{band}"], color=C["ours"], lw=1.2, ls="-.",
                label="mean $Z$ of stars < 20 Myr")
        bx.plot(b["redshift"], b[f"single_ZL_{band}"], color=C[band], lw=2, label=f"(3) $Z_L$ of the {band} band")
        bx.set_xlim(18.5, 3)
        bx.set_ylim(0.2 if band == "He" else 0.8, 1.3 if band == "He" else 1.05)
        bx.set_xlabel("Redshift $z$")
        bx.set_ylabel(f"{band}-band output, single $Z$ / ours")
        bx.set_title(f"{'H I' if band == 'H' else 'He II'} Lyman band", fontsize=12)
        bx.grid(True, color="#e1e0d9", lw=0.6)
        bx.legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
