"""
plot_cells_global_xray.py
=========================
21 cm global signal of the two 50 cMpc test runs (with / without MRT_XRAY): the 128^3 pixel maps at every render
(data/compare_xray.csv, compare_xray.py) as lines, and the gas-cell level (cells_Tb_xray.py: mean_cells_cell, tau_GP
from the cell) at every full snapshot that has a $BASE/<run>/cells_Tb/cells_Tb_NNN.npz, as markers. Run on ORCD to read
the npz files; without them (e.g. on a laptop) it falls back to the four pilot values of NOTES.md (job 29467469).
Writes data/xray_cells_global.png. Quick look (user, 2026-11-18).

  python plot_cells_global_xray.py [--base /orcd/data/mvogelsb/004/chcho/xray]
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"withoutXrays": ("no X-rays", "#1f77b4", "Tb_noX"), "withXrays": ("X-rays", "#d62728", "Tb_X")}
PILOT = {   # NOTES.md, "Results (job 29467469)", column "cells (cell)" [mK]
    "withoutXrays": {15.016: -8.005, 11.931: -29.710, 10.014: -21.492, 7.995: -3.623},
    "withXrays":    {15.016: -7.662, 11.931: -8.665, 10.014: +17.467, 7.995: +16.672},
}


def cells(base, run):
    out = {}
    for p in sorted(glob.glob(os.path.join(base, run, "cells_Tb", "cells_Tb_*.npz"))):
        d = np.load(p)
        out[float(d["z"])] = float(d["mean_cells_cell"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/orcd/data/mvogelsb/004/chcho/xray")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(HERE, "data", "compare_xray.csv"))
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 14})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="0.6", lw=1)
    for run, (lab, c, col) in RUNS.items():
        ax.plot(df["z"], df[col], color=c, lw=2, label=f"{lab}: $128^3$ pixels")
        cc = cells(a.base, run)
        src = "npz"
        if not cc:
            cc, src = PILOT[run], "NOTES.md pilot values"
        zc = np.array(sorted(cc)); tc = np.array([cc[z] for z in zc])
        print(f"{run}: {len(zc)} gas-cell points ({src}), z = {zc.max():.2f} ... {zc.min():.2f}")
        ax.plot(zc, tc, color=c, ls="--", lw=1.2, marker="o", ms=7, mfc="white", mew=2, label=f"{lab}: gas cells")
    ax.set_xlim(20, 4.8)
    ax.set_xlabel(r"$z$", fontsize=18)
    ax.set_ylabel(r"$\langle\delta T_b^{\rm H}\rangle$ [mK]", fontsize=18)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.legend(frameon=False, fontsize=12, loc="lower left")
    ax.text(0.02, 0.96, "50 cMpc test runs; gas cells only at full snapshots\n(none between z = 15.0 and 11.9)",
            transform=ax.transAxes, ha="left", va="top", fontsize=10, color="0.35")
    fig.tight_layout()
    out = os.path.join(HERE, "data", "xray_cells_global.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
