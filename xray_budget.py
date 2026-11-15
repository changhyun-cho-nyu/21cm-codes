#!/usr/bin/env python3
"""
xray_budget.py
==============
Radiative energy injected per source and per RT bin in an AREPO-RT run, from the RT_LOG files
<output>/photons_{stars,bhs,hmxb,ism}.txt (one row per log call: scale factor a, then the energy injected into each
bin during that call, then a trailing 0; MRT_XRAY runs: 6 bins with edges 13.6, 24.59, 54.42, 100, 200, 500, 2000 eV).

The energy injected in each redshift window is the sum of the rows inside it. Restarts append rows that repeat an
interval already logged: whenever a decreases, the earlier rows with a >= the new a are dropped (the later run wins).
The four files are written by the same routine family with the same units, so ratios between them are meaningful.

Usage (reads the log files only; prints to stdout):
  python xray_budget.py <run>/output [--z 30 15 12 10.4 9 8 7 6 5]
"""
import argparse
import os
import sys

import numpy as np

SOURCES = ("stars", "bhs", "hmxb", "ism")
EDGES_EV = (13.6, 24.59, 54.42, 100, 200, 500, 2000)


def read_log(path):
    """Rows (a, e_1 .. e_n) with restart overlaps removed."""
    rows = []
    with open(path) as fp:
        for line in fp:
            v = line.split()
            if not v:
                continue
            try:
                x = [float(t) for t in v]
            except ValueError:
                continue
            a = x[0]
            while rows and rows[-1][0] >= a:      # restart: drop the overlap logged before
                rows.pop()
            rows.append(x)
    if not rows:
        return None
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir", help="<run>/output (read only)")
    p.add_argument("--z", type=float, nargs="+", default=[30, 15, 12, 10.4, 9, 8, 7, 6, 5],
                   help="window edges in redshift, high to low")
    a = p.parse_args()

    logs = {}
    for s in SOURCES:
        f = os.path.join(a.outdir, f"photons_{s}.txt")
        if os.path.exists(f):
            d = read_log(f)
            if d is not None:
                logs[s] = d
    if not logs:
        sys.exit(f"no photons_*.txt in {a.outdir}")
    nb = min(d.shape[1] - 1 for d in logs.values())
    nb = min(nb, len(EDGES_EV) - 1) if nb >= len(EDGES_EV) - 1 else nb
    if nb == len(EDGES_EV) - 1:
        bins = [f"{EDGES_EV[i]:g}-{EDGES_EV[i + 1]:g}" for i in range(nb)]
    else:
        bins = [f"bin{i}" for i in range(nb)]
    print(f"{a.outdir}: sources {list(logs)}, {nb} bins: {', '.join(bins)} eV")
    for s, d in logs.items():
        print(f"  {s:5s}: {len(d)} rows, z = {1 / d[0, 0] - 1:.2f} .. {1 / d[-1, 0] - 1:.2f}")

    zs = a.z
    for zhi, zlo in zip(zs[:-1], zs[1:]):
        alo, ahi = 1 / (1 + zhi), 1 / (1 + zlo)
        E = {}
        for s, d in logs.items():
            m = (d[:, 0] > alo) & (d[:, 0] <= ahi)
            E[s] = d[m, 1:1 + nb].sum(axis=0)
        tot = sum(E.values())
        print(f"\nz = {zhi:g} .. {zlo:g}: injected energy per bin (log units), share of each source per bin")
        print(f"  {'bin [eV]':>12s} {'total':>11s} " + " ".join(f"{s:>7s}" for s in E))
        for i in range(nb):
            sh = " ".join(f"{(E[s][i] / tot[i] if tot[i] > 0 else 0):7.3f}" for s in E)
            print(f"  {bins[i]:>12s} {tot[i]:11.4e} {sh}")
        if nb >= 6:
            x = sum(E[s][3:6] for s in E)
            xt = x.sum()
            if xt > 0:
                print("  X-ray bins (>= 100 eV): share of the X-ray energy per bin "
                      + " ".join(f"{bins[3 + k]}: {x[k] / xt:.3f}" for k in range(3)))
                print("  X-ray energy by source: "
                      + " ".join(f"{s} {E[s][3:6].sum() / xt:.3f}" for s in E))


if __name__ == "__main__":
    main()
