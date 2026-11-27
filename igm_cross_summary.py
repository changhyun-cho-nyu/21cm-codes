"""
igm_cross_summary.py
====================
Table of run_cross_he_igm.sh (user, 2026-11-27): is the He III bubble imprint on the 3.46 cm signal carried by the IGM
or by the dense halo gas? For each full snapshot and each gas-cell He map (all cells; only cells with 1+delta < 100,
30, 10, 3), at k = 0.05, 0.1, 0.2, 0.5 h/cMpc (log-k interpolation between bins, as run_cross_he_rho.sh):
  share     = mean of the map / mean of the all-cell map (the map's share of <dT_b^He>)
  1-r2_rho  = 1 - r(He, density)^2: the share of the He power not linearly tied to the gas-cell density
  r_x       = r(He, x_HeIII)
  rp_x      = partial correlation of He and x_HeIII at fixed density,
              (r_Hx - r_Hd r_xd) / sqrt((1 - r_Hd^2)(1 - r_xd^2)), with r_xd = r(x_HeIII, density) from cross_x_rho_NNN.npz
              (the correlation of the parts of He and x_HeIII that density does not predict, band by band)
If the IGM-only maps keep a partial r as strong as the all-cell map, the bubble imprint lives in gas the simulation
resolves well. 1 - r^2 alone rises when dense gas is cut whatever the bubbles do (less of the map follows density),
so the partial r is the test.

  python igm_cross_summary.py --dir /orcd/data/mvogelsb/004/chcho/cross_he_igm [--csv out.csv]
  python igm_cross_summary.py --dir /orcd/data/mvogelsb/004/chcho/cross_he_igm --null     (null test, null_igm_map.py)
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

MAPS = ("dTb_cell", "dTb_cell_igm_lt100", "dTb_cell_igm_lt30", "dTb_cell_igm_lt10", "dTb_cell_igm_lt3")
MAPS_NULL = ("null_all", "null_lt100", "null_lt30", "null_lt10", "null_lt3")    # null_igm_map.py (run_null_igm_cross.sh)


def label(m):
    return "all" if m in ("dTb_cell", "null_all") else "<" + m.rsplit("lt", 1)[1]


LABEL = {m: label(m) for m in MAPS + MAPS_NULL}
K0 = (0.05, 0.1, 0.2, 0.5)


def at(k, v, k0):
    return float(np.interp(np.log(k0), np.log(k), v))


def partial(rHx, rHd, rxd):
    return (rHx - rHd * rxd) / np.sqrt((1.0 - rHd**2) * (1.0 - rxd**2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--summary", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cell_test_summary.csv"))
    p.add_argument("--csv", default=None, help="default <dir>/igm_cross_summary.csv (--null: igm_cross_null_summary.csv)")
    p.add_argument("--null", action="store_true", help="the null maps of null_igm_map.py (cross_igm_null_*_NNN.npz): "
                   "(1+delta)^2 with the same cuts, no x_HeIII; 'share' is then relative to null_all")
    a = p.parse_args()
    zs = {}
    if os.path.exists(a.summary):
        zs = {int(r["snap"]): float(r["z"]) for r in csv.DictReader(l for l in open(a.summary) if not l.startswith("#"))}
    snaps = sorted(int(re.search(r"_(\d{3})\.npz$", f).group(1)) for f in glob.glob(os.path.join(a.dir, "cross_x_rho_*.npz")))
    if not snaps:
        sys.exit(f"no cross_x_rho_*.npz in {a.dir}")
    maps = MAPS_NULL if a.null else MAPS
    rows = []
    for s in snaps:
        X = np.load(os.path.join(a.dir, f"cross_x_rho_{s:03d}.npz"))
        k, rxd = X["k"], X["r_0"]
        D = {}
        for m in maps:
            f = os.path.join(a.dir, f"cross_igm_{m}_{s:03d}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            if not np.allclose(d["k"], k, rtol=1e-12):
                sys.exit(f"{f}: k bins differ from cross_x_rho_{s:03d}.npz")
            if "HeIII" not in str(d["B0"]) or "rho" not in str(d["B1"]):
                sys.exit(f"{f}: B0 = {d['B0']}, B1 = {d['B1']} (expected HeIII_VolumeFraction, gas-cell rho)")
            D[m] = d
        if maps[0] not in D:
            continue
        miss = [m for m in maps if m not in D]
        if miss:
            print(f"snapshot {s}: missing {miss}")
        mean_all = float(D[maps[0]]["mean_A"])
        print(f"\nfull snapshot {s}" + (f", z = {zs[s]:.3f}" if s in zs else "") +
              f": modes at k = {', '.join(f'{k0:g}: {at(k, X['N_modes'], k0):.0f}' for k0 in K0)};  "
              f"r(x_HeIII, density) = {', '.join(f'{at(k, rxd, k0):+.3f}' for k0 in K0)}")
        print(f"  {'map':>5} {'share':>6} | 1 - r^2(He, density)      | r(He, x_HeIII)            | "
              f"partial r(He, x_HeIII | density)   (k = {' / '.join(f'{k0:g}' for k0 in K0)})")
        for m, d in D.items():
            rHx, rHd = d["r_0"], d["r_1"]
            rp = partial(rHx, rHd, rxd)
            share = float(d["mean_A"]) / mean_all
            v1 = [1 - at(k, rHd, k0)**2 for k0 in K0]
            v2 = [at(k, rHx, k0) for k0 in K0]
            v3 = [at(k, rp, k0) for k0 in K0]
            print(f"  {LABEL[m]:>5} {share:6.3f} | " + " ".join(f"{v:.3f}" for v in v1) + " | "
                  + " ".join(f"{v:+.3f}" for v in v2) + " | " + " ".join(f"{v:+.3f}" for v in v3))
            row = {"snap": s, "z": zs.get(s, np.nan), "map": LABEL[m], "share": share}
            for k0, u1, u2, u3 in zip(K0, v1, v2, v3):
                row.update({f"1-r2_rho_k{k0:g}": u1, f"r_x_k{k0:g}": u2, f"rp_x_k{k0:g}": u3})
            rows.append(row)
    out = a.csv or os.path.join(a.dir, "igm_cross_null_summary.csv" if a.null else "igm_cross_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
