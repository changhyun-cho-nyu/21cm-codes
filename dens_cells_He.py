"""
dens_cells_He.py
================
Whole-box gas state and 3.46 cm signal by gas density, from EVERY gas cell of a full snapshot (user, 2026-11-26).
Question: after He II reionization, which gas still holds He II, and which gas makes the 3.46 cm signal? The earlier
answer (check_switch_cells.py) came from the 64 test cubes of the He scans (~0.8% of the volume); this is the whole box.

scan (one array task = 1/ntask of the rows; reads the same fields and uses the same physics as test_cells_Tb.py scan,
      imported from it: Snapshot, prep_cells, Phys, J_at, choose_order, render_index):
  each cell: 1+delta = rho / rho_mean (rho_mean = Omega_b rho_crit, as test_cells_Tb.py), star-forming flag (SFR > 0),
  render conventions (star-forming gas: x_HII = He II = He III = 0), T from u and Ne, J of its 640^3 pixel,
  delta T_b^He from Phys.tb. Cells are binned in log10(1+delta) (0.05 dex, -3 ... 7, plus under/overflow; star-forming
  cells in their own bin), summing: count, m, V, m x_HII, m x_HeII, m x_HeIII (x = f / y for He), m n_e/n_H, m T, V dT_b.
  Writes <out>/dens_He_<snap>_<task>.npz (small).
merge: adds the task files (stops unless all ntask are present, from the same snapshot and settings, and their rows
  tile 0 ... N_cells exactly), prints per 1+delta class: mass and volume shares, mass-weighted x_HII, x_HeII, x_HeIII,
  n_e/n_H, T, the class's contribution to <dT_b^He> = sum V dT_b / sum V and its share; the whole-box mean (equal to
  the gas-cell mean of test_cells_Tb.py merge for the same J); and the local slope of x_HeII/x_HeIII against 1+delta
  (photoionization equilibrium with a uniform rate: slope ~ 1). Writes <out>/dens_He_<snap>.csv (fine bins) and
  dens_He_<snap>_classes.csv.

  python -u dens_cells_He.py scan --snapdir D --snap 158 --J <J stars> --J <J AGN> --render <ren_640> --render-snap 705 \
      --out <dir> --task T --ntask 32
  python dens_cells_He.py merge --snap 158 --out <dir> --ntask 32
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

from test_cells_Tb import Snapshot, Phys, prep_cells, J_at, choose_order, render_index
from tb_physics import Y_HE

import h5py

CLASS_EDGES = (0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)     # 1+delta classes of the summary table
# log10(1+delta) bin edges: 0.05 dex from -3 to 7, plus the class edges, so every fine bin lies inside one class
LOG_EDGES = np.unique(np.concatenate([np.round(np.arange(-3.0, 7.0001, 0.05), 6), np.log10(CLASS_EDGES)]))
LOG_EDGES = LOG_EDGES[np.concatenate([[True], np.diff(LOG_EDGES) > 1e-6])]   # (log10 1, 10, ... are on the grid)
NB = len(LOG_EDGES) + 1                                        # + underflow (0) and overflow (NB-1)
I_SF = NB                                                      # star-forming cells: their own bin
KEYS = ("n", "m", "V", "m_xHII", "m_xHeII", "m_xHeIII", "m_ne", "m_T", "V_tb")


def scan(a):
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    P = Phys(S, "He", 1)
    if not 0 <= a.task < a.ntask:
        sys.exit(f"ERROR: --task {a.task} outside 0..{a.ntask - 1}")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"dens_He_{a.snap:03d}_{a.task:03d}.npz")
    if os.path.exists(out):
        print(f"{out} exists: task done")
        return
    edges = np.linspace(0, S.n, a.ntask + 1).astype(np.int64)
    r0, r1 = int(edges[a.task]), int(edges[a.task + 1])
    p = os.path.join(a.render, "Density", f"Density_{a.render_snap:03d}.hdf5")
    with h5py.File(p if os.path.exists(p) else os.path.join(a.render, "All", f"All_{a.render_snap:03d}.hdf5"), "r") as f:
        Rd = f["Density"][()]
    N = Rd.shape[0]
    J, Jinfo = J_at(a, N, S.z)
    if J.shape != (N,) * 3:
        sys.exit(f"ERROR: J is {J.shape}, render {Rd.shape}")
    print(f"snapshot {a.snap} z = {S.z:.5f}, task {a.task}/{a.ntask}: rows {r0:,}-{r1:,} of {S.n:,}; "
          f"rho_mean (Omega_b rho_crit) = {P.rho_mean:.6e} code units")
    order = choose_order(S, (r0, min(r1, r0 + 5_000_000)), Rd, S.box)
    del Rd
    acc = {k: np.zeros(NB + 1) for k in KEYS}
    t0 = time.time()
    for i0 in range(r0, r1, a.chunk):
        i1 = min(r1, i0 + a.chunk)
        c = prep_cells(S, i0, i1)
        sf = S.read("StarFormationRate", i0, i1) > 0
        i, j, k = render_index(c["pos"], S.box, N, order)
        Jc = J[i, j, k].astype(np.float64)
        del i, j, k
        tb = P.tb(c["rho"], c["T"], c["xHII"], c["fII"], c["fIII"], Jc)
        if not np.all(np.isfinite(tb)):
            sys.exit(f"STOP: rows {i0}-{i1}: {np.sum(~np.isfinite(tb))} non-finite dT_b")
        b = np.searchsorted(LOG_EDGES, np.log10(c["rho"] / P.rho_mean), side="right")   # 0: below -3, NB-1: above 7
        b = np.where(sf, I_SF, b)
        m, V = c["m"], c["V"]
        w = {"n": np.ones_like(m), "m": m, "V": V, "m_xHII": m * c["xHII"], "m_xHeII": m * c["fII"] / Y_HE,
             "m_xHeIII": m * c["fIII"] / Y_HE, "m_ne": m * (c["xHII"] + c["fII"] + 2.0 * c["fIII"]),
             "m_T": m * c["T"], "V_tb": V * tb}
        for key in KEYS:
            acc[key] += np.bincount(b, w[key], minlength=NB + 1)
        print(f"  rows {i1:,} / {r1:,}: {time.time() - t0:.0f} s", flush=True)
    part = out + ".part.npz"
    np.savez(part, snap=a.snap, z=S.z, task=a.task, ntask=a.ntask, rows=np.array([r0, r1]), n_total=S.n,
             rho_mean=P.rho_mean, render_snap=a.render_snap, J_files=";".join(os.path.basename(x) for x in a.J),
             log_edges=LOG_EDGES, **acc)
    os.replace(part, out)
    print(f"wrote {out} ({time.time() - t0:.0f} s)")


def merge(a):
    files = sorted(glob.glob(os.path.join(a.out, f"dens_He_{a.snap:03d}_[0-9][0-9][0-9].npz")))
    if len(files) != a.ntask:
        sys.exit(f"STOP: {len(files)} task files in {a.out}, expected {a.ntask}")
    T = [np.load(f) for f in files]
    ref = T[0]
    for t, f in zip(T, files):
        for k in ("snap", "z", "ntask", "n_total", "rho_mean", "render_snap", "J_files"):
            if str(t[k]) != str(ref[k]):
                sys.exit(f"STOP: {f}: {k} = {t[k]} differs from {ref[k]}")
        if not np.array_equal(t["log_edges"], LOG_EDGES) or int(t["ntask"]) != a.ntask:
            sys.exit(f"STOP: {f}: other bins or ntask than this run")
    rows = sorted(tuple(int(v) for v in t["rows"]) for t in T)
    if rows[0][0] != 0 or rows[-1][1] != int(ref["n_total"]) or any(rows[i][1] != rows[i + 1][0] for i in range(len(rows) - 1)):
        sys.exit(f"STOP: the task rows do not tile 0 ... {int(ref['n_total'])}: {rows[:3]} ...")
    S = {k: sum(t[k] for t in T) for k in KEYS}
    ncell = int(round(S["n"].sum()))
    if ncell != int(ref["n_total"]):
        sys.exit(f"STOP: {ncell:,} cells binned, the snapshot has {int(ref['n_total']):,}")
    z = float(ref["z"])
    Vtot, Mtot = S["V"].sum(), S["m"].sum()
    mean = S["V_tb"].sum() / Vtot
    print(f"snapshot {int(ref['snap'])}, z = {z:.4f}: {ncell:,} cells (every cell), J {ref['J_files']} (render "
          f"{int(ref['render_snap'])}); sum V / box^3 is checked in test_cells_Tb.py merge")
    print(f"WHOLE BOX <dT_b^He> = {mean:.5e} uK  (compare: the gas-cell mean of test_cells_Tb.py merge with the same J)")
    print(f"star-forming cells: mass share {S['m'][I_SF] / Mtot:.4f}, contribution {S['V_tb'][I_SF] / Vtot:+.3e} uK "
          f"(He II = He III = 0 by the render convention)")
    # 1+delta classes (non-star-forming)
    lo = np.concatenate([[-np.inf], LOG_EDGES])                # bin b covers [lo[b], hi[b])
    hi = np.concatenate([LOG_EDGES, [np.inf]])
    mid = 10**(0.5 * (lo[1:-1] + hi[1:-1]))
    rows_out = []
    print("\n1+delta class (non-star-forming): mass share | volume share | x_HII  x_HeII  x_HeIII  n_e/n_H  T [K] | "
          "<dT_b> contribution [uK]  share of the mean")
    # fine bin b (1 ... NB-2) covers [LOG_EDGES[b-1], LOG_EDGES[b]); 0 = underflow, NB-1 = overflow
    cut = [0] + [int(np.argmin(np.abs(LOG_EDGES - np.log10(e)))) + 1 for e in CLASS_EDGES] + [NB]
    names = [f"<{CLASS_EDGES[0]:g}"] + [f"{a_:g}-{b_:g}" for a_, b_ in zip(CLASS_EDGES[:-1], CLASS_EDGES[1:])] \
        + [f">{CLASS_EDGES[-1]:g}"]
    for name, b0, b1 in zip(names, cut[:-1], cut[1:]):
        s = {k: S[k][b0:b1].sum() for k in KEYS}
        mm = max(s["m"], 1e-300)
        r = dict(cls=name, mass_share=s["m"] / Mtot, vol_share=s["V"] / Vtot, x_HII=s["m_xHII"] / mm,
                 x_HeII=s["m_xHeII"] / mm, x_HeIII=s["m_xHeIII"] / mm, ne_nH=s["m_ne"] / mm, T=s["m_T"] / mm,
                 dTb_contrib=s["V_tb"] / Vtot, share=s["V_tb"] / Vtot / mean)
        rows_out.append(r)
        print(f"  {r['cls']:>10s}  {r['mass_share']:.4f}  {r['vol_share']:.4f} | {r['x_HII']:.5f} {r['x_HeII']:.4f} "
              f"{r['x_HeIII']:.4f} {r['ne_nH']:.4f} {r['T']:9.4g} | {r['dTb_contrib']:+.4e}  {r['share']:.4f}")
    sf = {k: S[k][I_SF] for k in KEYS}
    rows_out.append(dict(cls="star-forming", mass_share=sf["m"] / Mtot, vol_share=sf["V"] / Vtot, x_HII=0.0, x_HeII=0.0,
                         x_HeIII=0.0, ne_nH=0.0, T=sf["m_T"] / max(sf["m"], 1e-300), dTb_contrib=sf["V_tb"] / Vtot,
                         share=sf["V_tb"] / Vtot / mean))
    print(f"  (sum of the shares, with star-forming: {sum(r['share'] for r in rows_out):.6f}; of the mass shares: "
          f"{sum(r['mass_share'] for r in rows_out):.6f})")
    # equilibrium slope: x_HeII / x_HeIII against 1+delta, fine bins with mass, non-star-forming
    ok = (S["m"][1:NB - 1] > 0) & (S["m_xHeIII"][1:NB - 1] > 0) & (S["m_xHeII"][1:NB - 1] > 0)
    q = np.where(ok, S["m_xHeII"][1:NB - 1] / np.where(ok, S["m_xHeIII"][1:NB - 1], 1.0), np.nan)
    print("\nx_HeII / x_HeIII and its local slope d ln / d ln(1+delta) (photoionization equilibrium, uniform rate: ~1):")
    for d0 in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        ib = int(np.argmin(np.abs(np.log10(mid) - np.log10(d0))))
        sl = slice(max(ib - 4, 0), min(ib + 5, len(mid)))
        good = np.isfinite(q[sl]) & (q[sl] > 0)
        slope = np.polyfit(np.log(mid[sl][good]), np.log(q[sl][good]), 1)[0] if good.sum() >= 3 else np.nan
        print(f"  1+delta = {d0:7g}: ratio {q[ib]:.4g}, slope {slope:.2f}")
    # outputs
    import csv
    fine = os.path.join(a.out, f"dens_He_{a.snap:03d}.csv")
    with open(fine, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["# z", z, "whole-box <dT_b^He> [uK]", mean, "cells", ncell])
        wr.writerow(["bin", "log10_1pd_lo", "log10_1pd_hi"] + list(KEYS))
        for b in range(NB + 1):
            name = "star-forming" if b == I_SF else str(b)
            l0 = "" if b == I_SF else lo[b]; h0 = "" if b == I_SF else hi[b]
            wr.writerow([name, l0, h0] + [S[k][b] for k in KEYS])
    cls = os.path.join(a.out, f"dens_He_{a.snap:03d}_classes.csv")
    with open(cls, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        wr.writeheader(); wr.writerows(rows_out)
    print(f"\nwrote {fine} and {cls}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("scan")
    q.add_argument("--snapdir", required=True)
    q.add_argument("--snap", type=int, required=True)
    q.add_argument("--gas-file", default=None)
    q.add_argument("--J", action="append", required=True, help="He J files at --render-snap (stars, AGN); or none")
    q.add_argument("--render", required=True)
    q.add_argument("--render-snap", type=int, required=True)
    q.add_argument("--max-dz", type=float, default=1e-4)
    q.add_argument("--out", required=True)
    q.add_argument("--task", type=int, required=True)
    q.add_argument("--ntask", type=int, required=True)
    q.add_argument("--chunk", type=int, default=20_000_000)
    m = sub.add_parser("merge")
    m.add_argument("--snap", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--ntask", type=int, required=True)
    a = p.parse_args()
    if a.mode == "scan":
        a.J_b, a.render_snap_b = [], None                         # J_at: no z interpolation (J at the render)
        scan(a)
    else:
        merge(a)


if __name__ == "__main__":
    main()
