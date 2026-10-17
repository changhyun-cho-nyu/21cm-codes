"""
decompose_cells_He.py
=====================
Which gas makes the cell-level delta T_b of test_cells_Tb.py? Uses the cells of the random cubes saved by the
scan tasks (scan_<line>_<snap>_NNN.hdf5: pos, m, rho, V, T, x_HII, f_HeII, f_HeIII, J, dT_b per cell).

Why: the cell-level <dT_b^He> is ~7x the production (640^3) value at z = 4 and the ladder 320^3 ... 10240^3 has
not converged. Whether that signal comes from the bulk ionized IGM (resolved, robust) or from dense gas near the
star-formation threshold (resolution- and subgrid-dependent; star-forming gas carries no He II in the renders)
decides whether the cell value is trustworthy, whether a clumping correction of the 640^3 maps is possible, and
what to state in the paper.

For classes of cells by n_H [cm^-3, proper], by 1 + delta, by T and by x_HeII = f_HeII / y it reports
  volume and mass fraction, share of the cell-level sum(V dT_b), mean dT_b in the class,
  share of the variance of the cell-level map at the render's pixels (640^3) (pixel map = sum over classes of the class maps;
  share_c = cov(map_c, map) / var(map), the shares add up to 1);
and how concentrated the signal is: the share of sum(V dT_b) from the brightest 0.1%, 1%, 10% of the volume.

Usage
-----
  python decompose_cells_He.py --snapdir <.../snapshots> --snap 136 --line He --out /orcd/data/mvogelsb/004/chcho/cell_test
  python decompose_cells_He.py ... --igm [--cuts 3 10 30 100]
      does pixel averaging reproduce the IGM component itself? For cells with 1 + delta < cut: the cell-level
      sum(V dT_b) vs the same cells grouped into N/2 ... 4N pixels (mass-weighted T and fractions, density
      sum(m)/sum(V), volume-weighted J), with jackknife errors over cubes; at N^3 pixels the variance ratio and
      correlation of the grouped IGM map and of the production map (cellmap_*.hdf5) with the cell-level IGM map.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

from test_cells_Tb import Snapshot, Phys, group
from tb_physics import X_H, Y_HE, M_H

CLASSES = {
    "n_H [cm^-3]": ("nH", [0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 0.106, np.inf]),
    "1 + delta":   ("dl", [0, 1, 10, 100, 1000, 1e4, np.inf]),
    "T [K]":       ("T",  [0, 1e3, 1e4, 2e4, 5e4, 1e5, 1e6, np.inf]),
    "x_HeII":      ("xh", [-np.inf, 0.01, 0.1, 0.5, 0.9, np.inf]),
}


def load_cube(files, q, keys):
    parts = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            g = f.get(f"cube_{q:03d}")
            if g is not None:
                parts.append({k: g[k][()].astype(np.float64) for k in keys})
    return {k: np.concatenate([p_[k] for p_ in parts]) for k in keys} if parts else None


def igm_test(a, S, P, files, ncube, N_PIX):
    """does pixel averaging reproduce the IGM component itself? cells below a cut in 1 + delta, at the cell
    level vs the same cells grouped into pixels (as the renders average), and vs the production map"""
    box = S.box
    cuts = a.cuts
    lv = [N_PIX // 2, N_PIX, 2 * N_PIX, 4 * N_PIX]
    cm = os.path.join(a.out, f"cellmap_{a.line}_{a.snap:03d}.hdf5")
    if not os.path.exists(cm):
        sys.exit(f"no {cm} (test_cells_Tb.py merge --save-map)")
    with h5py.File(cm, "r") as f:
        prod = f["dTb_production"][()]
        order = tuple(int(x) for x in f.attrs["order"])
    keys = ("pos", "m", "rho", "V", "T", "xHII", "fII", "fIII", "J", "tb")
    per = []                                   # per cube: {cut: {"cell": .., level: ..}, "totV", "all": ..}
    maps = {c: {"cell": [], "group": []} for c in cuts}
    maps_all, maps_prod = [], []
    for q in range(ncube):
        c = load_cube(files, q, keys)
        if c is None:
            continue
        V = c["V"]
        dl = c["rho"] / P.rho_mean
        uk, inv6, ix6 = group(c["pos"], box, N_PIX)
        npx = len(uk)
        Vpix = np.bincount(inv6, V)
        first = np.zeros(npx, np.int64); first[inv6[::-1]] = np.arange(len(inv6))[::-1]
        pix = ix6[first]
        maps_prod.append(prod[pix[:, order[0]], pix[:, order[1]], pix[:, order[2]]].astype(np.float64))
        maps_all.append(np.bincount(inv6, V * c["tb"]) / Vpix)
        r = {"totV": V.sum(), "all": (V * c["tb"]).sum(), "prod": maps_prod[-1].sum() * (box / N_PIX)**3}
        for cut in cuts:
            s = dl < cut
            rc = {"cell": (V[s] * c["tb"][s]).sum()}
            for Ng in lv:
                inv = inv6[s] if Ng == N_PIX else group(c["pos"][s], box, Ng)[1]
                n = npx if Ng == N_PIX else None
                Mp = np.bincount(inv, c["m"][s], minlength=n or 0)
                Vp = np.bincount(inv, V[s], minlength=n or 0)
                ok = Vp > 0
                avg = lambda x: np.bincount(inv, c["m"][s] * x, minlength=n or 0)[ok] / Mp[ok]
                tbp = np.zeros(len(Vp))
                tbp[ok] = P.tb(Mp[ok] / Vp[ok], avg(c["T"][s]), avg(c["xHII"][s]), avg(c["fII"][s]), avg(c["fIII"][s]),
                               np.bincount(inv, V[s] * c["J"][s], minlength=n or 0)[ok] / Vp[ok])
                rc[Ng] = (Vp * tbp).sum()
                if Ng == N_PIX:
                    maps[cut]["group"].append(Vp * tbp / Vpix)
                    maps[cut]["cell"].append(np.bincount(inv6[s], V[s] * c["tb"][s], minlength=npx) / Vpix)
            r[cut] = rc
        per.append(r)
        print(f"  cube {q:2d}: {len(V):,} cells", flush=True)

    n = len(per)
    tot = lambda key, sub=None: np.array([(p_[key] if sub is None else p_[key][sub]) for p_ in per])
    TV = tot("totV").sum()
    def ratio_err(num, den):
        R = num.sum() / den.sum()
        jk = np.array([(num.sum() - num[i]) / (den.sum() - den[i]) for i in range(n)])
        return R, np.sqrt((n - 1) / n * ((jk - jk.mean())**2).sum())
    MA, MP = np.concatenate(maps_all), np.concatenate(maps_prod)
    print(f"\n  {n} cubes. cell level, all gas: <dT_b> = {tot('all').sum() / TV:+.5e} {P.L.unit}; production map on the same "
          f"pixels {tot('prod').sum() / TV:+.5e}")
    print("  IGM = cells with 1 + delta < cut. Means over the cubes' volume; ratios with jackknife errors over cubes.")
    print(f"  {'cut':>5} {'IGM share':>10} {'<dT_b>_IGM cell':>16} " + " ".join(f"{'grouped ' + str(g) + '/cell':>16}" for g in lv)
          + f" {'production/IGM':>15}")
    for cut in cuts:
        cell = tot(cut, "cell")
        sh = cell.sum() / tot("all").sum()
        cols = []
        for g in lv:
            R, e = ratio_err(tot(cut, g), cell)
            cols.append(f"{R:.4f}+-{e:.4f}")
        Rp, ep = ratio_err(tot("prod"), cell)
        print(f"  {cut:>5g} {sh:10.4f} {cell.sum() / TV:+16.5e} " + " ".join(f"{x:>16}" for x in cols) + f" {Rp:.4f}+-{ep:.4f}")
    print(f"\n  at {N_PIX}^3 pixels ({len(MA):,}): variance ratio / correlation")
    print(f"  {'cut':>5} {'grouped IGM vs cell IGM':>26} {'production vs cell IGM':>24} {'cell IGM vs cell all':>22}")
    for cut in cuts:
        C, Gm = np.concatenate(maps[cut]["cell"]), np.concatenate(maps[cut]["group"])
        f = lambda x, y: f"{np.var(x) / np.var(y):.4f}/{np.corrcoef(x, y)[0, 1]:.4f}"
        print(f"  {cut:>5g} {f(Gm, C):>26} {f(MP, C):>24} {f(C, MA):>22}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapdir", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--line", choices=["H", "He"], default="He")
    p.add_argument("--out", required=True, help="directory of the scan task files")
    p.add_argument("--igm", action="store_true",
                   help="instead: does pixel averaging reproduce the IGM (1 + delta < cut) component? needs cellmap_*.hdf5")
    p.add_argument("--cuts", type=float, nargs="+", default=[3, 10, 30, 100])
    a = p.parse_args()

    S = Snapshot(a.snapdir, a.snap)
    P = Phys(S, a.line)
    box, z = S.box, S.z
    files = sorted(f for f in glob.glob(os.path.join(a.out, f"scan_{a.line}_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        sys.exit(f"no scan files in {a.out}")
    with h5py.File(files[0], "r") as f:
        ncube = int(f.attrs["n_cube"])
        N_PIX = f["V"].shape[0]                      # the render grid of the scan (640 for Lumina)
    nH_fac = X_H * P.rho_unit * (1.0 + z)**3 / M_H
    print(f"snapshot {a.snap}, z = {z:.4f}, line {a.line}: {len(files)} task files, {ncube} cubes; "
          f"<rho_b> = {P.rho_mean:.4e} code = n_H {P.rho_mean * nH_fac:.3e} cm^-3")
    if a.igm:
        return igm_test(a, S, P, files, ncube, N_PIX)

    acc = {name: {k: np.zeros(len(e) - 1) for k in ("V", "m", "Vtb")} for name, (_, e) in CLASSES.items()}
    pix = {name: [] for name in CLASSES}          # per cube: class maps at 640^3 (sum V tb / pixel V)
    tot_map = []
    allV, allTb = [], []
    totV = totm = totVtb = 0.0
    for q in range(ncube):
        parts = []
        for fn in files:
            with h5py.File(fn, "r") as f:
                g = f.get(f"cube_{q:03d}")
                if g is not None:
                    parts.append({k: g[k][()].astype(np.float64) for k in ("pos", "m", "rho", "V", "T", "fII", "tb")})
        if not parts:
            continue
        c = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
        prop = {"nH": c["rho"] * nH_fac, "dl": c["rho"] / P.rho_mean, "T": c["T"], "xh": c["fII"] / Y_HE}
        V, Vtb = c["V"], c["V"] * c["tb"]
        totV += V.sum(); totm += c["m"].sum(); totVtb += Vtb.sum()
        inv = group(c["pos"], box, N_PIX)[1]
        Vp = np.bincount(inv, V)
        tot_map.append(np.bincount(inv, Vtb) / Vp)
        for name, (key, edges) in CLASSES.items():
            idx = np.clip(np.digitize(prop[key], edges) - 1, 0, len(edges) - 2)
            acc[name]["V"] += np.bincount(idx, V, minlength=len(edges) - 1)
            acc[name]["m"] += np.bincount(idx, c["m"], minlength=len(edges) - 1)
            acc[name]["Vtb"] += np.bincount(idx, Vtb, minlength=len(edges) - 1)
            nc = len(edges) - 1
            pix[name].append(np.stack([np.bincount(inv[idx == j], Vtb[idx == j], minlength=len(Vp)) / Vp for j in range(nc)]))
        # concentration: keep a 1-in-10 sample of the cells (enough for the brightest-volume shares)
        allV.append(V[::10]); allTb.append(c["tb"][::10])
        print(f"  cube {q:2d}: {len(V):,} cells", flush=True)

    M = np.concatenate(tot_map)
    var = np.var(M)
    print(f"\n  cells used: volume-weighted <dT_b> = {totVtb / totV:+.5e} {P.L.unit}; {N_PIX}^3 pixel map: {len(M):,} pixels, "
          f"mean {M.mean():+.5e}, std {np.sqrt(var):.4e}")
    for name, (key, edges) in CLASSES.items():
        A = acc[name]
        maps = np.concatenate(pix[name], axis=1)
        vshare = [np.mean((maps[j] - maps[j].mean()) * (M - M.mean())) / var for j in range(len(edges) - 1)]
        print(f"\n  by {name}:")
        print(f"    {'class':>20} {'volume':>9} {'mass':>9} {'<dT_b> share':>13} {'mean dT_b':>12} {'variance share':>15}")
        for j in range(len(edges) - 1):
            lab = f"{edges[j]:g} - {edges[j + 1]:g}"
            mean_c = A["Vtb"][j] / A["V"][j] if A["V"][j] > 0 else np.nan
            print(f"    {lab:>20} {A['V'][j] / totV:9.4f} {A['m'][j] / totm:9.4f} {A['Vtb'][j] / totVtb:+13.4f} "
                  f"{mean_c:+12.4e} {vshare[j]:+15.4f}")
    V_, T_ = np.concatenate(allV), np.concatenate(allTb)
    o = np.argsort(-T_)
    cumV = np.cumsum(V_[o]) / V_.sum()
    cumS = np.cumsum((V_ * T_)[o]) / (V_ * T_).sum()
    print(f"\n  concentration (1-in-10 cell sample): share of sum(V dT_b) from the brightest "
          + ", ".join(f"{f * 100:g}%: {cumS[np.searchsorted(cumV, f)]:.3f}" for f in (1e-3, 1e-2, 1e-1))
          + " of the volume")


if __name__ == "__main__":
    main()
