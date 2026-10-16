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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapdir", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--line", choices=["H", "He"], default="He")
    p.add_argument("--out", required=True, help="directory of the scan task files")
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
