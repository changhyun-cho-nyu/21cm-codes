"""
decompose_cells_H.py
====================
Which gas makes the gas-cell 21 cm delta T_b of test_cells_Tb.py --line H? The H counterpart of
decompose_cells_He.py, from the cells of the random cubes saved by the scan tasks (scan_H_<snap>_NNN.hdf5:
pos, m, rho, V, T, x_HII, f_HeII, f_HeIII, J, dT_b per cell; spin model 1, tau_GP from the cell's n_HI).
No rescan: it only reads the scan files (keep them until this has run).

Why (user, 2026-11-20): at z = 23 (snapshot 011) the gas cells emit (+0.58 mK) while the 640^3 maps absorb, and
at z = 15 (024) the cells absorb (-6.8 mK) while the maps emit; the ladder changes sign between 1280^3 and 2560^3.
Expected but not yet measured: at z = 23 dense, warm, collisionally coupled gas (x_c >~ 1, T_K >> T_gamma) emits
and outweighs the weakly absorbing IGM (x_c ~ 1e-3, x_a ~ 1e-5); at z = 15 cold diffuse gas absorbs once x_a ~ 0.1.
This says which gas it is.

Reports, for the cells of the cubes:
  - the check: the volume-weighted <dT_b> of the cube cells, which must equal the merge log's "cell" line (CUBES);
  - emission (dT_b > 0) vs absorption (dT_b < 0): volume, H I mass, share of sum(V dT_b), mean dT_b;
  - classes of 1 + delta, T_K, x_c (collisional coupling, computed here from the cell, no T_S solve), x_HII and
    n_H, with "star-forming" (x_HII = f_HeII = f_HeIII = 0 exactly: the render convention for Sfr > 0 gas, which
    test_cells_Tb.prep_cells follows) as its own x_HII class: volume, H I mass, share of sum(V dT_b) with a
    jackknife error over cubes, mean dT_b, share of the variance of the cell-level map at the render's pixels (640^3);
  - 2D tables 1 + delta x T_K: volume fraction and share of sum(V dT_b);
  - concentration: the share of the emission (absorption) from the brightest (darkest) 0.1%, 1%, 10% of the
    volume (1-in-10 cell sample);
  - x_a = x~_a/S~_a of the cells (from their J) for reference.
Shares: class sum(V dT_b) / total sum(V dT_b); with emission and absorption both present they can exceed 1 or be
negative (e.g. +3.0 and -2.0). Writes decomp_H_<snap>.npz (per-cube class sums) next to the scan files.

Usage (ORCD; ~1-3 h and ~20-40 GB per snapshot):
  python decompose_cells_H.py --snapdir /orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots --snap 11 \
      --out /orcd/data/mvogelsb/004/chcho/cell_test_H_11
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

from test_cells_Tb import Snapshot, Phys, group
from tb_physics import X_H, M_H

KEYS = ("pos", "m", "rho", "V", "T", "xHII", "fII", "fIII", "J", "tb")
SF_LABEL = "star-forming (x = 0)"
CLASSES = {                      # name: (property, edges); x_HII also gets the star-forming class first
    "1 + delta":   ("dl", [0, 0.3, 1, 3, 10, 30, 100, 1e3, 1e4, np.inf]),
    "T_K [K]":     ("T",  [0, 5.5, 10, 20, 50, 100, 300, 1e3, 1e4, 1e5, np.inf]),
    "x_c":         ("xc", [0, 1e-3, 1e-2, 0.1, 1, 10, 100, np.inf]),
    "x_HII":       ("xi", [0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, np.inf]),
    "n_H [cm^-3]": ("nH", [0, 1e-4, 1e-3, 1e-2, 3e-2, 0.106, 1, np.inf]),
}
D2_DL = [0, 1, 3, 10, 100, 1e3, np.inf]
D2_T = [0, 5.5, 20, 100, 1e3, 1e4, np.inf]


def class_index(name, prop, sf):
    key, edges = CLASSES[name]
    idx = np.clip(np.digitize(prop[key], edges) - 1, 0, len(edges) - 2)
    if name == "x_HII":                      # class 0 = star-forming, the x_HII bins shifted by one
        idx = np.where(sf, 0, idx + 1)
    return idx


def n_classes(name):
    return len(CLASSES[name][1]) - 1 + (name == "x_HII")


def labels(name):
    e = CLASSES[name][1]
    lab = [f"{e[j]:g} - {e[j + 1]:g}" for j in range(len(e) - 1)]
    return ([SF_LABEL] + lab) if name == "x_HII" else lab


def load_cube(files, q):
    parts = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            g = f.get(f"cube_{q:03d}")
            if g is not None and g["V"].shape[0] > 0:
                parts.append({k: g[k][()].astype(np.float64) for k in KEYS})
    return {k: np.concatenate([p_[k] for p_ in parts]) for k in KEYS} if parts else None


def jack_share(num, den):
    """share sum(num)/sum(den) over cubes, jackknife error (num, den: per cube)"""
    n = len(den)
    R = num.sum() / den.sum()
    if n < 2:
        return R, np.nan
    jk = (num.sum() - num) / (den.sum() - den)
    return R, np.sqrt((n - 1) / n * ((jk - jk.mean())**2).sum())


def conc(V, T, sign):
    """share of the emission (sign +1) or absorption (-1) from the brightest/darkest 0.1, 1, 10% of the volume"""
    s = sign * T > 0
    if not s.any():
        return [np.nan] * 3
    o = np.argsort(-sign * T)
    cumV = np.cumsum(V[o]) / V.sum()
    w = np.where(sign * T[o] > 0, V[o] * T[o], 0.0)
    cumS = np.cumsum(w) / w.sum()
    return [cumS[min(np.searchsorted(cumV, f), len(cumS) - 1)] for f in (1e-3, 1e-2, 1e-1)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapdir", required=True)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--out", required=True, help="directory of the scan task files (scan_H_NNN_TTT.hdf5)")
    p.add_argument("--sample", type=int, default=10, help="1-in-N cell sample for the concentration (default 10)")
    a = p.parse_args()

    S = Snapshot(a.snapdir, a.snap)
    files = sorted(f for f in glob.glob(os.path.join(a.out, f"scan_H_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        sys.exit(f"no scan_H_{a.snap:03d}_*.hdf5 in {a.out}")
    with h5py.File(files[0], "r") as f:
        A = dict(f.attrs)
        N_PIX = f["V"].shape[0]
    if str(A.get("line", "H")) != "H":
        sys.exit(f"{files[0]} is line {A['line']}, not H")
    sm = int(A.get("spin_model", 1))
    ntask, ncube = int(A["ntask"]) if "ntask" in A else None, int(A["n_cube"])
    if ntask is not None and len(files) != ntask:
        sys.exit(f"{len(files)} scan files, the scan had ntask = {ntask}: incomplete")
    rows = []
    for fn in files:
        with h5py.File(fn, "r") as f:
            rows.append(tuple(int(x) for x in f.attrs["rows"]))
            for k in ("n_cube", "cube_side", "seed", "spin_model"):
                if k in A and not np.array_equal(f.attrs.get(k), A[k]):
                    sys.exit(f"{fn}: {k} = {f.attrs.get(k)} differs from {files[0]} ({A[k]})")
    rows.sort()                                    # the scans before 2026-11-19 have no ntask: check the rows instead
    gaps = [(r[1], q[0]) for r, q in zip(rows, rows[1:]) if r[1] != q[0]]
    if rows[0][0] != 0 or rows[-1][1] != S.n or gaps:
        sys.exit(f"scan rows do not cover 0-{S.n:,} once: first {rows[0]}, last {rows[-1]}, gaps/overlaps {gaps[:5]}")
    P = Phys(S, "H", spin_model=sm)
    L, z = P.L, S.z
    nH_fac = X_H * P.rho_unit * (1.0 + z)**3 / M_H
    print(f"snapshot {a.snap}, z = {z:.5f}, line H, spin model {sm}: {len(files)} task files, {ncube} cubes of "
          f"{int(A['cube_side'])}^3 of the {N_PIX // 2}^3 pixels; T_gamma = {L.Tg:.2f} K; <n_H> = {P.rho_mean * nH_fac:.4e} "
          f"cm^-3; render {A.get('render_snap')}" + (f" / {A.get('render_snap_b')} (J weight of b {A.get('J_weight_b')})"
                                                     if A.get("J_interp") else ""), flush=True)

    names = list(CLASSES)
    per = {k: [] for k in ("V", "mHI", "Vtb", "Vem", "Vtbem", "mHIem", "n")}
    cls = {nm: {k: [] for k in ("V", "mHI", "Vtb")} for nm in names}
    d2 = {k: [] for k in ("V", "Vtb")}
    pix = {nm: [] for nm in names}
    tot_map = []
    sV, sT = [], []
    xa_V = []                                      # per cube: sum V x_a (S~ = 1)
    sf_stats = []                                  # per cube: sum V, sum m, sum m T, sum m nH of star-forming cells
    used = []
    for q in range(ncube):
        c = load_cube(files, q)
        if c is None:
            continue
        used.append(q)
        V, tb, m = c["V"], c["tb"], c["m"]
        if not np.all(np.isfinite(tb)):
            sys.exit(f"cube {q}: non-finite dT_b in the scan files")
        mHI = m * (1.0 - c["xHII"])
        Vtb = V * tb
        nH = c["rho"] * nH_fac
        sf = (c["xHII"] == 0) & (c["fII"] == 0) & (c["fIII"] == 0)
        prop = {"dl": c["rho"] / P.rho_mean, "T": c["T"], "nH": nH, "xi": c["xHII"],
                "xc": L.xc_H(nH, c["T"], c["xHII"], c["fII"], c["fIII"])}
        em = tb > 0
        per["V"].append(V.sum()); per["mHI"].append(mHI.sum()); per["Vtb"].append(Vtb.sum())
        per["Vem"].append(V[em].sum()); per["Vtbem"].append(Vtb[em].sum()); per["mHIem"].append(mHI[em].sum())
        per["n"].append(len(V))
        inv = group(c["pos"], S.box, N_PIX)[1]
        Vp = np.bincount(inv, V)
        tot_map.append(np.bincount(inv, Vtb) / Vp)
        for nm in names:
            idx = class_index(nm, prop, sf)
            nc = n_classes(nm)
            cls[nm]["V"].append(np.bincount(idx, V, minlength=nc))
            cls[nm]["mHI"].append(np.bincount(idx, mHI, minlength=nc))
            cls[nm]["Vtb"].append(np.bincount(idx, Vtb, minlength=nc))
            k = inv * nc + idx                     # class maps at 640^3: sum V tb per (pixel, class) / pixel V
            pix[nm].append((np.bincount(k, Vtb, minlength=len(Vp) * nc).reshape(len(Vp), nc) / Vp[:, None]).T)
        i2 = (np.clip(np.digitize(prop["dl"], D2_DL) - 1, 0, len(D2_DL) - 2) * (len(D2_T) - 1)
              + np.clip(np.digitize(prop["T"], D2_T) - 1, 0, len(D2_T) - 2))
        n2 = (len(D2_DL) - 1) * (len(D2_T) - 1)
        d2["V"].append(np.bincount(i2, V, minlength=n2)); d2["Vtb"].append(np.bincount(i2, Vtb, minlength=n2))
        xa_V.append((V * L.xa_c * c["J"]).sum())
        sf_stats.append([V[sf].sum(), m[sf].sum(), (m * c["T"])[sf].sum(), (m * nH)[sf].sum(), sf.sum()])
        sV.append(V[::a.sample]); sT.append(tb[::a.sample])
        print(f"  cube {q:2d}: {len(V):,} cells, <dT_b> {Vtb.sum() / V.sum():+.4e} mK", flush=True)
    if not used:
        sys.exit("no cube cells in the scan files")

    per = {k: np.array(v, dtype=np.float64) for k, v in per.items()}
    TV, TM, TS = per["V"].sum(), per["mHI"].sum(), per["Vtb"].sum()
    M = np.concatenate(tot_map)
    var = np.var(M)
    print(f"\n  {len(used)} cubes, {int(per['n'].sum()):,} cells. CHECK, mean of the cubes' volume-weighted <dT_b> = "
          f"{np.mean(per['Vtb'] / per['V']):+.6e} mK (must equal the merge log's CUBES 'cell' line); all cube cells "
          f"together sum(V dT_b)/sum(V) = {TS / TV:+.6e} mK (the reference of the shares below)")
    print(f"  <x_a> (S~_a = 1, volume-weighted) = {np.sum(xa_V) / TV:.3e};  {N_PIX}^3 pixel map of the cube cells: "
          f"{len(M):,} pixels, mean {M.mean():+.5e}, std {np.sqrt(var):.4e} mK")

    print("\n  emission vs absorption (share of sum(V dT_b); jackknife over cubes):")
    for lab, Vs, Ms, Ss in (("dT_b > 0", per["Vem"], per["mHIem"], per["Vtbem"]),
                            ("dT_b <= 0", per["V"] - per["Vem"], per["mHI"] - per["mHIem"], per["Vtb"] - per["Vtbem"])):
        sh, e = jack_share(Ss, per["Vtb"])
        print(f"    {lab:>10}: volume {Vs.sum() / TV:.4f}, H I mass {Ms.sum() / TM:.4f}, share {sh:+9.4f} +- {e:.4f}, "
              f"mean dT_b in it {Ss.sum() / max(Vs.sum(), 1e-300):+.4e} mK, contributes {Ss.sum() / TV:+.4e} mK")

    out = {"cubes": np.array(used), "per_V": per["V"], "per_Vtb": per["Vtb"], "per_mHI": per["mHI"]}
    for nm in names:
        C = {k: np.array(v) for k, v in cls[nm].items()}           # (cubes, classes)
        maps = np.concatenate(pix[nm], axis=1)
        vshare = [np.mean((maps[j] - maps[j].mean()) * (M - M.mean())) / var for j in range(maps.shape[0])]
        print(f"\n  by {nm}:")
        print(f"    {'class':>22} {'volume':>9} {'H I mass':>9} {'share of sum(V dT_b)':>22} {'mean dT_b [mK]':>15} "
              f"{'variance share':>15}")
        for j, lab in enumerate(labels(nm)):
            sh, e = jack_share(C["Vtb"][:, j], per["Vtb"])
            Vj = C["V"][:, j].sum()
            mean_c = C["Vtb"][:, j].sum() / Vj if Vj > 0 else np.nan
            print(f"    {lab:>22} {Vj / TV:9.2e} {C['mHI'][:, j].sum() / TM:9.2e} {sh:+13.4f} +- {e:6.4f} "
                  f"{mean_c:+15.4e} {vshare[j]:+15.4f}")
        key = CLASSES[nm][0]                       # dl, T, xc, xi, nH
        for k, v in C.items():
            out[f"{key}_{k}"] = v
        out[f"{key}_edges"] = np.array(CLASSES[nm][1], dtype=np.float64)

    D = {k: np.array(v).sum(axis=0).reshape(len(D2_DL) - 1, len(D2_T) - 1) for k, v in d2.items()}
    out["d2_dl_edges"], out["d2_T_edges"] = np.array(D2_DL, float), np.array(D2_T, float)
    out["d2_V"], out["d2_Vtb"] = np.array(d2["V"]), np.array(d2["Vtb"])
    tl = [f"{D2_T[j]:g}-{D2_T[j + 1]:g}" for j in range(len(D2_T) - 1)]
    for title, X in (("volume fraction", D["V"] / TV), ("share of sum(V dT_b)", D["Vtb"] / TS)):
        print(f"\n  1 + delta (rows) x T_K [K] (columns): {title}")
        print(f"    {'':>12}" + "".join(f"{x:>11}" for x in tl))
        for i in range(len(D2_DL) - 1):
            print(f"    {f'{D2_DL[i]:g}-{D2_DL[i + 1]:g}':>12}" + "".join(f"{x:+11.2e}" for x in X[i]))

    sfs = np.array(sf_stats).sum(axis=0)
    if sfs[4] > 0:
        print(f"\n  star-forming cells (x = 0 exactly): {int(sfs[4]):,} cells, volume {sfs[0] / TV:.2e}, mass-weighted "
              f"T {sfs[2] / sfs[1]:.4g} K, n_H {sfs[3] / sfs[1]:.4g} cm^-3 (the renders and the maps treat them as fully neutral)")
    else:
        print("\n  no star-forming cells (x = 0 exactly) in the cubes")

    V_, T_ = np.concatenate(sV), np.concatenate(sT)
    ce, ca = conc(V_, T_, +1), conc(V_, T_, -1)
    print(f"\n  concentration (1-in-{a.sample} cell sample, {len(V_):,} cells): share of the emission from the brightest "
          + ", ".join(f"{f:g}%: {x:.3f}" for f, x in zip((0.1, 1, 10), ce)) + " of the volume;")
    print("    share of the absorption from the darkest "
          + ", ".join(f"{f:g}%: {x:.3f}" for f, x in zip((0.1, 1, 10), ca)) + " of the volume")

    fn = os.path.join(a.out, f"decomp_H_{a.snap:03d}.npz")
    np.savez(fn, z=z, snap=a.snap, spin_model=sm, **out)
    print(f"\n  wrote {fn}")


if __name__ == "__main__":
    main()
