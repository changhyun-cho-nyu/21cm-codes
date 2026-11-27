"""
null_igm_map.py
===============
Null test for the IGM-only 3.46 cm cross-spectra (run_cross_he_igm.sh; user, 2026-11-27). The IGM-only gas-cell He
maps (dTb_cell_igm_ltC = sum V dT_b [1+delta < C] / sum V per 640^3 pixel) anti-correlate with x_HeIII at fixed
density (partial r ~ -0.95 at k <= 0.1). Cutting the dense cells makes a map a non-linear function of density, and
x_HeIII correlates with density, so part of that partial r could come from the cut alone. Here the same cut and deposit
are applied to a quantity WITHOUT x_HeIII, the signal's density weighting:
    null_ltC(p) = sum V (1+delta)^2 [1+delta < C, not star-forming] / sum V      (C = inf ("all"), 100, 30, 10, 3)
(star-forming cells count 0, as in the He maps). If its partial r(null, x_HeIII | density) is ~ 0, the cut does not
make the He result.

scan:  one task = rows edges[task]:edges[task+1] of the snapshot (np.linspace, as test_cells_Tb.py); reads Coordinates,
       Masses, Density, StarFormationRate; pixel = the render index (choose_order on the render Density, as the He
       maps); writes <out>/null_NNN_TTT.hdf5 with the 640^3 sums V and V q [cut] (float64 in memory, stored float32,
       1.05 GB each, 6 grids; delete the task files after the merge).
merge: sums the ntask files (stops unless all are there, same order and cuts, rows tile 0 ... N_cells), checks
       sum V / box^3 and that the pixels with gas are those of the gas-cell density map (cellrho), writes
       <out>/nullmap_NNN.hdf5 (datasets null_all, null_lt100, ... float32, render index order) and deletes nothing.

  python -u null_igm_map.py scan --snapdir D --snap 158 --render <ren_640> --render-snap 705 --max-dz 1e-4 \
      --out <dir> --task T --ntask 8
  python null_igm_map.py merge --snap 158 --out <dir> --ntask 8 --cellrho <cell_test_158/cellrho_158.hdf5>
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

from test_cells_Tb import Snapshot, Phys, choose_order, render_index, render_z

CUTS = (np.inf, 100.0, 30.0, 10.0, 3.0)


def gname(c):
    return "null_all" if np.isinf(c) else f"null_lt{c:g}"


def scan(a):
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    if not 0 <= a.task < a.ntask:
        sys.exit(f"ERROR: --task {a.task} outside 0..{a.ntask - 1}")
    zr = render_z(a.render, a.render_snap)
    if abs(zr - S.z) > a.max_dz:
        sys.exit(f"ERROR: render {a.render_snap} z = {zr} vs snapshot z = {S.z} (|dz| > --max-dz {a.max_dz})")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"null_{a.snap:03d}_{a.task:03d}.hdf5")
    if os.path.exists(out):
        print(f"{out} exists: task done")
        return
    P = Phys(S, "He", 1)
    edges = np.linspace(0, S.n, a.ntask + 1).astype(np.int64)
    r0, r1 = int(edges[a.task]), int(edges[a.task + 1])
    p = os.path.join(a.render, "Density", f"Density_{a.render_snap:03d}.hdf5")
    with h5py.File(p if os.path.exists(p) else os.path.join(a.render, "All", f"All_{a.render_snap:03d}.hdf5"), "r") as f:
        Rd = f["Density"][()]
    N = Rd.shape[0]
    print(f"snapshot {a.snap} z = {S.z:.5f}, task {a.task}/{a.ntask}: rows {r0:,}-{r1:,} of {S.n:,}; render {a.render_snap}")
    order = choose_order(S, (r0, min(r1, r0 + 5_000_000)), Rd, S.box)
    del Rd
    G = {"V": np.zeros(N**3)}
    G.update({gname(c): np.zeros(N**3) for c in CUTS})
    t0 = time.time()
    for i0 in range(r0, r1, a.chunk):
        i1 = min(r1, i0 + a.chunk)
        pos = S.read("Coordinates", i0, i1)
        m, rho = S.read("Masses", i0, i1), S.read("Density", i0, i1)
        sf = S.read("StarFormationRate", i0, i1) > 0
        V = m / rho
        dl = rho / P.rho_mean
        i, j, k = render_index(pos, S.box, N, order)
        lin = (i * N + j) * N + k
        del pos, i, j, k
        G["V"] += np.bincount(lin, V, minlength=N**3)
        q = np.where(sf, 0.0, V * dl**2)
        for c in CUTS:
            G[gname(c)] += np.bincount(lin, q * (dl < c), minlength=N**3)
        print(f"  rows {i1:,} / {r1:,}: {time.time() - t0:.0f} s", flush=True)
    part = out + ".part"
    with h5py.File(part, "w") as f:
        f.attrs.update({"snap": a.snap, "z": S.z, "rows": (r0, r1), "n_total": S.n, "ntask": a.ntask, "order": order,
                        "render_snap": a.render_snap, "box": S.box, "rho_mean": P.rho_mean, "cuts": list(CUTS)})
        for g, v in G.items():
            f.create_dataset(g, data=v.reshape(N, N, N).astype(np.float32))    # 6.3 GB per task (float64: 12.6)
    os.replace(part, out)
    print(f"wrote {out} ({time.time() - t0:.0f} s)")


def merge(a):
    files = sorted(glob.glob(os.path.join(a.out, f"null_{a.snap:03d}_[0-9][0-9][0-9].hdf5")))
    if len(files) != a.ntask:
        sys.exit(f"STOP: {len(files)} task files in {a.out}, expected {a.ntask}")
    G, ref, rows = None, None, []
    for fn in files:
        with h5py.File(fn, "r") as f:
            fa = dict(f.attrs)
            if ref is None:
                ref = fa
            for k in ("snap", "n_total", "ntask", "render_snap", "box"):
                if fa[k] != ref[k]:
                    sys.exit(f"STOP: {fn}: {k} = {fa[k]}, first file {ref[k]}")
            if list(fa["order"]) != list(ref["order"]) or list(fa["cuts"]) != list(ref["cuts"]) or int(fa["ntask"]) != a.ntask:
                sys.exit(f"STOP: {fn}: order / cuts / ntask differ")
            rows.append(tuple(int(x) for x in fa["rows"]))
            if G is None:
                G = {g: f[g][()].astype(np.float64) for g in f}
            else:
                for g in G:
                    G[g] += f[g][()]
    rows.sort()
    n = int(ref["n_total"])
    if rows[0][0] != 0 or rows[-1][1] != n or any(rows[i][1] != rows[i + 1][0] for i in range(len(rows) - 1)):
        sys.exit(f"STOP: task rows do not tile 0 ... {n}: {rows}")
    N = G["V"].shape[0]
    box = float(ref["box"])
    cov = G["V"].sum() / box**3
    print(f"snapshot {a.snap}, z = {float(ref['z']):.4f}: {a.ntask} tasks, {n:,} cells; sum V / box^3 = {cov:.6f}; "
          f"{int((G['V'] == 0).sum())} pixels without gas")
    if abs(cov - 1) > 1e-3:
        sys.exit("STOP: the cells' volume does not fill the box")
    if a.cellrho:
        with h5py.File(a.cellrho, "r") as f:
            rho = f["rho"][()]
        if rho.shape != G["V"].shape or not np.array_equal(rho > 0, G["V"] > 0):
            sys.exit(f"STOP: the pixels with gas differ from {a.cellrho} (index order or deposit)")
        print(f"  pixels with gas = those of {os.path.basename(a.cellrho)}")
    out = os.path.join(a.out, f"nullmap_{a.snap:03d}.hdf5")
    with h5py.File(out + ".part", "w") as f:
        f.attrs.update({k: ref[k] for k in ("snap", "z", "render_snap", "box", "order", "rho_mean")})
        f.attrs["content"] = "sum V (1+delta)^2 [1+delta < cut, not star-forming] / sum V per pixel"
        Vs = np.maximum(G["V"], 1e-300)
        for c in CUTS:
            g = gname(c)
            mp = np.where(G["V"] > 0, G[g] / Vs, 0.0)
            f.create_dataset(g, data=mp.astype(np.float32))
            print(f"  {g}: mean {mp.mean():.5e} (volume-weighted <(1+delta)^2 [cut]> = {G[g].sum() / G['V'].sum():.5e})")
    os.replace(out + ".part", out)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("scan")
    q.add_argument("--snapdir", required=True)
    q.add_argument("--snap", type=int, required=True)
    q.add_argument("--gas-file", default=None)
    q.add_argument("--render", required=True)
    q.add_argument("--render-snap", type=int, required=True)
    q.add_argument("--max-dz", type=float, default=1e-4)
    q.add_argument("--out", required=True)
    q.add_argument("--task", type=int, required=True)
    q.add_argument("--ntask", type=int, required=True)
    q.add_argument("--chunk", type=int, default=50_000_000)
    m = sub.add_parser("merge")
    m.add_argument("--snap", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--ntask", type=int, required=True)
    m.add_argument("--cellrho", default=None, help="gas-cell density map of the snapshot (checks the pixels)")
    a = p.parse_args()
    scan(a) if a.mode == "scan" else merge(a)


if __name__ == "__main__":
    main()
