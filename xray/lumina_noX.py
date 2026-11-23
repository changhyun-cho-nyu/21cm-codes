"""
lumina_noX.py
=============
Global 21 cm signal of Lumina WITHOUT X-ray heating, option 3 (user, 2026-11-22; xray/NOTES.md steps 5-7): every
non-star-forming Lumina gas cell keeps its density, x_HII, He fractions and J_alpha, and gets a temperature drawn from
the 50 cMpc no-X-ray run's T distribution of its key (0.05-dex bin of 1+delta x x_HII class, the X run's state), i.e.
the training tables H_train_all of closure_noX_xray.py (pair_cells/closure_NNN.npz, same snapshot number and z).
Random placement inside the key: the closure / placement tests showed the global mean does not depend on it (spread
<= 0.06 mK) and that the method recovers the 50 cMpc no-X-ray global signal within +0.2 to +0.8 mK (1-7% of the X-ray
effect). Star-forming cells (render convention: all fractions 0) keep their state. dT_b with test_cells_Tb.Phys
(tb_physics, spin model of the scan, tau_GP from the cell), as the Lumina gas-cell scans.

Only the global mean is computed (user: global evolution only). Sample: the 64 random cubes (16^3 of the 320^3
pixels, seed 12345) of the H gas-cell scans, 1.7e9 cells per snapshot (0.8% of the volume):
  cubes --snap N : from the stored cube cells of cell_test_H_<N>/scan_H_NNN_*.hdf5 (snapshots with an H scan:
                   024 036 048 057 066); the stored dT_b (X-rays) is recomputed on every --check-every-th cell (check)
  scan  --snap N : the same cubes read from the full snapshot (snapshots without an H scan: 074 ... 106), dT_b with
                   and without X-rays only for the cube cells (no 640^3 grids); J as test_cells_Tb.py scan
  merge --snap N : sums the task files; cube means with and without X-rays, the X-ray effect with a jackknife over
                   the cubes, and, where the H scan files exist, the exact whole-box cell-level mean with X-rays
                   (sum of the scans' sum_V_tb / sum_V) and the whole-box no-X estimate = box X - cube effect
  summary        : all merged snapshots -> data/lumina_noX_global.{csv,png}, with the 50 cMpc cell means (cells_Tb)
Task files: <out>/lumina_noX/noX_<kind>_NNN_TTT.npz; merged: <out>/lumina_noX/noX_NNN.npz.
Outputs only under /orcd/data/mvogelsb/004/chcho; inputs read only.
"""
import argparse
import glob
import os
import re
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from test_cells_Tb import (Snapshot, Phys, prep_cells, render_index, choose_order, cube_origins, J_at,  # noqa: E402
                           render_redshifts)
from closure_noX_xray import keys_of, fallback, inv_cdf, NK, NX, LD, LT, XCL, XLAB  # noqa: E402
from pair_cells_xray import OUT, RUNS, X_H, check_out  # noqa: E402
from cells_Tb_xray import CBINS  # noqa: E402

LUMINA_DATA = "/orcd/data/mvogelsb/004/chcho"
SNAP_ABOVE = "/orcd/data/mvogelsb/005/Lumina/Lumina_above_z_4p75/snapshots"
GAS_OVER_OMB = 1.0042          # Lumina's mean gas density / (Omega_b rho_crit) before star formation (JALPHA_NOTES 6.2):
#                                the keys use 1+delta relative to the gas mean, as closure_noX_xray.py (X-run gas mean)
XNEU = 1e-3


class Sums:
    """per cube: n, V, V tb_X, V tb_noX; per key (neutral non-SF): H I weight, w/T_X, w/T_noX; per key: V tb_X, V tb_noX"""
    def __init__(self, ncube):
        self.cube = np.zeros((ncube, 4))
        self.key = np.zeros((NK, 5))
        self.sf = np.zeros(3)                    # V, V tb_X, V tb_noX of star-forming cells
        self.check = np.zeros(4)                 # n checked, sum |d|, sum |tb|, max |d|

    def add(self, q, c, tbX, tbN, TN, key, sf):
        V = c["V"]
        self.cube[q] += [len(V), V.sum(), np.sum(V * tbX), np.sum(V * tbN)]
        ns = ~sf
        self.key[:, 3] += np.bincount(key[ns], V[ns] * tbX[ns], NK)
        self.key[:, 4] += np.bincount(key[ns], V[ns] * tbN[ns], NK)
        neu = ns & (c["xHII"] < XNEU)
        w = X_H * c["m"][neu] * (1.0 - c["xHII"][neu])
        self.key[:, 0] += np.bincount(key[neu], w, NK)
        self.key[:, 1] += np.bincount(key[neu], w / c["T"][neu], NK)
        self.key[:, 2] += np.bincount(key[neu], w / TN[neu], NK)
        self.sf += [V[sf].sum(), np.sum(V[sf] * tbX[sf]), np.sum(V[sf] * tbN[sf])]


def load_table(out, snap, z):
    fn = os.path.join(out, "pair_cells", f"closure_{snap:03d}.npz")
    if not os.path.exists(fn):
        sys.exit(f"STOP: no training table {fn} (closure_noX_xray.py --snap {snap})")
    D = np.load(fn)
    if abs(float(D["z"]) - z) > 1e-3:
        sys.exit(f"STOP: {fn} is at z = {float(D['z'])}, Lumina snapshot {snap} at z = {z}")
    if not (np.allclose(D["ld"], LD) and np.allclose(D["lt"], LT) and np.allclose(D["xcl"], XCL)):
        sys.exit(f"STOP: {fn} has other bins than closure_noX_xray.py")
    H = D["H_train_all"].astype(np.float64)
    fb = fallback(D["n_train_all"])
    print(f"training table {fn}: z = {float(D['z']):.5f}, {int(np.sum(D['n_train_all'] >= D['min_train']))} keys with "
          f">= {int(D['min_train'])} training cells", flush=True)
    return H, fb, fn


def assign(c, P, H, fb, rng):
    """option 3: T_noX of the non-SF cells drawn from the key's training distribution; SF cells keep T"""
    sf = (c["xHII"] == 0) & (c["fII"] == 0) & (c["fIII"] == 0)      # render convention of the scan cells
    key = keys_of(c["rho"] / (P.rho_mean * GAS_OVER_OMB), np.clip(c["xHII"], 0.0, 1.0))
    TN = c["T"].copy()
    ns = np.flatnonzero(~sf)
    if len(ns):
        TN[ns] = 10**inv_cdf(H, fb[key[ns]], rng.random(len(ns)))
    return TN, key, sf


def tb_chunked(P, c, T, chunk, sel=None):
    idx = np.arange(len(T)) if sel is None else sel
    out = np.empty(len(idx))
    for i0 in range(0, len(idx), chunk):
        s = idx[i0:i0 + chunk]
        out[i0:i0 + len(s)] = P.tb(c["rho"][s], T[s], c["xHII"][s], c["fII"][s], c["fIII"][s], c["J"][s])
    if not np.all(np.isfinite(out)):
        sys.exit("STOP: non-finite dT_b")
    return out


def task_file(a, kind):
    od = os.path.join(a.out, "lumina_noX")
    check_out(od, a.allow_any_out)
    os.makedirs(od, exist_ok=True)
    return os.path.join(od, f"noX_{kind}_{a.snap:03d}_{a.task:03d}.npz")


def save(fn, S, **meta):
    np.savez(fn + ".part.npz", cube=S.cube, key=S.key, sf=S.sf, check=S.check, **meta)
    os.replace(fn + ".part.npz", fn)
    print(f"wrote {fn}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def cubes(a):
    d = a.dir or os.path.join(LUMINA_DATA, f"cell_test_H_{a.snap}")
    files = sorted(f for f in glob.glob(os.path.join(d, f"scan_H_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        sys.exit(f"no scan_H_{a.snap:03d}_*.hdf5 in {d}")
    with h5py.File(files[0], "r") as f:
        A = dict(f.attrs)
    if "ntask" in A and len(files) != int(A["ntask"]):
        sys.exit(f"{len(files)} scan files, the scan had ntask = {int(A['ntask'])}: incomplete")
    S = Snapshot(a.snapdir, a.snap)
    sm = int(A.get("spin_model", 1))
    P = Phys(S, "H", sm)
    if abs(S.z - float(A["z"])) > 1e-6 or abs(P.rho_mean / float(A["rho_mean"]) - 1) > 1e-9:
        sys.exit(f"STOP: snapshot z {S.z} / rho_mean {P.rho_mean} vs scan files {A['z']} / {A['rho_mean']}")
    H, fb, tfile = load_table(a.out, a.snap, S.z)
    fn = task_file(a, "cubes")
    ncube = int(A["n_cube"])
    mine = [q for q in range(ncube) if q % a.ntask == a.task]
    print(f"Lumina snapshot {a.snap}: z = {S.z:.5f}, spin model {sm}, {len(files)} scan files; task {a.task}/{a.ntask}: "
          f"cubes {mine}", flush=True)
    rng = np.random.default_rng([a.seed, a.snap, a.task])
    Su = Sums(ncube)
    t0 = time.time()
    for q in mine:
        parts = []
        for f_ in files:
            with h5py.File(f_, "r") as f:
                g = f.get(f"cube_{q:03d}")
                if g is not None and g["m"].shape[0] > 0:
                    parts.append({k: g[k][()].astype(np.float64) for k in ("m", "rho", "V", "T", "xHII", "fII", "fIII",
                                                                           "J", "tb")})
        if not parts:
            sys.exit(f"STOP: cube {q} has no cells")
        c = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
        del parts
        TN, key, sf = assign(c, P, H, fb, rng)
        tbN = tb_chunked(P, c, TN, a.chunk)
        chk = np.arange(0, len(TN), a.check_every)
        tbr = tb_chunked(P, c, c["T"], a.chunk, chk)
        dd = np.abs(tbr - c["tb"][chk])
        Su.check += [len(chk), dd.sum(), np.abs(c["tb"][chk]).sum(), dd.max()]
        Su.add(q, c, c["tb"], tbN, TN, key, sf)
        print(f"  cube {q:2d}: {len(TN):,} cells, <dT_b> X {Su.cube[q, 2] / Su.cube[q, 1]:+.4f}, no X "
              f"{Su.cube[q, 3] / Su.cube[q, 1]:+.4f} mK; check (every {a.check_every}th cell) max |tb_recomputed - "
              f"tb_stored| {dd.max():.2e} ({time.time() - t0:.0f} s)", flush=True)
        del c
    save(fn, Su, kind="cubes", snap=a.snap, z=S.z, task=a.task, ntask=a.ntask, cubes_done=np.array(mine),
         n_cube=ncube, cube_side=int(A["cube_side"]), seed_cubes=int(A["seed"]), spin_model=sm, table=tfile, src=d)


def scan(a):
    """cube cells of a full snapshot (no H scan there): rows task::ntask, only cells inside the 64 cubes"""
    S = Snapshot(a.snapdir, a.snap, a.gas_file)
    P = Phys(S, "H", a.spin_model)
    H, fb, tfile = load_table(a.out, a.snap, S.z)
    if a.render_snap is None:                     # the render at the snapshot's z (z <= 12: exact matches)
        rz = render_redshifts([a.render])
        if not rz:
            sys.exit(f"no renders in {a.render}/All")
        a.render_snap = min(rz, key=lambda r: abs(rz[r] - S.z))
        print(f"render at the snapshot's z: {a.render_snap} (z = {rz[a.render_snap]:.5f}, dz = {rz[a.render_snap] - S.z:+.2e})")
    if not a.J:
        if not a.jdir:
            sys.exit("give --J files or --jdir")
        a.J = [os.path.join(a.jdir, f"Jalpha_H_{a.render_snap}.hdf5"), os.path.join(a.jdir, f"Jalpha_AGN_H_{a.render_snap}.hdf5")]
        for f_ in a.J:
            if not os.path.exists(f_):
                sys.exit(f"STOP: no {f_}")
    fn = task_file(a, "scan")
    if os.path.exists(fn):
        print(f"{fn} exists: task done")
        return
    edges = np.linspace(0, S.n, a.ntask + 1).astype(np.int64)
    r0, r1 = int(edges[a.task]), int(edges[a.task + 1])
    p = os.path.join(a.render, "Density", f"Density_{a.render_snap:03d}.hdf5")
    Rd = h5py.File(p if os.path.exists(p) else os.path.join(a.render, "All", f"All_{a.render_snap:03d}.hdf5"), "r")["Density"][()]
    N = Rd.shape[0]
    J, Jinfo = J_at(a, N, S.z)
    if J.shape != (N,) * 3:
        sys.exit(f"ERROR: J is {J.shape}, render {Rd.shape}")
    order = choose_order(S, (r0, min(r1, r0 + 5_000_000)), Rd, S.box)
    del Rd
    Nc = N // 2
    nl, cubeid = cube_origins(a.n_cube, a.cube_side, a.seed_cubes, Nc)
    lut = np.full(nl**3, -1, np.int64)
    for k_, v in cubeid.items():
        lut[k_] = v
    print(f"Lumina snapshot {a.snap}: z = {S.z:.5f}, task {a.task}/{a.ntask}: rows {r0:,}-{r1:,} of {S.n:,}; axis order "
          f"{order}; {a.n_cube} cubes of {a.cube_side}^3 of the {Nc}^3 pixels (seed {a.seed_cubes})", flush=True)
    rng = np.random.default_rng([a.seed, a.snap, a.task])
    Su = Sums(a.n_cube)
    t0, nsel, nread = time.time(), 0, 0
    for i0 in range(r0, r1, a.chunk):
        i1 = min(r1, i0 + a.chunk)
        pos = S.read("Coordinates", i0, i1)
        cl = np.floor(pos * (Nc / S.box)).astype(np.int64) % Nc // a.cube_side
        cid = lut[(cl[:, 0] * nl + cl[:, 1]) * nl + cl[:, 2]]
        nread += i1 - i0
        sel = np.flatnonzero(cid >= 0)
        if len(sel):
            c_all = prep_cells(S, i0, i1)
            c = {k: v[sel] for k, v in c_all.items()}
            del c_all
            i, j, k = render_index(c["pos"], S.box, N, order)
            c["J"] = J[i, j, k].astype(np.float64)
            TN, key, sf = assign(c, P, H, fb, rng)
            tbX = tb_chunked(P, c, c["T"], 4_000_000)
            tbN = tb_chunked(P, c, TN, 4_000_000)
            for q in np.unique(cid[sel]):
                m_ = cid[sel] == q
                Su.add(q, {k_: v[m_] for k_, v in c.items()}, tbX[m_], tbN[m_], TN[m_], key[m_], sf[m_])
            nsel += len(sel)
        if (i0 - r0) // a.chunk % 20 == 0:
            el = time.time() - t0
            print(f"  rows {i1 - r0:,}/{r1 - r0:,}: {nsel:,} cube cells, {el / 60:.1f} min, ~{el * (r1 - r0) / (i1 - r0) / 3600:.1f} h "
                  f"total", flush=True)
    save(fn, Su, kind="scan", snap=a.snap, z=S.z, task=a.task, ntask=a.ntask, rows=np.array([r0, r1]), n_cube=a.n_cube,
         cube_side=a.cube_side, seed_cubes=a.seed_cubes, spin_model=a.spin_model, table=tfile, order=np.array(order),
         render_snap=a.render_snap, J_files=";".join(os.path.basename(x) for x in a.J),
         J_b_files=";".join(os.path.basename(x) for x in a.J_b), **{k: v for k, v in Jinfo.items()})
    print(f"done: {nsel:,} cube cells of {nread:,} rows, {(time.time() - t0) / 3600:.2f} h")


# ---------------------------------------------------------------------------------------------------------------------
def merge(a):
    od = os.path.join(a.out, "lumina_noX")
    fs = sorted(glob.glob(os.path.join(od, f"noX_*_{a.snap:03d}_*.npz")))
    fs = [f for f in fs if not f.endswith(".part.npz")]
    if not fs:
        sys.exit(f"no task files for snapshot {a.snap} in {od}")
    kinds = {re.search(r"noX_(cubes|scan)_", f).group(1) for f in fs}
    if len(kinds) != 1:
        sys.exit(f"STOP: both kinds of task files for snapshot {a.snap}: {kinds}")
    D = [np.load(f) for f in fs]
    nt = {int(d["ntask"]) for d in D}
    if len(nt) != 1 or len(D) != nt.pop():
        sys.exit(f"STOP: {len(D)} task files, ntask {sorted({int(d['ntask']) for d in D})}: incomplete or mixed")
    if len({str(d["table"]) for d in D}) != 1 or len({int(d["spin_model"]) for d in D}) != 1:
        sys.exit("STOP: the task files used different training tables or spin models")
    z = float(D[0]["z"])
    cube = sum(d["cube"] for d in D)
    key = sum(d["key"] for d in D)
    sf = sum(d["sf"] for d in D)
    chk = sum(d["check"] for d in D)
    if np.any(cube[:, 0] == 0):
        sys.exit(f"STOP: cubes without cells: {np.flatnonzero(cube[:, 0] == 0)}")
    V, VX, VN = cube[:, 1], cube[:, 2], cube[:, 3]
    mX, mN = VX.sum() / V.sum(), VN.sum() / V.sum()
    n = len(V)
    jk = lambda num: np.array([(num.sum() - num[i]) / (V.sum() - V[i]) for i in range(n)])
    jX, jN = jk(VX), jk(VN)
    jE = jX - jN
    err = lambda j: np.sqrt((n - 1) / n * np.sum((j - j.mean())**2))
    print(f"snapshot {a.snap}, z = {z:.4f}: {len(D)} task files ({kinds.pop()}), {int(cube[:, 0].sum()):,} cells in {n} cubes")
    if chk[0] > 0:
        print(f"  check: dT_b with X-rays recomputed on {int(chk[0]):,} cells vs stored: mean |diff| / mean |tb| "
              f"{chk[1] / chk[2]:.2e}, max |diff| {chk[3]:.2e} mK")
    print(f"  cubes: <dT_b> X-rays {mX:+.4f} +- {err(jX):.4f} mK, no X-rays {mN:+.4f} +- {err(jN):.4f} mK, X-ray effect "
          f"{mX - mN:+.4f} +- {err(jE):.4f} mK (jackknife over the cubes)")
    res = dict(z=z, snap=a.snap, cube=cube, key=key, sf=sf, check=chk, mean_X_cubes=mX, mean_noX_cubes=mN,
               err_X_cubes=err(jX), err_noX_cubes=err(jN), effect_cubes=mX - mN, err_effect=err(jE))
    # exact whole-box cell level with X-rays from the H scan files
    hd = a.hdir or os.path.join(LUMINA_DATA, f"cell_test_H_{a.snap}")
    hs = sorted(f for f in glob.glob(os.path.join(hd, f"scan_H_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if hs:
        sv = svt = 0.0
        for f_ in hs:
            with h5py.File(f_, "r") as f:
                sv += float(f.attrs["sum_V"]); svt += float(f.attrs["sum_V_tb"])
        box = svt / sv
        print(f"  whole box with X-rays (exact, {len(hs)} H scan files): {box:+.4f} mK (cubes {mX:+.4f}); whole-box "
              f"no-X estimate = box X - cube effect = {box - (mX - mN):+.4f} +- {err(jE):.4f} mK")
        res.update(mean_X_box=box, mean_noX_box=box - (mX - mN))
    else:
        print("  (no H scan files: no exact whole-box value; the cube means are the estimate)")
    # by class of 1+delta x x_HII (key sums), neutral-IGM temperatures
    c = 0.5 * (LD[1:] + LD[:-1])
    kd = np.arange(NK) // NX
    kx = np.arange(NK) % NX
    print("  share of the cube sum V dT_b [mK x cube volume] by the cell's class: X-rays | no X-rays | difference")
    for lo, hi in zip(CBINS[:-1], CBINS[1:]):
        for ix, xl in enumerate(XLAB):
            s = (10**c[kd] >= lo) & (10**c[kd] < hi) & (kx == ix)
            x_, n_ = key[s, 3].sum() / V.sum(), key[s, 4].sum() / V.sum()
            if max(abs(x_), abs(n_)) > 1e-3 * max(abs(mN), 1e-3):
                print(f"     1+d {lo:5g}-{hi:<5g} {xl:9s}: {x_:+9.4f} | {n_:+9.4f} | {n_ - x_:+9.4f}")
    print(f"     star-forming                : {sf[1] / V.sum():+9.4f} | {sf[2] / V.sum():+9.4f}")
    Tg = 2.7255 * (1.0 + z)
    for lab, lo, hi in (("all", 0, np.inf), ("voids 1+d<1", 0, 1), ("1+d 1-2", 1, 2)):
        s = (10**c[kd] >= lo) & (10**c[kd] < hi)
        w, iX, iN = key[s, 0].sum(), key[s, 1].sum(), key[s, 2].sum()
        print(f"  neutral IGM, {lab:11s}: <1/T>^-1 X {w / iX:8.2f} K, no X {w / iN:8.2f} K; <1-Tg/T> {1 - Tg * iX / w:+.3f} "
              f"-> {1 - Tg * iN / w:+.3f}")
    fn = os.path.join(od, f"noX_{a.snap:03d}.npz")
    np.savez(fn, **res)
    print(f"wrote {fn}")


def summary(a):
    od = os.path.join(a.out, "lumina_noX")
    fs = sorted(glob.glob(os.path.join(od, "noX_[0-9][0-9][0-9].npz")))
    if not fs:
        sys.exit(f"no merged noX_NNN.npz in {od}")
    rows = []
    print(" snap      z | Lumina X: box (cells) | cubes          | Lumina no X: box est.  | cubes          | X-ray effect"
          "     | 50 cMpc cells: X | no X")
    for f_ in fs:
        D = np.load(f_)
        s, z = int(D["snap"]), float(D["z"])
        pr = []
        for run in RUNS[::-1]:
            p_ = os.path.join(a.out, run, "cells_Tb", f"cells_Tb_{s:03d}.npz")
            pr.append(float(np.load(p_)["mean_cells_cell"]) if os.path.exists(p_) else np.nan)
        bx = float(D["mean_X_box"]) if "mean_X_box" in D else np.nan
        bn = float(D["mean_noX_box"]) if "mean_noX_box" in D else np.nan
        r = (s, z, bx, float(D["mean_X_cubes"]), float(D["err_X_cubes"]), bn, float(D["mean_noX_cubes"]),
             float(D["err_noX_cubes"]), float(D["effect_cubes"]), float(D["err_effect"]), pr[0], pr[1])
        ok = z >= a.zmin
        rows.append(r + (int(ok),))
        print(f"  {s:03d} {z:7.3f} | {bx:+10.4f}            | {r[3]:+8.4f}+-{r[4]:.4f} | {bn:+10.4f}             | "
              f"{r[6]:+8.4f}+-{r[7]:.4f} | {r[8]:+8.4f}+-{r[9]:.4f} | {pr[0]:+8.4f} | {pr[1]:+8.4f}"
              + ("" if ok else f"  <- no X not valid (z < --zmin {a.zmin}): not in the csv no-X columns or the figure"))
    os.makedirs(a.data, exist_ok=True)
    fc = os.path.join(a.data, "lumina_noX_global.csv")
    with open(fc, "w") as f:
        f.write("snap,z,X_box,X_cubes,err_X_cubes,noX_box_est,noX_cubes,err_noX_cubes,effect,err_effect,pair_X,pair_noX,"
                "noX_valid\n")
        for r in rows:
            v = list(r[1:12])
            if not r[12]:                                       # no-X columns blank below --zmin
                v[4:9] = [np.nan] * 5
            f.write(",".join([str(r[0])] + [f"{x:.6g}" for x in v] + [str(r[12])]) + "\n")
    print(f"wrote {fc}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    R = np.array([r[1:12] for r in rows], float)
    ok = np.array([r[12] for r in rows], bool)
    zz = R[:, 0]
    X = np.where(np.isfinite(R[:, 1]), R[:, 1], R[:, 2])            # box where exact, else cubes
    Nn = np.where(np.isfinite(R[:, 4]), R[:, 4], R[:, 5])
    eN = np.where(np.isfinite(R[:, 4]), R[:, 8], R[:, 6])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(zz, X, "o-", color="#2a6fb0", label="Lumina, X-rays (gas cells)")
    ax.errorbar(zz[ok], Nn[ok], yerr=eN[ok], fmt="D--", color="#1b9e77", mfc="none", capsize=2,
                label="Lumina, no X-rays (option 3)")
    ax.plot(zz, R[:, 9], "s:", color="#c4572a", ms=4, label="50 cMpc, X-rays (cells)")
    ax.plot(zz, R[:, 10], "s:", color="#777777", ms=4, label="50 cMpc, no X-rays (cells)")
    ax.set(xlabel="z", ylabel=r"$\langle\delta T_b\rangle$ [mK]", title="global 21 cm signal, gas-cell level")
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    fig.tight_layout()
    fp = os.path.join(a.data, "lumina_noX_global.png")
    fig.savefig(fp, dpi=200)
    print(f"wrote {fp}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("cubes", "scan"):
        q = sp.add_parser(name)
        q.add_argument("--snap", type=int, required=True)
        q.add_argument("--snapdir", default=SNAP_ABOVE)
        q.add_argument("--task", type=int, default=0)
        q.add_argument("--ntask", type=int, default=1)
        q.add_argument("--seed", type=int, default=2026, help="seed of the temperature draws")
        q.add_argument("--out", default=os.environ.get("BASE", OUT))
        q.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    q = sp.choices["cubes"]
    q.add_argument("--dir", default=None, help="H scan directory (default <data>/cell_test_H_<snap>)")
    q.add_argument("--chunk", type=int, default=4_000_000)
    q.add_argument("--check-every", type=int, default=50)
    q = sp.choices["scan"]
    q.add_argument("--gas-file", default=None)
    q.add_argument("--spin-model", type=int, choices=[0, 1], default=1)
    q.add_argument("--J", action="append", default=[], help="J_alpha files (stars, AGN); default from --jdir")
    q.add_argument("--jdir", default=None, help="directory of Jalpha_H_RRR.hdf5 and Jalpha_AGN_H_RRR.hdf5")
    q.add_argument("--J-b", action="append", default=[])
    q.add_argument("--render", required=True)
    q.add_argument("--render-snap", type=int, default=None, help="default: the render nearest in z (checked by --max-dz)")
    q.add_argument("--render-snap-b", type=int)
    q.add_argument("--max-dz", type=float, default=1e-4)
    q.add_argument("--chunk", type=int, default=20_000_000)
    q.add_argument("--n-cube", type=int, default=64)
    q.add_argument("--cube-side", type=int, default=16)
    q.add_argument("--seed-cubes", type=int, default=12345, help="cube lattice seed (the H scans' --seed)")
    q = sp.add_parser("merge")
    q.add_argument("--snap", type=int, required=True)
    q.add_argument("--hdir", default=None, help="H scan directory for the exact whole-box X value")
    q.add_argument("--out", default=os.environ.get("BASE", OUT))
    q = sp.add_parser("summary")
    q.add_argument("--out", default=os.environ.get("BASE", OUT))
    q.add_argument("--data", default="data")
    q.add_argument("--zmin", type=float, default=5.7,
                   help="no-X points only at z >= zmin: below it the 50 cMpc training run has no neutral gas left "
                        "(z = 5.50: no populated x_HII < 1e-3 key; user decision 2026-11-22: end at z = 5.73)")
    a = p.parse_args()
    {"cubes": cubes, "scan": scan, "merge": merge, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
