"""
void_heating_test.py
====================
Are Lumina's warmer neutral voids at z = 15 and 12 X-ray heated? (user, 2026-11-20; xray/NOTES.md step 4.)
At z = 15.02 Lumina's void (1+delta < 1) <1/T>^-1 is 6.95 K against 5.99 K in the 50 cMpc X-ray run, where the X-ray
effect is only +0.05 K (no X-rays 5.94 K). The own-rank no-X-ray map would remove that excess. It is right to remove it
only if it is X-ray heating (e.g. Lumina's earlier, rarer sources). Two tests, both with the two boxes' own data:

 1. Per unit of 8.46 cMpc/h: the void excess against the unit's X-ray photon density, now and integrated over time
    ("exposure", sum over all renders up to the snapshot of the mean photon density per bin x dt). Calibrated on the
    50 cMpc pair, where the X-ray excess of each unit is known exactly (X run - no-X run, same cells): fit
    excess = k_b E_b through the origin per X-ray bin (and the three bins together), then predict Lumina's excess from
    Lumina's exposure. Lumina's excess = its void <1/T>^-1 minus the pair's no-X-ray value at the unit's mean density
    (fit across the pair units). UV bins and density are the controls (rank correlations, also partial on density).
 2. In the void cells, x_HII against T (H I-weighted median per log T bin): X-ray heating comes with ionisation
    (secondary ionisations), so X-ray-heated void cells sit above the no-X-ray x_HII at the same T. The pair X / no-X
    runs give both curves; Lumina's warm void cells are placed against them.
The same unit sums are kept for 1+delta = 1-2 and > 2 (Lumina's 1-2 gas is 12% cooler than the pair's at z = 15 and
11% at z = 12, which X-ray heating cannot explain).

  lumina --snap N : the cells of the 64 random cubes of the H gas-cell scans (cell_test_H_<N>/scan_H_NNN_*.hdf5, read
                    only; 16^3 of the 320^3 pixels, positions from the scan), each split into 2^3 sub-cubes (512 units).
                    The cube lattice is rebuilt from the scan's n_cube, cube_side, seed (test_cells_Tb.cube_origins) and
                    checked cell by cell. From the 640^3 renders (ren_640/IonEnergies/IonEnergies_RRR.hdf5; render axes
                    = coordinate axes in the scan's 'order'): the photon number per bin summed over each unit's 16^3
                    pixels / unit volume (IonEnergies = deposit of PhotonDensity x V), at renders 0..R (R = the scan's
                    render_snap) -> exposure; the render Density over each unit at R (mapping check).
  pair --snap N   : all gas cells of both 50 cMpc runs, file by file, 64 units of 8.46 cMpc/h (4^3); the X run's 128^3
                    raw renders 0..R (R = SNAP_RENDER[N]), axis order from the NGP mass deposit vs the render Density.
  compare         : tables per snapshot and data/void_heating.png.
Selection as compare_TX_lumina.py: neutral IGM = non-star-forming with x_HII < 1e-3, H I weight X m (1 - x_HII),
1+delta = rho / (Omega_b rho_crit); Lumina star-forming = all fractions 0 (render convention of the scan files).
Outputs: <out>/void_heat/void_heat_{lumina,pair}_NNN.npz (written only under /orcd/data/mvogelsb/004/chcho).

  python void_heating_test.py lumina --snap 24
  python void_heating_test.py pair --snap 24
  python void_heating_test.py compare
"""
import argparse
import glob
import os
import re
import sys
import time

import h5py
import numpy as np

try:                                   # compressed Lumina renders (Density) need the filters
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from pair_cells_xray import (GAMMA_M1, K_B, M_P, OUT, RUNS, SNAP_RENDER, SRC, X_H, XNEU, axis_order, check_out,
                             read_render, snap_files)
from compare_TX_lumina import LUMINA_DATA, rho_crit_b, temperature

LUM_REN = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL/Flagship_new/postprocessing/ren_640"
NC = 320                                   # the scans' cube lattice grid (N/2 of the 640^3 renders)
NSUB = 2                                   # sub-cubes per cube side (16 -> 8 pixels of 320^3 = 8.46 cMpc/h)
NPAIR = 4                                  # pair units per side (33.83 / 4 = 8.46 cMpc/h)
CEDGE = (0.0, 1.0, 2.0, np.inf)            # 1+delta classes: voids, 1-2, > 2
CLAB = ("1+d<1", "1-2", ">2")
LTX = np.arange(0.0, 4.0001, 0.02)         # log10 T of the x-T histogram (void cells)
LXX = np.arange(-7.0, -0.9999, 0.05)       # log10 x_HII (x < 1e-7 goes to the first bin)
XMAX_XT = 0.1                              # void cells with x_HII < 0.1 enter the x-T histogram
NB = 6                                     # RT bins: 13.6 / 24.59 / 54.42 / 100 / 200 / 500 / 2000 eV
XB = (3, 4, 5)                             # X-ray bins
BLAB = ("13.6-24.6", "24.6-54.4", "54.4-100", "100-200", "200-500", "500-2000")
SEC_MYR = 3.15576e13


class Units:
    """per-unit H I sums of the neutral IGM by class of 1+delta, gas mass and volume; box x-T histogram of voids"""
    def __init__(self, nu):
        self.nu = nu
        self.sw = np.zeros((nu, 3))
        self.swinv = np.zeros((nu, 3))
        self.m = np.zeros(nu)
        self.V = np.zeros(nu)
        self.xt = np.zeros((len(LTX) - 1, len(LXX) - 1))
        self.clip = np.zeros(2)                   # void x-T cells outside the T range, cells outside the units

    def add(self, u, one, T, m, V, x, sf):
        nu = self.nu
        bad = (u < 0) | (u >= nu)
        self.clip[1] += bad.sum()
        u, one, T, m, V, x, sf = (v[~bad] for v in (u, one, T, m, V, x, sf))
        hi = X_H * m * (1.0 - x)
        self.m += np.bincount(u, m, nu)
        self.V += np.bincount(u, V, nu)
        s = (~sf) & (x < XNEU)
        c = np.searchsorted(CEDGE, one[s], side="right") - 1
        k = u[s] * 3 + c
        self.sw += np.bincount(k, hi[s], nu * 3).reshape(nu, 3)
        self.swinv += np.bincount(k, hi[s] / T[s], nu * 3).reshape(nu, 3)
        v = (~sf) & (one < 1.0) & (x < XMAX_XT)
        it = np.searchsorted(LTX, np.log10(T[v]), side="right") - 1
        ix = np.clip(np.searchsorted(LXX, np.log10(np.maximum(x[v], 1e-30)), side="right") - 1, 0, len(LXX) - 2)
        ok = (it >= 0) & (it < len(LTX) - 1)
        self.clip[0] += (~ok).sum()
        nx = len(LXX) - 1
        self.xt += np.bincount(it[ok] * nx + ix[ok], hi[v][ok], self.xt.size).reshape(self.xt.shape)

    def dump(self, pre=""):
        return {pre + "sw": self.sw, pre + "swinv": self.swinv, pre + "m": self.m, pre + "V": self.V,
                pre + "xt": self.xt, pre + "clip": self.clip}


def cosmic_time_myr(z, h, om, ol):
    """flat matter + Lambda (radiation negligible at z < 30 for a time difference)"""
    H0 = 100.0 * h * 1e5 / 3.085678e24
    a = 1.0 / (1.0 + np.asarray(z, float))
    return 2.0 / (3.0 * H0 * np.sqrt(ol)) * np.arcsinh(np.sqrt(ol / om) * a**1.5) / SEC_MYR


def exposure(nh, tr):
    """time integral of the photon density over the renders (trapezoid), per unit and bin: nh (nu, nr, NB)"""
    if len(tr) < 2:
        return np.zeros((nh.shape[0], nh.shape[2]))
    return np.trapz(nh, tr, axis=1) if not hasattr(np, "trapezoid") else np.trapezoid(nh, tr, axis=1)


def hist_renders(R, every):
    r = list(range(0, R + 1, max(1, every)))
    if r[-1] != R:
        r.append(R)
    return r


# ---------------------------------------------------------------------------------------------------------------------
def lumina(a):
    d = a.dir or os.path.join(LUMINA_DATA, f"cell_test_H_{a.snap}")
    files = sorted(f for f in glob.glob(os.path.join(d, f"scan_H_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        sys.exit(f"no scan_H_{a.snap:03d}_*.hdf5 in {d}")
    with h5py.File(files[0], "r") as f:
        A = dict(f.attrs)
    for k in ("z", "rho_mean", "n_cube", "cube_side", "seed", "order", "render_snap"):
        if k not in A:
            sys.exit(f"STOP: {files[0]} has no attribute {k}")
    if "ntask" in A and len(files) != int(A["ntask"]):
        sys.exit(f"{len(files)} scan files, the scan had ntask = {int(A['ntask'])}: incomplete")
    for fn in files[1:]:
        with h5py.File(fn, "r") as f:
            for k in ("z", "rho_mean", "n_cube", "cube_side", "seed", "order", "render_snap"):
                if not np.array_equal(np.asarray(f.attrs.get(k)), np.asarray(A[k])):
                    sys.exit(f"{fn}: {k} = {f.attrs.get(k)} differs from {files[0]} ({A[k]})")
    z, rho_b = float(A["z"]), float(A["rho_mean"])
    ncube, side, seed = int(A["n_cube"]), int(A["cube_side"]), int(A["seed"])
    order = tuple(int(v) for v in A["order"])
    inv = tuple(int(v) for v in np.argsort(order))
    R = int(A["render_snap"]) if a.render is None else a.render
    if side % NSUB:
        sys.exit(f"cube_side {side} not divisible by {NSUB}")
    sub = side // NSUB
    ren = a.render_dir
    fion = lambda r: os.path.join(ren, "IonEnergies", f"IonEnergies_{r:03d}.hdf5")
    if not os.path.exists(fion(R)):
        sys.exit(f"no {fion(R)}")
    with h5py.File(fion(R), "r") as f:
        hd = dict(f["Header"].attrs)
        Nr = int(hd["NumPixels"])
        shp = f["IonEnergies"].shape
    box, h, om, ol = float(hd["BoxSize"]), float(hd["HubbleParam"]), float(hd["Omega0"]), float(hd["OmegaLambda"])
    if shp != (Nr, Nr, Nr, NB) or Nr % 2 or (Nr // 2) % side:
        sys.exit(f"STOP: render {R}: NumPixels {Nr}, IonEnergies {shp}, cube_side {side}")
    nc = Nr // 2                                                                     # 320 for Lumina
    if nc != NC and not a.allow_any_out:
        sys.exit(f"STOP: render grid {Nr}, the Lumina scans use {2 * NC}")
    nl = nc // side
    pick = np.random.default_rng(seed).choice(nl**3, size=ncube, replace=False)      # = test_cells_Tb.cube_origins
    corner = np.stack(np.unravel_index(pick, (nl, nl, nl)), 1) * side                # coordinate axes, nc^3 pixels
    od = os.path.join(a.out, "void_heat")
    check_out(od, a.allow_any_out)
    nu = ncube * NSUB**3
    U = Units(nu)
    print(f"Lumina snapshot {a.snap}: z = {z:.5f}, {len(files)} scan files, {ncube} cubes of {side}^3 of the {nc}^3 "
          f"pixels -> {nu} units of {sub}^3 ({sub * box / nc / 1e3:.2f} cMpc/h); render {R} (z {float(hd['Redshift']):.4f}); "
          f"axis order {order}; box {box}", flush=True)
    t0 = time.time()
    nmis = 0
    for q in range(ncube):
        parts = []
        for fn in files:
            with h5py.File(fn, "r") as f:
                g = f.get(f"cube_{q:03d}")
                if g is not None and g["m"].shape[0] > 0:
                    parts.append({k: g[k][()].astype(np.float64) for k in ("m", "rho", "V", "T", "xHII", "fII", "fIII",
                                                                           "pos")})
        if not parts:
            sys.exit(f"STOP: cube {q} has no cells")
        c = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
        pix = np.floor(c["pos"] * (nc / box)).astype(np.int64) % nc                 # coordinate axes, nc^3
        rel = pix - corner[q]
        mis = np.any((rel < 0) | (rel >= side), axis=1)
        nmis += int(mis.sum())
        if mis.mean() > 1e-6:
            sys.exit(f"STOP: cube {q}: {mis.sum():,} of {len(mis):,} cells outside the rebuilt cube {corner[q]} "
                     f"(lattice or box size wrong)")
        s3 = np.clip(rel // sub, 0, NSUB - 1)
        u = q * NSUB**3 + (s3[:, 0] * NSUB + s3[:, 1]) * NSUB + s3[:, 2]
        sf = (c["xHII"] == 0) & (c["fII"] == 0) & (c["fIII"] == 0)
        U.add(u, c["rho"] / rho_b, c["T"], c["m"], c["V"], c["xHII"], sf)
        if q % 8 == 0 or q == ncube - 1:
            print(f"  cube {q:2d}: {len(c['m']):,} cells ({time.time() - t0:.0f} s)", flush=True)
        del c, parts
    print(f"  cells outside their rebuilt cube: {nmis}; void x-T cells outside the T range: {int(U.clip[0])}", flush=True)

    vunit = (sub * box / nc)**3

    def unit_sums(ds, extra_axis):
        """sum over each unit's (2 sub)^3 render pixels -> (nu, k) in coordinate-axis sub-cube order"""
        out = np.zeros((nu, ds.shape[3] if extra_axis else 1))
        for q in range(ncube):
            sl = tuple(slice(2 * corner[q][order[i]], 2 * corner[q][order[i]] + 2 * side) for i in range(3))
            b = ds[sl] if extra_axis else ds[sl][..., None]
            b = np.transpose(b.astype(np.float64), inv + (3,))                    # coordinate axes
            s_ = 2 * sub
            b = b.reshape(NSUB, s_, NSUB, s_, NSUB, s_, b.shape[-1]).sum((1, 3, 5))
            out[q * NSUB**3:(q + 1) * NSUB**3] = b.reshape(NSUB**3, -1)
        return out

    # mapping check: render Density mean over each unit vs the cells' mass / volume
    fden = os.path.join(ren, "Density", f"Density_{R:03d}.hdf5")
    dens = np.full(nu, np.nan)
    if os.path.exists(fden):
        with h5py.File(fden, "r") as f:
            dens = unit_sums(f["Density"], False)[:, 0] / (2 * sub)**3
        ok = (dens > 0) & (U.V > 0)
        lr = np.log10(U.m[ok] / U.V[ok]) - np.log10(dens[ok])
        print(f"  mapping check: corr(log cell mass/volume, log render Density) over units "
              f"{np.corrcoef(np.log10(U.m[ok] / U.V[ok]), np.log10(dens[ok]))[0, 1]:.4f}; ratio median "
              f"{10**np.median(lr):.4f}, p5-p95 {10**np.percentile(lr, 5):.4f}-{10**np.percentile(lr, 95):.4f}", flush=True)
    else:
        print(f"  (no {fden}: mapping check skipped)")

    rr = hist_renders(R, a.hist_every)
    nh = np.zeros((nu, len(rr), NB))
    zr = np.zeros(len(rr))
    for i, r in enumerate(rr):
        with h5py.File(fion(r), "r") as f:
            zr[i] = float(f["Header"].attrs["Redshift"])
            nh[:, i] = unit_sums(f["IonEnergies"], True) / vunit
        if i % 10 == 0 or r == R:
            print(f"  render {r:3d} (z {zr[i]:.3f}): mean photon density per bin " +
                  " ".join(f"{v:.3e}" for v in nh[:, i].mean(0)) + f" ({time.time() - t0:.0f} s)", flush=True)
    tr = cosmic_time_myr(zr, h, om, ol)
    E = exposure(nh, tr)
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"void_heat_lumina_{a.snap:03d}.npz")
    np.savez(fn, **U.dump(), nR=nh[:, -1], nh=nh, zr=zr, tr=tr, rr=np.array(rr), E=E, dens_render=dens, z=z,
             snap=a.snap, render=R, rho_mean=rho_b, box=box, unit_side=sub * box / nc, cube=np.repeat(np.arange(ncube),
             NSUB**3), src=d)
    print(f"wrote {fn} ({time.time() - t0:.0f} s)")


def pair(a):
    R = SNAP_RENDER.get(a.snap) if a.render is None else a.render
    if R is None:
        sys.exit(f"no render for snapshot {a.snap} in SNAP_RENDER")
    od = os.path.join(a.out, "void_heat")
    check_out(od, a.allow_any_out)
    zR, N, RR = read_render(a.src, "withXrays", R, ["Density"])
    if N % NPAIR:
        sys.exit(f"render N = {N} not divisible by {NPAIR}")
    nu = NPAIR**3
    res = {}
    t0 = time.time()
    for run in RUNS:
        fs = snap_files(a.src, run, a.snap)
        with h5py.File(fs[0], "r") as f:
            hd = dict(f["Header"].attrs)
            par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
        z, box = float(hd["Redshift"]), float(hd["BoxSize"])
        uv = float(hd.get("UnitVelocity_in_cm_per_s", par.get("UnitVelocity_in_cm_per_s", 1e5)))
        rho_b, h, ob = rho_crit_b(hd, par)
        om = float(hd.get("Omega0", par.get("Omega0", 0.30964144)))
        ol = float(hd.get("OmegaLambda", par.get("OmegaLambda", 1.0 - om)))
        U = Units(nu)
        dep = np.zeros(N**3) if run == "withXrays" else None
        ncell = 0
        for p in fs:
            with h5py.File(p, "r") as f:
                if "PartType0" not in f or f["PartType0"]["Masses"].shape[0] == 0:
                    continue
                g = f["PartType0"]
                n = g["Masses"].shape[0]
                ip = g["IntCoordinates"][()]
                m = g["Masses"][()].astype(np.float64)
                rho = g["Density"][()].astype(np.float64)
                x = g["HII_Fraction"][()].astype(np.float64)
                f2 = g["HeII_Fraction"][()].astype(np.float64) if "HeII_Fraction" in g else np.zeros(n)
                f3 = g["HeIII_Fraction"][()].astype(np.float64) if "HeIII_Fraction" in g else np.zeros(n)
                ne = g["ElectronAbundance"][()].astype(np.float64) if "ElectronAbundance" in g else x + f2 + 2.0 * f3
                T = temperature(g["InternalEnergy"][()].astype(np.float64), ne, uv)
                sf = (g["StarFormationRate"][()] > 0) if "StarFormationRate" in g else np.zeros(n, bool)
            iu = ((ip.astype(np.uint64) * np.uint64(NPAIR)) >> np.uint64(32)).astype(np.int64)
            u = (iu[:, 0] * NPAIR + iu[:, 1]) * NPAIR + iu[:, 2]
            U.add(u, rho / rho_b, T, m, m / rho, x, sf)
            if dep is not None:
                ix = ((ip.astype(np.uint64) * np.uint64(N)) >> np.uint64(32)).astype(np.int64)
                dep += np.bincount(np.ravel_multi_index((ix[:, 0], ix[:, 1], ix[:, 2]), (N, N, N)), m, N**3)
            ncell += n
        print(f"{run} snapshot {a.snap}: z = {z:.5f}, {len(fs)} files, {ncell:,} cells; void <1/T>^-1 per unit "
              f"p16/p50/p84 " + "/".join(f"{v:.2f}" for v in np.nanpercentile(U.sw[:, 0] / U.swinv[:, 0],
                                                                                 [16, 50, 84]))
              + f" K ({time.time() - t0:.0f} s)", flush=True)
        res.update(U.dump(f"{run}_"))
        res.update({f"{run}_z": z, f"{run}_rho_mean": rho_b})
        if dep is not None:
            o, cc = axis_order(dep.reshape(N, N, N), RR["Density"])
            res["order"] = np.array(o)
    o = tuple(int(v) for v in res["order"])
    inv = tuple(int(v) for v in np.argsort(o))
    rr = hist_renders(R, a.hist_every)
    nh = np.zeros((nu, len(rr), NB))
    zr = np.zeros(len(rr))
    vunit = (box / NPAIR)**3
    s_ = N // NPAIR
    for i, r in enumerate(rr):
        zr[i], Nr, Q = read_render(a.src, "withXrays", r, ["IonEnergies"])
        ion = Q["IonEnergies"]
        if Nr != N or ion.shape[-1] != NB:
            sys.exit(f"STOP: render {r}: N {Nr}, bins {ion.shape[-1]}")
        b = np.transpose(ion, inv + (3,)).reshape(NPAIR, s_, NPAIR, s_, NPAIR, s_, NB).sum((1, 3, 5))
        nh[:, i] = b.reshape(nu, NB) / vunit
        if i % 10 == 0 or r == R:
            print(f"  render {r:3d} (z {zr[i]:.3f}): mean photon density per bin " +
                  " ".join(f"{v:.3e}" for v in nh[:, i].mean(0)) + f" ({time.time() - t0:.0f} s)", flush=True)
    tr = cosmic_time_myr(zr, h, om, ol)
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"void_heat_pair_{a.snap:03d}.npz")
    np.savez(fn, **res, nR=nh[:, -1], nh=nh, zr=zr, tr=tr, rr=np.array(rr), E=exposure(nh, tr), z=res["withXrays_z"],
             snap=a.snap, render=R, box=box, unit_side=box / NPAIR, src=a.src)
    print(f"wrote {fn} ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------------------------------------------------------------
def rank(v):
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r


def spear(a_, b_):
    ok = np.isfinite(a_) & np.isfinite(b_)
    if ok.sum() < 5 or np.std(a_[ok]) == 0 or np.std(b_[ok]) == 0:
        return np.nan
    return np.corrcoef(rank(a_[ok]), rank(b_[ok]))[0, 1]


def spear_partial(a_, b_, c_):
    """rank correlation of a and b with the rank of c regressed out of both"""
    ok = np.isfinite(a_) & np.isfinite(b_) & np.isfinite(c_)
    if ok.sum() < 6:
        return np.nan
    ra, rb, rc = rank(a_[ok]), rank(b_[ok]), rank(c_[ok])
    A = np.c_[np.ones(len(rc)), rc]
    ea = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    eb = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    return np.corrcoef(ea, eb)[0, 1] if np.std(ea) > 0 and np.std(eb) > 0 else np.nan


def xt_median(xt):
    """H I-weighted median log10 x_HII per log T row (nan where empty)"""
    c = np.cumsum(xt, 1)
    tot = c[:, -1]
    out = np.full(xt.shape[0], np.nan)
    for i in np.flatnonzero(tot > 0):
        out[i] = np.interp(0.5, np.r_[0.0, c[i]] / tot[i], LXX)
    return out, tot


def compare(a):
    od = os.path.join(a.out, "void_heat")
    snaps = sorted(int(re.search(r"_(\d+)\.npz$", f).group(1)) for f in glob.glob(os.path.join(od, "void_heat_lumina_*.npz")))
    snaps = [s for s in snaps if os.path.exists(os.path.join(od, f"void_heat_pair_{s:03d}.npz"))]
    if not snaps:
        sys.exit(f"no snapshot with both void_heat_lumina_NNN.npz and void_heat_pair_NNN.npz in {od}")
    figd = []
    for s in snaps:
        L = np.load(os.path.join(od, f"void_heat_lumina_{s:03d}.npz"))
        P = np.load(os.path.join(od, f"void_heat_pair_{s:03d}.npz"))
        z = float(L["z"])
        if abs(z - float(P["z"])) > 1e-3:
            sys.exit(f"STOP: snapshot {s}: Lumina z {z}, pair z {float(P['z'])}")
        print(f"\n=== snapshot {s}: z = {z:.4f}; renders: Lumina {int(L['render'])} (z {L['zr'][-1]:.4f}, {len(L['rr'])} in "
              f"the history), pair {int(P['render'])} (z {P['zr'][-1]:.4f}, {len(P['rr'])}); units: Lumina "
              f"{len(L['m'])} of {float(L['unit_side']) / 1e3:.2f} cMpc/h, pair {len(P['withXrays_m'])} of "
              f"{float(P['unit_side']) / 1e3:.2f} cMpc/h; pair axis order {tuple(int(v) for v in P['order'])}")
        rbL, rbP = float(L["rho_mean"]), float(P["withXrays_rho_mean"])
        dL = L["m"] / L["V"] / rbL
        dP = P["withXrays_m"] / P["withXrays_V"] / rbP
        hm = lambda sw, si, k: np.where(si[:, k] > 0, sw[:, k] / np.where(si[:, k] > 0, si[:, k], 1.0), np.nan)
        hL = [hm(L["sw"], L["swinv"], k) for k in range(3)]
        hX = [hm(P["withXrays_sw"], P["withXrays_swinv"], k) for k in range(3)]
        hN = [hm(P["withoutXrays_sw"], P["withoutXrays_swinv"], k) for k in range(3)]
        box_hm = lambda sw, si, k: sw[:, k].sum() / si[:, k].sum()
        print("  whole sets, <1/T>^-1 [K] of the neutral IGM by class:   " + " | ".join(CLAB))
        for lab, sw, si in (("Lumina (all units)", L["sw"], L["swinv"]),
                            ("pair X", P["withXrays_sw"], P["withXrays_swinv"]),
                            ("pair no X", P["withoutXrays_sw"], P["withoutXrays_swinv"])):
            print(f"    {lab:20s} " + " | ".join(f"{box_hm(sw, si, k):8.2f}" for k in range(3)))
        # exposures and photon densities: whole-set means (equal unit volumes)
        print("  photon density now (n) and exposure (E = int n dt, Myr), unit means, per bin "
              + ", ".join(BLAB) + " eV:")
        for lab, D in (("Lumina", L), ("pair X", P)):
            print(f"    {lab:7s} n " + " ".join(f"{v:9.3e}" for v in D["nR"].mean(0)) + "   E "
                  + " ".join(f"{v:9.3e}" for v in D["E"].mean(0)))
        print("    Lumina / pair: n " + " ".join(f"{v:9.3f}" for v in L["nR"].mean(0) / P["nR"].mean(0)) + "   E "
              + " ".join(f"{v:9.3f}" for v in L["E"].mean(0) / P["E"].mean(0)))
        # baseline (no X-rays) of the void <1/T>^-1 at the unit's mean density, from the pair's no-X units
        ok = np.isfinite(hN[0]) & (dP > 0)
        cb = np.polyfit(np.log10(dP[ok]), np.log10(hN[0][ok]), 1)
        base = lambda d_: 10**np.polyval(cb, np.log10(d_))
        resid = np.std(np.log10(hN[0][ok]) - np.polyval(cb, np.log10(dP[ok])))
        exL = hL[0] - base(dL)
        exP = hX[0] - hN[0]
        print(f"  no-X baseline: log void <1/T>^-1 = {cb[1]:.4f} + {cb[0]:.4f} log <1+delta>_unit over the pair units "
              f"(rms {resid:.4f} dex = {100 * (10**resid - 1):.1f}%); Lumina unit densities {np.percentile(dL, 1):.3f}-"
              f"{np.percentile(dL, 99):.3f} (p1-p99), pair {dP.min():.3f}-{dP.max():.3f}")
        print(f"  void excess over no X-rays [K], unit p16/p50/p84 (mean): Lumina "
              + "/".join(f"{v:.3f}" for v in np.nanpercentile(exL, [16, 50, 84])) + f" ({np.nanmean(exL):.3f}); pair "
              + "/".join(f"{v:.3f}" for v in np.nanpercentile(exP, [16, 50, 84])) + f" ({np.nanmean(exP):.3f}, exact)")
        # rank correlations of the excess
        print("  rank correlation of the void excess with:     " + "  ".join(f"{b:>9s}" for b in BLAB) + "  density")
        lgd = np.log10(dL)
        for lab, ex, D, dd in (("Lumina, E", exL, L["E"], lgd), ("Lumina, n now", exL, L["nR"], lgd),
                               ("Lumina, E | density", exL, L["E"], lgd), ("pair, E", exP, P["E"], np.log10(dP)),
                               ("pair, n now", exP, P["nR"], np.log10(dP))):
            if "|" in lab:
                v = [spear_partial(ex, D[:, b], dd) for b in range(NB)] + [np.nan]
            else:
                v = [spear(ex, D[:, b]) for b in range(NB)] + [spear(ex, dd)]
            print(f"    {lab:22s}                    " + "  ".join(f"{x_:9.3f}" for x_ in v))
        # calibration on the pair, prediction for Lumina
        print("  pair calibration excess = k E (through the origin) and the prediction for Lumina:")
        preds = {}
        for lab, cols in [(BLAB[b], [b]) for b in XB] + [("bins 3-5 jointly", list(XB))]:
            EP, EL = P["E"][:, cols], L["E"][:, cols]
            okp = np.isfinite(exP)
            k, *_ = np.linalg.lstsq(EP[okp], exP[okp], rcond=None)
            fitP = EP @ k
            r2 = 1.0 - np.sum((exP[okp] - fitP[okp])**2) / np.sum((exP[okp] - exP[okp].mean())**2)
            pL = EL @ k
            preds[lab] = pL
            out = np.mean(np.any(EL > EP.max(0), axis=1))
            print(f"    {lab:17s}: k = " + ", ".join(f"{v:.3e}" for v in k) + f"; pair R^2 {r2:.3f}; Lumina predicted "
                  f"excess mean {np.nanmean(pL):.3f} K (measured {np.nanmean(exL):.3f}), rank corr with measured "
                  f"{spear(exL, pL):.3f}, partial on density {spear_partial(exL, pL, lgd):.3f}; Lumina units beyond the "
                  f"pair's largest E {100 * out:.0f}%")
        # cubes (64, 16.9 cMpc/h) from the sub-units
        cube = L["cube"]
        nq = cube.max() + 1
        csw = np.bincount(cube, L["sw"][:, 0], nq); csi = np.bincount(cube, L["swinv"][:, 0], nq)
        cm = np.bincount(cube, L["m"], nq); cV = np.bincount(cube, L["V"], nq)
        cE = np.stack([np.bincount(cube, L["E"][:, b], nq) for b in range(NB)], 1) / np.bincount(cube, None, nq)[:, None]
        cex = csw / csi - base(cm / cV / rbL)
        print(f"  {nq} cubes ({NSUB * float(L['unit_side']) / 1e3:.1f} cMpc/h): rank corr of the void excess with E per bin " +
              " ".join(f"{spear(cex, cE[:, b]):.3f}" for b in range(NB)) + f"; with density {spear(cex, cm / cV):.3f}")
        # 1+delta = 1-2: Lumina vs pair X at matched unit density
        ok1 = np.isfinite(hX[1])
        c1 = np.polyfit(np.log10(dP[ok1]), np.log10(hX[1][ok1]), 1)
        r1 = hL[1] / 10**np.polyval(c1, lgd)
        print(f"  1+delta = 1-2: Lumina / pair-X fit at the unit's density, p16/p50/p84 "
              + "/".join(f"{v:.3f}" for v in np.nanpercentile(r1, [16, 50, 84]))
              + "; rank corr with E per bin " + " ".join(f"{spear(r1, L['E'][:, b]):.3f}" for b in range(NB))
              + f"; pair X/no X in that class {box_hm(P['withXrays_sw'], P['withXrays_swinv'], 1) / box_hm(P['withoutXrays_sw'], P['withoutXrays_swinv'], 1):.4f}")
        # x-T of the void cells
        mL, tL = xt_median(L["xt"]); mX, tX = xt_median(P["withXrays_xt"]); mN, tN = xt_median(P["withoutXrays_xt"])
        ct = 0.5 * (LTX[1:] + LTX[:-1])
        print("  void cells (non-SF, x_HII < 0.1): H I-weighted median x_HII at T [K] (H I share of the row):")
        print("        T     Lumina              pair X              pair no X")
        for Tq in (5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 13.0, 16.0, 20.0, 30.0, 50.0, 100.0):
            i = int(np.searchsorted(LTX, np.log10(Tq), side="right") - 1)
            if 0 <= i < len(ct):
                f_ = lambda m_, t_: (f"{10**m_[i]:9.2e} ({t_[i] / t_.sum():.3f})" if np.isfinite(m_[i]) else
                                     "        -          ")
                print(f"    {Tq:6.1f}   {f_(mL, tL)}  {f_(mX, tX)}  {f_(mN, tN)}")
        for lab, xt in (("Lumina", L["xt"]), ("pair X", P["withXrays_xt"]), ("pair no X", P["withoutXrays_xt"])):
            cx = 0.5 * (LXX[1:] + LXX[:-1])
            print(f"    {lab:9s}: H I-weighted mean x_HII of the void cells {np.sum(xt * 10**cx[None, :]) / xt.sum():.3e}")
        print(f"  clipped: Lumina x-T {int(L['clip'][0])}, outside units {int(L['clip'][1])}; pair X "
              f"{int(P['withXrays_clip'][0])}/{int(P['withXrays_clip'][1])}, no X {int(P['withoutXrays_clip'][0])}/"
              f"{int(P['withoutXrays_clip'][1])}")
        figd.append(dict(s=s, z=z, L=L, P=P, hL=hL[0], hX=hX[0], hN=hN[0], exL=exL, exP=exP,
                         pred=preds["bins 3-5 jointly"], kfit=None, mL=mL, mX=mX, mN=mN, ct=ct))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    fig, ax = plt.subplots(len(figd), 3, figsize=(13, 3.8 * len(figd)), squeeze=False)
    for r, F in enumerate(figd):
        EL = F["L"]["E"][:, list(XB)].sum(1)
        EP = F["P"]["E"][:, list(XB)].sum(1)
        ax[r, 0].scatter(EL, F["hL"], s=4, color="#2a6fb0", alpha=0.5, label="Lumina units")
        ax[r, 0].scatter(EP, F["hX"], s=12, color="#c4572a", label="50 cMpc, X-rays")
        ax[r, 0].scatter(EP, F["hN"], s=12, color="#777777", label="50 cMpc, no X-rays")
        ax[r, 0].set(xscale="log", yscale="log", xlabel="X-ray exposure, bins 100-2000 eV (sum) [code x Myr]",
                     ylabel=r"void $\langle 1/T\rangle^{-1}$ [K]", title=f"z = {F['z']:.2f}, 8.46 cMpc/h units")
        lo = np.nanmin(np.r_[F["exL"], F["exP"], F["pred"]]); hi = np.nanmax(np.r_[F["exL"], F["exP"], F["pred"]])
        ax[r, 1].plot([lo, hi], [lo, hi], color="k", lw=0.8, ls=":")
        ax[r, 1].scatter(F["pred"], F["exL"], s=4, color="#2a6fb0", alpha=0.5, label="Lumina (measured vs predicted)")
        ax[r, 1].scatter(F["P"]["E"][:, list(XB)] @ np.linalg.lstsq(F["P"]["E"][:, list(XB)], F["exP"], rcond=None)[0],
                         F["exP"], s=12, color="#c4572a", label="50 cMpc (exact vs fit)")
        ax[r, 1].set(xlabel="excess predicted from the X-ray exposure [K]", ylabel="void excess over no X-rays [K]",
                     title="pair-calibrated k E, bins 3-5")
        for m_, lab, col in ((F["mL"], "Lumina", "#2a6fb0"), (F["mX"], "50 cMpc, X-rays", "#c4572a"),
                             (F["mN"], "50 cMpc, no X-rays", "#777777")):
            ax[r, 2].plot(10**F["ct"], 10**m_, color=col, marker="o", ms=2, label=lab)
        ax[r, 2].set(xscale="log", yscale="log", xlim=(4.5, 300), xlabel="T [K]", ylabel="median x_HII (void cells)",
                     title="ionisation vs temperature, 1+delta < 1")
        for j in range(3):
            ax[r, j].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(a.data, exist_ok=True)
    fp = os.path.join(a.data, "void_heating.png")
    fig.savefig(fp, dpi=180)
    print(f"\nwrote {fp}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("lumina")
    s.add_argument("--snap", type=int, required=True)
    s.add_argument("--dir", default=None, help="scan directory (default <data>/cell_test_H_<snap>)")
    s.add_argument("--render-dir", default=LUM_REN)
    s.add_argument("--render", type=int, default=None, help="default: the scan's render_snap")
    s.add_argument("--hist-every", type=int, default=1, help="use every n-th render of the history (always the last)")
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    s = sp.add_parser("pair")
    s.add_argument("--snap", type=int, required=True)
    s.add_argument("--src", default=SRC)
    s.add_argument("--render", type=int, default=None, help="default: SNAP_RENDER")
    s.add_argument("--hist-every", type=int, default=1)
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    s = sp.add_parser("compare")
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--data", default="data")
    a = p.parse_args()
    {"lumina": lumina, "pair": pair, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
