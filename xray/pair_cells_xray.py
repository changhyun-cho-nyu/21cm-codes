"""
pair_cells_xray.py
==================
Cell-by-cell comparison of the two X-ray test runs (same ICs, differ only in MRT_XRAY) at a full snapshot: the first
step towards a post-processed "Lumina without X-rays" (user, 2026-11-20). Questions:
  1. Can the gas cells of the two runs be paired by ParticleIDs, and do the pairs sit at the same place?
  2. In the neutral IGM, what did X-rays change per cell: the specific energy u (the X-ray heat), the density
     (pressure feedback), x_HII (X-ray ionization)? Is the heat ~additive (u_X = u_noX + Delta u, Delta u smooth)?
  3. How accurately must Delta u be known? Best-case test of "subtract the X-ray heat": T_noX predicted as
     max(5 K, T_X - Delta T_est) with Delta T_est the TRUE Delta u averaged over 1+delta bins, 128^3 pixels, 16^3 pixels
     or the whole box, against the true T_noX, in <1/T>^-1 and <1 - T_gamma/T> of the H I.
  4. Is the render field IonEnergies the cube deposit of PhotonDensity x cell volume (AREPO RT_movie_3D_rendering.cc:
     ion_energies[bin] = SphP.Cons_DensPhot[bin]; snapshot PhotonDensity = SphP.DensPhot)? Tested per bin on the
     128^3 renders of both runs (box totals and pixel correlation; NGP vs cube deposit, so pixels agree only roughly).
     Lumina's renders have the same field (640^3 x 6, bins 13.6 / 24.59 / 54.42 / 100 / 200 / 500 / 2000 eV, as here).

scan (one snapshot, both runs; a batch job, ~2.2e8 cells per run):
  per cell: T = (gamma-1) u mu m_p / k_B, mu = 4/(1 + 3X + 4X Ne), Ne = ElectronAbundance or HII + HeII + 2 HeIII;
  pairs = IDs present once in each run (duplicated IDs are dropped and counted); separation of the pair (periodic,
  from IntCoordinates) in units of the no-X-ray cell size V^(1/3); "good" pair: separation < SEP_MAX cell sizes.
  Statistics over good pairs of non-star-forming cells that are neutral without X-rays (x_HII < 1e-3), H I weight
  X m_noX (1 - x_noX), by the no-X-ray cell's 1+delta:
    Delta T = (gamma-1) (u_X - u_noX) mu_n m_p / k_B with mu_n = 4/(1+3X) (the X-ray heat in K of neutral gas),
    T_noX and T_X, rho_X / rho_noX, x_HII with X-rays, share with u_X < u_noX.
  the Delta T field: H I-weighted mean per 128^3 pixel and per 16^3 block, against the render IonEnergies X-ray bins.
  writes <out>/pair_cells/pair_cells_NNN.npz (tables, maps, 2-D histograms, a random subsample of 2e6 good neutral
  pairs with both runs' state and the X run's PhotonDensity, for fitting a T_noX model later).
report: tables of every snapshot found, data/pair_cells_xray.png.
fit: T_noX predictors that use only X-run quantities (what Lumina has), scored on the subsample against the true T_noX
  (H I weight; classes by 1+delta; 16^3-block <1 - T_gamma/T> for the spatial pattern). Models, by 0.05-dex bins of
  1+delta_X: median T_noX | 1+delta; quantile map T_X -> T_noX | 1+delta with the training set's T_X ranks or with the
  target's own ranks (what transfers to a box with different X-ray heating); median and random draw of T_noX |
  (1+delta, T_X) (0.05-dex T_X bins, absolute, so they do not transfer); own-rank quantile map | (1+delta, quartile of
  the X-run PhotonDensity 100-500 eV). Splits: in-sample; random halves of the 16^3 blocks; train on the blocks whose
  void cells (1+delta < 1) have the coolest T_X, test on the hottest (a transfer test). Writes pair_fit_NNN.npz with
  the quantile tables fitted on the whole subsample. Limitation: the subsample holds pairs neutral WITHOUT X-rays.

  python pair_cells_xray.py scan --snap 48            (render taken from SNAP_RENDER)
  python pair_cells_xray.py report
  python pair_cells_xray.py fit [--snaps 36 48]
Inputs are read only (Oliver's runs); outputs only under /orcd/data/mvogelsb/004/chcho.
"""
import argparse
import glob
import itertools
import os
import re
import sys
import time

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

SRC = "/nfs/mvogelsblab002/Users/ozier/SonjaXrayPaper"
OUT = "/orcd/data/mvogelsb/004/chcho/xray"
ALLOWED = "/orcd/data/mvogelsb/004/chcho"
RUNS = ("withoutXrays", "withXrays")
# full snapshot -> render at the same z (|dz| <= 5e-5, xray/NOTES.md)
SNAP_RENDER = {24: 71, 36: 112, 48: 154, 57: 182, 66: 207, 74: 224, 80: 254, 86: 278, 92: 302, 97: 324, 102: 344,
               106: 362, 111: 384, 116: 404}
X_H, GAMMA_M1, K_B, M_P = 0.76, 2.0 / 3.0, 1.380649e-16, 1.67262192e-24
MU_N = 4.0 / (1.0 + 3.0 * X_H)                                    # neutral H + He, no electrons
XNEU, T_FLOOR, SEP_MAX = 1e-3, 5.0, 2.0
XBINS = (3, 4, 5)                                                 # 100-200, 200-500, 500-2000 eV
CBINS = (0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, np.inf)       # 1+delta
CLASSES = (("1+d<1", 0.0, 1.0), ("1-2", 1.0, 2.0), (">2", 2.0, np.inf), ("all", 0.0, np.inf))
NSUB = 2_000_000
BLOCK = 8                                                         # 128^3 -> 16^3
LD_EDGES = np.linspace(-1.5, 3.0, 91)
LT_EDGES = np.linspace(0.0, 6.0, 121)


def check_out(path, allow_any):
    rp = os.path.realpath(path)
    if not allow_any and os.path.commonpath([rp, ALLOWED]) != ALLOWED:
        sys.exit(f"refusing to write outside {ALLOWED}: {rp}")


def wpct(x, w, q):
    if len(x) == 0 or w.sum() <= 0:
        return np.full(len(q), np.nan)
    o = np.argsort(x)
    c = np.cumsum(w[o].astype(np.float64))
    return np.interp(np.asarray(q) / 100.0 * c[-1], c, x[o])


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def snap_files(src, run, snap):
    fs = glob.glob(os.path.join(src, run, "output", f"snapdir_{snap:03d}", f"snap_{snap:03d}.*.hdf5"))
    fs = sorted(fs, key=lambda p: int(re.search(r"\.(\d+)\.hdf5$", p).group(1)))
    if not fs:
        sys.exit(f"no full snapshot {snap} for {run} under {src}")
    return fs


def read_render(src, run, render, fields):
    """Raw render pieces (flat C-order pixel list, only file 0 has the Header) -> N^3 (x k) cubes."""
    d = os.path.join(src, run, "output", "movie", "renderings", f"render_{render:03d}")
    fs = sorted(glob.glob(os.path.join(d, f"render_{render:03d}.*.hdf5")), key=lambda p: int(p.rsplit(".", 2)[-2]))
    if not fs:
        sys.exit(f"no render {render} for {run} under {d}")
    with h5py.File(fs[0], "r") as f:
        hdr = dict(f["Header"].attrs)
    if len(fs) != int(hdr["NumFiles"]):
        sys.exit(f"STOP: render {render} of {run}: {len(fs)} files, header NumFiles = {int(hdr['NumFiles'])}")
    N = int(hdr["NumPixels"])
    parts = {k: [] for k in fields}
    for p in fs:
        with h5py.File(p, "r") as f:
            for k in fields:
                if k not in f:
                    sys.exit(f"STOP: {k} not in render {render} of {run}")
                parts[k].append(f[k][()])
    out = {}
    for k, v in parts.items():
        v = np.concatenate(v).astype(np.float64)
        if v.shape[0] != N**3:
            sys.exit(f"STOP: render {render} {k}: {v.shape[0]} pixels, expected {N}^3")
        if k == "Density" and v.ndim == 2:                       # 2x2x2 sub-cells: mean (equal volumes)
            v = v.mean(1)
        out[k] = v.reshape((N, N, N) + v.shape[1:])
    return float(hdr["Redshift"]), N, out


def read_snap(src, run, snap, N):
    """All gas cells of one run; NGP deposit (coordinate order) of mass and of PhotonDensity x volume per bin."""
    fs = snap_files(src, run, snap)
    with h5py.File(fs[0], "r") as f:
        hd = dict(f["Header"].attrs)
        par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
    z, box = float(hd["Redshift"]), float(hd["BoxSize"])
    nfiles = int(hd.get("NumFilesPerSnapshot", len(fs)))
    ntot = int(hd["NumPart_Total"][0]) + (int(hd["NumPart_Total_HighWord"][0]) << 32 if "NumPart_Total_HighWord" in hd else 0)
    uv = float(hd.get("UnitVelocity_in_cm_per_s", par.get("UnitVelocity_in_cm_per_s", 1e5)))
    if len(fs) != nfiles:
        sys.exit(f"STOP: {run}: {len(fs)} files found, the header says {nfiles}")
    print(f"{run} snapshot {snap}: z = {z:.5f}, BoxSize {box}, {len(fs)} files, {ntot:,} gas cells", flush=True)
    keys = ("id", "ip", "m", "rho", "u", "x", "ne", "sf", "ph")
    C = {k: [] for k in keys}
    dep_m = np.zeros(N**3)
    dep_ph = None
    nb = None
    nread = 0
    for p in fs:
        with h5py.File(p, "r") as f:
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            n = g["Masses"].shape[0]
            if n == 0:
                continue
            for k in ("ParticleIDs", "Masses", "Density", "InternalEnergy", "HII_Fraction", "PhotonDensity"):
                if k not in g:
                    sys.exit(f"STOP: {k} missing in {p}; available: {sorted(g.keys())}")
            if "IntCoordinates" in g:
                ip = g["IntCoordinates"][()]
                if ip.dtype != np.uint32:
                    sys.exit(f"STOP: IntCoordinates is {ip.dtype}, expected uint32")
            elif "Coordinates" in g:
                ip = np.floor(np.mod(g["Coordinates"][()].astype(np.float64), box) / box * 2.0**32)
                ip = np.minimum(ip, 2.0**32 - 1).astype(np.uint32)
            else:
                sys.exit(f"STOP: no positions in {p}")
            m = g["Masses"][()].astype(np.float32)
            rho = g["Density"][()].astype(np.float32)
            x = g["HII_Fraction"][()].astype(np.float32)
            if "ElectronAbundance" in g:
                ne = g["ElectronAbundance"][()].astype(np.float32)
            else:
                fII = g["HeII_Fraction"][()] if "HeII_Fraction" in g else 0.0
                fIII = g["HeIII_Fraction"][()] if "HeIII_Fraction" in g else 0.0
                ne = (x + fII + 2.0 * fIII).astype(np.float32)
            ph = g["PhotonDensity"][()].astype(np.float64)
            if nb is None:
                nb = ph.shape[1]
                dep_ph = np.zeros((nb, N**3))
            idx = ((ip.astype(np.uint64) * np.uint64(N)) >> np.uint64(32)).astype(np.int64)
            flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), (N, N, N))
            vol = m.astype(np.float64) / rho
            dep_m += np.bincount(flat, m, minlength=N**3)
            for b in range(nb):
                dep_ph[b] += np.bincount(flat, ph[:, b] * vol, minlength=N**3)
            C["id"].append(g["ParticleIDs"][()].astype(np.uint64))
            C["ip"].append(ip)
            C["m"].append(m); C["rho"].append(rho); C["x"].append(x); C["ne"].append(ne)
            C["u"].append(g["InternalEnergy"][()].astype(np.float32))
            C["sf"].append((g["StarFormationRate"][()] > 0) if "StarFormationRate" in g else np.zeros(n, bool))
            C["ph"].append(ph[:, list(XBINS)].astype(np.float32) if nb == 6 else np.zeros((n, 0), np.float32))
            nread += n
    if nread != ntot:
        sys.exit(f"STOP: {run}: read {nread:,} cells, header says {ntot:,}")
    C = {k: np.concatenate(v) for k, v in C.items()}
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * C["ne"].astype(np.float64)) * M_P
    C["T"] = (GAMMA_M1 * C["u"] * uv**2 * mu / K_B).astype(np.float32)
    del mu
    meta = dict(z=z, box=box, uv=uv, ntot=ntot, nb=nb, rho_mean=C["m"].astype(np.float64).sum() / box**3)
    print(f"  read {nread:,} cells, PhotonDensity bins {nb}, mean gas density {meta['rho_mean']:.6g} (code)", flush=True)
    return C, meta, dep_m, dep_ph


def axis_order(dep_m, rden):
    """Render index (j0,j1,j2) = coordinate indices (i_o0, i_o1, i_o2): o from the data."""
    ok = (dep_m > 0) & (rden > 0)
    best = []
    for o in itertools.permutations(range(3)):
        d = np.transpose(dep_m, o)
        k = ok & (d > 0)
        best.append((corr(np.log10(d[k]), np.log10(rden[k])), o))
    best.sort(reverse=True)
    for c, o in best:
        print(f"    render[j] = coords {o}: corr(log NGP mass, log render Density) = {c:.4f}")
    if best[0][0] - best[1][0] < 0.05:
        sys.exit("STOP: the render axis order is not clear from the data")
    return best[0][1], best[0][0]


def unique_sorted(ids):
    o = np.argsort(ids, kind="stable")
    s = ids[o]
    d = np.zeros(len(s), bool)
    eq = s[1:] == s[:-1]
    d[1:] |= eq
    d[:-1] |= eq
    return s[~d], o[~d], int(d.sum())


def separation(ipa, ipb, ia, ib, box):
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), 1 << 24):
        e = min(s + (1 << 24), len(ia))
        d = ipa[ia[s:e]].astype(np.int64) - ipb[ib[s:e]].astype(np.int64)
        d = (d + 2**31) % 2**32 - 2**31
        out[s:e] = np.sqrt((d.astype(np.float64)**2).sum(1)) * (box / 2.0**32)
    return out


def scan(a):
    od = os.path.join(a.out, "pair_cells")
    check_out(od, a.allow_any_out)
    render = a.render if a.render is not None else SNAP_RENDER[a.snap]
    t0 = time.time()
    R, Cs, Ms = {}, {}, {}
    for run in RUNS:
        zr, N, R[run] = read_render(a.src, run, render, ["Density", "IonEnergies"])
        print(f"{run} render {render}: z = {zr:.5f}, N = {N}, IonEnergies bins {R[run]['IonEnergies'].shape[-1]}", flush=True)
        C, M, dep_m, dep_ph = read_snap(a.src, run, a.snap, N)
        if abs(M["z"] - zr) > 1e-3:
            sys.exit(f"STOP: render z {zr} vs snapshot z {M['z']}")
        dep_m = dep_m.reshape(N, N, N)
        o, co = axis_order(dep_m, R[run]["Density"])
        M["order"], M["corr_mass"] = o, co
        # IonEnergies = cube deposit of Cons_DensPhot; test Cons_DensPhot = PhotonDensity x V (per bin)
        ion = R[run]["IonEnergies"]
        rows = []
        for b in range(M["nb"]):
            d = np.transpose(dep_ph[b].reshape(N, N, N), o)
            r = ion[..., b]
            k = (d > 0) & (r > 0)
            tot = d.sum() / r.sum() if r.sum() > 0 else np.nan
            c = corr(np.log10(d[k]), np.log10(r[k])) if k.sum() > 2 else np.nan
            med = np.median(d[k] / r[k]) if k.any() else np.nan
            rows.append([b, tot, c, med, k.mean()])
            print(f"    bin {b}: sum PhotonDensity*V (cells) / sum IonEnergies (render) = {tot:.5f}; pixels: "
                  f"corr(log) {c:.4f}, median ratio {med:.4f} (NGP vs cube deposit), both > 0 in {k.mean():.4f}")
        M["ion_check"] = np.array(rows)
        M["N"] = N
        Cs[run], Ms[run] = C, M
        del dep_ph, dep_m
    A, B = Cs["withoutXrays"], Cs["withXrays"]
    MA, MB = Ms["withoutXrays"], Ms["withXrays"]
    N, box, uv, z = MA["N"], MA["box"], MA["uv"], MA["z"]
    if MA["order"] != MB["order"]:
        sys.exit("STOP: the two runs' render axis orders differ")
    if abs(MA["z"] - MB["z"]) > 1e-4 or MA["box"] != MB["box"]:
        sys.exit("STOP: the two snapshots differ in z or BoxSize")
    print(f"read both runs in {time.time() - t0:.0f} s", flush=True)

    # 1. pairing by ParticleIDs
    sa, oa, dA = unique_sorted(A["id"])
    sb, ob, dB = unique_sorted(B["id"])
    pos = np.searchsorted(sb, sa)
    pos = np.minimum(pos, len(sb) - 1)
    hit = sb[pos] == sa
    ia, ib = oa[hit], ob[pos[hit]]
    del sa, sb, oa, ob, pos, hit
    o = np.argsort(ia)                                           # file order of the no-X-ray run
    ia, ib = ia[o], ib[o]
    del o
    sep = separation(A["ip"], B["ip"], ia, ib, box)
    scell = np.cbrt(A["m"][ia].astype(np.float64) / A["rho"][ia]).astype(np.float32)
    rsep = sep / scell
    good = rsep < SEP_MAX
    wA = X_H * A["m"].astype(np.float64) * (1.0 - A["x"])
    neuA = (~A["sf"]) & (A["x"] < XNEU)
    pn = neuA[ia] & ~B["sf"][ib]
    mA = A["m"].astype(np.float64)
    match = dict(nA=len(A["id"]), nB=len(B["id"]), dupA=dA, dupB=dB, npair=len(ia), ngood=int(good.sum()),
                 f_mass=mA[ia].sum() / mA.sum(), f_mass_good=mA[ia[good]].sum() / mA.sum(),
                 f_HI=wA[ia].sum() / wA.sum(),
                 f_neuHI_good=wA[ia[good & pn]].sum() / wA[neuA].sum())
    print(f"\n1. pairs by ParticleIDs: cells {match['nA']:,} (no X) / {match['nB']:,} (X); duplicated IDs dropped "
          f"{dA:,} / {dB:,}; pairs {match['npair']:,}, good (separation < {SEP_MAX:g} cell sizes) {match['ngood']:,}")
    print(f"   paired share of the no-X-ray gas mass {match['f_mass']:.4f} (good {match['f_mass_good']:.4f}), of its H I "
          f"{match['f_HI']:.4f}; of its neutral-IGM H I in good pairs (non-SF in both) {match['f_neuHI_good']:.4f}")
    sep_rows = []
    for lab, s in (("all pairs", np.ones(len(ia), bool)), ("neutral IGM pairs", pn)):
        q = wpct(rsep[s].astype(np.float64), wA[ia[s]], (50, 90, 99)) if s.any() else [np.nan] * 3
        fg = wA[ia[s & good]].sum() / wA[ia[s]].sum() if s.any() else np.nan
        sep_rows.append([*q, fg])
        print(f"   separation / cell size, {lab} (H I-weighted): p50 {q[0]:.3f}, p90 {q[1]:.3f}, p99 {q[2]:.3f}; "
              f"good share {fg:.4f}")

    # 2. neutral IGM pairs: what X-rays changed
    P = good & pn
    ja, jb = ia[P], ib[P]
    rsP = rsep[P]
    del ia, ib, rsep, sep, scell, good, pn
    w = wA[ja]
    one = A["rho"][ja].astype(np.float64) / MA["rho_mean"]
    oneB = B["rho"][jb].astype(np.float64) / MB["rho_mean"]
    TA, TB = A["T"][ja].astype(np.float64), B["T"][jb].astype(np.float64)
    du = (B["u"][jb].astype(np.float64) - A["u"][ja]) * GAMMA_M1 * uv**2 * MU_N * M_P / K_B   # Delta T [K]
    xB = B["x"][jb].astype(np.float64)
    rratio = B["rho"][jb].astype(np.float64) / A["rho"][ja]
    Tg = 2.7255 * (1.0 + z)
    W = w.sum()
    print(f"\n2. good neutral IGM pairs (no X-rays: x_HII < {XNEU:g}, non-SF in both): {len(ja):,}; T_gamma = {Tg:.2f} K")
    print("   by the no-X-ray cell's 1+delta (H I-weighted): H I share | Delta T p10 p50 p90 [K] | Delta T/(1+d)^2/3 p50 "
          "| T_noX p50 <1/T>^-1 | T_X p50 <1/T>^-1 | rho_X/rho_noX p10 p50 p90 | x_HII,X p50 p90 | f(u_X<u_noX)")
    tab = []
    for lo, hi in zip(CBINS[:-1], CBINS[1:]):
        s = (one >= lo) & (one < hi)
        ws = w[s]
        if ws.sum() <= 0:
            tab.append([lo, hi] + [np.nan] * 17)
            continue
        d10, d50, d90 = wpct(du[s], ws, (10, 50, 90))
        dn = wpct(du[s] / one[s]**(2.0 / 3.0), ws, (50,))[0]
        ta50 = wpct(TA[s], ws, (50,))[0]; tb50 = wpct(TB[s], ws, (50,))[0]
        ha = ws.sum() / np.sum(ws / TA[s]); hb = ws.sum() / np.sum(ws / TB[s])
        r10, r50, r90 = wpct(rratio[s], ws, (10, 50, 90))
        x50, x90 = wpct(xB[s], ws, (50, 90))
        fneg = ws[du[s] < 0].sum() / ws.sum()
        f55 = ws[TA[s] < 5.5].sum() / ws.sum()
        tab.append([lo, hi, ws.sum() / W, d10, d50, d90, dn, ta50, ha, tb50, hb, r10, r50, r90, x50, x90, fneg, f55,
                    wpct(one[s], ws, (50,))[0]])
        print(f"   {lo:5g}-{hi:<5g} | {ws.sum() / W:6.4f} | {d10:8.2f} {d50:8.2f} {d90:8.2f} | {dn:8.2f} | "
              f"{ta50:8.1f} {ha:8.1f} | {tb50:8.1f} {hb:8.1f} | {r10:5.3f} {r50:5.3f} {r90:5.3f} | {x50:.1e} {x90:.1e} | "
              f"{fneg:.3f}")

    # Delta T field: H I-weighted mean per 128^3 pixel / 16^3 block (render order)
    o_ = MA["order"]
    idx = ((A["ip"][ja].astype(np.uint64) * np.uint64(N)) >> np.uint64(32)).astype(np.int64)
    flat = np.ravel_multi_index((idx[:, o_[0]], idx[:, o_[1]], idx[:, o_[2]]), (N, N, N))
    del idx
    sw = np.bincount(flat, w, minlength=N**3).reshape(N, N, N)
    swd = np.bincount(flat, w * du, minlength=N**3).reshape(N, N, N)
    dpix = np.where(sw > 0, swd / np.where(sw > 0, sw, 1.0), np.nan)
    nbk = N // BLOCK
    bw = sw.reshape(nbk, BLOCK, nbk, BLOCK, nbk, BLOCK).sum((1, 3, 5))
    bwd = swd.reshape(nbk, BLOCK, nbk, BLOCK, nbk, BLOCK).sum((1, 3, 5))
    dblk = np.where(bw > 0, bwd / np.where(bw > 0, bw, 1.0), np.nan)
    dbox = swd.sum() / sw.sum()
    print(f"\n   Delta T field: H I-weighted box mean {dbox:.3f} K; pixels with pairs {np.mean(sw > 0):.4f}; "
          f"scatter (H I-weighted std / mean) of the pixel means {np.sqrt(np.nansum(sw * (dpix - dbox)**2) / sw.sum()) / dbox:.3f}, "
          f"of the 16^3 block means {np.sqrt(np.nansum(bw * (dblk - dbox)**2) / bw.sum()) / dbox:.3f}, "
          f"of the cells {np.sqrt(np.sum(w * (du - dbox)**2) / W) / dbox:.3f}")
    ionB = R["withXrays"]["IonEnergies"]
    xcorr = []
    if ionB.shape[-1] == 6:
        for b in XBINS:
            e = ionB[..., b]
            k = (sw > 0) & (dpix > 0) & (e > 0)
            c1 = corr(np.log10(dpix[k]), np.log10(e[k])) if k.sum() > 2 else np.nan
            eb = e.reshape(nbk, BLOCK, nbk, BLOCK, nbk, BLOCK).sum((1, 3, 5))
            kb = (bw > 0) & (dblk > 0) & (eb > 0)
            c2 = corr(np.log10(dblk[kb]), np.log10(eb[kb])) if kb.sum() > 2 else np.nan
            xcorr.append([b, c1, c2])
            print(f"   corr(log Delta T, log IonEnergies bin {b}, X run): 128^3 {c1:.3f}, 16^3 {c2:.3f} "
                  "(instantaneous field vs accumulated heat)")

    # 3. best-case subtraction: T_noX = max(5 K, T_X - Delta T_est), Delta T_est from the true Delta T, smoothed
    bins_x = np.searchsorted(np.array(CBINS[1:-1]), oneB, side="right")       # by the X run's own 1+delta
    med_x = np.array([wpct(du[bins_x == i], w[bins_x == i], (50,))[0] if np.any(bins_x == i) else dbox
                      for i in range(len(CBINS) - 1)])
    i0, i1, i2 = np.unravel_index(flat, (N, N, N))
    blk = np.ravel_multi_index((i0 // BLOCK, i1 // BLOCK, i2 // BLOCK), (nbk, nbk, nbk))
    del i0, i1, i2
    est = {"truth": None, "X (none)": np.zeros_like(du), "box mean": np.full_like(du, dbox),
           "1+delta_X bin": med_x[bins_x], "16^3 blocks": dblk.ravel()[blk], "128^3 pixels": dpix.ravel()[flat]}
    print(f"\n3. best-case subtraction of the true Delta T, smoothed: T = max({T_FLOOR:g} K, T_X - Delta T_est) "
          "(H I-weighted, good neutral pairs)")
    print("   estimate        | " + " | ".join(f"{c[0]:>6s}: <1/T>^-1  <1-Tg/T>" for c in CLASSES) + " | rms log10(T/T_true)")
    sens = []
    for name, e in est.items():
        Tp = TA if e is None else np.maximum(T_FLOOR, TB - e)
        row = []
        for lab, lo, hi in CLASSES:
            s = (one >= lo) & (one < hi)
            ws = w[s]
            row += [ws.sum() / np.sum(ws / Tp[s]), np.sum(ws * (1.0 - Tg / Tp[s])) / ws.sum()]
        rms = np.sqrt(np.sum(w * np.log10(Tp / TA)**2) / W)
        sens.append(row + [rms])
        print(f"   {name:15s} | " + " | ".join(f"{row[2 * i]:16.2f} {row[2 * i + 1]:8.3f}" for i in range(len(CLASSES)))
              + f" | {rms:.3f}")

    # subsample and histograms
    rng = np.random.default_rng(12345)
    k = rng.choice(len(ja), min(NSUB, len(ja)), replace=False)
    sub = dict(T_noX=TA[k].astype(np.float32), T_X=TB[k].astype(np.float32), dT=du[k].astype(np.float32),
               one_noX=one[k].astype(np.float32), one_X=oneB[k].astype(np.float32),
               x_noX=A["x"][ja[k]], x_X=B["x"][jb[k]], ne_noX=A["ne"][ja[k]], ne_X=B["ne"][jb[k]],
               w=w[k].astype(np.float32), pix=flat[k].astype(np.int32), rsep=rsP[k],
               ph_X=B["ph"][jb[k]], m_noX=A["m"][ja[k]])
    h_du = np.histogram2d(np.log10(one), np.log10(np.maximum(du, 1e-3)), [LD_EDGES, np.linspace(-3, 5, 161)],
                          weights=w / W)[0]
    h_TT = np.histogram2d(np.log10(np.maximum(TA, 1.0)), np.log10(np.maximum(TB, 1.0)), [LT_EDGES, LT_EDGES],
                          weights=w / W)[0]
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"pair_cells_{a.snap:03d}.npz")
    np.savez(fn, z=z, snap=a.snap, render=render, Tgamma=Tg, order=np.array(o_), cbins=np.array(CBINS),
             match=np.array([match[k_] for k_ in ("nA", "nB", "dupA", "dupB", "npair", "ngood", "f_mass", "f_mass_good",
                                                   "f_HI", "f_neuHI_good")], dtype=np.float64),
             sep_rows=np.array(sep_rows), table=np.array(tab), dbox=dbox, dpix=dpix.astype(np.float32),
             dblk=dblk.astype(np.float32), hiw_pix=sw.astype(np.float32), xcorr=np.array(xcorr),
             sens=np.array(sens), sens_names=np.array(list(est.keys())), classes=np.array([c[0] for c in CLASSES]),
             ion_check_noX=MA["ion_check"], ion_check_X=MB["ion_check"],
             corr_mass=np.array([MA["corr_mass"], MB["corr_mass"]]), h_du=h_du, h_TT=h_TT, ld_edges=LD_EDGES,
             lt_edges=LT_EDGES, **{f"sub_{k_}": v for k_, v in sub.items()})
    print(f"\nwrote {fn} ({time.time() - t0:.0f} s)")


def report(a):
    fs = sorted(glob.glob(os.path.join(a.out, "pair_cells", "pair_cells_*.npz")))
    if not fs:
        sys.exit("no scan outputs found")
    D = [np.load(f) for f in fs]
    for d in D:
        mt = d["match"]
        print(f"\nsnapshot {int(d['snap'])} (z = {float(d['z']):.3f}, render {int(d['render'])}): pairs {int(mt[4]):,} of "
              f"{int(mt[0]):,} / {int(mt[1]):,}, good {int(mt[5]):,}; neutral-IGM H I in good pairs {mt[9]:.4f}; "
              f"Delta T box mean {float(d['dbox']):.2f} K")
        for r in d["table"]:
            print(f"  1+delta {r[0]:5g}-{r[1]:<5g}: H I {r[2]:.4f}, Delta T p50 {r[4]:8.2f} K (p10 {r[3]:.2f}, p90 {r[5]:.2f}), "
                  f"T_noX {r[7]:7.1f}, T_X {r[9]:7.1f} K, rho ratio {r[12]:.3f}, f(cooler) {r[16]:.3f}")
        for name, s in zip(d["sens_names"], d["sens"]):
            print(f"  {str(name):15s}: all <1/T>^-1 {s[6]:7.2f} K, <1-Tg/T> {s[7]:7.3f}; 1+d<1: {s[0]:7.2f} K, "
                  f"{s[1]:7.3f}; rms log {s[8]:.3f}")
        for lab, ic in (("no X", d["ion_check_noX"]), ("X", d["ion_check_X"])):
            print(f"  IonEnergies check ({lab}): " + ", ".join(f"bin {int(r[0])} sum {r[1]:.4f} corr {r[2]:.3f}" for r in ic))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(D)))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for c, d in zip(cols, D):
        t = d["table"]
        ok = np.isfinite(t[:, 4])
        xm = t[ok, 18]
        lab = f"z = {float(d['z']):.2f}"
        ax[0].plot(xm, t[ok, 4], color=c, marker="o", label=lab)
        ax[0].fill_between(xm, np.maximum(t[ok, 3], 1e-2), t[ok, 5], color=c, alpha=0.15)
        ref = t[ok, 4][np.argmin(abs(np.log(xm / 0.7)))] * (xm / xm[np.argmin(abs(np.log(xm / 0.7)))])**(2 / 3)
        ax[0].plot(xm, ref, color=c, ls=":", lw=1)
        ax[1].plot(xm, t[ok, 7], color=c, marker="o", label=f"{lab}, no X")
        ax[1].plot(xm, t[ok, 9], color=c, marker="s", ls="--", mfc="none", label=f"{lab}, X")
        ax[1].axhline(float(d["Tgamma"]), color=c, lw=0.8, ls="-.")
    ax[0].set(xscale="log", yscale="log", xlabel=r"$1+\delta$ (no X-rays)", ylabel=r"$\Delta T$ [K] (median, 10-90%)",
              title="X-ray heat per cell (dotted: $(1+\\delta)^{2/3}$)")
    ax[1].set(xscale="log", yscale="log", xlabel=r"$1+\delta$ (no X-rays)", ylabel="median T [K]",
              title="neutral IGM pairs ($T_\\gamma$ dash-dot)")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=6.5, ncol=2)
    names = [str(n) for n in D[0]["sens_names"]]
    xs = np.arange(len(names))
    for i, (c, d) in enumerate(zip(cols, D)):
        ax[2].plot(xs + 0.08 * i, d["sens"][:, 7], color=c, marker="o", ls="none", label=f"z = {float(d['z']):.2f}")
        ax[2].axhline(d["sens"][0, 7], color=c, lw=0.8, ls=":")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
    ax[2].set(ylabel=r"$\langle 1 - T_\gamma/T\rangle_{\rm HI}$", title="subtract smoothed true heat (dotted: truth)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(a.data, exist_ok=True)
    fig.savefig(os.path.join(a.data, "pair_cells_xray.png"), dpi=200)
    print(f"\nwrote {a.data}/pair_cells_xray.png")


# ---------------------------------------------------------------------------------------------------------------------
# fit: T_noX predictors from X-run quantities only, scored against the true T_noX of the paired subsample
FIT_LD = np.arange(-1.5, 3.0001, 0.05)       # log10(1+delta_X) bins
FIT_LT = np.arange(0.0, 5.0001, 0.05)        # log10 T_X bins (2-D tables)
NLEV = 1001                                  # quantile levels
NQ = 4                                       # photon-density classes (quartiles within a 1+delta bin)
MIN_TRAIN = 50                               # cells a bin needs to be used; otherwise the nearest such bin / 1-D model
MIN_BLK = 20                                 # test cells a 16^3 block needs in the block statistics
LEV = np.linspace(0.0, 1.0, NLEV)


def wquant(y, w, lev):
    """weighted quantiles (mid-point CDF) of y at levels lev"""
    o = np.argsort(y, kind="stable")
    c = np.cumsum(w[o].astype(np.float64))
    c = (c - 0.5 * w[o]) / c[-1]
    return np.interp(lev, c, y[o])


def groups(key):
    o = np.argsort(key, kind="stable")
    ks = key[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    en = np.r_[st[1:], len(ks)]
    return o, ks[st], st, en


def qtable(key, nkey, y, w):
    Q = np.full((nkey, NLEV), np.nan)
    n = np.zeros(nkey, np.int64)
    o, ku, st, en = groups(key)
    for k, a0, a1 in zip(ku, st, en):
        j = o[a0:a1]
        n[k] = a1 - a0
        Q[k] = wquant(y[j], w[j], LEV)
    return Q, n


def nearest_ok(ok):
    """index of the nearest True entry (ties: lower)"""
    good = np.flatnonzero(ok)
    if len(good) == 0:
        sys.exit("STOP: no populated bin in the training set")
    i = np.arange(len(ok))
    p = np.clip(np.searchsorted(good, i), 1, max(len(good) - 1, 1))
    lo, hi = good[np.maximum(p - 1, 0)], good[np.minimum(p, len(good) - 1)]
    return np.where(np.abs(i - lo) <= np.abs(hi - i), lo, hi)


def own_rank(key, y, w):
    """weighted mid-point rank (0..1) of y within its own key group"""
    p = np.empty(len(y))
    o, ku, st, en = groups(key)
    for a0, a1 in zip(st, en):
        j = o[a0:a1]
        jj = j[np.argsort(y[j], kind="stable")]
        c = np.cumsum(w[jj].astype(np.float64))
        p[jj] = (c - 0.5 * w[jj]) / c[-1]
    return p


def eval_q(Q, key, p):
    """Q[key] evaluated at levels p (per group)"""
    out = np.empty(len(p))
    o, ku, st, en = groups(key)
    for k, a0, a1 in zip(ku, st, en):
        j = o[a0:a1]
        out[j] = np.interp(p[j], LEV, Q[k])
    return out


def ph_class(kd, lph, w, nkd, edges=None):
    """photon-density quartile within each 1+delta bin; edges from (kd, lph, w) unless given"""
    if edges is None:
        edges = np.full((nkd, NQ - 1), np.nan)
        o, ku, st, en = groups(kd)
        for k, a0, a1 in zip(ku, st, en):
            j = o[a0:a1]
            edges[k] = wquant(lph[j], w[j], np.arange(1, NQ) / NQ)
    q = np.zeros(len(kd), np.int64)
    for i in range(NQ - 1):
        q += lph > edges[kd, i]
    return q, edges


def predictors(tr, te, S, rng):
    """log10 T_noX predictions for the test cells from X-run quantities; models trained on tr"""
    ld, lt, lph, w, ltn = S["ld"], S["lt"], S["lph"], S["w"], S["ltn"]
    nd, nt = len(FIT_LD) - 1, len(FIT_LT) - 1
    kd = np.clip(np.searchsorted(FIT_LD, ld, side="right") - 1, 0, nd - 1)
    kt = np.clip(np.searchsorted(FIT_LT, lt, side="right") - 1, 0, nt - 1)
    # 1-D (1+delta) tables
    QN, n1 = qtable(kd[tr], nd, ltn[tr], w[tr])
    QX, _ = qtable(kd[tr], nd, lt[tr], w[tr])
    near = nearest_ok(n1 >= MIN_TRAIN)
    kd_tr = near[kd[te]]
    # 2-D (1+delta, T_X) tables of T_noX
    k2 = kd * nt + kt
    Q2, n2 = qtable(k2[tr], nd * nt, ltn[tr], w[tr])
    ok2 = n2[k2[te]] >= MIN_TRAIN
    # photon-density classes: training edges for the training table, the test set's own edges for its own ranks
    near4 = nearest_ok(n1 >= MIN_TRAIN * NQ)
    q_tr, _ = ph_class(kd[tr], lph[tr], w[tr], nd)
    Q4, _ = qtable(kd[tr] * NQ + q_tr, nd * NQ, ltn[tr], w[tr])
    kd4 = near4[kd[te]]
    q_te, _ = ph_class(kd4, lph[te], w[te], nd)
    P = {}
    P["X (none)"] = lt[te]
    P["median | d"] = QN[kd_tr, NLEV // 2]
    p = np.empty(len(te))
    lte = lt[te]
    o, ku, st, en = groups(kd_tr)
    for k, a0, a1 in zip(ku, st, en):
        j = o[a0:a1]
        p[j] = np.interp(lte[j], QX[k], LEV)
    P["qmap | d, train ranks"] = eval_q(QN, kd_tr, p)
    P["qmap | d, own ranks"] = eval_q(QN, kd_tr, own_rank(kd_tr, lt[te], w[te]))
    med2 = Q2[k2[te], NLEV // 2]
    P["median | d,T_X"] = np.where(ok2, med2, P["median | d"])
    u = rng.random(len(te))
    d2 = np.full(len(te), np.nan)
    if ok2.any():
        d2[ok2] = eval_q(Q2, k2[te][ok2], u[ok2])
    d1 = eval_q(QN, kd_tr, u)
    P["draw | d,T_X"] = np.where(ok2, d2, d1)
    k4 = kd4 * NQ + q_te
    ok4 = np.all(np.isfinite(Q4), axis=1)[k4]                    # an empty quartile (ties in ph): 1-D own ranks
    P["qmap | d,ph, own ranks"] = np.where(ok4, eval_q(np.nan_to_num(Q4), k4, own_rank(k4, lt[te], w[te])),
                                           P["qmap | d, own ranks"])
    return P, dict(n1=n1, f2=float(np.sum(w[te][ok2]) / np.sum(w[te])))


def scores(lp, te, S, blk, nblk):
    """class <1/T>^-1, <1-Tg/T>; rms log10 error; 16^3-block <1-Tg/T>: corr, rms / std of the truth"""
    w, one, Tg = S["w"][te], S["one"][te], S["Tg"]
    T, Tt = 10.0**lp, 10.0**S["ltn"][te]
    row = []
    for lab, lo, hi in CLASSES:
        s = (one >= lo) & (one < hi)
        ws = w[s]
        row += [ws.sum() / np.sum(ws / T[s]), np.sum(ws * (1.0 - Tg / T[s])) / ws.sum()]
    row.append(np.sqrt(np.sum(w * (lp - S["ltn"][te])**2) / w.sum()))
    b = blk[te]
    sw = np.bincount(b, w, nblk)
    nb = np.bincount(b, minlength=nblk)
    ap = np.bincount(b, w * (1.0 - Tg / T), nblk)
    at = np.bincount(b, w * (1.0 - Tg / Tt), nblk)
    k = nb >= MIN_BLK
    ap, at = ap[k] / sw[k], at[k] / sw[k]
    row += [corr(ap, at) if k.sum() > 2 else np.nan,
            np.sqrt(np.mean((ap - at)**2)) / np.std(at) if k.sum() > 2 else np.nan, int(k.sum())]
    return row


def fit(a):
    fs = sorted(glob.glob(os.path.join(a.out, "pair_cells", "pair_cells_*.npz")))
    if a.snaps:
        fs = [f for f in fs if int(re.search(r"_(\d+)\.npz$", f).group(1)) in a.snaps]
    if not fs:
        sys.exit("no scan outputs found")
    od = os.path.join(a.out, "pair_cells")
    check_out(od, a.allow_any_out)
    for f in fs:
        d = np.load(f)
        snap, z, Tg = int(d["snap"]), float(d["z"]), float(d["Tgamma"])
        N = d["dpix"].shape[0]
        nbk = N // BLOCK
        T_noX, T_X = d["sub_T_noX"].astype(np.float64), d["sub_T_X"].astype(np.float64)
        w = d["sub_w"].astype(np.float64)
        ph = d["sub_ph_X"].astype(np.float64)
        if ph.shape[1] < 2:
            sys.exit(f"STOP: snapshot {snap}: the X run's subsample has no X-ray PhotonDensity bins")
        if np.any(T_noX <= 0) or np.any(T_X <= 0):
            sys.exit(f"STOP: snapshot {snap}: non-positive temperatures in the subsample")
        phs = ph[:, 0] + ph[:, 1]
        lph = np.log10(np.where(phs > 0, phs, np.nan))
        lph = np.where(np.isfinite(lph), lph, np.nanmin(lph) - 1.0)
        S = dict(ld=np.log10(d["sub_one_X"].astype(np.float64)), lt=np.log10(T_X), ltn=np.log10(T_noX), lph=lph, w=w,
                 one=d["sub_one_noX"].astype(np.float64), Tg=Tg)
        i0, i1, i2 = np.unravel_index(d["sub_pix"].astype(np.int64), (N, N, N))
        blk = np.ravel_multi_index((i0 // BLOCK, i1 // BLOCK, i2 // BLOCK), (nbk,) * 3)
        nblk = nbk**3
        n = len(w)
        xX = d["sub_x_X"].astype(np.float64)
        print(f"\nsnapshot {snap} (z = {z:.3f}, T_gamma = {Tg:.2f} K): {n:,} paired neutral-IGM cells (subsample); "
              f"H I share with x_HII,X > 1e-3: {w[xX > 1e-3].sum() / w.sum():.4f}")
        # diagnostic: within-bin rank correlation of T_X and T_noX
        kd = np.clip(np.searchsorted(FIT_LD, S["ld"], side="right") - 1, 0, len(FIT_LD) - 2)
        rx, rn = own_rank(kd, S["lt"], w), own_rank(kd, S["ltn"], w)
        rph = own_rank(kd, lph, w)
        msg = []
        for lab, lo, hi in CLASSES:
            s = (S["one"] >= lo) & (S["one"] < hi)
            ws = w[s]
            cw = lambda x_, y_: np.sum(ws * (x_ - 0.5) * (y_ - 0.5)) / np.sqrt(np.sum(ws * (x_ - 0.5)**2) * np.sum(ws * (y_ - 0.5)**2))
            msg.append(f"{lab}: r(T_X, T_noX) {cw(rx[s], rn[s]):.3f}, r(ph, T_noX) {cw(rph[s], rn[s]):.3f}")
        print("   within-bin rank correlation (H I-weighted, 0.05 dex in 1+delta_X): " + "; ".join(msg))
        # splits: in-sample; random half of the 16^3 blocks; train on the blocks with the coolest void T_X, test on the hottest
        rng = np.random.default_rng(a.seed)
        perm = rng.permutation(nblk)
        tr_b = np.zeros(nblk, bool); tr_b[perm[:nblk // 2]] = True
        vs = S["one"] < 1.0
        bw = np.bincount(blk[vs], w[vs], nblk)
        bn = np.bincount(blk[vs], minlength=nblk)
        bt = np.bincount(blk[vs], w[vs] * S["lt"][vs], nblk) / np.where(bw > 0, bw, 1.0)
        has = bn >= MIN_BLK
        cut = np.median(bt[has])
        hot = has & (bt > cut)
        splits = {"in-sample": (np.arange(n), np.arange(n)),
                  "random blocks": (np.flatnonzero(tr_b[blk]), np.flatnonzero(~tr_b[blk])),
                  "cool -> hot blocks": (np.flatnonzero(~hot[blk]), np.flatnonzero(hot[blk]))}
        res, names = {}, None
        for sn, (tr, te) in splits.items():
            P, info = predictors(tr, te, S, np.random.default_rng(a.seed + 1))
            truth = scores(S["ltn"][te], te, S, blk, nblk)
            rows = [truth] + [scores(lp, te, S, blk, nblk) for lp in P.values()]
            names = ["truth"] + list(P.keys())
            res[sn] = np.array(rows)
            extra = ""
            if sn == "cool -> hot blocks":
                def hm(idx):
                    s = idx[vs[idx]]
                    return w[s].sum() / np.sum(w[s] / T_X[s]), w[s].sum() / np.sum(w[s] / T_noX[s])
                (hx0, hn0), (hx1, hn1) = hm(tr), hm(te)
                extra = (f"; 1+d<1 <1/T>^-1 train {hx0:.1f} K with X / {hn0:.2f} K without, "
                         f"test {hx1:.1f} / {hn1:.2f} K")
            print(f"\n   split '{sn}': train {len(tr):,}, test {len(te):,} cells; 2-D table bins used for "
                  f"{info['f2']:.3f} of the test H I{extra}")
            print("   model                    |   all: <1/T>^-1  <1-Tg/T>  (err) |  1+d<1: <1-Tg/T> (err) |"
                  "    1-2: <1-Tg/T> (err) |     >2: <1-Tg/T> (err) | rms log | 16^3 blocks: r  rms/sd")
            for nm, r in zip(names, rows):
                err = lambda i: (r[i] - truth[i]) / abs(truth[i])
                print(f"   {nm:24s} | {r[6]:15.2f} {r[7]:9.3f} ({err(7):+6.1%}) | {r[1]:15.3f} ({err(1):+6.1%}) | "
                      f"{r[3]:15.3f} ({err(3):+6.1%}) | {r[5]:15.3f} ({err(5):+6.1%}) | {r[8]:7.3f} | "
                      f"{r[9]:6.3f} {r[10]:6.3f} (n {int(r[11])})")
        # tables fitted on the whole subsample, for applying to Lumina later
        nd = len(FIT_LD) - 1
        QN, n1 = qtable(kd, nd, S["ltn"], w)
        QX, _ = qtable(kd, nd, S["lt"], w)
        q4, e4 = ph_class(kd, lph, w, nd)
        Q4, n4 = qtable(kd * NQ + q4, nd * NQ, S["ltn"], w)
        fn = os.path.join(od, f"pair_fit_{snap:03d}.npz")
        np.savez(fn, z=z, snap=snap, Tgamma=Tg, ld_edges=FIT_LD, lev=LEV, QN_logT_noX=QN, QX_logT_X=QX, n1=n1,
                 Q4_logT_noX=Q4.reshape(nd, NQ, NLEV), n4=n4.reshape(nd, NQ), ph_edges=e4,
                 models=np.array(names), splits=np.array(list(res.keys())), scores=np.array(list(res.values())),
                 score_cols=np.array([f"{c[0]} {q}" for c in CLASSES for q in ("hm", "a")] +
                                     ["rms_log", "blk_r", "blk_rms_sd", "blk_n"]))
        print(f"\n   wrote {fn}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--snap", type=int, required=True)
    s.add_argument("--render", type=int, default=None)
    s.add_argument("--src", default=SRC)
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)   # mock tests only
    r = sp.add_parser("report")
    r.add_argument("--out", default=os.environ.get("BASE", OUT))
    r.add_argument("--data", default="data")
    f = sp.add_parser("fit")
    f.add_argument("--snaps", type=int, nargs="*", default=None)
    f.add_argument("--out", default=os.environ.get("BASE", OUT))
    f.add_argument("--seed", type=int, default=2026)
    f.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)   # mock tests only
    a = p.parse_args()
    if a.cmd == "scan" and a.render is None and a.snap not in SNAP_RENDER:
        sys.exit(f"no render known for snapshot {a.snap}: pass --render")
    {"scan": scan, "report": report, "fit": fit}[a.cmd](a)


if __name__ == "__main__":
    main()
