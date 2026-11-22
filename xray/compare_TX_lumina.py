"""
compare_TX_lumina.py
====================
Does Lumina's neutral IGM have the same temperature distribution at fixed density as the 50 cMpc X-ray test run
(SonjaXrayPaper withXrays, Lumina's physics and gas resolution)? Step 3 of the post-processed "Lumina without X-rays"
(user, 2026-11-20; xray/NOTES.md). The own-rank quantile map of pair_cells_xray.py fit gives, in every 0.05-dex bin of
1+delta, exactly the 50 cMpc box's no-X-ray T distribution; the cell's T_X only sets WHERE each T_noX goes. So the
global error of the map applied to Lumina is the difference between Lumina's and the 50 cMpc box's T_noX | 1+delta,
which cannot be measured; what can be measured is T_X | 1+delta (heating and environment mix) in both, here.

Same selection and binning in both boxes: non-star-forming cells with x_HII < 1e-3 (the X-ray run's own state),
H I weight X m (1 - x_HII), 1+delta = rho / (Omega_b rho_crit) (both from the header cosmology; Lumina's from the
scan files' rho_mean), T = (gamma-1) u mu m_p / k, mu = 4/(1 + 3X + 4X Ne), Ne = HII + HeII + 2 HeIII. Star-forming:
Lumina's scan files carry the render convention (x_HII = f_HeII = f_HeIII = 0 exactly for Sfr > 0); the 50 cMpc run
has StarFormationRate (the count of non-SF cells with all three fractions exactly 0 is printed as a check).

  lumina --snap N : the cells of the 64 random cubes (16^3 of the 320^3 pixels = 16.9 cMpc/h on a side) saved by
                    test_cells_Tb.py --line H in cell_test_H_<N>/scan_H_<NNN>_*.hdf5 (read only); one unit per cube
  pair --snap N   : all gas cells of both 50 cMpc runs (33.8 cMpc/h); one unit per octant (16.9 cMpc/h, as the
                    Lumina cubes)
  compare         : per class of 1+delta: H I share, T quantiles, <1/T>^-1, <1 - T_gamma/T>; the scatter of the void
                    (1+delta < 1) <1/T>^-1 between units (Lumina cubes vs 50 cMpc octants); the global <1 - T_gamma/T>
                    the own-rank map would give Lumina (Lumina's H I shares x the 50 cMpc per-bin no-X-ray values from
                    pair_fit_NNN.npz) next to the 50 cMpc truth; data/TX_lumina_pair.png.
Per-unit 2-D histograms (H I weight) of log10(1+delta) (0.05 dex, as the fit) x log10 T (0.01 dex) and exact
per-bin sums of w and w/T go to <out>/TX_compare/TX_{lumina,pair_withXrays,pair_withoutXrays}_NNN.npz.

  python compare_TX_lumina.py lumina --snap 48
  python compare_TX_lumina.py pair --snap 48
  python compare_TX_lumina.py compare
  python compare_TX_lumina.py evolution   (T vs z: Lumina, Lumina without X-rays as predicted, the 50 cMpc pair;
                                           data/T_evolution_noX.{png,csv})
  python compare_TX_lumina.py optionb     (option B: Lumina's T minus the pair's X-ray heating at the same rank,
                                           scaled by the exposure ratio from void_heating_test.py when present;
                                           data/T_evolution_optionB.{png,csv})
Inputs read only; outputs only under /orcd/data/mvogelsb/004/chcho.
"""
import argparse
import glob
import os
import re
import sys
import time

import h5py
import numpy as np

from pair_cells_xray import (CBINS, GAMMA_M1, K_B, M_P, OUT, RUNS, SRC, X_H, XNEU, check_out, snap_files)

LUMINA_DATA = "/orcd/data/mvogelsb/004/chcho"
LD = np.arange(-1.5, 3.0001, 0.05)           # log10(1+delta) edges (as pair_cells_xray.FIT_LD)
LT = np.arange(0.0, 7.0001, 0.01)            # log10 T edges
ND, NT = len(LD) - 1, len(LT) - 1
G_CGS, MPC_CM = 6.6743e-8, 3.085678e24
QS = (10, 25, 50, 75, 90)
CLS = (("1+d<1", 0.0, 1.0), ("1-2", 1.0, 2.0), (">2", 2.0, np.inf), ("all", 0.0, np.inf))


class Acc:
    """per-unit H I-weighted histograms and sums"""
    def __init__(self, nunit):
        self.h = np.zeros((nunit, ND, NT))
        self.sw = np.zeros((nunit, ND))
        self.swinv = np.zeros((nunit, ND))
        # per unit: H I of all cells, of star-forming cells, of non-SF cells with x >= XNEU, n cells, n neutral IGM,
        #           n clipped in 1+delta, n clipped in T, n non-SF cells with all fractions exactly 0
        self.extra = np.zeros((nunit, 8))

    def add(self, u, one, T, w, sf, x, zero_frac):
        hi = X_H * w                              # w = m (1 - x) here; H I mass = X m (1 - x)
        e = self.extra[u]
        e[0] += hi.sum()
        e[1] += hi[sf].sum()
        ion = (~sf) & (x >= XNEU)
        e[2] += hi[ion].sum()
        e[3] += len(w)
        s = (~sf) & (x < XNEU)
        e[4] += s.sum()
        e[7] += np.sum((~sf) & zero_frac)
        ld, lt, ws, Ts = np.log10(one[s]), np.log10(T[s]), hi[s], T[s]
        il = np.searchsorted(LD, ld, side="right") - 1
        it = np.searchsorted(LT, lt, side="right") - 1
        e[5] += np.sum((il < 0) | (il >= ND))
        e[6] += np.sum((it < 0) | (it >= NT))
        il = np.clip(il, 0, ND - 1)
        it = np.clip(it, 0, NT - 1)
        self.h[u] += np.bincount(il * NT + it, ws, ND * NT).reshape(ND, NT)
        self.sw[u] += np.bincount(il, ws, ND)
        self.swinv[u] += np.bincount(il, ws / Ts, ND)

    def save(self, fn, **meta):
        np.savez(fn, h=self.h.astype(np.float32), sw=self.sw, swinv=self.swinv, extra=self.extra, ld=LD, lt=LT,
                 **meta)


def rho_crit_b(hd, par):
    """Omega_b rho_crit,0 in comoving code density units (as test_cells_Tb.Phys)"""
    get = lambda k, d=None: hd.get(k, par.get(k, d))
    h, ob = float(get("HubbleParam")), float(get("OmegaBaryon"))
    UL, UM = float(get("UnitLength_in_cm", 3.085678e21)), float(get("UnitMass_in_g", 1.989e43))
    H0 = 100.0 * h * 1e5 / MPC_CM
    return ob * 3.0 * H0**2 / (8.0 * np.pi * G_CGS) / (UM * h**2 / UL**3), h, ob


def temperature(u, ne, uv):
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * ne)
    return GAMMA_M1 * u * uv**2 * mu * M_P / K_B


# ---------------------------------------------------------------------------------------------------------------------
def lumina(a):
    d = a.dir or os.path.join(LUMINA_DATA, f"cell_test_H_{a.snap}")
    files = sorted(f for f in glob.glob(os.path.join(d, f"scan_H_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        sys.exit(f"no scan_H_{a.snap:03d}_*.hdf5 in {d}")
    with h5py.File(files[0], "r") as f:
        A = dict(f.attrs)
    for k in ("z", "rho_mean", "n_cube", "cube_side"):
        if k not in A:
            sys.exit(f"STOP: {files[0]} has no attribute {k}")
    if str(A.get("line", "H")) != "H":
        sys.exit(f"{files[0]} is line {A['line']}, not H")
    if "ntask" in A and len(files) != int(A["ntask"]):
        sys.exit(f"{len(files)} scan files, the scan had ntask = {int(A['ntask'])}: incomplete")
    z, rho_b, ncube = float(A["z"]), float(A["rho_mean"]), int(A["n_cube"])
    for fn in files[1:]:
        with h5py.File(fn, "r") as f:
            for k in ("z", "rho_mean", "n_cube", "cube_side", "seed"):
                if k in A and not np.allclose(f.attrs.get(k), A[k]):
                    sys.exit(f"{fn}: {k} = {f.attrs.get(k)} differs from {files[0]} ({A[k]})")
    od = os.path.join(a.out, "TX_compare")
    check_out(od, a.allow_any_out)
    print(f"Lumina snapshot {a.snap}: z = {z:.5f}, {len(files)} scan files, {ncube} cubes of {int(A['cube_side'])}^3 of "
          f"the 320^3 pixels; rho_mean (Omega_b rho_crit, code) {rho_b:.6e}", flush=True)
    acc = Acc(ncube)
    t0 = time.time()
    msum = np.zeros(ncube); vsum = np.zeros(ncube)
    for q in range(ncube):
        parts = []
        for fn in files:
            with h5py.File(fn, "r") as f:
                g = f.get(f"cube_{q:03d}")
                if g is not None and g["m"].shape[0] > 0:
                    parts.append({k: g[k][()].astype(np.float64) for k in ("m", "rho", "V", "T", "xHII", "fII", "fIII")})
        if not parts:
            sys.exit(f"STOP: cube {q} has no cells")
        c = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
        zero = (c["xHII"] == 0) & (c["fII"] == 0) & (c["fIII"] == 0)
        sf = zero                                   # render convention of the scan files (Sfr > 0 -> all 0)
        acc.add(q, c["rho"] / rho_b, c["T"], c["m"] * (1.0 - c["xHII"]), sf, c["xHII"], np.zeros(len(sf), bool))
        msum[q], vsum[q] = c["m"].sum(), c["V"].sum()
        print(f"  cube {q:2d}: {len(c['m']):,} cells, <1+delta>_V {msum[q] / vsum[q] / rho_b:.4f}, neutral-IGM cells "
              f"{int(acc.extra[q, 4]):,}, void <1/T>^-1 {void_hm(acc, q):.2f} K ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"TX_lumina_{a.snap:03d}.npz")
    acc.save(fn, z=z, snap=a.snap, box="lumina", rho_mean=rho_b, unit_mass=msum, unit_vol=vsum,
             side=float(A["cube_side"]) / 320.0, src=d)
    print(f"wrote {fn}; clipped in 1+delta {int(acc.extra[:, 5].sum()):,}, in T {int(acc.extra[:, 6].sum()):,} cells")


def void_hm(acc, u):
    k = LD[1:] <= 0.0 + 1e-9                         # 1+delta < 1
    return acc.sw[u, k].sum() / acc.swinv[u, k].sum() if acc.swinv[u, k].sum() > 0 else np.nan


def pair(a):
    od = os.path.join(a.out, "TX_compare")
    check_out(od, a.allow_any_out)
    for run in RUNS:
        fs = snap_files(a.src, run, a.snap)
        with h5py.File(fs[0], "r") as f:
            hd = dict(f["Header"].attrs)
            par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
        z, box = float(hd["Redshift"]), float(hd["BoxSize"])
        uv = float(hd.get("UnitVelocity_in_cm_per_s", par.get("UnitVelocity_in_cm_per_s", 1e5)))
        for k in ("HubbleParam", "OmegaBaryon"):
            if k not in hd and k not in par:
                sys.exit(f"STOP: {k} not in the header or parameters of {fs[0]}")
        rho_b, h, ob = rho_crit_b(hd, par)
        print(f"{run} snapshot {a.snap}: z = {z:.5f}, BoxSize {box}, h {h}, Omega_b {ob}, {len(fs)} files; "
              f"rho_mean (Omega_b rho_crit, code) {rho_b:.6e}", flush=True)
        acc = Acc(8)
        msum, vsum = np.zeros(8), np.zeros(8)
        for p in fs:
            with h5py.File(p, "r") as f:
                if "PartType0" not in f or f["PartType0"]["Masses"].shape[0] == 0:
                    continue
                g = f["PartType0"]
                if "IntCoordinates" in g:
                    ip = g["IntCoordinates"][()]
                    if ip.dtype != np.uint32:
                        sys.exit(f"STOP: IntCoordinates is {ip.dtype}")
                    oc = ((ip >> np.uint32(31)).astype(np.int64) * np.array([4, 2, 1])).sum(1)
                else:
                    pos = np.mod(g["Coordinates"][()].astype(np.float64), box)
                    oc = ((pos >= 0.5 * box).astype(np.int64) * np.array([4, 2, 1])).sum(1)
                m = g["Masses"][()].astype(np.float64)
                rho = g["Density"][()].astype(np.float64)
                x = g["HII_Fraction"][()].astype(np.float64)
                fII = g["HeII_Fraction"][()].astype(np.float64) if "HeII_Fraction" in g else np.zeros_like(x)
                fIII = g["HeIII_Fraction"][()].astype(np.float64) if "HeIII_Fraction" in g else np.zeros_like(x)
                ne = g["ElectronAbundance"][()].astype(np.float64) if "ElectronAbundance" in g else x + fII + 2.0 * fIII
                T = temperature(g["InternalEnergy"][()].astype(np.float64), ne, uv)
                sf = (g["StarFormationRate"][()] > 0) if "StarFormationRate" in g else np.zeros(len(m), bool)
                zero = (x == 0) & (fII == 0) & (fIII == 0)
                for u in np.unique(oc):
                    s = oc == u
                    acc.add(u, rho[s] / rho_b, T[s], m[s] * (1.0 - x[s]), sf[s], x[s], zero[s])
                    msum[u] += m[s].sum(); vsum[u] += (m[s] / rho[s]).sum()
        os.makedirs(od, exist_ok=True)
        fn = os.path.join(od, f"TX_pair_{run}_{a.snap:03d}.npz")
        acc.save(fn, z=z, snap=a.snap, box=run, rho_mean=rho_b, unit_mass=msum, unit_vol=vsum, side=0.5, src=a.src)
        print(f"  octants: <1+delta>_V " + " ".join(f"{msum[u] / vsum[u] / rho_b:.4f}" for u in range(8)))
        print(f"  octants: void <1/T>^-1 [K] " + " ".join(f"{void_hm(acc, u):.2f}" for u in range(8)))
        print(f"  box <1+delta> (gas mass / volume / Omega_b rho_crit) {msum.sum() / vsum.sum() / rho_b:.5f}; "
              f"non-SF cells with x_HII = f_HeII = f_HeIII = 0 exactly: {int(acc.extra[:, 7].sum()):,}; clipped in "
              f"1+delta {int(acc.extra[:, 5].sum()):,}, in T {int(acc.extra[:, 6].sum()):,}")
        print(f"wrote {fn}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def stats(D, units=None, lo=0.0, hi=np.inf, Tg=None):
    """H I share, T quantiles, <1/T>^-1, <1 - Tg/T> for 1+delta in [lo, hi) (bin centres), summed over units"""
    u = slice(None) if units is None else units
    c = 0.5 * (LD[1:] + LD[:-1])
    k = (10**c >= lo) & (10**c < hi)
    h = D["h"][u].sum(0)[k].sum(0).astype(np.float64)
    sw = D["sw"][u].sum(0)[k].sum()
    si = D["swinv"][u].sum(0)[k].sum()
    if sw <= 0:
        return dict(share=0.0, q=[np.nan] * len(QS), hm=np.nan, a=np.nan)
    cdf = np.r_[0.0, np.cumsum(h)] / h.sum()
    q = [10**np.interp(p / 100.0, cdf, LT) for p in QS]
    return dict(share=sw / D["sw"][u].sum(), q=q, hm=sw / si, a=1.0 - Tg * si / sw)


def compare(a):
    od = os.path.join(a.out, "TX_compare")
    fl = sorted(glob.glob(os.path.join(od, "TX_lumina_*.npz")))
    if not fl:
        sys.exit(f"no TX_lumina_*.npz in {od}")
    figdat = []
    for f in fl:
        snap = int(re.search(r"_(\d+)\.npz$", f).group(1))
        fx = os.path.join(od, f"TX_pair_withXrays_{snap:03d}.npz")
        fn = os.path.join(od, f"TX_pair_withoutXrays_{snap:03d}.npz")
        if not os.path.exists(fx):
            print(f"\nsnapshot {snap}: no {os.path.basename(fx)}, skipped")
            continue
        L, X = np.load(f), np.load(fx)
        Nx = np.load(fn) if os.path.exists(fn) else None
        z = float(L["z"])
        if abs(z - float(X["z"])) > 1e-3:
            sys.exit(f"STOP: snapshot {snap}: Lumina z {z} vs 50 cMpc z {float(X['z'])}")
        Tg = 2.7255 * (1.0 + z)
        print(f"\nsnapshot {snap}: z = {z:.4f} (50 cMpc {float(X['z']):.4f}), T_gamma = {Tg:.2f} K")
        for lab, D in (("Lumina cubes", L), ("50 cMpc X", X)) + ((("50 cMpc no X", Nx),) if Nx is not None else ()):
            e = D["extra"].sum(0)
            print(f"   {lab:13s}: {int(e[3]):,} cells ({len(D['sw'])} units), <1+delta> of the units' gas "
                  f"{D['unit_mass'].sum() / D['unit_vol'].sum() / float(D['rho_mean']):.4f}; H I shares: neutral IGM "
                  f"{D['sw'].sum() / e[0]:.4f}, star-forming {e[1] / e[0]:.4f}, non-SF with x_HII >= {XNEU:g} "
                  f"{e[2] / e[0]:.4f}; clipped 1+delta {int(e[5]):,}, T {int(e[6]):,}; non-SF with all fractions 0 "
                  f"{int(e[7]):,}")
        print("   neutral IGM by 1+delta (H I-weighted):         H I share | T p10 p25 p50 p75 p90 [K] | <1/T>^-1 [K] | "
              "<1-Tg/T>")
        rows = []
        for lo, hi in list(zip(CBINS[:-1], CBINS[1:])) + [(0.0, np.inf)]:
            for lab, D in (("Lumina", L), ("50 cMpc X", X)) + ((("50 cMpc no X", Nx),) if Nx is not None else ()):
                s = stats(D, None, lo, hi, Tg)
                rows.append((lo, hi, lab, s))
                print(f"   {lo:5g}-{hi:<5g} {lab:13s}: {s['share']:8.4f} | " + " ".join(f"{v:8.2f}" for v in s["q"])
                      + f" | {s['hm']:9.2f} | {s['a']:8.3f}")
        # bin-by-bin in 1+delta = 0.3-3 (0.05 dex): Lumina / 50 cMpc of the median T and of <1/T>^-1
        c = 0.5 * (LD[1:] + LD[:-1])
        k = (10**c >= 0.3) & (10**c < 3.0)
        hl, hx = L["h"].sum(0)[k].astype(np.float64), X["h"].sum(0)[k].astype(np.float64)
        ok = (hl.sum(1) > 0) & (hx.sum(1) > 0)
        medr = []
        for i in np.flatnonzero(ok):
            ml = 10**np.interp(0.5, np.r_[0, np.cumsum(hl[i])] / hl[i].sum(), LT)
            mx = 10**np.interp(0.5, np.r_[0, np.cumsum(hx[i])] / hx[i].sum(), LT)
            medr.append(ml / mx)
        hmr = (L["sw"].sum(0)[k] / L["swinv"].sum(0)[k]) / (X["sw"].sum(0)[k] / X["swinv"].sum(0)[k])
        hmr = hmr[ok]
        print(f"   0.05-dex bins in 1+delta = 0.3-3: Lumina / 50 cMpc median T_X {np.min(medr):.3f}-{np.max(medr):.3f} "
              f"(H I-weighted mean {np.average(medr, weights=L['sw'].sum(0)[k][ok]):.3f}); <1/T>^-1 "
              f"{np.min(hmr):.3f}-{np.max(hmr):.3f} (mean {np.average(hmr, weights=L['sw'].sum(0)[k][ok]):.3f})")
        # unit-to-unit scatter of the void <1/T>^-1 (16.9 cMpc/h units in both boxes)
        def spread(D):
            v = np.array([void_hm(D_acc(D), u) for u in range(len(D["sw"]))])
            v = v[np.isfinite(v)]
            return v, np.percentile(v, [0, 16, 50, 84, 100])
        vl, pl = spread(L)
        vx, px = spread(X)
        print(f"   void (1+delta < 1) <1/T>^-1 per 16.9 cMpc/h unit [K]: Lumina {len(vl)} cubes min/p16/p50/p84/max "
              + "/".join(f"{v:.2f}" for v in pl) + f"; 50 cMpc X {len(vx)} octants " + "/".join(f"{v:.2f}" for v in px)
              + f"; whole boxes {stats(L, None, 0, 1, Tg)['hm']:.2f} / {stats(X, None, 0, 1, Tg)['hm']:.2f}")
        if Nx is not None:
            vn, pn = spread(Nx)
            print(f"   50 cMpc no X octants: " + "/".join(f"{v:.2f}" for v in pn)
                  + f"; corr(void <1/T>^-1 X, no X) over the octants {np.corrcoef(vx, vn)[0, 1]:.3f}")
        # what the own-rank map gives Lumina: Lumina's H I per 0.05-dex bin x the 50 cMpc no-X-ray <1 - Tg/T> per bin
        ff = os.path.join(od, "..", "pair_cells", f"pair_fit_{snap:03d}.npz")
        if os.path.exists(ff):
            F = np.load(ff)
            if not np.allclose(F["ld_edges"], LD):
                sys.exit("STOP: pair_fit 1+delta bins differ from these")
            QN = F["QN_logT_noX"]
            okq = np.all(np.isfinite(QN), axis=1)
            a_bin = np.where(okq, np.mean(1.0 - Tg / 10**np.nan_to_num(QN), axis=1), np.nan)   # equal-weight levels
            near = np.flatnonzero(okq)
            fill = near[np.abs(np.arange(ND)[:, None] - near[None, :]).argmin(1)]
            a_bin = a_bin[fill]
            print("   own-rank map, <1 - Tg/T> of the neutral IGM = sum over 0.05-dex bins of the H I share x the 50 cMpc "
                  "no-X-ray per-bin value (pair_fit tables, pairs selected neutral without X-rays):")
            for lab, lo, hi in CLS:
                kk = (10**c >= lo) & (10**c < hi)
                wl, wx = L["sw"].sum(0) * kk, X["sw"].sum(0) * kk
                pl_ = np.sum(wl * a_bin) / wl.sum()
                px_ = np.sum(wx * a_bin) / wx.sum()
                cub = np.array([np.sum(L["sw"][u] * kk * a_bin) / np.sum(L["sw"][u] * kk) for u in range(len(L["sw"]))
                                if np.sum(L["sw"][u] * kk) > 0])
                print(f"     {lab:6s}: Lumina {pl_:8.3f} (cubes p16-p84 {np.percentile(cub, 16):.3f} to "
                      f"{np.percentile(cub, 84):.3f}); 50 cMpc H I shares {px_:8.3f}; with X-rays Lumina "
                      f"{stats(L, None, lo, hi, Tg)['a']:8.3f}, 50 cMpc {stats(X, None, lo, hi, Tg)['a']:8.3f}")
        else:
            print(f"   (no {ff}: run pair_cells_xray.py fit for the predicted no-X-ray values)")
        figdat.append((snap, z, L, X, Nx))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    fig, ax = plt.subplots(len(figdat), 2, figsize=(10, 3.6 * len(figdat)), squeeze=False)
    c = 0.5 * (LD[1:] + LD[:-1])
    for r, (snap, z, L, X, Nx) in enumerate(figdat):
        vv = np.array([void_hm(D_acc(D), u) for D in (L, X) for u in range(len(D["sw"]))])
        vb = np.geomspace(np.nanmin(vv) * 0.9, np.nanmax(vv) * 1.1, 31)
        for D, lab, col in ((L, "Lumina cubes", "#2a6fb0"), (X, "50 cMpc, X-rays", "#c4572a"),
                            (Nx, "50 cMpc, no X-rays", "#777777")):
            if D is None:
                continue
            h = D["h"].sum(0).astype(np.float64)
            ok = h.sum(1) > h.sum() * 1e-5
            qq = np.array([[10**np.interp(p / 100, np.r_[0, np.cumsum(h[i])] / h[i].sum(), LT) for p in (10, 50, 90)]
                           for i in np.flatnonzero(ok)])
            x = 10**c[ok]
            ax[r, 0].plot(x, qq[:, 1], color=col, label=lab)
            ax[r, 0].fill_between(x, qq[:, 0], qq[:, 2], color=col, alpha=0.15)
            if D is not Nx:                          # the void scatter: Lumina cubes vs 50 cMpc X-ray octants
                v = np.array([void_hm(D_acc(D), u) for u in range(len(D["sw"]))])
                ax[r, 1].hist(v[np.isfinite(v)], bins=vb, color=col, alpha=0.5, density=True, label=f"{lab} ({len(v)})")
        ax[r, 0].set(xscale="log", yscale="log", xlabel=r"$1+\delta$", ylabel="T [K] (median, 10-90%)",
                     title=f"neutral IGM, z = {z:.2f}")
        ax[r, 1].set(xscale="log", xlabel=r"void ($1+\delta<1$) $\langle 1/T\rangle^{-1}$ per 16.9 cMpc/h unit [K]",
                     ylabel="density", title="unit-to-unit scatter")
        ax[r, 0].legend(fontsize=7); ax[r, 1].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(a.data, exist_ok=True)
    fig.savefig(os.path.join(a.data, "TX_lumina_pair.png"), dpi=200)
    print(f"\nwrote {a.data}/TX_lumina_pair.png")


def predict_noX(wl, N, Tg):
    """Own-rank prediction from 1+delta bins: weights wl (H I per 0.05-dex bin of the target) x the 50 cMpc no-X-ray
    per-bin T distribution N (npz of TX_pair_withoutXrays). Empty bins of N borrow the nearest filled bin.
    Returns <1/T>^-1, <1 - Tg/T>, T quantiles QS."""
    swN, siN, hN = N["sw"].sum(0), N["swinv"].sum(0), N["h"].sum(0).astype(np.float64)
    ok = (swN > 0) & (hN.sum(1) > 0)
    near = np.flatnonzero(ok)
    fill = near[np.abs(np.arange(ND)[:, None] - near[None, :]).argmin(1)]
    r = (siN / np.where(ok, swN, 1.0))[fill]                       # <1/T> per bin
    cond = (hN / np.where(ok, hN.sum(1), 1.0)[:, None])[fill]      # T distribution per bin
    W = wl.sum()
    if W <= 0:
        return np.nan, np.nan, [np.nan] * len(QS)
    inv = np.sum(wl * r) / W
    h = (wl[:, None] * cond).sum(0)
    cdf = np.r_[0.0, np.cumsum(h)] / h.sum()
    return 1.0 / inv, 1.0 - Tg * inv, [10**np.interp(p / 100.0, cdf, LT) for p in QS]


def evolution(a):
    """T evolution of the neutral IGM: Lumina (X-rays), Lumina without X-rays as predicted by the own-rank map
    (Lumina's H I per 0.05-dex bin of 1+delta x the 50 cMpc no-X-ray T distribution in that bin), and the 50 cMpc pair.
    The same prediction applied to the 50 cMpc X run's H I shares is printed against its true no-X-ray run (check)."""
    od = os.path.join(a.out, "TX_compare")
    snaps = sorted(int(re.search(r"_(\d+)\.npz$", f).group(1)) for f in glob.glob(os.path.join(od, "TX_lumina_*.npz")))
    snaps = [s for s in snaps if all(os.path.exists(os.path.join(od, f"TX_{k}_{s:03d}.npz"))
                                     for k in ("pair_withXrays", "pair_withoutXrays"))]
    if not snaps:
        sys.exit(f"no complete snapshot (Lumina + both pair runs) in {od}")
    c = 0.5 * (LD[1:] + LD[:-1])
    ECL = (("all neutral IGM", 0.0, np.inf), ("voids, 1+delta < 1", 0.0, 1.0), ("1+delta = 1-2", 1.0, 2.0))
    rows = []
    for s in snaps:
        L, X, N = (np.load(os.path.join(od, f"TX_{k}_{s:03d}.npz")) for k in ("lumina", "pair_withXrays",
                                                                              "pair_withoutXrays"))
        z = float(L["z"])
        Tg = 2.7255 * (1.0 + z)
        print(f"\nsnapshot {s}: z = {z:.3f}, T_gamma = {Tg:.2f} K   (<1/T>^-1 [K] | median T [K] (p10-p90) | <1-Tg/T>)")
        for lab, lo, hi in ECL:
            k = (10**c >= lo) & (10**c < hi)
            sL, sX, sN = (stats(D, None, lo, hi, Tg) for D in (L, X, N))
            pL = predict_noX(L["sw"].sum(0) * k, N, Tg)
            pX = predict_noX(X["sw"].sum(0) * k, N, Tg)
            cub = np.array([predict_noX(L["sw"][u] * k, N, Tg)[0] for u in range(len(L["sw"]))])
            cub = cub[np.isfinite(cub)]
            print(f"  {lab}:")
            for nm, hm, aa, q in (("Lumina, X-rays", sL["hm"], sL["a"], sL["q"]),
                                  ("Lumina, no X-rays (pred.)", pL[0], pL[1], pL[2]),
                                  ("50 cMpc, X-rays", sX["hm"], sX["a"], sX["q"]),
                                  ("50 cMpc, no X-rays", sN["hm"], sN["a"], sN["q"]),
                                  ("  check: pred. from 50 cMpc X", pX[0], pX[1], pX[2])):
                print(f"    {nm:30s} {hm:9.2f} | {q[2]:9.2f} ({q[0]:8.2f}-{q[4]:9.2f}) | {aa:8.3f}")
            print(f"    Lumina no X-rays (pred.), cube-to-cube <1/T>^-1 p16-p84: {np.percentile(cub, 16):.2f}-"
                  f"{np.percentile(cub, 84):.2f} K (from the cubes' density mix only)")
            for nm, hm, aa, q in (("lumina_X", sL["hm"], sL["a"], sL["q"]), ("lumina_noX_pred", pL[0], pL[1], pL[2]),
                                  ("pair_X", sX["hm"], sX["a"], sX["q"]), ("pair_noX", sN["hm"], sN["a"], sN["q"]),
                                  ("pair_noX_pred", pX[0], pX[1], pX[2])):
                rows.append((s, z, lab, nm, hm, aa, *q))
    os.makedirs(a.data, exist_ok=True)
    fc = os.path.join(a.data, "T_evolution_noX.csv")
    with open(fc, "w") as f:
        f.write("snap,z,class,case,hm_K,one_minus_Tg_over_T," + ",".join(f"T_p{p}" for p in QS) + "\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]},{r[3]}," + ",".join(f"{v:.5g}" for v in r[4:]) + "\n")
    print(f"\nwrote {fc}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    # drawn in this order: the predicted no-X-ray Lumina last and on top (it lies within ~1% of the 50 cMpc no-X run)
    sty = {"pair_noX": dict(color="#777777", ls="-", lw=2.5, marker="s", ms=5, label="50 cMpc, no X-rays"),
           "pair_X": dict(color="#c4572a", ls="-", marker="s", ms=4, label="50 cMpc, X-rays"),
           "lumina_X": dict(color="#2a6fb0", ls="-", marker="o", label="Lumina, X-rays"),
           "lumina_noX_pred": dict(color="#1b9e77", ls="--", lw=1.6, marker="D", ms=6, mfc="none", mew=1.4, zorder=5,
                                   label="Lumina, no X-rays (predicted)")}
    fig, ax = plt.subplots(3, len(ECL), figsize=(4.0 * len(ECL), 9.6), squeeze=False,
                           gridspec_kw={"height_ratios": [1, 1, 0.55]})
    zz = np.array(sorted({r[1] for r in rows}))
    get = lambda lab, nm: sorted((r for r in rows if r[2] == lab and r[3] == nm), key=lambda r: r[1])
    for j, (lab, lo, hi) in enumerate(ECL):
        for nm, st in sty.items():
            R = get(lab, nm)
            zr = np.array([r[1] for r in R])
            ax[0, j].plot(zr, [r[4] for r in R], **st)
            ax[1, j].plot(zr, [r[8] for r in R], **st)
            if nm in ("lumina_X", "pair_noX"):
                ax[1, j].fill_between(zr, [r[6] for r in R], [r[10] for r in R], color=st["color"], alpha=0.10)
        zg = np.linspace(zz.min(), zz.max(), 50)
        for i in range(2):
            ax[i, j].plot(zg, 2.7255 * (1 + zg), color="k", lw=0.8, ls=":", label=r"$T_\gamma$")
            ax[i, j].set(yscale="log", xlabel="z")
        # ratio panel: predicted no-X Lumina / 50 cMpc no-X run (and the check on the pair)
        base = {r[1]: r[4] for r in get(lab, "pair_noX")}
        for nm, col, mk, lb in (("lumina_noX_pred", "#1b9e77", "D", "Lumina no X (pred.) / 50 cMpc no X"),
                                ("pair_noX_pred", "#777777", "s", "check: pred. from 50 cMpc X / 50 cMpc no X")):
            R = get(lab, nm)
            ax[2, j].plot([r[1] for r in R], [r[4] / base[r[1]] for r in R], color=col, marker=mk, mfc="none", label=lb)
        ax[2, j].axhline(1.0, color="k", lw=0.6)
        ax[2, j].set(xlabel="z", ylabel=r"ratio of $\langle 1/T\rangle^{-1}$")
        for i in range(3):
            ax[i, j].invert_xaxis()
        ax[0, j].set(title=lab, ylabel=r"$\langle 1/T\rangle^{-1}$ of the H I [K]")
        ax[1, j].set(ylabel="median T [K] (band: 10-90%)")
    ax[2, 0].legend(fontsize=7)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("neutral IGM (non-star-forming, x_HII < 1e-3), gas cells, H I-weighted; "
                 "no-X-ray Lumina = own-rank map from the 50 cMpc pair", fontsize=9)
    fig.tight_layout()
    fp = os.path.join(a.data, "T_evolution_noX.png")
    fig.savefig(fp, dpi=200)
    print(f"wrote {fp}")


def option_b(L, X, N, Tg, s, lo, hi, tfloor=5.0, nl=1000):
    """Option B (rank-matched subtraction of the pair's X-ray heating): in every 0.05-dex bin of 1+delta,
    T_noX,L(p) = max(tfloor, T_X,L(p) - s [T_X,pair(p) - T_noX,pair(p)]) at quantile p of the H I-weighted T
    distribution; s = Lumina / pair X-ray exposure. Lumina's H I per bin as weight; empty pair bins borrow the nearest
    filled bin. Returns <1/T>^-1, <1 - Tg/T>, T quantiles QS, H I share clipped at the floor."""
    c = 0.5 * (LD[1:] + LD[:-1])
    k = (10**c >= lo) & (10**c < hi)
    p = (np.arange(nl) + 0.5) / nl
    hL, hX, hN = (D["h"].sum(0).astype(np.float64) for D in (L, X, N))
    okP = (hX.sum(1) > 0) & (hN.sum(1) > 0)
    near = np.flatnonzero(okP)
    fill = near[np.abs(np.arange(ND)[:, None] - near[None, :]).argmin(1)]
    wl = L["sw"].sum(0) * k
    Q = lambda h: 10**np.interp(p, np.r_[0.0, np.cumsum(h)] / h.sum(), LT)
    Ts, ws = [], []
    clip = 0.0
    for i in np.flatnonzero((wl > 0) & (hL.sum(1) > 0)):
        j = fill[i]
        TB = Q(hL[i]) - s * (Q(hX[j]) - Q(hN[j]))
        clip += wl[i] * np.mean(TB < tfloor)
        Ts.append(np.maximum(TB, tfloor))
        ws.append(np.full(nl, wl[i] / nl))
    if not Ts:
        return np.nan, np.nan, [np.nan] * len(QS), np.nan
    T, w = np.concatenate(Ts), np.concatenate(ws)
    inv = np.sum(w / T) / w.sum()
    o = np.argsort(T)
    cw = np.cumsum(w[o]) / w.sum()
    q = [float(np.interp(pp / 100.0, cw, T[o])) for pp in QS]
    return 1.0 / inv, 1.0 - Tg * inv, q, clip / w.sum()


def exposure_ratio(out, snap, b=5):
    """Lumina / pair mean X-ray exposure in RT bin b (500-2000 eV) from void_heating_test.py, or nan"""
    fl = os.path.join(out, "void_heat", f"void_heat_lumina_{snap:03d}.npz")
    fp = os.path.join(out, "void_heat", f"void_heat_pair_{snap:03d}.npz")
    if not (os.path.exists(fl) and os.path.exists(fp)):
        return np.nan
    return float(np.load(fl)["E"][:, b].mean() / np.load(fp)["E"][:, b].mean())


def optionb(a):
    """Option B next to the current own-rank model (option 3) for Lumina's neutral IGM, per snapshot and class"""
    od = os.path.join(a.out, "TX_compare")
    snaps = sorted(int(re.search(r"_(\d+)\.npz$", f).group(1)) for f in glob.glob(os.path.join(od, "TX_lumina_*.npz")))
    snaps = [s for s in snaps if all(os.path.exists(os.path.join(od, f"TX_{k}_{s:03d}.npz"))
                                     for k in ("pair_withXrays", "pair_withoutXrays"))]
    if not snaps:
        sys.exit(f"no complete snapshot (Lumina + both pair runs) in {od}")
    c = 0.5 * (LD[1:] + LD[:-1])
    ECL = (("all neutral IGM", 0.0, np.inf), ("voids, 1+delta < 1", 0.0, 1.0), ("1+delta = 1-2", 1.0, 2.0))
    rows = []
    for s in snaps:
        L, X, N = (np.load(os.path.join(od, f"TX_{k}_{s:03d}.npz")) for k in ("lumina", "pair_withXrays",
                                                                              "pair_withoutXrays"))
        z = float(L["z"])
        Tg = 2.7255 * (1.0 + z)
        sE = exposure_ratio(a.out, s)
        print(f"\nsnapshot {s}: z = {z:.3f}, T_gamma = {Tg:.2f} K; Lumina / pair X-ray exposure (500-2000 eV) "
              f"{sE:.3f}" + ("" if np.isfinite(sE) else " (no void_heat files: only s = 1)")
              + "   (<1/T>^-1 [K] | median T (p10-p90) [K] | <1-Tg/T> | H I share clipped at 5 K)")
        for lab, lo, hi in ECL:
            k = (10**c >= lo) & (10**c < hi)
            sL, sN = stats(L, None, lo, hi, Tg), stats(N, None, lo, hi, Tg)
            p3 = predict_noX(L["sw"].sum(0) * k, N, Tg)
            cases = [("Lumina, X-rays", sL["hm"], sL["a"], sL["q"], np.nan, "lumina_X"),
                     ("option 3 (current)", p3[0], p3[1], p3[2], np.nan, "option3"),
                     ("option B, s = 1", *option_b(L, X, N, Tg, 1.0, lo, hi), "optionB_s1")]
            if np.isfinite(sE):
                cases.append((f"option B, s = {sE:.3f}", *option_b(L, X, N, Tg, sE, lo, hi), "optionB_sE"))
            cases += [("50 cMpc, no X-rays", sN["hm"], sN["a"], sN["q"], np.nan, "pair_noX"),
                      ("  check: B on 50 cMpc X, s = 1", *option_b(X, X, N, Tg, 1.0, lo, hi), "check")]
            print(f"  {lab}:")
            for nm, hm, aa, q, cl, tag in cases:
                print(f"    {nm:30s} {hm:9.2f} | {q[2]:8.2f} ({q[0]:7.2f}-{q[4]:8.2f}) | {aa:8.3f} | "
                      + (f"{cl:.4f}" if np.isfinite(cl) else "   -"))
                rows.append((s, z, lab, tag, hm, aa, *q, cl))
    os.makedirs(a.data, exist_ok=True)
    fc = os.path.join(a.data, "T_evolution_optionB.csv")
    with open(fc, "w") as f:
        f.write("snap,z,class,case,hm_K,one_minus_Tg_over_T," + ",".join(f"T_p{p}" for p in QS) + ",clipped\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]},{r[3]}," + ",".join(f"{v:.5g}" for v in r[4:]) + "\n")
    print(f"\nwrote {fc}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    sty = {"pair_noX": dict(color="#777777", ls="-", lw=2.5, marker="s", ms=5, label="50 cMpc, no X-rays"),
           "lumina_X": dict(color="#2a6fb0", ls="-", marker="o", label="Lumina, X-rays"),
           "option3": dict(color="#1b9e77", ls="--", marker="D", ms=6, mfc="none", mew=1.4, zorder=5,
                           label="Lumina no X: option 3 (current)"),
           "optionB_s1": dict(color="#d95f02", ls="-", marker="^", ms=6, zorder=6, label="Lumina no X: option B, s = 1"),
           "optionB_sE": dict(color="#7570b3", ls=":", marker="v", ms=6, zorder=6,
                              label="Lumina no X: option B, s = exposure ratio")}
    fig, ax = plt.subplots(3, len(ECL), figsize=(4.0 * len(ECL), 9.6), squeeze=False,
                           gridspec_kw={"height_ratios": [1, 1, 0.6]})
    get = lambda lab, tag: sorted((r for r in rows if r[2] == lab and r[3] == tag), key=lambda r: r[1])
    for j, (lab, lo, hi) in enumerate(ECL):
        for tag, st in sty.items():
            R = get(lab, tag)
            if not R:
                continue
            zr = [r[1] for r in R]
            ax[0, j].plot(zr, [r[4] for r in R], **st)
            ax[1, j].plot(zr, [r[5] for r in R], **st)
        base = {r[1]: r[4] for r in get(lab, "option3")}
        for tag in ("optionB_s1", "optionB_sE"):
            R = get(lab, tag)
            if R:
                ax[2, j].plot([r[1] for r in R], [r[4] / base[r[1]] for r in R], **{k_: v for k_, v in sty[tag].items()
                                                                                    if k_ != "label"})
        ax[2, j].axhline(1.0, color="k", lw=0.6)
        zz = np.array(sorted({r[1] for r in rows}))
        ax[0, j].plot(zz, 2.7255 * (1 + zz), color="k", lw=0.8, ls=":", label=r"$T_\gamma$")
        ax[0, j].set(yscale="log", title=lab, ylabel=r"$\langle 1/T\rangle^{-1}$ of the H I [K]")
        ax[1, j].set(ylabel=r"$\langle 1 - T_\gamma/T\rangle$")
        ax[2, j].set(xlabel="z", ylabel=r"option B / option 3, $\langle 1/T\rangle^{-1}$")
        for i in range(3):
            ax[i, j].invert_xaxis()
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Lumina without X-rays, neutral IGM (gas cells, H I-weighted): option 3 (pair's no-X T per 1+delta bin) "
                 "vs option B (Lumina's T minus the pair's X-ray heating at the same rank)", fontsize=8)
    fig.tight_layout()
    fp = os.path.join(a.data, "T_evolution_optionB.png")
    fig.savefig(fp, dpi=200)
    print(f"wrote {fp}")


class D_acc:
    """read-only view of a saved npz as an Acc (for void_hm)"""
    def __init__(self, D):
        self.sw, self.swinv = D["sw"], D["swinv"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("lumina")
    s.add_argument("--snap", type=int, required=True)
    s.add_argument("--dir", default=None, help="scan directory (default <data>/cell_test_H_<snap>)")
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    s = sp.add_parser("pair")
    s.add_argument("--snap", type=int, required=True)
    s.add_argument("--src", default=SRC)
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    s = sp.add_parser("compare")
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--data", default="data")
    s = sp.add_parser("evolution")
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--data", default="data")
    s = sp.add_parser("optionb")
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--data", default="data")
    a = p.parse_args()
    {"lumina": lumina, "pair": pair, "compare": compare, "evolution": evolution, "optionb": optionb}[a.cmd](a)


if __name__ == "__main__":
    main()
