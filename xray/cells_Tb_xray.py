"""
cells_Tb_xray.py
================
21 cm delta T_b of every gas cell of a full snapshot of the X-ray test runs, against the render pixels.
Question (user, 2026-11-16): at z >~ 10 the pixel maps mix cold and warm neutral gas (pixel <1/T>^-1 too high by
x2-7 without X-rays, x1.2-1.9 with X-rays at z = 10; cells_T_xray.py). How much absorption do the gas cells give?
Pilot for Lumina (test_cells_Tb.py --line H): 50 cMpc runs with Lumina's physics and gas resolution, ~2e8 cells.

scan (one run, one snapshot; a batch job):
  cells, with the render conventions (as test_cells_Tb.py and cells_T_xray.py):
    T = (gamma-1) u mu m_p / k_B, mu = 4/(1 + 3X + 4X Ne), Ne = ElectronAbundance (or HII + HeII + 2 HeIII);
    star-forming cells (SFR > 0) neutral: x_HII = He II = He III = 0;
    J_alpha = stars + AGN (128^3 files of run_Jalpha_xray.sh) at the cell's pixel;
    sig = rho (1 - x_HII) / <rho>_gas, n_H = X rho / m_H (proper);
    delta T_b from tb_physics.Line('H', spin_model=1), the physics of the production maps, twice:
      "cell": tau_GP from the cell's own n_HI (tb_physics default);
      "pix":  tau_GP from the n_HI of the cell's 128^3 render pixel (what the production map used there).
    Neither is the cell-level physics: the photons that set S~_a scatter within ~(a tau_GP)^1/3 b ~ 7-24 km/s of line
    centre, a Hubble length of ~40-110 ckpc/h (1-2 void cells, 0.15-0.4 of a 128^3 pixel), and tau_GP = n_HI/(dv/dr)
    with the local velocity gradient, which is not used (Hubble flow only). The two options bracket that choice
    (S~_a changes 13-17% for x3 in tau_GP at 5 K). tau_21 (x_CMB) from the cell's own n_HI, Hubble flow only (as the
    real-space maps), so collapsing gas (dv/dr < H) would have a larger tau_21 than used here.
    Star-forming cells: T from u with the snapshot's Ne (as test_cells_Tb.py), then neutral; negligible for dT_b
    (x_c >> 1 there).
  pixel references, the same physics:
      "pix128": the 128^3 render (mass-weighted T and fractions) = the production map; checked against RIN_<render>.hdf5;
      "pix64":  the render coarsened 2x2x2 to 64^3 (mean density and J, mass-weighted T and fractions): 0.53 cMpc/h,
                Lumina's 640^3 pixel size.
  cell-level maps: each cell at its nearest pixel, volume-weighted (sum V dT_b / sum V), at 128^3 and 64^3.
  like-for-like pixel baseline "pixngp": pixel fields built from the same cells with the same nearest-pixel deposit
      and the same normalisation as the cell map (density = sum m / sum V of the cells in the pixel; T and fractions
      mass-weighted; sig = rho (1 - x)/<rho>_gas), then the same physics (J of the pixel). In the linear limit
      (dT_b proportional to n_HI) the cell map and pixngp are identical pixel by pixel, so cells vs pixngp is the
      sub-pixel nonlinearity at every k. cells vs the render pixels also contains the deposit difference (the render
      deposits each cell as a cube of its own volume): up to ~10-25% in Delta^2 in the top k bins.
  means: sum V dT_b / box^3 (sum V sig = sum m_HI/<rho> exactly, so in the linear limit this equals the pixel mean).
      The printed means of the cell map and of pixngp are plain pixel averages of sum/sum V maps (for the spectra) and
      differ from the volume mean by the cell-count weighting; compare means with the volume mean only.
  caveats: the cells keep the 128^3 J (and, for "pix", the 128^3 n_HI) in the 64^3 comparison (the block J scatter is
      printed); "neutral" = x_HII < 1e-3, so X-ray-warmed partly ionized gas is in the x_HII >= 1e-3 row.
  validation snapshot 066 (z = 8): only the X-ray run is a check (T_S >> T_gamma there, so cells and pixels must agree);
      without X-rays the void cells are still cold (~6-7 K at z = 10), so a cell-pixel difference there is physics.
  writes <out>/<run>/cells_Tb/cells_Tb_<snap>.npz: means, Delta^2(k) and r(k) cells vs pixels, the cell-level
  mean split by cell class / 1+delta / T and into absorbing and emitting gas, diagnostics (S~_a, tau_GP, tau_21,
  x_CMB, share of H I with T_S < T_gamma), and the 128^3 / 64^3 cell maps.
report (after the scans): tables of both runs at every snapshot found, data/cells_Tb_xray.png.

  python cells_Tb_xray.py scan --run withXrays --snap 24 --render 71
  python cells_Tb_xray.py report
Inputs are read only (Oliver's runs); outputs only under /orcd/data/mvogelsb/004/chcho.
"""
import argparse
import glob
import itertools
import os
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from tb_physics import Line, X_H, M_H, SRC as CONST_SRC          # noqa: E402
from cells_T_xray import (SRC, OUT, RUNS, LABELS, COLORS, INK, INK2, GRID, snap_files, pick, check_out,  # noqa: E402
                          GAMMA_M1, K_B, M_P)
from ps_xray import spectra                                          # noqa: E402

XNEU = 1e-3
CBINS = (0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0, np.inf)        # cell (pixel) 1+delta
OPTS = ("cell", "pix")                                              # tau_GP from the cell's / its pixel's n_HI
LGP_EDGES = np.linspace(-2.0, 12.0, 141)                            # log10 tau_GP (H I-weighted histograms)


def tbins(Tg):
    """T bins of the neutral IGM cells [K]: floor, below T_gamma, then decades"""
    return (0.0, 5.5, Tg, 100.0, 1e3, 1e4, np.inf)


def csum8(a):
    """sum over 2x2x2 blocks (N^3 -> (N/2)^3)"""
    n = a.shape[0] // 2
    return a.reshape(n, 2, n, 2, n, 2).sum(axis=(1, 3, 5))


def hdr_value(hd, par, k, default=None):
    if k in hd:
        return float(hd[k])
    if k in par:
        return float(par[k])
    if default is None:
        sys.exit(f"STOP: {k} neither in the snapshot Header nor in Parameters")
    return default


def read_cells(fs, box, uv, N):
    """all gas cells: pixel indices (coordinate axes, uint8 at N <= 256), rho, V, w (H I mass), T, x_HII, fII, fIII, sf"""
    if N > 256:
        sys.exit("STOP: uint8 pixel indices need N <= 256")
    cols = {k: [] for k in ("rho", "V", "w", "T", "x", "fII", "fIII", "sf", "i0", "i1", "i2")}
    names, nread, msum, nclip = None, 0, 0.0, [0, 0]
    for p in fs:
        with h5py.File(p, "r") as f:
            if "PartType0" not in f:
                continue
            g = f["PartType0"]
            if names is None:
                names = dict(x=pick(g, ("HII_Fraction",)), fII=pick(g, ("HeII_Fraction",)),
                             fIII=pick(g, ("HeIII_Fraction",)), ne=pick(g, ("ElectronAbundance",)),
                             sfr=pick(g, ("StarFormationRate",)), pos=pick(g, ("Coordinates", "IntCoordinates")))
                need = ("Masses", "Density", "InternalEnergy")
                if any(k not in g for k in need) or None in (names["x"], names["pos"], names["fII"], names["fIII"],
                                                             names["sfr"]):
                    sys.exit(f"STOP: missing gas fields; available: {sorted(g.keys())}")
                pdt = g[names["pos"]].dtype
                if np.issubdtype(pdt, np.integer) and not np.issubdtype(pdt, np.unsignedinteger):
                    sys.exit(f"STOP: {names['pos']} is {pdt} (signed): the integer -> position mapping is not documented")
                print(f"positions from {names['pos']} ({pdt}); Ne from "
                      f"{'ElectronAbundance' if names['ne'] else 'HII + HeII + 2 HeIII'}")
            n = g["Masses"].shape[0]
            if n == 0:
                continue
            pos = g[names["pos"]][()]
            if np.issubdtype(pos.dtype, np.integer):                # AREPO IntCoordinates: int x BoxSize / 2^bits
                pos = pos.astype(np.float64) * (box / 2.0**(8 * pos.dtype.itemsize))
            else:
                pos = pos.astype(np.float64)
            if pos.min() < 0.0 or pos.max() > box * (1.0 + 1e-6):
                sys.exit(f"STOP: positions outside [0, BoxSize] in {p}: {pos.min()} .. {pos.max()}")
            m = g["Masses"][()].astype(np.float64)
            rho = g["Density"][()].astype(np.float64)
            u = g["InternalEnergy"][()].astype(np.float64)
            x = g[names["x"]][()].astype(np.float64)
            fII = g[names["fII"]][()].astype(np.float64)
            fIII = g[names["fIII"]][()].astype(np.float64)
            ne = g[names["ne"]][()].astype(np.float64) if names["ne"] else x + fII + 2.0 * fIII
            sf = g[names["sfr"]][()] > 0
        mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * ne) * M_P
        T = GAMMA_M1 * u * uv**2 * mu / K_B
        nclip[0] += int(np.sum(x > 1.0)); nclip[1] += int(np.sum(x < 0.0))
        x = np.clip(x, 0.0, 1.0)                                    # float32 rounding (n_HI < 0 would give NaN in tau_GP)
        x, fII, fIII = np.where(sf, 0.0, x), np.where(sf, 0.0, fII), np.where(sf, 0.0, fIII)   # render convention
        idx = np.floor(pos / box * N).astype(np.int64) % N
        cols["rho"].append(rho.astype(np.float32)); cols["V"].append((m / rho).astype(np.float32))
        cols["w"].append((X_H * m * (1.0 - x)).astype(np.float32)); cols["T"].append(T.astype(np.float32))
        cols["x"].append(x.astype(np.float32)); cols["fII"].append(fII.astype(np.float32))
        cols["fIII"].append(fIII.astype(np.float32)); cols["sf"].append(sf)
        for k in range(3):
            cols[f"i{k}"].append(idx[:, k].astype(np.uint8))
        nread += n
        msum += m.sum()
    print(f"x_HII clipped to [0, 1] in {nclip[0]} cells above 1 and {nclip[1]} below 0")
    return {k: np.concatenate(v) for k, v in cols.items()}, nread, msum


def load_3d(path, prefer):
    with h5py.File(path, "r") as f:
        k = prefer if prefer in f else next(k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3)
        return f[k][()].astype(np.float64)


def ps_pair(m1, m2, L, nbins):
    N = m1.shape[0]
    edges = np.geomspace(2.0 * np.pi / L, np.pi * N / L, nbins + 1)
    P1, P2, X, kc, nind = spectra(m1, m2, L, edges)
    f = kc**3 / (2.0 * np.pi**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = X / np.sqrt(P1 * P2)
    return kc, f * P1, f * P2, r, nind


def pixel_stats(tb, spin, w, one, x, sfp, Tg, T):
    """the same split as the cells, for a pixel map: class shares of the mean, absorbing / emitting, H I diagnostics"""
    out = {"V_tb_abs": tb[tb < 0].sum() / tb.size, "V_tb_emi": tb[tb > 0].sum() / tb.size,
           "HI_Ts_below_Tg": w[spin < 0].sum() / w.sum(), "HI_spin_mean": (w * spin).sum() / w.sum()}
    cls = [tb[sfp].sum(), tb[~sfp & (x >= XNEU)].sum()]
    neu = ~sfp & (x < XNEU)
    for lo, hi in zip(CBINS[:-1], CBINS[1:]):
        cls.append(tb[neu & (one >= lo) & (one < hi)].sum())
    out["class"] = np.array(cls) / tb.size
    tcl = [tb[neu & (T >= lo) & (T < hi)].sum() for lo, hi in zip(tbins(Tg)[:-1], tbins(Tg)[1:])]
    out["tclass"] = np.array(tcl) / tb.size
    return out


def scan(a):
    od = os.path.join(a.out, a.run, "cells_Tb")
    check_out(od, a.allow_any_out)                                   # before the long read
    t00 = time.time()
    fs = snap_files(a.src, a.run, a.snap)
    with h5py.File(fs[0], "r") as f:
        hd = dict(f["Header"].attrs)
        par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
    z, box = float(hd["Redshift"]), float(hd["BoxSize"])
    h = hdr_value(hd, par, "HubbleParam")
    om, ob = hdr_value(hd, par, "Omega0"), hdr_value(hd, par, "OmegaBaryon")
    ol = hdr_value(hd, par, "OmegaLambda", 0.6902672)
    UL = hdr_value(hd, par, "UnitLength_in_cm", 3.085678e21)
    UM = hdr_value(hd, par, "UnitMass_in_g", 1.989e43)
    uv = hdr_value(hd, par, "UnitVelocity_in_cm_per_s", 1e5)
    nfiles = int(hd.get("NumFilesPerSnapshot", len(fs)))
    ntot = int(hd["NumPart_Total"][0]) + (int(hd["NumPart_Total_HighWord"][0]) << 32 if "NumPart_Total_HighWord" in hd else 0)
    print(f"{a.run} snapshot {a.snap}: z = {z:.5f}, BoxSize {box}, h {h}, Om {om}, Ob {ob}, OL {ol}, {len(fs)} files "
          f"(header: {nfiles}), {ntot:,} gas cells; constants: {CONST_SRC}")
    if len(fs) != nfiles:
        sys.exit(f"STOP: {len(fs)} files found, the header says {nfiles}")
    rho_unit = UM * h**2 / UL**3                                     # code density -> g cm^-3 comoving

    # render, J, production map
    base = os.path.join(a.out, a.run)
    ren = os.path.join(base, "ren_128", "All", f"All_{a.render:03d}.hdf5")
    with h5py.File(ren, "r") as f:
        zr = float(f["Header"].attrs["Redshift"])
        R = {k: f[k][()].astype(np.float64) for k in ("Density", "Temperature", "HII_Fraction", "HeII_Fraction",
                                                      "HeIII_Fraction")}
        sfr_r = f["StarFormationRate"][()] if "StarFormationRate" in f else None
        tcgs = float(f["Density"].attrs["to_cgs"]) if "to_cgs" in f["Density"].attrs else None
    N = R["Density"].shape[0]
    print(f"render {a.render}: z = {zr:.5f} (dz = {z - zr:+.1e}), N = {N}")
    if abs(z - zr) > 1e-3:
        sys.exit("STOP: the render is not at the snapshot's redshift")
    if tcgs is not None:
        rr = tcgs / (UM / UL**3)
        print(f"check: render Density to_cgs / (UnitMass/UnitLength^3) = {rr:.6f}")
        if abs(rr - 1.0) > 1e-3:
            sys.exit("STOP: the render density is not in the snapshot's code units")
    jdir = os.path.join(base, "Jalpha")
    J = load_3d(os.path.join(jdir, f"Jalpha_H_{a.render}.hdf5"), "J_alpha") \
        + load_3d(os.path.join(jdir, f"Jalpha_AGN_H_{a.render}.hdf5"), "Jalpha_AGN_H")
    rin_path = os.path.join(base, "Tb_21cm", f"RIN_{a.render}.hdf5")
    with h5py.File(rin_path, "r") as f:
        C128 = f["rin"][()].astype(np.float64)
        content = f["rin"].attrs.get("content", b"")
    content = content.decode() if isinstance(content, bytes) else str(content)
    print(f"J: <J> = {J.mean():.4e} (stars + AGN); map {rin_path}: {content}")
    if "SPIN_MODEL 1" not in content or "real space" not in content:
        sys.exit("STOP: the production map is not the real-space SPIN_MODEL 1 map")
    if J.shape != (N,) * 3 or C128.shape != (N,) * 3:
        sys.exit(f"STOP: J {J.shape}, map {C128.shape}, render N = {N}")

    L = Line("H", z, h, om, ob, ol, spin_model=1)
    Tg = L.Tg
    nfac = X_H * rho_unit * (1.0 + z)**3 / M_H                       # code density -> proper n_H [cm^-3]

    # pixel references (tb_physics on the render: the production physics)
    rbar = R["Density"].mean()
    sig128 = R["Density"] * (1.0 - R["HII_Fraction"]) / rbar
    tb128, _, xa128, ex128 = L.tb(sig128, nfac * R["Density"], R["Temperature"], R["HII_Fraction"], R["HeII_Fraction"],
                                  R["HeIII_Fraction"], J, full=True)
    rel = np.abs(C128 - tb128) / np.maximum(np.abs(tb128), 1e-6 * np.abs(tb128).max())
    print(f"check: production map vs tb_physics on the render: <C> {C128.mean():+.6f}, here {tb128.mean():+.6f} mK "
          f"(ratio {C128.mean() / tb128.mean():.7f}); per pixel max |diff|/|here| {rel.max():.2e}, 99.9% {np.percentile(rel, 99.9):.2e}")
    if not (np.all(np.isfinite(C128)) and np.all(np.isfinite(tb128))):
        sys.exit(f"STOP: non-finite values: production map {np.sum(~np.isfinite(C128))}, tb_physics {np.sum(~np.isfinite(tb128))}")
    if abs(C128.mean() / tb128.mean() - 1.0) > 1e-3 or np.percentile(rel, 99.9) > 1e-3:
        sys.exit("STOP: tb_physics does not reproduce the production map (mean or per pixel)")
    m8 = csum8(R["Density"])
    D64 = m8 / 8.0
    T64 = csum8(R["Density"] * R["Temperature"]) / m8
    x64 = csum8(R["Density"] * R["HII_Fraction"]) / m8
    fII64 = csum8(R["Density"] * R["HeII_Fraction"]) / m8
    fIII64 = csum8(R["Density"] * R["HeIII_Fraction"]) / m8
    J64 = csum8(J) / 8.0
    tb64, _, _, ex64 = L.tb(D64 * (1.0 - x64) / D64.mean(), nfac * D64, T64, x64, fII64, fIII64, J64, full=True)
    if not np.all(np.isfinite(tb64)):
        sys.exit("STOP: non-finite dT_b in the 64^3 pixels")
    sfp128 = (sfr_r > 0) if sfr_r is not None else np.zeros((N,) * 3, bool)
    sfp64 = csum8(sfp128.astype(np.float64)) > 0
    pst = {"pix128": pixel_stats(tb128, 1.0 - Tg / ex128["TS"], R["Density"] * (1.0 - R["HII_Fraction"]),
                                 R["Density"] / rbar, R["HII_Fraction"], sfp128, Tg, R["Temperature"]),
           "pix64": pixel_stats(tb64, 1.0 - Tg / ex64["TS"], D64 * (1.0 - x64), D64 / D64.mean(), x64, sfp64, Tg, T64)}
    print(f"pixels: <dT_b> 128^3 {tb128.mean():+.5f} mK, 64^3 {tb64.mean():+.5f} mK  (T_gamma = {Tg:.2f} K)")

    # cells
    C, nread, msum = read_cells(fs, box, uv, N)
    print(f"read {nread:,} cells in {time.time() - t00:.0f} s" + ("" if nread == ntot else f"  STOP: header says {ntot:,}"))
    if nread != ntot:
        sys.exit(1)
    Vsum = C["V"].astype(np.float64).sum()
    rho_bar = msum / box**3
    print(f"check: gas mass cells / render {msum / (rbar * box**3):.6f}; sum of cell volumes / box^3 {Vsum / box**3:.6f}")

    # render axis order from the data (as cells_T_xray.py)
    rng = np.random.default_rng(1)
    s = rng.choice(nread, min(nread, 3_000_000), replace=False)
    lr = np.log10(C["rho"][s].astype(np.float64))
    best = []
    for o in itertools.permutations(range(3)):
        rd = R["Density"][C[f"i{o[0]}"][s], C[f"i{o[1]}"][s], C[f"i{o[2]}"][s]]
        cc = np.corrcoef(lr, np.log10(np.maximum(rd, 1e-30)))[0, 1]
        best.append((cc, o))
    best.sort(reverse=True)
    print("axis order: " + ", ".join(f"{o}: {cc:.4f}" for cc, o in best))
    if best[0][0] - best[1][0] < 0.05:
        sys.exit("STOP: the render axis order is not clear from the data")
    o = best[0][1]
    flat = np.ravel_multi_index((C[f"i{o[0]}"].astype(np.int64), C[f"i{o[1]}"].astype(np.int64),
                                 C[f"i{o[2]}"].astype(np.int64)), (N, N, N)).astype(np.int32)
    for k in range(3):
        del C[f"i{k}"]

    # per-cell delta T_b, both tau_GP options, in chunks
    nHIpix = (nfac * R["Density"] * (1.0 - R["HII_Fraction"])).ravel()
    Jf = J.ravel()
    Tb_ = tbins(Tg)
    acc = {op: {"Vtb": np.zeros(N**3), "abs": 0.0, "emi": 0.0, "class": np.zeros(len(CBINS) + 1),
                "tclass": np.zeros(len(Tb_) - 1), "HI_Ts_below_Tg": 0.0, "HI_spin": 0.0, "HI_S": 0.0, "HI_SJ": 0.0,
                "h_lgp": np.zeros(len(LGP_EDGES) + 1), "xcmb_min": 1.0, "tau21_max": 0.0, "HI_tau21_gt01": 0.0}
           for op in OPTS}
    Vgrid = np.bincount(flat, C["V"].astype(np.float64), minlength=N**3)
    Wtot = C["w"].astype(np.float64).sum()

    # like-for-like pixel baseline: pixel fields from the same cells, nearest-pixel deposit, the same physics
    mC = C["rho"].astype(np.float64) * C["V"]
    Mg = np.bincount(flat, mC, minlength=N**3)
    Sg = {k: np.bincount(flat, mC * C[k], minlength=N**3) for k in ("T", "x", "fII", "fIII")}
    del mC
    vpix = (box / N)**3
    eg = (Mg <= 0) | (Vgrid <= 0)
    if eg.any():
        print(f"WARNING: {eg.sum()} of {N**3} pixels hold no cell centre: NGP fields and cell map there are the render's")
    Veff = np.where(eg, vpix, Vgrid)                                 # the cell map's normaliser (sum V of the cells)
    Mg = np.where(eg, R["Density"].ravel() * vpix, Mg)
    for k, rk in (("T", "Temperature"), ("x", "HII_Fraction"), ("fII", "HeII_Fraction"), ("fIII", "HeIII_Fraction")):
        Sg[k] = np.where(eg, R[rk].ravel() * Mg, Sg[k])

    def ngp_tb(M_, S_, J_, V_):
        rho_ = M_ / V_                                               # = sum m / sum V: the cell map's normalisation
        xx = S_["x"] / M_
        return L.tb(rho_ * (1.0 - xx) / rho_bar, nfac * rho_, S_["T"] / M_, xx, S_["fII"] / M_, S_["fIII"] / M_, J_)[0]
    sh = (N,) * 3
    tbN128 = ngp_tb(Mg.reshape(sh), {k: v.reshape(sh) for k, v in Sg.items()}, J, Veff.reshape(sh))
    tbN64 = ngp_tb(csum8(Mg.reshape(sh)), {k: csum8(v.reshape(sh)) for k, v in Sg.items()}, J64, csum8(Veff.reshape(sh)))
    del Mg, Sg
    if not (np.all(np.isfinite(tbN128)) and np.all(np.isfinite(tbN64))):
        sys.exit("STOP: non-finite dT_b in the NGP pixels")
    Jb = np.repeat(np.repeat(np.repeat(J64, 2, 0), 2, 1), 2, 2)
    wr = R["Density"] * (1.0 - R["HII_Fraction"])
    jsc = np.sqrt((wr * (J / Jb - 1.0)**2).sum() / wr.sum())
    print(f"NGP pixels from the cells: <dT_b> 128^3 {tbN128.mean():+.5f}, 64^3 {tbN64.mean():+.5f} mK; J scatter inside "
          f"64^3 blocks (H I-weighted rms of J_128/J_64 - 1): {jsc:.4f}")
    t0 = time.time()
    for i0 in range(0, nread, a.chunk):
        sl = slice(i0, min(nread, i0 + a.chunk))
        rho = C["rho"][sl].astype(np.float64)
        x = C["x"][sl].astype(np.float64)
        T = C["T"][sl].astype(np.float64)
        V = C["V"][sl].astype(np.float64)
        w = C["w"][sl].astype(np.float64)
        sf = C["sf"][sl]
        fl = flat[sl]
        one = rho / rho_bar
        nH = nfac * rho
        sig = rho * (1.0 - x) / rho_bar
        Jc = Jf[fl]
        neu = ~sf & (x < XNEU)
        for op in OPTS:
            try:
                tb, _, xa, ex = L.tb(sig, nH, T, x, C["fII"][sl].astype(np.float64), C["fIII"][sl].astype(np.float64),
                                     Jc, nHI_gp=(nHIpix[fl] if op == "pix" else None), full=True)
            except ValueError as e:
                sys.exit(f"STOP: cells {sl.start}-{sl.stop}: {e}")
            if not np.all(np.isfinite(tb)):
                sys.exit(f"STOP: cells {sl.start}-{sl.stop}: {np.sum(~np.isfinite(tb))} non-finite dT_b "
                         f"(T range {T.min():.3g}-{T.max():.3g} K)")
            A = acc[op]
            Vtb = V * tb
            A["Vtb"] += np.bincount(fl, Vtb, minlength=N**3)
            A["abs"] += Vtb[tb < 0].sum(); A["emi"] += Vtb[tb > 0].sum()
            cls = [Vtb[sf].sum(), Vtb[~sf & (x >= XNEU)].sum()]
            for lo, hi in zip(CBINS[:-1], CBINS[1:]):
                cls.append(Vtb[neu & (one >= lo) & (one < hi)].sum())
            A["class"] += np.array(cls)
            A["tclass"] += np.array([Vtb[neu & (T >= lo) & (T < hi)].sum() for lo, hi in zip(Tb_[:-1], Tb_[1:])])
            spin = 1.0 - Tg / ex["TS"]
            A["HI_Ts_below_Tg"] += w[spin < 0].sum(); A["HI_spin"] += (w * spin).sum()
            jp = Jc > 0
            A["HI_S"] += (w[jp] * xa[jp] / (L.xa_c * Jc[jp])).sum(); A["HI_SJ"] += w[jp].sum()
            A["h_lgp"] += np.bincount(np.searchsorted(LGP_EDGES, np.log10(np.maximum(ex["tauGP"], 1e-300)), side="right"),
                                      w, minlength=len(LGP_EDGES) + 1)       # [0]: below the edges, [-1]: above
            A["xcmb_min"] = min(A["xcmb_min"], float(ex["xcmb"][w > 0].min()) if (w > 0).any() else 1.0)
            A["tau21_max"] = max(A["tau21_max"], float(ex["tau21"].max()))
            A["HI_tau21_gt01"] += w[ex["tau21"] > 0.1].sum()
        if i0 == 0 or (i0 // a.chunk) % 10 == 0:
            print(f"  cells {sl.stop:,} / {nread:,}: {time.time() - t0:.0f} s", flush=True)

    # maps, means, spectra
    Lb = box / 1000.0                                                # cMpc/h
    V64 = csum8(Veff.reshape((N,) * 3))
    fill = np.where(eg, tbN128.ravel() * vpix, 0.0)                  # empty pixels: the render's value
    res = {"z": z, "render": a.render, "run": a.run, "snap": a.snap, "Tgamma": Tg, "N": N, "box": box,
           "axis_order": np.array(o), "cbins": np.array(CBINS), "tbins": np.array(Tb_), "lgp_edges": LGP_EDGES,
           "mean_C128": C128.mean(), "mean_pix128": tb128.mean(), "mean_pix64": tb64.mean(),
           "mean_pixngp128": tbN128.mean(), "mean_pixngp64": tbN64.mean(), "J_block_scatter": jsc,
           "checks": np.array([msum / (rbar * box**3), Vsum / box**3, C128.mean() / tb128.mean(), rel.max()])}
    for k, P in pst.items():
        for kk, v in P.items():
            res[f"{k}_{kk}"] = v
    for tag, pn, pr, nb in (("128", tbN128, tb128, 10), ("64", tbN64, tb64, 8)):
        kc, Dn, Dr, r, nind = ps_pair(pn, pr, Lb, nb)
        res.update({f"D2pixngp{tag}": Dn, f"r_ngp_render{tag}": r})
        print(f"deposit control at {tag}^3, NGP pixels / render pixels: Delta^2 "
              + " ".join(f"{k:.2f}:{d / q:.3f}" for k, d, q, nn in zip(kc, Dn, Dr, nind) if nn > 0)
              + f"; min r {np.nanmin(r):.3f}")
    print(f"\n{a.run}, snapshot {a.snap}, z = {z:.3f}: <dT_b> [mK] production map {C128.mean():+.5f} | tb_physics pixels "
          f"128^3 {tb128.mean():+.5f}, 64^3 {tb64.mean():+.5f} | NGP pixels from the cells 128^3 {tbN128.mean():+.5f}, "
          f"64^3 {tbN64.mean():+.5f}")
    for op in OPTS:
        A = acc[op]
        m = A["Vtb"].sum() / box**3
        g128 = ((A["Vtb"] + fill) / Veff).reshape((N,) * 3)
        g64 = csum8((A["Vtb"] + fill).reshape((N,) * 3)) / V64
        res.update({f"mean_cells_{op}": m, f"map128_{op}": g128.astype(np.float32), f"map64_{op}": g64.astype(np.float32),
                    f"abs_{op}": A["abs"] / box**3, f"emi_{op}": A["emi"] / box**3, f"class_{op}": A["class"] / box**3,
                    f"tclass_{op}": A["tclass"] / box**3, f"HI_Ts_below_Tg_{op}": A["HI_Ts_below_Tg"] / Wtot,
                    f"HI_spin_{op}": A["HI_spin"] / Wtot, f"HI_S_{op}": A["HI_S"] / max(A["HI_SJ"], 1e-300),
                    f"h_lgp_{op}": A["h_lgp"] / Wtot, f"xcmb_min_{op}": A["xcmb_min"], f"tau21_max_{op}": A["tau21_max"],
                    f"HI_tau21_gt01_{op}": A["HI_tau21_gt01"] / Wtot})
        for tag, cm, pm, pn, nb in (("128", g128, tb128, tbN128, 10), ("64", g64, tb64, tbN64, 8)):
            kc, Dc, Dp, r, nind = ps_pair(cm, pm, Lb, nb)
            res.update({f"k{tag}": kc, f"D2cells{tag}_{op}": Dc, f"D2pix{tag}": Dp, f"r{tag}_{op}": r, f"nind{tag}": nind})
            res[f"rngp{tag}_{op}"] = ps_pair(cm, pn, Lb, nb)[3]
        print(f"  cells, tau_GP {op:4s}: <dT_b> {m:+.5f} mK  (cells / pixels: {m / tb128.mean():.3f} at 128^3, "
              f"{m / tb64.mean():.3f} at 64^3); absorbing {A['abs'] / box**3:+.5f}, emitting {A['emi'] / box**3:+.5f}; "
              f"H I with T_S < T_gamma {A['HI_Ts_below_Tg'] / Wtot:.3f}, <1 - T_g/T_S>_HI {A['HI_spin'] / Wtot:+.3f}, "
              f"<S~_a>_HI {A['HI_S'] / max(A['HI_SJ'], 1e-300):.3f}; min x_CMB {A['xcmb_min']:.4f}, max tau_21 "
              f"{A['tau21_max']:.3g} (H I with tau_21 > 0.1: {A['HI_tau21_gt01'] / Wtot:.4f}); H I with log tau_GP outside "
              f"[{LGP_EDGES[0]:g}, {LGP_EDGES[-1]:g}]: {A['h_lgp'][0] / Wtot:.4f} below, {A['h_lgp'][-1] / Wtot:.4f} above")
        for tag in ("128", "64"):
            for ref, lab, rk in (("pixngp", "NGP pixels   ", "rngp"), ("pix", "render pixels", "r")):
                print(f"    Delta^2 cells / {lab} at {tag}^3: " + " ".join(
                    f"{k:.2f}:{d / p:.3f}" for k, d, p, nn in
                    zip(res[f'k{tag}'], res[f'D2cells{tag}_{op}'], res[f'D2{ref}{tag}'], res[f'nind{tag}']) if nn > 0)
                    + f"; min r {np.nanmin(res[f'{rk}{tag}_{op}']):.3f}")
    names = ["star-forming (pix: any)", f"x_HII >= {XNEU:g}"] + [f"neutral 1+d {lo:g}-{hi:g}" for lo, hi in zip(CBINS[:-1], CBINS[1:])]
    print("  share of <dT_b> [mK], cells (tau_GP cell | pix) | render pixels 128^3 | 64^3 (pixels split by the pixel's own\n"
          "  values; 'star-forming' pixels are all pixels holding any star-forming gas, IGM included):")
    for i, nm in enumerate(names):
        print(f"    {nm:24s} {res['class_cell'][i]:+.5f} | {res['class_pix'][i]:+.5f} | "
              f"{res['pix128_class'][i]:+.5f} | {res['pix64_class'][i]:+.5f}")
    for i, (lo, hi) in enumerate(zip(Tb_[:-1], Tb_[1:])):
        print(f"    neutral, T {lo:7.1f}-{hi:<7.4g} K {res['tclass_cell'][i]:+.5f} | {res['tclass_pix'][i]:+.5f} | "
              f"{res['pix128_tclass'][i]:+.5f} | {res['pix64_tclass'][i]:+.5f}")
    print(f"  pixels: absorbing / emitting 128^3 {res['pix128_V_tb_abs']:+.5f} / {res['pix128_V_tb_emi']:+.5f}, 64^3 "
          f"{res['pix64_V_tb_abs']:+.5f} / {res['pix64_V_tb_emi']:+.5f}; H I with T_S < T_gamma "
          f"{res['pix128_HI_Ts_below_Tg']:.3f} / {res['pix64_HI_Ts_below_Tg']:.3f}")
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"cells_Tb_{a.snap:03d}.npz")
    np.savez(out + ".part.npz", **res)
    os.replace(out + ".part.npz", out)
    print(f"\nwrote {out}  ({time.time() - t00:.0f} s)")


def report(a):
    D = {}
    for run in RUNS:
        for p in sorted(glob.glob(os.path.join(a.out, run, "cells_Tb", "cells_Tb_*.npz"))):
            d = np.load(p)
            D[(run, int(d["snap"]))] = d
    if not D:
        sys.exit("no scan outputs found")
    print("<dT_b> [mK]: production map | pixels 128^3 | 64^3 | NGP pixels 128^3 | cells (tau_GP cell) | cells (tau_GP pix) | "
          "cells/pix128 | cells/pix64 (tau_GP cell); H I with T_S < T_gamma cells / pix128")
    for (run, snap), d in sorted(D.items(), key=lambda t: (RUNS.index(t[0][0]), -float(t[1]["z"]))):
        print(f"  {LABELS[run]:12s} z = {float(d['z']):6.3f}: {float(d['mean_C128']):+8.4f} | {float(d['mean_pix128']):+8.4f} | "
              f"{float(d['mean_pix64']):+8.4f} | {float(d['mean_pixngp128']):+8.4f} | {float(d['mean_cells_cell']):+8.4f} | {float(d['mean_cells_pix']):+8.4f} | "
              f"{float(d['mean_cells_cell']) / float(d['mean_pix128']):7.3f} | "
              f"{float(d['mean_cells_cell']) / float(d['mean_pix64']):7.3f}; "
              f"{float(d['HI_Ts_below_Tg_cell']):.3f} / {float(d['pix128_HI_Ts_below_Tg']):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for run in RUNS:
        ks = sorted((k for k in D if k[0] == run), key=lambda k: float(D[k]["z"]))
        if not ks:
            continue
        zz = [float(D[k]["z"]) for k in ks]
        c = COLORS[run]
        ax[0].plot(zz, [float(D[k]["mean_pix64"]) for k in ks], color=c, lw=1, ls=":", marker="s", mfc="none",
                   label=f"{LABELS[run]}: pixels 64$^3$")
        ax[0].plot(zz, [float(D[k]["mean_pix128"]) for k in ks], color=c, lw=1, ls="--", marker="s",
                   label=f"{LABELS[run]}: pixels 128$^3$")
        ax[0].plot(zz, [float(D[k]["mean_cells_cell"]) for k in ks], color=c, lw=2, marker="o",
                   label=f"{LABELS[run]}: cells")
        ax[0].plot(zz, [float(D[k]["mean_cells_pix"]) for k in ks], color=c, lw=0, marker="x", ms=7,
                   label=f"{LABELS[run]}: cells, pixel $\\tau_{{GP}}$")
        for i, k in enumerate(sorted(ks, key=lambda k: -float(D[k]["z"]))):
            d = D[k]
            lab = f"{LABELS[run]}, z = {float(d['z']):.1f}"
            ls = ("-", "--", ":", "-.")[i % 4]
            ax[1].plot(d["k128"], d["D2cells128_cell"] / d["D2pixngp128"], color=c, lw=1.2, ls=ls, label=lab)
            ax[2].plot(d["k128"], d["rngp128_cell"], color=c, lw=1.2, ls=ls)
    ax[0].axhline(0, color=GRID, lw=1)
    ax[0].set_xlabel("z"); ax[0].set_ylabel(r"$\langle\delta T_b\rangle$ [mK]")
    ax[0].legend(fontsize=6.5, loc="best")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].axhline(1, color=GRID, lw=1)
    ax[1].set_xlabel(r"$k$ [h/cMpc]"); ax[1].set_ylabel(r"$\Delta^2_{\rm cells}/\Delta^2_{\rm pixels}$ (128$^3$, same deposit)")
    ax[1].legend(fontsize=6.5)
    ax[2].set_xscale("log"); ax[2].axhline(1, color=GRID, lw=1)
    ax[2].set_xlabel(r"$k$ [h/cMpc]"); ax[2].set_ylabel(r"$r(k)$ cells vs pixels (128$^3$, same deposit)")
    from matplotlib.ticker import NullFormatter
    for x_ in ax[1:]:
        x_.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    os.makedirs(a.data, exist_ok=True)
    fig.savefig(os.path.join(a.data, "cells_Tb_xray.png"), dpi=200)
    print(f"\nwrote {a.data}/cells_Tb_xray.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--run", choices=RUNS, required=True)
    s.add_argument("--snap", type=int, required=True)
    s.add_argument("--render", type=int, required=True)
    s.add_argument("--chunk", type=int, default=4_000_000)
    s.add_argument("--src", default=SRC)
    s.add_argument("--out", default=os.environ.get("BASE", OUT))
    s.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)   # mock tests only
    r = sp.add_parser("report")
    r.add_argument("--out", default=os.environ.get("BASE", OUT))
    r.add_argument("--data", default="data")
    a = p.parse_args()
    scan(a) if a.cmd == "scan" else report(a)


if __name__ == "__main__":
    main()
