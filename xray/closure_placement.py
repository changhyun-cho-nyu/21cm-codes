"""
closure_placement.py
====================
Which cell gets which no-X-ray temperature? Placement test of the own-rank model on the 50 cMpc pair (user,
2026-11-20, after closure_noX_xray.py: the global no-X dT_b is 1-7% of the X-ray effect too warm, and the no-X power at
k <= 0.4 h/cMpc 2-3.4x too high, because ranking by T_X imprints the X-ray heating pattern; xray/NOTES.md step 5).
The per-key T_noX distribution is kept exactly (key = 0.05-dex 1+delta bin x x_HII class of the X run, training =
good non-SF ID pairs, X-run H I weight, as closure_noX_xray.py); only the score that ranks the target cells inside
their key changes:
    tx     : log T_X (the current model; its mean is checked against closure_NNN.npz pred_all)
    random : a random number (no placement information: structure from density, x_HII and J only)
    txloc  : log T_X minus its local mean at fixed key: d = log T_X - <log T_X>_key, score = d - G_R[d] (G_R = H I-weighted
             Gaussian smoothing over R at 128^3); keeps small-scale shock structure, removes the smooth X-ray pattern
    denv   : log10 of the gas density smoothed over R (Gaussian, 128^3): the large-scale environment, which both boxes
             share; its sign is taken from the training pairs (within-key correlation with log T_noX)
Every variant's T replaces the X-run T of the non-SF cells; dT_b of every cell as in closure_noX_xray.py (spin model
1, tau_GP from the cell; density, x_HII, He and J_alpha of the X run; SF cells unchanged). Ranks are exact H I-weighted
ranks within the key (ties broken at random), not histograms.
Reported per variant: paired non-SF <1/T>^-1 against the truth, rms log T error, within-key correlation of the score with
log T_noX on the pairs; box-mean dT_b, predicted / true X-ray effect, error of the no-X mean; the error by 1+delta x
x_HII class; Delta^2 ratio and r(k) at 64^3 (diagnostic). Writes <out>/pair_cells/closure_place_NNN.npz.

  python closure_placement.py --snap 57 [--smooth 1.0]      (R in cMpc/h)
Inputs read only; outputs only under /orcd/data/mvogelsb/004/chcho.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from pair_cells_xray import (SRC, OUT, RUNS, SNAP_RENDER, SEP_MAX, X_H, read_render, read_snap, axis_order,  # noqa: E402
                             unique_sorted, separation, check_out, snap_files)
from cells_Tb_xray import hdr_value, load_3d, ps_pair, Line, M_H, CBINS  # noqa: E402
from closure_noX_xray import (keys_of, hist, fallback, inv_cdf, tb_cells, cmap, NK, XCL, XLAB,  # noqa: E402
                              MIN_TRAIN)
import h5py                                                                       # noqa: E402

VARIANTS = ("tx", "random", "txloc", "denv")


def rank_p(key, score, w, rng):
    """H I-weighted mid-rank quantile of each cell inside its key: (W_below + w/2) / W_key; ties broken at random"""
    o = np.lexsort((rng.random(len(key)), score, key))
    ks, ws = key[o], w[o]
    cw = np.cumsum(ws)
    st = np.r_[True, ks[1:] != ks[:-1]]
    s0 = np.flatnonzero(st)
    seg = np.cumsum(st) - 1
    before = (cw - ws)[s0][seg]
    tot = np.add.reduceat(ws, s0)[seg]
    p = np.empty(len(key))
    p[o] = np.where(tot > 0, (cw - 0.5 * ws - before) / np.where(tot > 0, tot, 1.0), 0.5)
    del o, ks, ws, cw, st, seg, before, tot
    return np.clip(p, 0.0, 1.0)


def gsmooth(f, sig_pix):
    """periodic Gaussian smoothing of a cube, sigma in pixels"""
    n = f.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n)
    k2 = k[:, None, None]**2 + k[None, :, None]**2 + kz[None, None, :]**2
    return np.fft.irfftn(np.fft.rfftn(f) * np.exp(-0.5 * k2 * sig_pix**2), s=f.shape, axes=(0, 1, 2))


def within_key_corr(key, a_, b_, w):
    """pooled weighted correlation of a and b after removing each key's weighted mean from both"""
    sw = np.bincount(key, w, NK)
    ok = sw > 0
    ma = np.bincount(key, w * a_, NK) / np.where(ok, sw, 1.0)
    mb = np.bincount(key, w * b_, NK) / np.where(ok, sw, 1.0)
    da, db = a_ - ma[key], b_ - mb[key]
    return float(np.sum(w * da * db) / np.sqrt(np.sum(w * da**2) * np.sum(w * db**2)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--render", type=int, default=None)
    p.add_argument("--smooth", type=float, default=1.0, help="Gaussian sigma R [cMpc/h] for txloc and denv")
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", default=os.environ.get("BASE", OUT))
    p.add_argument("--chunk", type=int, default=4_000_000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--allow-any-out", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    render = a.render if a.render is not None else SNAP_RENDER.get(a.snap)
    if render is None:
        sys.exit(f"no render known for snapshot {a.snap}: pass --render")
    od = os.path.join(a.out, "pair_cells")
    check_out(od, a.allow_any_out)
    rng = np.random.default_rng(a.seed)
    t00 = time.time()

    fs = snap_files(a.src, RUNS[1], a.snap)
    with h5py.File(fs[0], "r") as f:
        hd = dict(f["Header"].attrs)
        par = dict(f["Parameters"].attrs) if "Parameters" in f else {}
    z = float(hd["Redshift"])
    h = hdr_value(hd, par, "HubbleParam")
    om, ob = hdr_value(hd, par, "Omega0"), hdr_value(hd, par, "OmegaBaryon")
    ol = hdr_value(hd, par, "OmegaLambda", 0.6902672)
    UL = hdr_value(hd, par, "UnitLength_in_cm", 3.085678e21)
    UM = hdr_value(hd, par, "UnitMass_in_g", 1.989e43)
    nfac = X_H * UM * h**2 / UL**3 * (1.0 + z)**3 / M_H
    L = Line("H", z, h, om, ob, ol, spin_model=1)
    print(f"snapshot {a.snap}, render {render}: z = {z:.5f}, T_gamma = {L.Tg:.2f} K; smoothing R = {a.smooth} cMpc/h",
          flush=True)

    C, M, F = {}, {}, {}
    for run in RUNS:
        zr, N, R = read_render(a.src, run, render, ["Density"])
        if abs(zr - z) > 1e-3:
            sys.exit(f"STOP: render z {zr} vs snapshot z {z}")
        c, m, dep_m, _ = read_snap(a.src, run, a.snap, N)
        o, co = axis_order(dep_m.reshape(N, N, N), R["Density"])
        del dep_m, R
        idx = ((c["ip"].astype(np.uint64) * np.uint64(N)) >> np.uint64(32)).astype(np.int64)
        F[run] = np.ravel_multi_index((idx[:, o[0]], idx[:, o[1]], idx[:, o[2]]), (N,) * 3).astype(np.int32)
        del idx
        c["V"] = (c["m"].astype(np.float64) / c["rho"]).astype(np.float32)
        jd = os.path.join(a.out, run, "Jalpha")
        J = load_3d(os.path.join(jd, f"Jalpha_H_{render}.hdf5"), "J_alpha") \
            + load_3d(os.path.join(jd, f"Jalpha_AGN_H_{render}.hdf5"), "Jalpha_AGN_H")
        if J.shape != (N,) * 3:
            sys.exit(f"STOP: J {J.shape}, render N = {N}")
        m["J"], m["order"], m["N"] = J.ravel(), o, N
        C[run], M[run] = c, m
        print(f"  {run}: <J> {J.mean():.4e}, axis order {o}", flush=True)
    A, B = C[RUNS[0]], C[RUNS[1]]
    MA, MB = M[RUNS[0]], M[RUNS[1]]
    N, box = MA["N"], MA["box"]
    if MA["order"] != MB["order"] or abs(MA["z"] - MB["z"]) > 1e-4:
        sys.exit("STOP: the runs differ in axis order or z")

    # pairs (training), as closure_noX_xray.py
    sa, oa, dA = unique_sorted(A["id"])
    sb, ob_, dB = unique_sorted(B["id"])
    pos = np.minimum(np.searchsorted(sb, sa), len(sb) - 1)
    hit = sb[pos] == sa
    ia, ib = oa[hit], ob_[pos[hit]]
    del sa, sb, oa, ob_, pos, hit
    sep = separation(A["ip"], B["ip"], ia, ib, box)
    good = sep / np.cbrt(A["V"][ia].astype(np.float64)) < SEP_MAX
    ia, ib = ia[good], ib[good]
    del sep, good
    nsf = ~A["sf"][ia] & ~B["sf"][ib]
    wB = X_H * B["m"].astype(np.float64) * (1.0 - np.clip(B["x"], 0, 1))
    wA = X_H * A["m"].astype(np.float64) * (1.0 - np.clip(A["x"], 0, 1))
    print(f"pairs: {len(ia):,} good of {len(A['id']):,} / {len(B['id']):,} cells; H I share of the no-X run in good pairs "
          f"{wA[ia].sum() / wA.sum():.4f}", flush=True)
    ja, jb = ia[nsf], ib[nsf]

    # model tables (X-run state only), as closure_noX_xray.py
    oneB = B["rho"].astype(np.float64) / MB["rho_mean"]
    keyB = keys_of(oneB, np.clip(B["x"], 0, 1))
    ltB = np.log10(np.maximum(B["T"].astype(np.float64), 1.0))
    ltA = np.log10(np.maximum(A["T"].astype(np.float64), 1.0))
    tgt = ~B["sf"]
    Htr, ntr = hist(keyB[jb], ltA[ja], wB[jb])
    fb = fallback(ntr)
    kt, wt = keyB[tgt], wB[tgt]
    FB = F[RUNS[1]]

    # scores
    sig = a.smooth / (box / 1000.0 / N)
    scores = {"tx": ltB}
    scores["random"] = rng.random(len(ltB))
    swk = np.bincount(kt, wt, NK)
    mk = np.bincount(kt, wt * ltB[tgt], NK) / np.where(swk > 0, swk, 1.0)
    d = ltB - mk[keyB]
    num = np.bincount(FB[tgt], wt * d[tgt], N**3).reshape((N,) * 3)
    den = np.bincount(FB[tgt], wt, N**3).reshape((N,) * 3)
    sm_den = gsmooth(den, sig)
    dsm = gsmooth(num, sig) / np.where(sm_den > 1e-30 * sm_den.max(), sm_den, np.inf)
    scores["txloc"] = d - dsm.ravel()[FB]
    del num, den, sm_den, dsm
    mass = np.bincount(FB, B["m"].astype(np.float64), N**3).reshape((N,) * 3)
    msm = gsmooth(mass, sig)
    env = np.log10(np.maximum(msm, 1e-12 * msm.max())).ravel()[FB]
    del mass, msm
    r_env = within_key_corr(keyB[jb], env[jb], ltA[ja], wB[jb])
    scores["denv"] = env if r_env >= 0 else -env
    print(f"  denv: within-key corr(smoothed log density, log T_noX) on the pairs {r_env:+.3f} -> sign "
          f"{'+' if r_env >= 0 else '-'}", flush=True)
    del env, d

    lts = {}
    for v in VARIANTS:
        lt_ = ltB.copy()
        p_ = rank_p(kt, scores[v][tgt], wt, rng)
        lt_[tgt] = inv_cdf(Htr, fb[kt], p_)
        lts[v] = lt_
        w_ = wA[ja]
        hm_p = w_.sum() / np.sum(w_ / 10**lt_[jb])
        hm_t = w_.sum() / np.sum(w_ / 10**ltA[ja])
        rr = within_key_corr(keyB[jb], scores[v][jb], ltA[ja], wB[jb])
        print(f"  {v:7s}: paired non-SF H I <1/T>^-1 {hm_p:9.3f} K (truth {hm_t:.3f}); rms log10(T/T_true) "
              f"{np.sqrt(np.sum(w_ * (lt_[jb] - ltA[ja])**2) / w_.sum()):.3f}; within-key corr(score, log T_noX) "
              f"{rr:+.3f}; within-key corr(pred, true log T) {within_key_corr(keyB[jb], lt_[jb], ltA[ja], wB[jb]):+.3f}",
              flush=True)
        del p_
    del scores

    # dT_b
    tbs = {"noX": tb_cells(L, A, A["T"], MA["rho_mean"], nfac, MA["J"], F[RUNS[0]], a.chunk, "noX"),
           "X": tb_cells(L, B, B["T"], MB["rho_mean"], nfac, MB["J"], FB, a.chunk, "X")}
    for v in VARIANTS:
        tbs[v] = tb_cells(L, B, 10**lts[v], MB["rho_mean"], nfac, MB["J"], FB, a.chunk, v)
    VA, VB = A["V"].astype(np.float64), B["V"].astype(np.float64)
    Vc = {k: (VA if k == "noX" else VB) for k in tbs}
    fl = {k: (F[RUNS[0]] if k == "noX" else FB) for k in tbs}
    mean = {k: float(np.sum(Vc[k] * tbs[k]) / Vc[k].sum()) for k in tbs}
    eff = mean["X"] - mean["noX"]
    print(f"\nsnapshot {a.snap}, z = {z:.3f}: <dT_b> [mK] no X (truth) {mean['noX']:+.4f} | X {mean['X']:+.4f}; "
          f"true X-ray effect {eff:+.4f}")
    print("   variant  | predicted no X | error of the no-X mean | predicted / true X-ray effect")
    for v in VARIANTS:
        e = mean["X"] - mean[v]
        print(f"   {v:8s} | {mean[v]:+14.4f} | {mean[v] - mean['noX']:+22.4f} | {e / eff if eff != 0 else np.nan:.4f}")
    fc = os.path.join(od, f"closure_{a.snap:03d}.npz")
    if os.path.exists(fc):
        print(f"   check: tx here {mean['tx']:+.4f} vs closure_noX_xray pred_all {float(np.load(fc)['means'][2]):+.4f} "
              f"(histogram ranks there, exact ranks here)")

    # error by X-run class (paired cells)
    xB = np.clip(B["x"], 0, 1)
    Vtot = Vc["noX"].sum()
    print("   paired cells by the X run's class, share of sum V dT_b [mK x box]: truth | " + " | ".join(VARIANTS))
    rows = []
    for lo, hi in zip(CBINS[:-1], CBINS[1:]):
        for ix, xl in enumerate(XLAB):
            xlo = 0.0 if ix == 0 else XCL[ix - 1]
            xhi = XCL[ix] if ix < len(XCL) else 2.0
            s = (oneB[ib] >= lo) & (oneB[ib] < hi) & (xB[ib] >= xlo) & (xB[ib] < xhi) & ~B["sf"][ib]
            if not s.any():
                continue
            t_ = np.sum(Vc["noX"][ia[s]] * tbs["noX"][ia[s]]) / Vtot
            pv = [np.sum(VB[ib[s]] * tbs[v][ib[s]]) / Vtot for v in VARIANTS]
            rows.append([lo, hi, ix, t_, *pv])
            if max(abs(t_), *map(abs, pv)) > 1e-3 * max(abs(mean["noX"]), 1e-3):
                print(f"     1+d {lo:5g}-{hi:<5g} {xl:9s}: {t_:+9.4f} | " + " | ".join(f"{q:+9.4f}" for q in pv))

    # spectra (diagnostic)
    Lb = box / 1000.0
    maps = {k: cmap(fl[k], Vc[k], tbs[k].astype(np.float64), N)[1] for k in tbs}
    res = dict(z=z, snap=a.snap, render=render, smooth=a.smooth, r_env=r_env, variants=np.array(VARIANTS),
               means=np.array([mean[k] for k in ("noX", "X") + VARIANTS]), class_rows=np.array(rows))
    mt, mx = maps["noX"], maps["X"]
    np.seterr(invalid="ignore", divide="ignore")
    for v in VARIANTS:
        kc, Dp, Dt, r, nind = ps_pair(maps[v], mt, Lb, 8)
        _, De_p, De_t, re, _ = ps_pair(mx - maps[v], mx - mt, Lb, 8)
        ok = nind > 0
        print(f"   64^3, {v:7s}: Delta^2 pred / true " + " ".join(f"{q:.2f}:{x_:.3f}" for q, x_ in zip(kc[ok], (Dp / Dt)[ok]))
              + f"; r min {np.nanmin(r[ok]):.3f} | effect map " + " ".join(f"{x_:.2f}" for x_ in (De_p / De_t)[ok])
              + f"; r min {np.nanmin(re[ok]):.3f}")
        res.update({"k64": kc, "D2true64": Dt, f"D2_{v}64": Dp, f"r_{v}64": r, f"D2eff_{v}64": De_p, "D2eff_true64": De_t,
                    f"reff_{v}64": re})
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"closure_place_{a.snap:03d}.npz")
    np.savez(fn + ".part.npz", **res)
    os.replace(fn + ".part.npz", fn)
    print(f"\nwrote {fn} ({time.time() - t00:.0f} s)")


if __name__ == "__main__":
    main()
