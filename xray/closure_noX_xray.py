"""
closure_noX_xray.py
===================
Closure test of the post-processed "no X-rays" model on the 50 cMpc pair (user, 2026-11-20: Lumina without X-rays at
gas-cell level down to z ~ 5.5; xray/NOTES.md). The model uses only what Lumina has (the X-ray run's cells): every
non-star-forming cell gets T_noX from an own-rank quantile map, keyed by its 0.05-dex bin of 1+delta and its x_HII
class (< 1e-3, 1e-3-0.1, 0.1-0.9, >= 0.9), all from the X-ray run's state:
    p = H I-weighted CDF of T_X among the target cells of the same key (the target box's own ranks),
    T_noX = the training no-X-ray T at quantile p (H I-weighted with the X run's H I, as Lumina can), training = pairs
    by ParticleIDs (good: separation < 2 cell sizes) non-star-forming in both runs.
x_HII, f_HeII, f_HeIII, density and J_alpha stay those of the X-ray run (the X-ray effect is almost all heating:
H I term <= 0.09 mK; J_alpha <= 1-3%). Star-forming cells keep their X-run state (neutral, x_c >> 1).
Then the 21 cm delta T_b of EVERY cell (tb_physics.Line spin model 1, tau_GP from the cell's n_HI: the "cell" option
of cells_Tb_xray.py), four times:
    noX  = the true no-X-ray run (its own cells, J of the no-X run),
    X    = the X-ray run,
    pred_all   = X-run cells with T from the model trained on all pairs (what is applied to Lumina),
    pred_cross = the same, trained on a random half of the 16^3 blocks and applied to the other half (both ways):
                 an out-of-sample check (the target's own ranks come from the target half).
Reported: the box means (sum V dT_b / sum V), the predicted vs true X-ray effect (X - pred vs X - noX), the error by
class of 1+delta x x_HII (paired cells, X-run classes), Delta^2 ratios and r(k) of the 128^3 and 64^3 cell maps (each
cell at its nearest pixel, sum V dT_b / sum V) and of the X-ray-effect maps. Check: the no-X and X box means against
cells_Tb_xray.py (mean_cells_cell) where that output exists.
Writes <out>/pair_cells/closure_NNN.npz with the training tables of pred_all (cumulative H I-weighted histograms of
log10 T_noX per key) for applying to Lumina, the means, spectra and 64^3 maps.

  python closure_noX_xray.py --snap 57            (render from SNAP_RENDER; J_alpha from <out>/<run>/Jalpha)
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
from cells_Tb_xray import hdr_value, load_3d, ps_pair, csum8, Line, M_H, CBINS  # noqa: E402
import h5py                                                                       # noqa: E402

LD = np.arange(-1.5, 3.0001, 0.05)                 # log10(1+delta) edges (X run, gas mean)
LT = np.arange(0.0, 7.00001, 0.001)                # log10 T edges (0.001 dex: the 5 K floor is a spike)
ND, NT = len(LD) - 1, len(LT) - 1
XCL = np.array([1e-3, 0.1, 0.9])                   # x_HII class edges
NX = len(XCL) + 1
NK = ND * NX
MIN_TRAIN = 50
BLOCK = 8                                          # 128^3 -> 16^3 blocks
XLAB = ("x<1e-3", "1e-3-0.1", "0.1-0.9", ">=0.9")


def keys_of(one, x):
    kd = np.clip(np.searchsorted(LD, np.log10(np.maximum(one, 1e-30)), side="right") - 1, 0, ND - 1)
    return kd * NX + np.searchsorted(XCL, x, side="right")


def tbin(lt):
    return np.clip(np.searchsorted(LT, lt, side="right") - 1, 0, NT - 1)


def hist(key, lt, w):
    H = np.bincount(key * NT + tbin(lt), w, NK * NT).reshape(NK, NT)
    return H, np.bincount(key, minlength=NK)


def fallback(n):
    """for each key: itself if it has >= MIN_TRAIN training cells, else the nearest 1+delta bin of the same x class
    that has, else the nearest key of any class"""
    ok = n >= MIN_TRAIN
    if not ok.any():
        sys.exit("STOP: no populated key in the training set")
    out = np.arange(NK)
    kd, kx = out // NX, out % NX
    for k in np.flatnonzero(~ok):
        same = np.flatnonzero(ok & (kx == kx[k]))
        cand = same if len(same) else np.flatnonzero(ok)
        out[k] = cand[np.argmin(np.abs(cand // NX - kd[k]) * NX + np.abs(cand % NX - kx[k]))]
    return out


def own_p(H, key, lt):
    """H I-weighted CDF of the cell's own key at lt (linear within the bin)"""
    c = np.cumsum(H, axis=1)
    it = tbin(lt)
    lo = c[key, it] - H[key, it]
    frac = np.clip((lt - LT[it]) / (LT[1] - LT[0]), 0.0, 1.0)
    tot = c[key, -1]
    return np.where(tot > 0, (lo + frac * H[key, it]) / np.where(tot > 0, tot, 1.0), 0.5)


def inv_cdf(H, key, p):
    """log10 T at quantile p of H[key] (per key)"""
    out = np.empty(len(p))
    o = np.argsort(key, kind="stable")
    ks = key[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    en = np.r_[st[1:], len(ks)]
    for a0, a1 in zip(st, en):
        k = ks[a0]
        j = o[a0:a1]
        cum = np.r_[0.0, np.cumsum(H[k])]
        if cum[-1] <= 0:
            sys.exit(f"STOP: empty training key {k} after fallback")
        out[j] = np.interp(p[j], cum / cum[-1], LT)
    return out


def predict(Htrain, ntrain, Htarget, key, lt):
    fb = fallback(ntrain)
    return inv_cdf(Htrain, fb[key], own_p(Htarget, key, lt))


def tb_cells(L, C, T, rho_bar, nfac, J, flat, chunk, label):
    """dT_b [mK] of every cell (render convention: SF cells neutral); T overrides C's T"""
    n = len(T)
    tb = np.empty(n, np.float32)
    t0 = time.time()
    for i0 in range(0, n, chunk):
        sl = slice(i0, min(n, i0 + chunk))
        sf = C["sf"][sl]
        rho = C["rho"][sl].astype(np.float64)
        x = np.where(sf, 0.0, np.clip(C["x"][sl].astype(np.float64), 0.0, 1.0))
        f2 = np.where(sf, 0.0, C["f2"][sl].astype(np.float64))
        f3 = np.where(sf, 0.0, C["f3"][sl].astype(np.float64))
        try:
            t = L.tb(rho * (1.0 - x) / rho_bar, nfac * rho, T[sl].astype(np.float64), x, f2, f3, J[flat[sl]])[0]
        except ValueError as e:
            sys.exit(f"STOP: {label} cells {sl.start}-{sl.stop}: {e}")
        if not np.all(np.isfinite(t)):
            sys.exit(f"STOP: {label}: non-finite dT_b in cells {sl.start}-{sl.stop}")
        tb[sl] = t
    print(f"  dT_b {label}: {n:,} cells in {time.time() - t0:.0f} s", flush=True)
    return tb


def cmap(flat, V, tb, N):
    sv = np.bincount(flat, V, minlength=N**3)
    st = np.bincount(flat, V * tb, minlength=N**3)
    m128 = np.where(sv > 0, st / np.where(sv > 0, sv, 1.0), 0.0).reshape((N,) * 3)
    sv8, st8 = csum8(sv.reshape((N,) * 3)), csum8(st.reshape((N,) * 3))
    m64 = np.where(sv8 > 0, st8 / np.where(sv8 > 0, sv8, 1.0), 0.0)
    return m128, m64, int(np.sum(sv == 0)), int(np.sum(sv8 == 0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--render", type=int, default=None)
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
    Tg = L.Tg
    print(f"snapshot {a.snap}, render {render}: z = {z:.5f}, T_gamma = {Tg:.2f} K", flush=True)

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

    # pairs (training)
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
    sf_mis = wA[ia[(~A["sf"][ia]) & B["sf"][ib]]].sum() + wA[ia[A["sf"][ia] & ~B["sf"][ib]]].sum()
    print(f"pairs: {len(ia):,} good of {len(A['id']):,} / {len(B['id']):,} cells (duplicated IDs {dA} / {dB}); "
          f"H I share of the no-X run in good pairs {wA[ia].sum() / wA.sum():.4f}; star-forming in one run only "
          f"{sf_mis / wA.sum():.5f} of the H I", flush=True)
    ja, jb = ia[nsf], ib[nsf]

    # keys and the model (X-run state only)
    keyB = keys_of(B["rho"].astype(np.float64) / MB["rho_mean"], np.clip(B["x"], 0, 1))
    ltB = np.log10(np.maximum(B["T"].astype(np.float64), 1.0))
    ltA = np.log10(np.maximum(A["T"].astype(np.float64), 1.0))
    tgt = ~B["sf"]                                                   # cells that get a model T
    Htr, ntr = hist(keyB[jb], ltA[ja], wB[jb])
    Htg, _ = hist(keyB[tgt], ltB[tgt], wB[tgt])
    lt_all = ltB.copy()
    lt_all[tgt] = predict(Htr, ntr, Htg, keyB[tgt], ltB[tgt])
    # cross: 16^3 blocks in two random halves
    nb_ = N // BLOCK
    i0, i1, i2 = np.unravel_index(F[RUNS[1]], (N,) * 3)
    blk = np.ravel_multi_index((i0 // BLOCK, i1 // BLOCK, i2 // BLOCK), (nb_,) * 3)
    del i0, i1, i2
    half = np.zeros(nb_**3, np.int8)
    half[np.random.default_rng(a.seed).permutation(nb_**3)[: nb_**3 // 2]] = 1
    hB = half[blk]
    lt_cross = ltB.copy()
    for hh in (0, 1):
        tr = hB[jb] == 1 - hh
        Ht, nt = hist(keyB[jb[tr]], ltA[ja[tr]], wB[jb[tr]])
        tg = tgt & (hB == hh)
        Hg, _ = hist(keyB[tg], ltB[tg], wB[tg])
        lt_cross[tg] = predict(Ht, nt, Hg, keyB[tg], ltB[tg])
    del blk
    print(f"model: {NK} keys, {int(np.sum(ntr >= MIN_TRAIN))} with >= {MIN_TRAIN} training cells "
          f"({wB[jb][ntr[keyB[jb]] >= MIN_TRAIN].sum() / wB[jb].sum():.4f} of their H I)", flush=True)
    # cell-level T check on the paired cells
    for nm, lt_ in (("pred_all", lt_all), ("pred_cross", lt_cross), ("X", ltB)):
        w_ = wA[ja]
        hm_p = w_.sum() / np.sum(w_ / 10**lt_[jb])
        hm_t = w_.sum() / np.sum(w_ / 10**ltA[ja])
        print(f"  {nm:10s}: paired non-SF H I <1/T>^-1 {hm_p:9.3f} K (truth {hm_t:.3f}); rms log10(T/T_true) "
              f"{np.sqrt(np.sum(w_ * (lt_[jb] - ltA[ja])**2) / w_.sum()):.3f}")

    # dT_b of every cell
    TA, TB = A["T"], B["T"]
    tbs = {"noX": tb_cells(L, A, TA, MA["rho_mean"], nfac, MA["J"], F[RUNS[0]], a.chunk, "noX"),
           "X": tb_cells(L, B, TB, MB["rho_mean"], nfac, MB["J"], F[RUNS[1]], a.chunk, "X"),
           "pred_all": tb_cells(L, B, 10**lt_all, MB["rho_mean"], nfac, MB["J"], F[RUNS[1]], a.chunk, "pred_all"),
           "pred_cross": tb_cells(L, B, 10**lt_cross, MB["rho_mean"], nfac, MB["J"], F[RUNS[1]], a.chunk,
                                  "pred_cross")}
    VA, VB = A["V"].astype(np.float64), B["V"].astype(np.float64)
    Vc = {"noX": VA, "X": VB, "pred_all": VB, "pred_cross": VB}             # shared arrays, not copies
    fl = {"noX": F[RUNS[0]]}
    for k in ("X", "pred_all", "pred_cross"):
        fl[k] = F[RUNS[1]]
    mean = {k: float(np.sum(Vc[k] * tbs[k]) / Vc[k].sum()) for k in tbs}
    print(f"\nsnapshot {a.snap}, z = {z:.3f}: <dT_b> [mK] over all cells: no X (truth) {mean['noX']:+.4f} | X {mean['X']:+.4f}"
          f" | predicted no X: all {mean['pred_all']:+.4f}, cross {mean['pred_cross']:+.4f}")
    eff = mean["X"] - mean["noX"]
    for k in ("pred_all", "pred_cross"):
        e = mean["X"] - mean[k]
        print(f"   X-ray effect: true {eff:+.4f}, {k} {e:+.4f} (ratio {e / eff if eff != 0 else np.nan:.4f}); error of "
              f"the no-X mean {mean[k] - mean['noX']:+.4f} mK")
    ref = {}
    for run, k in ((RUNS[0], "noX"), (RUNS[1], "X")):
        fp = os.path.join(a.out, run, "cells_Tb", f"cells_Tb_{a.snap:03d}.npz")
        if os.path.exists(fp):
            ref[k] = float(np.load(fp)["mean_cells_cell"])
            print(f"   check against cells_Tb_xray.py ({fp}): {k} {ref[k]:+.5f} vs here {mean[k]:+.5f}")

    # error by X-run class, paired cells
    oneB = B["rho"].astype(np.float64) / MB["rho_mean"]
    xB = np.clip(B["x"], 0, 1)
    print("   paired cells by the X run's class: share of sum V dT_b (no X truth | predicted all | cross) [mK x box]")
    rows = []
    Vtot = Vc["noX"].sum()
    for lo, hi in zip(CBINS[:-1], CBINS[1:]):
        for ix, xl in enumerate(XLAB):
            xlo = 0.0 if ix == 0 else XCL[ix - 1]
            xhi = XCL[ix] if ix < len(XCL) else 2.0
            s = (oneB[ib] >= lo) & (oneB[ib] < hi) & (xB[ib] >= xlo) & (xB[ib] < xhi) & ~B["sf"][ib]
            if not s.any():
                continue
            t_ = np.sum(Vc["noX"][ia[s]] * tbs["noX"][ia[s]]) / Vtot
            pa = np.sum(Vc["pred_all"][ib[s]] * tbs["pred_all"][ib[s]]) / Vtot
            pc = np.sum(Vc["pred_cross"][ib[s]] * tbs["pred_cross"][ib[s]]) / Vtot
            rows.append([lo, hi, ix, t_, pa, pc])
            if max(abs(t_), abs(pa)) > 1e-3 * max(abs(mean["noX"]), 1e-3):
                print(f"     1+d {lo:5g}-{hi:<5g} {xl:9s}: {t_:+9.4f} | {pa:+9.4f} | {pc:+9.4f}")
    rest = {}
    for nm, sa_, sb_ in (("star-forming (X run)", None, B["sf"][ib]),):
        rest[nm] = [np.sum(Vc["noX"][ia[sb_]] * tbs["noX"][ia[sb_]]) / Vtot,
                    np.sum(Vc["pred_all"][ib[sb_]] * tbs["pred_all"][ib[sb_]]) / Vtot,
                    np.sum(Vc["pred_cross"][ib[sb_]] * tbs["pred_cross"][ib[sb_]]) / Vtot]
    pa_ = np.zeros(len(A["V"]), bool); pa_[ia] = True
    pb_ = np.zeros(len(B["V"]), bool); pb_[ib] = True
    rest["unpaired cells"] = [np.sum(Vc["noX"][~pa_] * tbs["noX"][~pa_]) / Vtot,
                              np.sum(Vc["pred_all"][~pb_] * tbs["pred_all"][~pb_]) / Vtot,
                              np.sum(Vc["pred_cross"][~pb_] * tbs["pred_cross"][~pb_]) / Vtot]
    for nm, v in rest.items():
        print(f"     {nm:29s}: {v[0]:+9.4f} | {v[1]:+9.4f} | {v[2]:+9.4f}")
        rows.append([np.nan, np.nan, -1 if nm.startswith("star") else -2, *v])
    del pa_, pb_

    # maps and spectra
    Lb = box / 1000.0
    maps = {k: cmap(fl[k], Vc[k], tbs[k].astype(np.float64), N) for k in tbs}
    res = dict(z=z, snap=a.snap, render=render, Tgamma=Tg, means=np.array([mean[k] for k in ("noX", "X", "pred_all",
                                                                                              "pred_cross")]),
               ref=np.array([ref.get("noX", np.nan), ref.get("X", np.nan)]), class_rows=np.array(rows),
               ld=LD, lt=LT, xcl=XCL, H_train_all=Htr.astype(np.float32), n_train_all=ntr, min_train=MIN_TRAIN)
    for tag, i, nbins in (("128", 0, 10), ("64", 1, 8)):
        mt, mx = maps["noX"][i], maps["X"][i]
        for k in ("pred_all", "pred_cross"):
            mp = maps[k][i]
            kc, Dp, Dt, r, nind = ps_pair(mp, mt, Lb, nbins)
            _, De_p, De_t, re, _ = ps_pair(mx - mp, mx - mt, Lb, nbins)
            ok = nind > 0
            print(f"   {tag}^3, {k}: Delta^2 pred / true " + " ".join(f"{q:.2f}:{x_:.3f}" for q, x_ in zip(kc[ok], (Dp / Dt)[ok]))
                  + f"; r min {np.nanmin(r[ok]):.3f}")
            print(f"   {tag}^3, {k}: X-ray effect map Delta^2 pred / true "
                  + " ".join(f"{q:.2f}:{x_:.3f}" for q, x_ in zip(kc[ok], (De_p / De_t)[ok])) + f"; r min {np.nanmin(re[ok]):.3f}")
            res.update({f"k{tag}": kc, f"D2true{tag}": Dt, f"D2_{k}{tag}": Dp, f"r_{k}{tag}": r,
                        f"D2eff_true{tag}": De_t, f"D2eff_{k}{tag}": De_p, f"reff_{k}{tag}": re, f"nind{tag}": nind})
        print(f"   empty pixels at {tag}^3: " + ", ".join(f"{k} {maps[k][2 + i]}" for k in maps))
    for k in maps:
        res[f"map64_{k}"] = maps[k][1].astype(np.float32)
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, f"closure_{a.snap:03d}.npz")
    np.savez(fn + ".part.npz", **res)
    os.replace(fn + ".part.npz", fn)
    print(f"\nwrote {fn} ({time.time() - t00:.0f} s)")


if __name__ == "__main__":
    main()
