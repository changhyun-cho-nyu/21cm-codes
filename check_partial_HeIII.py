"""
check_partial_HeIII.py
======================
Where is the partially ionized He III at the start of He II reionization?

Context (2026-11-23): at renders 406 / 428 (x_HeIII,V = 0.10 / 0.15) the x_HeII bispectrum flips sign at R ~ 1.9 h^-1 cMpc,
while the binary field 1 - [x_HeIII > 0.5] (the split the MFP uses) flips at 4.4 / 5.2, near the MFP mean. About 35% / 28%
of the He III sits in pixels with x_HeIII < 0.5 there (18% / 8% at 452 / 476, where the two agree). This script says where
that gas is, on the WHOLE 640^3 render (no sampling):

  1. x_HeIII histogram: volume share and share of the He III (sum x) per x bin, and the median 1 + delta of each bin.
  2. For the partially ionized pixels (0.01 < x <= 0.5): distance d_ion to the nearest ionized pixel (x > 0.5) and
     d_neu to the nearest neutral one (x <= 0.01); Euclidean distance transforms on the periodic box (wrap-padded by
     --pad cells: exact up to pad, larger values reported as "> pad"). Joint table of their He III share.
  3. Summary (D = 2 cells = 1.06 h^-1 cMpc): front = d_ion <= D and d_neu <= D; inside a bubble (clumps, residual He II
     cores) = d_ion <= D < d_neu; ahead of a bubble = D < d_ion <= 8; far from every bubble = d_ion > 8.

x_HeIII = HeIII_VolumeFraction / y (y = n_He/n_H = 0.0789, the render field is per H nucleus), as in bispec_maps.py and the
MFP code. 1 + delta = Density / mean(Density) of the same render.

Usage:  python check_partial_HeIII.py --snaps 406 428 452 476 [--pad 48] [--out <dir>]
        python check_partial_HeIII.py --mock          # self-test on a small synthetic box with known answers
Output: <out>/partial_HeIII_<snap>.txt (tables) and .npz (histograms).
"""
import argparse
import os
import sys
import time

import numpy as np
import h5py
from scipy import ndimage

try:
    import hdf5plugin  # noqa: F401   (compressed renders)
except ImportError:
    pass

U = "/orcd/data/mvogelsb/004/chcho"
REN = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
DIR_A = f"{REN}/Flagship_new/postprocessing/ren_640"                # render snapshots <= 428
DIR_B = f"{REN}/Flagship_below_z_4p75_new/postprocessing/ren_640"   # render snapshots > 428
SNAP_SWITCH = 428
X_H = 0.76
Y_HE = (1.0 - X_H) / (4.0 * X_H)
BOX = 338.3                                     # h^-1 cMpc
THR = 0.5                                       # ionized: x_HeIII > 0.5 (the MFP's split)
X_EDGES = np.array([0.0, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.9, 1.0 + 1e-6])
D_EDGES = np.array([0.0, 1.01, 2.01, 4.01, 8.01, 16.01, 32.01])   # cells; edges below --pad are used, then pad, "> pad"
DN_EDGES = np.array([0.0, 1.01, 2.01, 4.01, 8.01])                # d_neu bins, last one open
PARTIAL = (0.01, 0.5)                           # "partially ionized"; x <= 0.01 counts as neutral
D_CAT = 2.01                                    # cells, for the summary categories


def read_render(field, snap, dir_a, dir_b):
    d0 = dir_a if snap <= SNAP_SWITCH else dir_b
    for fn in (os.path.join(d0, field, f"{field}_{snap:03d}.hdf5"), os.path.join(d0, "All", f"All_{snap:03d}.hdf5")):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                if field in f:
                    z = float(f["Header"].attrs["Redshift"]) if "Header" in f else np.nan
                    return f[field][()], z, fn
    raise FileNotFoundError(f"no {field} for render {snap} in {d0}")


def distance_to(mask, pad):
    """Euclidean distance (cells) of every pixel to the nearest True pixel of mask, periodic box; exact for distances up to
    `pad` (every periodic image within pad is inside the padded box), and > pad whenever the value returned is > pad"""
    ion = mask
    big = np.pad(ion, pad, mode="wrap")
    d = ndimage.distance_transform_edt(~big)
    n = ion.shape[0]
    d = d[pad:pad + n, pad:pad + n, pad:pad + n].astype(np.float32)
    if not ion.any():
        d[:] = np.inf
    return d


def analyse(x, dens, pad):
    """all the numbers for one box; x = x_HeIII, dens = 1 + delta"""
    N3 = x.size
    tot = float(x.sum(dtype=np.float64))
    ion = x > THR
    res = {"xbar": tot / N3, "f_ion_vol": ion.mean(), "share_below_thr": float(x[~ion].sum(dtype=np.float64)) / tot}
    rows = []
    for lo, hi in zip(X_EDGES[:-1], X_EDGES[1:]):
        m = (x > lo) & (x <= hi) if lo > 0 else (x <= hi)
        nm = int(m.sum())
        rows.append((lo, hi, nm / N3, float(x[m].sum(dtype=np.float64)) / tot,
                     float(np.median(dens[m])) if nm else np.nan))
    res["xbins"] = rows
    part = (x > PARTIAL[0]) & (x <= PARTIAL[1])
    dp = distance_to(ion, pad)[part]
    dn = distance_to(x <= PARTIAL[0], pad)[part]
    xp, rp = x[part].astype(np.float64), dens[part]
    tp = xp.sum()
    dedges = np.concatenate([D_EDGES[D_EDGES < pad], [pad + 0.01, np.inf]])
    nedges = np.append(DN_EDGES, np.inf)
    joint = np.zeros((len(dedges) - 1, len(nedges) - 1))
    for i, (lo, hi) in enumerate(zip(dedges[:-1], dedges[1:])):
        mi = (dp > lo) & (dp <= hi) if lo > 0 else (dp <= hi)
        for j, (a_, b_) in enumerate(zip(nedges[:-1], nedges[1:])):
            mj = (dn > a_) & (dn <= b_) if a_ > 0 else (dn <= b_)
            joint[i, j] = xp[mi & mj].sum() / tp if tp else np.nan
    res["joint"], res["dedges"], res["nedges"] = joint, dedges, nedges
    cats = []
    for lab, sel in (("front (near ionized and neutral)", (dp <= D_CAT) & (dn <= D_CAT)),
                     ("inside a bubble (near ionized only)", (dp <= D_CAT) & (dn > D_CAT)),
                     ("ahead of a bubble (2 < d_ion <= 8)", (dp > D_CAT) & (dp <= 8.01)),
                     ("far from every bubble (d_ion > 8)", dp > 8.01)):
        cats.append((lab, float(sel.sum()) / max(part.sum(), 1), float(xp[sel].sum()) / tp if tp else np.nan,
                     float(np.median(rp[sel])) if sel.any() else np.nan))
    res["cats"] = cats
    res["partial_share"] = tp / tot
    res["hist_x"] = np.histogram(x, bins=np.linspace(0, 1, 101))[0]
    res["hist_d_partial"] = np.histogram(np.minimum(dp, pad + 1), bins=np.arange(0, pad + 3), weights=xp)[0]
    return res


def report(res, head, cell):
    L = [head,
         f"x_HeIII,V = {res['xbar']:.5f}; ionized volume (x > {THR}) = {res['f_ion_vol']:.5f}; "
         f"share of the He III in pixels with x <= {THR}: {res['share_below_thr']:.4f}; "
         f"in {PARTIAL[0]} < x <= {PARTIAL[1]}: {res['partial_share']:.4f}",
         "", "1. x_HeIII bins:  volume share | share of the He III (sum x) | median 1+delta"]
    for lo, hi, fv, fx, md in res["xbins"]:
        L.append(f"   {lo:6.3f} < x <= {min(hi, 1):5.3f}   {fv:10.6f}   {fx:8.5f}   {md:8.3f}")
    de, ne = res["dedges"], res["nedges"]
    lab = lambda lo, hi: f"> {lo - 0.01:.0f}" if np.isinf(hi) else f"{lo:.0f}-{hi:.0f}"
    L += ["", f"2. partially ionized pixels ({PARTIAL[0]} < x <= {PARTIAL[1]}): share of their He III by d_ion (rows) and d_neu",
          f"   (columns), in cells (1 cell = {cell:.4f} h^-1 cMpc); ionized: x > {THR}, neutral: x <= {PARTIAL[0]}",
          "   d_ion \\ d_neu " + "".join(f"{lab(a, b):>9s}" for a, b in zip(ne[:-1], ne[1:])) + "      all"]
    for i, (a, b) in enumerate(zip(de[:-1], de[1:])):
        L.append(f"   {lab(a, b):>12s}  " + "".join(f"{v:9.4f}" for v in res["joint"][i]) + f"{res['joint'][i].sum():9.4f}")
    L += ["", "3. summary:  share of the partially ionized pixels | share of their He III | median 1+delta"]
    for lab_, fn, fx, md in res["cats"]:
        L.append(f"   {lab_:38s} {fn:8.5f}   {fx:8.5f}   {md:8.3f}")
    return "\n".join(L)


def mock():
    """synthetic periodic box: fully ionized spheres with partially ionized shells, a dense partially ionized clump inside
    one sphere, and a separate partially ionized patch far from every sphere; checks the tables against direct counts"""
    n, pad = 64, 16
    g = np.arange(n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    def r_to(c):
        d = [np.minimum(np.abs(A - ci), n - np.abs(A - ci)) for A, ci in zip((X, Y, Z), c)]
        return np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
    x = np.zeros((n, n, n), np.float32)
    c1, c2 = (2, 2, 2), (40, 40, 40)                          # c1 straddles the periodic boundary
    for c in (c1, c2):
        r = r_to(c)
        x[r <= 8] = 1.0
        x[(r > 8) & (r <= 9)] = 0.3                           # partially ionized shell, 1 cell thick (a front)
    x[r_to(c2) <= 1.0] = 0.2                                  # clump inside sphere 2 (7 pixels)
    patch = r_to((20, 52, 20)) <= 2.0                          # isolated patch, far from both spheres
    x[patch] = 0.25
    dens = np.ones_like(x); dens[r_to(c2) <= 1.0] = 50.0
    res = analyse(x, dens, pad)
    ok = True
    def chk(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol * max(1, abs(want))
        ok &= good
        print(f"  {name:55s} got {got:.6f} want {want:.6f} {'ok' if good else 'FAIL'}")
    tot = x.sum(dtype=np.float64)
    part = (x > 0.01) & (x <= 0.5)
    chk("x_HeIII,V", res["xbar"], tot / x.size)
    chk("share of He III in partial pixels", res["partial_share"], x[part].sum(dtype=np.float64) / tot)
    # direct periodic distance from each patch pixel to the nearest ionized pixel (brute force)
    ion = np.argwhere(x > 0.5)
    pp = np.argwhere(patch)
    dmin = min(np.min(np.sqrt((np.minimum(np.abs(ion - p), n - np.abs(ion - p))**2).sum(1))) for p in pp)
    print(f"  (patch: nearest ionized pixel at {dmin:.2f} cells, so it must fall in the > 8 bins)")
    cat = dict((lab, (fn, fx, md)) for lab, fn, fx, md in res["cats"])
    tpart = x[part].sum(dtype=np.float64)
    chk("far from every bubble (= the patch)", cat["far from every bubble (d_ion > 8)"][1], x[patch].sum(dtype=np.float64) / tpart)
    chk("inside a bubble (= the clump, 7 pixels)", cat["inside a bubble (near ionized only)"][1], 7 * 0.2 / tpart)
    chk("median 1+delta inside a bubble", cat["inside a bubble (near ionized only)"][2], 50.0)
    shell = np.zeros_like(patch)
    for c in (c1, c2):
        r = r_to(c); shell |= (r > 8) & (r <= 9)
    chk("front (= the two shells)", cat["front (near ionized and neutral)"][1], x[shell].sum(dtype=np.float64) / tpart)
    chk("joint table sums to 1", float(res["joint"].sum()), 1.0)
    # periodic wrap: the distance field around c1 must be symmetric across the box edge
    d = distance_to(x > 0.5, pad)
    chk("periodic distance, across the edge vs inside", float(d[2, 2, 11]), float(d[2, 2, n - 7]))
    print(report(res, "mock box", 1.0))
    print("mock test", "passed" if ok else "FAILED")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snaps", type=int, nargs="+")
    p.add_argument("--pad", type=int, default=48, help="periodic padding [cells]; distances above it are '> pad'")
    p.add_argument("--out", default=f"{U}/partial_HeIII")
    p.add_argument("--dir-a", default=DIR_A)
    p.add_argument("--dir-b", default=DIR_B)
    p.add_argument("--mock", action="store_true")
    a = p.parse_args()
    if a.mock:
        sys.exit(0 if mock() else 1)
    if not a.snaps:
        sys.exit("--snaps needed")
    os.makedirs(a.out, exist_ok=True)
    for s in a.snaps:
        t0 = time.time()
        f3, z, fn3 = read_render("HeIII_VolumeFraction", s, a.dir_a, a.dir_b)
        rho, z2, fnd = read_render("Density", s, a.dir_a, a.dir_b)
        if f3.shape != rho.shape or f3.ndim != 3:
            sys.exit(f"render {s}: shapes {f3.shape} {rho.shape}")
        x = (f3 / Y_HE).astype(np.float32)
        dens = (rho / rho.mean(dtype=np.float64)).astype(np.float32)
        del f3, rho
        cell = BOX / x.shape[0]
        res = analyse(x, dens, a.pad)
        head = f"render {s}, z = {z:.4f}; {fn3}; {fnd}; N = {x.shape[0]}; pad {a.pad} cells; {time.time() - t0:.0f} s"
        txt = report(res, head, cell)
        print(txt, flush=True)
        with open(os.path.join(a.out, f"partial_HeIII_{s:03d}.txt"), "w") as f:
            f.write(txt + "\n")
        np.savez(os.path.join(a.out, f"partial_HeIII_{s:03d}.npz"), hist_x=res["hist_x"], hist_d_partial=res["hist_d_partial"],
                 x_edges=np.linspace(0, 1, 101), d_edges=np.arange(0, a.pad + 3), z=z)


if __name__ == "__main__":
    main()
