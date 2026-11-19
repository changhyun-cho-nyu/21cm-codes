"""
check_cells_HeIII.py
====================
Is the gas-cell x_HeIII the same as the render's (HeIII_VolumeFraction / y), pixel by pixel? (user, 2026-11-18)
The whole-box scan sums (test_cells_Tb.py) hold x_HII and f_HeII only (mass-weighted), not He III. But every scan task
file also keeps all gas cells of --n-cube random cubes (cube_QQQ groups: pos, V, xHII, fII, fIII, ...), with the
render conventions (star-forming cells x_HII = f_HeII = f_HeIII = 0). From those cells this builds, in every 640^3
pixel of a cube, the volume-weighted fractions sum(V f) / sum(V) with the merge's pixel assignment
(render_index(pos, box, N, order), order from the scan files), and compares them with the render's volume-weighted
fields at the same pixels:
    x_HII   : HII_VolumeFraction        vs cells xHII
    x_HeII  : HeII_VolumeFraction / y   vs cells fII / y
    x_HeIII : HeIII_VolumeFraction / y  vs cells fIII / y      <- the question
Per field: median and 99th percentile of |render - cells| per pixel, the correlation of the two pixel sets, and the
means. Also the gas volume per pixel / pixel volume (the cells must fill the cube's pixels). The deposits differ
(render: each cell as a cube of its own volume, overlap-weighted; here: the cell centre's pixel), so small differences
at pixel level are expected; the question is whether they are as small as for x_HII and x_HeII (the merge's
mass-weighted check: 1e-7 - 1e-5 per pixel).

Usage
-----
  python check_cells_HeIII.py --scandir /orcd/data/mvogelsb/004/chcho/cell_test_116 --snap 116 \\
      --render .../Flagship_new/postprocessing/ren_640 [--n-cubes 8]
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

Y_HE = (1.0 - 0.76) / (4.0 * 0.76)
FIELDS = (("x_HII", "HII_VolumeFraction", "xHII", 1.0),
          ("x_HeII", "HeII_VolumeFraction", "fII", 1.0 / Y_HE),
          ("x_HeIII", "HeIII_VolumeFraction", "fIII", 1.0 / Y_HE))


def render_index(pos, box, N, order):          # as test_cells_Tb.py
    ix = np.floor(pos * (N / box)).astype(np.int64) % N
    return ix[:, order[0]], ix[:, order[1]], ix[:, order[2]]


def field_dset(render, name, snap):
    p = os.path.join(render, name, f"{name}_{snap:03d}.hdf5")
    f = h5py.File(p if os.path.exists(p) else os.path.join(render, "All", f"All_{snap:03d}.hdf5"), "r")
    return f, f[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scandir", required=True, help="directory with scan_He_NNN_TTT.hdf5")
    ap.add_argument("--snap", type=int, required=True, help="full snapshot")
    ap.add_argument("--line", default="He")
    ap.add_argument("--render", required=True, help="ren_640 directory that holds the scan's render_snap")
    ap.add_argument("--n-cubes", type=int, default=8, help="cubes to use (each ~1 GB of cells)")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.scandir, f"scan_{a.line}_{a.snap:03d}_*.hdf5")) if not f.endswith(".part"))
    if not files:
        sys.exit(f"no scan files for snapshot {a.snap} in {a.scandir}")
    hs = [h5py.File(f, "r") for f in files]
    A = dict(hs[0].attrs)
    order, rsnap = tuple(int(x) for x in A["order"]), int(A["render_snap"])
    with h5py.File(os.path.join(a.render, "All", f"All_{rsnap:03d}.hdf5"), "r") as f:
        hd = dict(f["Header"].attrs)
    box, zr = float(hd["BoxSize"]), float(hd["Redshift"])
    R = {k: field_dset(a.render, k, rsnap) for _, k, _, _ in FIELDS}
    N = R["HII_VolumeFraction"][1].shape[0]
    cubes = sorted({g for h in hs for g in h if g.startswith("cube_")})
    use = cubes[:a.n_cubes]
    print(f"full snapshot {a.snap} (z = {float(A['z']):.4f}), render {rsnap} (z = {zr:.4f}), {len(files)} task files, "
          f"{len(cubes)} cubes, using {len(use)}; axis order {order}; N = {N}, box {box:g}")
    acc = {k: ([], []) for k, _, _, _ in FIELDS}
    cover = []
    vpix = (box / N)**3
    for q in use:
        parts = [h[q] for h in hs if q in h]
        pos = np.concatenate([g["pos"][()] for g in parts])
        V = np.concatenate([g["V"][()] for g in parts]).astype(np.float64)
        i, j, k = render_index(pos, box, N, order)
        lo = (i.min(), j.min(), k.min()); hi = (i.max() + 1, j.max() + 1, k.max() + 1)
        if any(h_ - l_ > N // 2 for l_, h_ in zip(lo, hi)):
            sys.exit(f"{q}: spans more than half the box (wraps?): {lo} - {hi}")
        sh = tuple(h_ - l_ for l_, h_ in zip(lo, hi))
        lin = ((i - lo[0]) * sh[1] + (j - lo[1])) * sh[2] + (k - lo[2])
        n = int(np.prod(sh))
        Vp = np.bincount(lin, V, minlength=n)
        full = Vp > 0
        cover.append(Vp[full] / vpix)
        for name, rname, cname, sc in FIELDS:
            f = np.concatenate([g[cname][()] for g in parts]).astype(np.float64)
            cell = np.bincount(lin, V * f, minlength=n)[full] / Vp[full] * sc
            rend = R[rname][1][lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64).ravel()[full] * sc
            acc[name][0].append(cell); acc[name][1].append(rend)
        print(f"  {q}: {len(V):,} cells, {int(full.sum()):,} pixels ({sh[0]}x{sh[1]}x{sh[2]})", flush=True)
    cov = np.concatenate(cover)
    print(f"\n  gas volume per pixel / pixel volume: median {np.median(cov):.4f}, 1-99% {np.percentile(cov, 1):.4f}-"
          f"{np.percentile(cov, 99):.4f} (cube-edge pixels can be partial)")
    print(f"\n  {'field':8s} {'pixels':>9s} {'median|d|':>10s} {'99%|d|':>10s} {'corr':>8s} {'<render>':>10s} {'<cells>':>10s}")
    for name, _, _, _ in FIELDS:
        c = np.concatenate(acc[name][0]); r = np.concatenate(acc[name][1]); d = np.abs(r - c)
        cc = np.corrcoef(c, r)[0, 1] if c.std() > 0 and r.std() > 0 else float("nan")
        print(f"  {name:8s} {len(c):9,d} {np.median(d):10.2e} {np.percentile(d, 99):10.2e} {cc:8.5f} {r.mean():10.5f} {c.mean():10.5f}")
    for h in hs:
        h.close()


if __name__ == "__main__":
    main()
