"""
check_cone_mapping.py
=====================
Where in the simulation box is each light-cone pixel? Needed to sample box fields (the production J_alpha) onto the
light cone. From AREPO src/reionization_lightcone (RT_lightcone_rendering.cc, RT_lightcone.cc): a gas cell at box
position P is drawn at pos = P - Center (periodic, tiled +-1 box across the sky) + (0, 0, UpperDistance), in the
pixel with theta_x = pos_x / pos_z, theta_y = pos_y / pos_z and in the slice that contains |pos|. Inverting, the
pixel (ix, iy) of slice j (centre distance r) is at
    t = ((i + 0.5) / N - 0.5) OpeningAngle,   p = r / sqrt(1 + t_x^2 + t_y^2),
    P = (O_x + t_x p, O_y + t_y p, O_z + p)  mod BoxSize
for a fixed origin O. The merged rlc files keep one Center per run and no UpperDistance, so O is found here
empirically: for a light-cone slice at the time of a render snapshot, the slice's central part (|t| < OpeningAngle/4,
where the sheet is flat to ~1 render cell) is folded onto the box grid and cross-correlated (log density, FFT over
x, y shifts) with every z plane of the render's Density. The best (plane, shift) gives O, and its contrast with the
best peak elsewhere shows whether it is unique. That O is good to ~1 cell; it is then refined in steps of 1/4 cell
on the full slice j (log density, r at the pixel scale), and the tests below use the refined O. First run (2026-10-26, snapshots 156 300 406 501 705): the same
O = (91.711, 39.909, 7.47-7.61) cMpc/h in both runs; the header Center is not O (Center - O = (166, -81, -107) above
z = 4.75, (-21, 81, 139) below); full-slice r 0.55-0.63 for log density at the cell scale.

Then, with that O (or --O), the tests that decide whether the mapping is right below the scales J varies on:
  (a) r against smoothing: cone and sampled render averaged over b x b pixel blocks (b = 1, 2, 4, 8, 16). A right
      mapping: r rises towards 1 (the cell-scale r is the two deposits' noise); a wrong one stays low.
  (b) r against distance from the slice centre, |t| / (OpeningAngle/2) in rings: flat for the right angular geometry;
      a wrong pixel convention or slice radius makes it fall towards the edges.
  (c) O shifted by +-0.5 and +-1 render cell along each axis: r must be highest at 0 shift (O pinned within a cell).
      For slice j's density this holds by construction (the refinement); HII_Fraction and the other slices are the
      independent check.
  (d) the same for HII_Fraction (linear, not log): ionized regions are large-scale, like J. --xmid adds the render
      snapshot with <x_HII>_V closest to that value (ionization_history.csv).
Slices j-20, j-5, j, j+5, j+20 (those that exist; the index runs towards the observer, z decreasing) are compared at
b = 1 and b = 4 (time offset: r should fall only slowly). The default snapshots include 427-430 around the join of
the two runs (428 is the last render of the z > 4.75 run).

Usage
-----
  python check_cone_mapping.py [--snaps 156 300 406 427 428 429 430 501 705] [--xmid 0.5] [--O x y z]
Snapshots <= 428 are matched with the z > 4.75 light cone and render directory, later ones with the z < 4.75 ones.
"""
import argparse
import os

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

BASE = "/orcd/data/mvogelsb/005/mkilab001/mvogelsblab003/Lab/Thesan-XL"
RUNS = {"above": (f"{BASE}/Flagship_new/postprocessing/rlc_640", f"{BASE}/Flagship_new/postprocessing/ren_640"),
        "below": (f"{BASE}/Flagship_below_z_4p75_new/postprocessing/rlc_640",
                  f"{BASE}/Flagship_below_z_4p75_new/postprocessing/ren_640")}
ION = "/orcd/data/mvogelsb/004/chcho/ion_history/ionization_history.csv"
SNAP_SWITCH = 428
BLOCKS = (1, 2, 4, 8, 16)
RINGS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)          # |t| / (OpeningAngle/2); > 1 is the corners
SHIFTS = (-1.0, -0.5, 0.5, 1.0)                   # render cells
OFFSETS = (-20, -5, 0, 5, 20)                     # light-cone slices from j


def render(d, snap, field="Density"):
    """(field, z) of a render snapshot"""
    p = os.path.join(d, field, f"{field}_{snap:03d}.hdf5")
    a = os.path.join(d, "All", f"All_{snap:03d}.hdf5")
    with h5py.File(p if os.path.exists(p) else a, "r") as f:
        x = f[field][()].astype(np.float32)
    with h5py.File(a, "r") as f:
        z = float(f["Header"].attrs["Redshift"])
    return x, z


def cone_geometry(N, theta, r):
    t = ((np.arange(N) + 0.5) / N - 0.5) * theta
    tx, ty = np.meshgrid(t, t, indexing="ij")
    p = r / np.sqrt(1.0 + tx**2 + ty**2)
    return tx, ty, p


def standard(a, log=True):
    a = np.log10(np.maximum(a, 1e-30), dtype=np.float32) if log else a.astype(np.float32)   # float32: 1 GB renders
    m, s = a.mean(dtype=np.float64), a.std(dtype=np.float64)
    return ((a - m) / s).astype(np.float32) if s > 0 else np.zeros_like(a)


def search(cs, tx, ty, p, theta, rho_l, L):
    """best (plane k, shift sx, sy, peak, next-best peak) of the folded central slice against the render planes"""
    Nb = rho_l.shape[0]
    dx = L / Nb
    m = (np.abs(tx) < theta / 4) & (np.abs(ty) < theta / 4)
    ib = np.floor(tx[m] * p[m] / dx).astype(np.int64) % Nb
    jb = np.floor(ty[m] * p[m] / dx).astype(np.int64) % Nb
    cell = ib * Nb + jb
    n = np.bincount(cell, minlength=Nb * Nb)
    s = np.bincount(cell, cs[m], minlength=Nb * Nb)
    occ = n > 0
    C = np.zeros(Nb * Nb)
    C[occ] = s[occ] / n[occ]
    C[occ] -= C[occ].mean()
    C = C.reshape(Nb, Nb)
    FC = np.conj(np.fft.rfft2(C))
    norm = np.sqrt(np.sum(C**2) * occ.sum())
    best = []
    for k in range(Nb):
        R = rho_l[:, :, k]
        cc = np.fft.irfft2(FC * np.fft.rfft2(R), s=(Nb, Nb)) / norm      # ~ Pearson r of C with R shifted
        i = int(np.argmax(cc))
        best.append((cc.flat[i], k, i // Nb, i % Nb))
    best.sort(reverse=True)
    top = best[0]
    # next-best peak away from the best plane (+-3 planes, the same structure)
    other = next(b for b in best[1:] if min(abs(b[1] - top[1]), Nb - abs(b[1] - top[1])) > 3)
    return top, other


def sample(rho_l, O, tx, ty, p, L):
    Nb = rho_l.shape[0]
    dx = L / Nb
    i = np.floor(((O[0] + tx * p) % L) / dx).astype(np.int64) % Nb
    j = np.floor(((O[1] + ty * p) % L) / dx).astype(np.int64) % Nb
    k = np.floor(((O[2] + p) % L) / dx).astype(np.int64) % Nb
    return rho_l[i, j, k]


def refine(c, rho_l, O, g, L, step=0.25, reach=1.5, sweeps=3):
    """O refined below a cell: coordinate sweeps over offsets of +-reach render cells in steps of step cells,
    maximising r (b = 1) of the standardized cone slice c with the render sampled at g = cone_geometry(...)"""
    dx = L / rho_l.shape[0]
    O = O.copy()
    offs = np.arange(-reach, reach + 1e-9, step) * dx
    for _ in range(sweeps):
        for ax in range(3):
            rs = []
            for d in offs:
                O2 = O.copy()
                O2[ax] += d
                rs.append(corr(c, sample(rho_l, O2, *g, L)))
            O[ax] += offs[int(np.nanargmax(rs))]
    return O % L


def block(a, b):
    """mean over b x b pixel blocks (cropped to a multiple of b)"""
    n = (a.shape[0] // b) * b
    return a[:n, :n].reshape(n // b, b, n // b, b).mean(axis=(1, 3))


def corr(a, b, mask=None):
    a, b = (a, b) if mask is None else (a[mask], b[mask])
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def ion_snap(path, x):
    """render snapshot with <x_HII>_V closest to x"""
    import pandas as pd
    ion = pd.read_csv(path, comment="#")
    i = int(np.argmin(np.abs(ion["xHII_V"].values - x)))
    return int(ion["snap"].values[i]), float(ion["xHII_V"].values[i]), float(ion["z"].values[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snaps", type=int, nargs="+", default=[156, 300, 406, 427, 428, 429, 430, 501, 705])
    ap.add_argument("--xmid", type=float, default=0.5, help="add the render with this <x_HII>_V (negative: none)")
    ap.add_argument("--ion", default=ION)
    ap.add_argument("--O", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="origin [cMpc/h] used for the tests (default: each snapshot's own search result)")
    ap.add_argument("--runs", nargs=2, metavar=("CONE_ABOVE", "REN_ABOVE"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--runs-below", nargs=2, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    runs = dict(RUNS)
    if a.runs:
        runs["above"] = tuple(a.runs)
    if a.runs_below:
        runs["below"] = tuple(a.runs_below)
    snaps = list(a.snaps)
    if a.xmid >= 0:
        s, x, z = ion_snap(a.ion, a.xmid)
        print(f"<x_HII>_V = {a.xmid}: render {s} (z = {z:.4f}, <x_HII>_V = {x:.4f})")
        snaps.append(s)
    snaps = sorted(set(snaps))

    found = []
    for snap in snaps:
        run = "above" if snap <= SNAP_SWITCH else "below"
        cdir, rdir = runs[run]
        rho, zr = render(rdir, snap)
        rho_l = standard(rho)
        del rho
        xh, _ = render(rdir, snap, "HII_Fraction")
        with h5py.File(os.path.join(cdir, "All.hdf5"), "r") as f:
            hd = f["Header"].attrs
            L, theta, N = float(hd["BoxSize"]), float(hd["OpeningAngle"]), int(hd["NumPixels"])
            C = f["Header/Center"][()]
            ze, De = f["Redshifts"][()], f["Distances"][()]
            zc = 0.5 * (ze[:-1] + ze[1:])
            j = int(np.argmin(np.abs(zc - zr)))
            js = [j + o for o in OFFSETS if 0 <= j + o < len(zc)]
            slices = {(fl, jj): f[fl][:, :, jj] for jj in js for fl in ("Density", "HII_Fraction")}
        r = 0.5 * (De[j] + De[j + 1])
        tx, ty, p = cone_geometry(N, theta, r)
        dx = L / rho_l.shape[0]
        print(f"\nsnapshot {snap} (z = {zr:.4f}, {run} z = 4.75): light-cone slice {j} (z = {zc[j]:.4f}, r = {r / 1e3:.2f} cMpc/h, "
              f"pixel {theta * r / N / 1e3:.3f} cMpc/h; render cell {dx / 1e3:.3f}); header Center {np.round(C / 1e3, 3)} cMpc/h")

        res = {}
        cs = standard(slices["Density", j])
        for lab, c in (("as stored (ix -> x)", cs), ("transposed", cs.T)):
            top, other = search(c, tx, ty, p, theta, rho_l, L)
            res[lab] = top
            print(f"  {lab:>20}: best plane {top[1]}, shift ({top[2]}, {top[3]}) cells, peak r {top[0]:.3f}; "
                  f"best elsewhere {other[0]:.3f} (plane {other[1]})")
        lab = max(res, key=lambda q: res[q][0])
        pk, k, sx, sy = res[lab]
        O0 = np.array([(sx + 0.5) * dx, (sy + 0.5) * dx, ((k + 0.5) * dx - r)]) % L
        tr = lambda x: x.T if lab.startswith("trans") else x                       # noqa: E731
        O = refine(standard(tr(slices["Density", j])), rho_l, O0, (tx, ty, p), L)
        found.append((snap, run, O))
        print(f"  search ({lab}): O = {np.round(O0 / 1e3, 3)} cMpc/h (to a cell);  refined on slice j's density "
              f"(steps of 1/4 cell): O = {np.round(O / 1e3, 3)} cMpc/h;  Center - O = {np.round(((C - O + L / 2) % L - L / 2) / 1e3, 3)} cMpc/h")
        if a.O is not None:
            O = np.array(a.O) * 1e3
            print(f"  tests with --O = {np.round(O / 1e3, 3)} cMpc/h")

        # the cone slices and the render sampled at the predicted positions, both standardized
        def pair(fl, jj, O_):
            g = cone_geometry(N, theta, 0.5 * (De[jj] + De[jj + 1]))
            if fl == "Density":
                return standard(tr(slices[fl, jj])), sample(rho_l, O_, *g, L)
            return standard(tr(slices[fl, jj]), log=False), standard(sample(xh, O_, *g, L), log=False)

        xmean = float(slices["HII_Fraction", j].mean())
        print(f"  light-cone slice <x_HII> (mass-weighted pixels) = {xmean:.4f}"
              + ("  (HII_Fraction r means little: nearly uniform)" if not 0.02 < xmean < 0.98 else ""))
        for fl in ("Density", "HII_Fraction"):
            c, s = pair(fl, j, O)
            # (a) smoothing
            print(f"  (a) {fl:<12} r vs block [cMpc/h]: " + ", ".join(
                f"{b * theta * r / N / 1e3:.2f}: {corr(block(c, b), block(s, b)):.3f}" for b in BLOCKS))
            # (b) rings
            trr = np.sqrt(tx**2 + ty**2) / (theta / 2)
            out = []
            for b in (1, 4):
                cb, sb, tb = block(c, b), block(s, b), block(trr, b)
                out.append(", ".join(f"{lo:.2f}-{hi:.2f}: {corr(cb, sb, (tb >= lo) & (tb < hi)):.3f}"
                                     for lo, hi in zip(RINGS[:-1], RINGS[1:])))
            print(f"  (b) {fl:<12} r vs |t|/(theta/2), b = 1: {out[0]}\n  {'':>17}b = 4: {out[1]}")
            # (c) shifts of O
            line = []
            for ax, name in enumerate("xyz"):
                for sh in SHIFTS:
                    O2 = O.copy()
                    O2[ax] += sh * dx
                    c2, s2 = pair(fl, j, O2)
                    line.append(f"{name}{sh:+.1f}: {corr(c2, s2):.3f}/{corr(block(c2, 4), block(s2, 4)):.3f}")
            print(f"  (c) {fl:<12} r (b = 1 / b = 4) at 0 shift {corr(c, s):.3f}/{corr(block(c, 4), block(s, 4)):.3f}; "
                  f"O shifted by render cells: " + ", ".join(line))
            # other slices
            line = []
            for jj in js:
                c2, s2 = pair(fl, jj, O)
                line.append(f"{jj - j:+d} (z = {zc[jj]:.4f}): {corr(c2, s2):.3f}/{corr(block(c2, 4), block(s2, 4)):.3f}")
            print(f"      {fl:<12} slices j + n, r (b = 1 / b = 4): " + ", ".join(line))
        del rho_l, xh

    print("\nsummary: refined O [cMpc/h]")
    for snap, run, O in found:
        print(f"  {snap:4d} {run:>5}: {np.round(O / 1e3, 3)}")
    Os = np.array([O for _, _, O in found])
    for run in ("above", "below"):
        sel = [i for i, (_, rr, _) in enumerate(found) if rr == run]
        if sel:
            print(f"  {run}: mean {np.round(Os[sel].mean(0) / 1e3, 3)}, max - min {np.round(np.ptp(Os[sel], 0) / 1e3, 3)} cMpc/h")
    print(f"  all: mean {np.round(Os.mean(0) / 1e3, 3)}, max - min {np.round(np.ptp(Os, 0) / 1e3, 3)} cMpc/h "
          f"(wrapped coordinates: compare by eye if a component sits near 0 or the box size)")


if __name__ == "__main__":
    main()
