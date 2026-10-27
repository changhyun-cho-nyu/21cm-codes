"""
check_cone_mapping.py
=====================
Where in the simulation box is each light-cone pixel? Needed to sample box fields (the production J_alpha) onto the
light cone. From AREPO src/reionization_lightcone (RT_lightcone_rendering.cc, RT_lightcone.cc): a gas cell at box
position P is drawn at pos = P - Center (periodic, tiled +-1 box across the sky) + (0, 0, UpperDistance), in the
pixel with theta_x = pos_x / pos_z, theta_y = pos_y / pos_z and in the slice that contains |pos|. Inverting, the
pixel (ix, iy) of slice j (centre distance r) is at
    t = ((i + 0.5) / N - 0.5) OpeningAngle,   p = r / sqrt(1 + t_x^2 + t_y^2),
    P = (O_x + t_x p, O_y + t_y p, O_z + p)  mod BoxSize,
with O_x, O_y = Center_x, Center_y and O_z = Center_z - UpperDistance, which stays the same from frame to frame
(both drop by the frame depth each time). The merged rlc files keep one Center and no UpperDistance, and the two
runs have different Centers, so O is found here empirically: for a light-cone slice at the time of a render
snapshot, the slice's central part (|t| < OpeningAngle/4, where the sheet is flat to ~1 render cell) is folded onto
the box grid and cross-correlated (log density, FFT over x, y shifts) with every z plane of the render's Density.
The best (plane, shift) gives O, and its contrast with the best peak elsewhere shows whether it is unique. Then the
full slice, and the slices 5 and 20 further out (1 + z larger), are sampled at their predicted positions (nearest
render cell) and correlated with the light cone: a correct, frame-independent O gives a high r for all three
(a wrong one ~ 0); r falls slowly with distance from the render's time.

Usage
-----
  python check_cone_mapping.py [--snaps 156 300 406 501 705]
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
SNAP_SWITCH = 428


def render(d, snap):
    """(Density, z) of a render snapshot"""
    p = os.path.join(d, "Density", f"Density_{snap:03d}.hdf5")
    a = os.path.join(d, "All", f"All_{snap:03d}.hdf5")
    with h5py.File(p if os.path.exists(p) else a, "r") as f:
        rho = f["Density"][()].astype(np.float32)
    with h5py.File(a, "r") as f:
        z = float(f["Header"].attrs["Redshift"])
    return rho, z


def cone_geometry(N, theta, r):
    t = ((np.arange(N) + 0.5) / N - 0.5) * theta
    tx, ty = np.meshgrid(t, t, indexing="ij")
    p = r / np.sqrt(1.0 + tx**2 + ty**2)
    return tx, ty, p


def standard(a):
    a = np.log10(np.maximum(a, 1e-30), dtype=np.float32)       # float32: a 640^3 render is 1 GB
    m, s = a.mean(dtype=np.float64), a.std(dtype=np.float64)
    return ((a - m) / s).astype(np.float32)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snaps", type=int, nargs="+", default=[156, 300, 406, 501, 705])
    ap.add_argument("--runs", nargs=2, metavar=("CONE_ABOVE", "REN_ABOVE"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--runs-below", nargs=2, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    runs = dict(RUNS)
    if a.runs:
        runs["above"] = tuple(a.runs)
    if a.runs_below:
        runs["below"] = tuple(a.runs_below)

    for snap in a.snaps:
        run = "above" if snap <= SNAP_SWITCH else "below"
        cdir, rdir = runs[run]
        rho, zr = render(rdir, snap)
        rho_l = standard(rho)
        del rho
        with h5py.File(os.path.join(cdir, "All.hdf5"), "r") as f:
            hd = f["Header"].attrs
            L, theta, N = float(hd["BoxSize"]), float(hd["OpeningAngle"]), int(hd["NumPixels"])
            C = f["Header/Center"][()]
            ze, De = f["Redshifts"][()], f["Distances"][()]
            zc = 0.5 * (ze[:-1] + ze[1:])
            j = int(np.argmin(np.abs(zc - zr)))
            r = 0.5 * (De[j] + De[j + 1])
            js = [jj for jj in (j, j + 5, j + 20) if jj < len(zc)]
            slices = {jj: f["Density"][:, :, jj] for jj in js}
        tx, ty, p = cone_geometry(N, theta, r)
        cs = standard(slices[j])
        print(f"\nsnapshot {snap} (z = {zr:.4f}, {run} z = 4.75): light-cone slice {j} (z = {zc[j]:.4f}, r = {r / 1e3:.2f} cMpc/h); "
              f"header Center {np.round(C / 1e3, 3)} cMpc/h")
        res = {}
        for lab, c in (("as stored (ix -> x)", cs), ("transposed", cs.T)):
            top, other = search(c, tx, ty, p, theta, rho_l, L)
            res[lab] = top
            print(f"  {lab:>20}: best plane {top[1]}, shift ({top[2]}, {top[3]}) cells, peak r {top[0]:.3f}; "
                  f"best elsewhere {other[0]:.3f} (plane {other[1]})")
        lab = max(res, key=lambda q: res[q][0])
        pk, k, sx, sy = res[lab]
        dx = L / rho_l.shape[0]
        O = np.array([(sx + 0.5) * dx, (sy + 0.5) * dx, ((k + 0.5) * dx - r)]) % L
        print(f"  mapping ({lab}): O = {np.round(O / 1e3, 3)} cMpc/h;  Center - O = {np.round(((C - O + L / 2) % L - L / 2) / 1e3, 3)} "
              f"cMpc/h (O_z = Center_z - UpperDistance: Center_z - O_z = {((C[2] - O[2]) % L) / 1e3:.3f} cMpc/h mod box)")
        for jj in js:
            c = slices[jj] if lab.startswith("as") else slices[jj].T
            tx2, ty2, p2 = cone_geometry(N, theta, 0.5 * (De[jj] + De[jj + 1]))
            rr = np.corrcoef(standard(c).ravel(), sample(rho_l, O, tx2, ty2, p2, L).ravel())[0, 1]
            print(f"    full slice {jj} (z = {zc[jj]:.4f}) vs the render at the predicted positions: r = {rr:.3f}")
        del rho_l


if __name__ == "__main__":
    main()
