"""
cross_spectrum.py
=================
Auto- and cross-power spectra of one reference 640^3 map A against one or more maps B, and the correlation
coefficient
    r_AB(k) = P_AB(k) / sqrt(P_AA(k) P_BB(k)),   P_AB(k) = < Re[ A~(k) B~*(k) ] >_shell  (mean of each map removed)
in log bins of |k| (rfft half-space, each independent mode once). r = +1: B is A times a positive factor in that k
bin (up to the bin's mode scatter); r = -1: a negative factor; r ~ 0: no linear relation at that scale. Also prints
the "bias" b_AB = P_AB / P_BB (A ~ b B + uncorrelated part) and the 1-sigma error of r for Gaussian fields,
(1 - r^2) / sqrt(2 N_modes) (each complex mode has two real degrees of freedom; before 2026-11-19 the code printed
(1 - r^2) / sqrt(N_modes), sqrt(2) too large). Bin edges at 10^(j/per_decade) h/cMpc (from 2026-11-19; before,
k_f 10^(j/per_decade)), so r(k) and the notebooks' rebinned Delta^2(k) share bins.

For the 3.46 cm question (user, 2026-11-18): does the gas-cell delta T_b^He map carry the He III bubbles? A =
cell_test*/RIN_cell_He_NNN.hdf5 (gas-cell delta T_b^He), B = the render fields at the matched render:
HeIII_VolumeFraction (bubbles), HeII_VolumeFraction (where He II is), Density (matter / haloes). Same 640^3 grid and
index order (the merge writes the cell maps in the production index order).

Usage
-----
  python cross_spectrum.py --a RIN_cell_He_116.hdf5:rin \\
      --b ren_640/HeIII_VolumeFraction/HeIII_VolumeFraction_406.hdf5:HeIII_VolumeFraction \\
      --b ren_640/HeII_VolumeFraction/HeII_VolumeFraction_406.hdf5:HeII_VolumeFraction \\
      --b ren_640/Density/Density_406.hdf5:Density [--box 338300] [--per-decade 10] [--out cross_116.npz]
Each map is file:dataset. k in h/cMpc. Memory ~12 GB at 640^3.
"""
import argparse
import os

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


def load(spec):
    path, _, dset = spec.partition(":")
    with h5py.File(path, "r") as f:
        if not dset:
            names = [k for k in f if isinstance(f[k], h5py.Dataset) and f[k].ndim == 3]
            if len(names) != 1:
                raise SystemExit(f"{path}: give the dataset as {path}:<name> (3-D datasets: {names})")
            dset = names[0]
        m = f[dset][()].astype(np.float64)
    if m.ndim != 3 or len(set(m.shape)) != 1:
        raise SystemExit(f"{spec}: not an N^3 map ({m.shape})")
    if not np.all(np.isfinite(m)):
        raise SystemExit(f"{spec}: {np.count_nonzero(~np.isfinite(m))} non-finite values")
    return m, f"{os.path.basename(path)}:{dset}"


class Bins:
    def __init__(self, N, L, per_decade):
        kf = 2 * np.pi / L
        ii = np.abs(np.fft.fftfreq(N, 1.0 / N))
        k = kf * np.sqrt(ii[:, None, None]**2 + ii[None, :, None]**2 + np.arange(N // 2 + 1)[None, None, :]**2)
        lo, hi = kf, kf * N / 2
        # edges at 10^(j/per_decade) h/cMpc, the grid of the notebooks' rebinned auto spectra (audit 2026-11-19: they
        # were at k_f 10^(j/per_decade), 0.07 dex off), plus k_f and k_f N/2 at the ends (partial first/last bins)
        j = np.arange(np.ceil(np.log10(lo) * per_decade - 1e-9), np.floor(np.log10(hi) * per_decade + 1e-9) + 1)
        e = 10.0**(j / per_decade)
        e = e[(e > lo * (1 + 1e-9)) & (e < hi * (1 - 1e-9))]
        self.edges = np.concatenate([[lo], e, [hi]])
        self.idx = np.digitize(k, self.edges).ravel()            # 0: k = 0; len(edges): beyond the last edge
        # each independent mode once: on the planes kz = 0 and kz = N/2 the half (ky, kx) < 0 duplicates the rest
        w = np.ones(k.shape)
        f = np.fft.fftfreq(N, 1.0 / N)
        ix, iy = np.meshgrid(f, f, indexing="ij")              # [i, j] = (n_x, n_y) of the array element
        dup = (ix < 0) | ((ix == 0) & (iy < 0))                # the conjugate of (n_x, n_y, 0) is (-n_x, -n_y, 0)
        for p in (0, N // 2):
            w[:, :, p][dup] = 0.0
        self.w = w.ravel()
        self.nb = len(self.edges) - 1
        self.n = self.sum(np.ones(k.size))
        self.k = np.exp(self.sum(np.log(np.where(k.ravel() > 0, k.ravel(), 1.0))) / np.maximum(self.n, 1))

    def sum(self, a):
        return np.bincount(self.idx, weights=a * self.w, minlength=self.nb + 2)[1:self.nb + 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="reference map, file:dataset")
    p.add_argument("--b", action="append", required=True, help="map to cross with A, file:dataset (repeat)")
    p.add_argument("--box", type=float, default=338300.0, help="BoxSize [ckpc/h]")
    p.add_argument("--per-decade", type=int, default=10)
    p.add_argument("--out", help="npz with k, N_modes, P_AA, and per B: P_BB, P_AB, r")
    a = p.parse_args()

    A, nameA = load(a.a)
    N, L = A.shape[0], a.box / 1e3
    bins = Bins(N, L, a.per_decade)
    FA = np.fft.rfftn(A - A.mean()).ravel()
    del A
    norm = L**3 / float(N)**6
    PAA = bins.sum(np.abs(FA)**2) / np.maximum(bins.n, 1) * norm
    use = bins.n > 0
    rec = {"k": bins.k[use], "N_modes": bins.n[use], "P_AA": PAA[use], "A": nameA}
    print(f"A = {nameA}: N = {N}, box {L:.1f} cMpc/h, {int(use.sum())} k bins")
    for j, spec in enumerate(a.b):
        B, nameB = load(spec)
        if B.shape != (N, N, N):
            raise SystemExit(f"{spec}: shape {B.shape} differs from A")
        FB = np.fft.rfftn(B - B.mean()).ravel()
        del B
        PBB = bins.sum(np.abs(FB)**2) / np.maximum(bins.n, 1) * norm
        PAB = bins.sum((FA * np.conj(FB)).real) / np.maximum(bins.n, 1) * norm
        del FB
        with np.errstate(invalid="ignore", divide="ignore"):        # bins without modes (not printed or saved)
            r = PAB / np.sqrt(PAA * PBB)
        err = (1 - r**2) / np.sqrt(2.0 * np.maximum(bins.n, 1))      # 2 real d.o.f. per complex mode (audit 2026-11-19)
        print(f"\n  B{j} = {nameB}")
        print(f"  {'k [h/cMpc]':>10} {'modes':>9} {'r_AB':>8} {'+-':>6} {'b = P_AB/P_BB':>14}")
        for i in np.flatnonzero(use):
            print(f"  {bins.k[i]:10.4f} {int(bins.n[i]):9d} {r[i]:+8.3f} {err[i]:6.3f} {PAB[i] / PBB[i]:14.4e}")
        rec.update({f"B{j}": nameB, f"P_BB_{j}": PBB[use], f"P_AB_{j}": PAB[use], f"r_{j}": r[use]})
    if a.out:
        np.savez(a.out, **rec)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
