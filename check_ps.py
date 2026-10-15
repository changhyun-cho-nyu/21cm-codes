"""
check_ps.py
===========
Is the power spectrum in PS_NNN.out (PS_21cm_mpi.c / PS_3cm_mpi.c, output_PS) what the map gives?

The C code writes, after a 5-line header (z, number of bins, sqrt(totmass*tot1mass), totvolume, N^3),
lines "k  delta  SumPower  CountModes  shotnoise" in 2000 log bins of k (k in h/ckpc, the render
units), and the notebooks (load_ps) use Delta^2 = delta x (line 3 / line 4)^2.
Here Delta^2(k) = k^3 P(k) / (2 pi^2) is computed independently from the RIN_NNN.hdf5 map with numpy
(P = |FFT(dT_b - <dT_b>)|^2 V / N^6, the same half-space modes and bins as the C code), and compared.
Agreement (ratio 1 in every bin with modes) confirms the normalisation convention end to end.

Usage
-----
  python check_ps.py --ps Tb_maps_prod/H_real/PS_156.out --rin Tb_maps_prod/H_real/RIN_156.hdf5 --box 338300
--box: BoxSize in the render units (ckpc/h; the C code's L). Prints the ratio in a few k ranges.
"""
import argparse
import sys

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ps", required=True)
    p.add_argument("--rin", required=True)
    p.add_argument("--box", type=float, default=338300.0, help="BoxSize [ckpc/h], as the render header")
    a = p.parse_args()

    with open(a.ps) as f:
        head = [next(f) for _ in range(5)]
    z, nb, l3, l4 = float(head[0]), int(head[1]), float(head[2]), float(head[3])
    t = np.loadtxt(a.ps, skiprows=5)
    kC, deltaC, countC = t[:, 0], t[:, 1], t[:, 3]
    D2C = deltaC * (l3 / l4)**2

    with h5py.File(a.rin, "r") as f:
        m = f["rin"][()].astype(np.float64)
    N = m.shape[0]
    L = a.box
    F = np.fft.rfftn(m - m.mean())
    P = np.abs(F)**2 * L**3 / float(N)**6                    # P(k) [map units^2 (length)^3]
    kf = 2 * np.pi / L
    ii = np.fft.fftfreq(N, 1.0 / N)
    kx = np.abs(ii)[:, None, None] * kf
    ky = np.abs(ii)[None, :, None] * kf
    kz = np.arange(N // 2 + 1)[None, None, :] * kf
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    Kmin, Kmax = kf, kf * N / 2
    binfac = nb / (np.log(Kmax) - np.log(Kmin))
    ok = (k >= Kmin) & (k < Kmax)
    b = (np.log(k[ok] / Kmin) * binfac).astype(np.int64)
    cnt = np.bincount(b, minlength=nb)
    Pb = np.bincount(b, P[ok], minlength=nb) / np.maximum(cnt, 1)
    kc = np.exp((np.arange(nb) + 0.5) / binfac) * Kmin
    D2 = kc**3 * Pb / (2 * np.pi**2)

    use = (countC > 0) & (cnt > 0)
    if not np.allclose(cnt[use], countC[use]):
        print(f"WARNING: mode counts differ in {(cnt[use] != countC[use]).sum()} bins (binning convention)")
    r = D2C[use] / D2[use]
    kk = kC[use]
    print(f"z = {z:.4f}, N = {N}, {use.sum()} bins with modes; k in h/ckpc ({kk.min():.2e} .. {kk.max():.2e})")
    for lo, hi in ((kk.min(), 3e-5), (3e-5, 3e-4), (3e-4, 3e-3), (3e-3, kk.max() * 1.01)):
        s = (kk >= lo) & (kk < hi)
        if s.any():
            print(f"  k {lo:.1e}-{hi:.1e} h/ckpc ({lo * 1e3:.3f}-{hi * 1e3:.3f} h/cMpc): Delta^2_C / Delta^2_numpy "
                  f"median {np.median(r[s]):.6f}, range {r[s].min():.6f}-{r[s].max():.6f} ({s.sum()} bins)")
    print("OK" if np.all(np.abs(r - 1) < 1e-3) else "DIFFERENT: the .out normalisation or binning differs from k^3 P/(2 pi^2)")


if __name__ == "__main__":
    main()
